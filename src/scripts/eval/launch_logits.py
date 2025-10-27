import argparse
import copy
import inspect
import json
import logging
import os
import re
import subprocess
import sys
from typing import List
import torch.nn.functional as F

from tqdm import tqdm

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

## This is the main launching script for running evaluations on logits.

_parser = argparse.ArgumentParser()
_parser.add_argument("--model", type=str, help="Name of model from model library")
_parser.add_argument(
    "--task", type=str, nargs="+", required=False, help="Task spec(s) from library or jsonl file"
)
_parser.add_argument("--eval-dir", type=str, default=None, help="Directory corresponding to eval directory")
_parser.add_argument("--batch-size", type=str, default=None, help="Override batch size")
_parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs to use")
_parser.add_argument("--use_correct_only", action='store_true', help="Use only correct sequences for evaluation")


logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger()

def find_file(directory, substring):
    """Finds all files in directory that contain substring in their filename."""
    found_arr = []
    for root, _, files in os.walk(directory):
        for file in files:
            if substring in file:
                found_arr += [os.path.join(root, file)]
    return found_arr

def load_jsonl_file(file_path):
    """Loads a jsonl file and returns a list of json objects."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_prompt_sequences_for_evaluation(eval_dataset_name, eval_folder):
    # general matching rule
    requests_files = find_file(eval_folder, f"{eval_dataset_name}-requests")
    predictions_files = find_file(eval_folder, f"{eval_dataset_name}-predictions")

    assert len(requests_files) == 1, f"Found {len(requests_files)} request files for {eval_dataset_name} in {eval_folder}, expected 1"
    assert len(predictions_files) == 1, f"Found {len(predictions_files)} prediction files for {eval_dataset_name} in {eval_folder}, expected 1"

    requests_file = requests_files[0]
    predictions_file = predictions_files[0]

    # load the jsonl file
    requests_data = load_jsonl_file(requests_file)
    predictions_data = load_jsonl_file(predictions_file)

    prompts, correct = [], []

    if eval_dataset_name == "hellaswag:mc":
        assert (len(requests_data) == 4 * len(predictions_data)), f"Found {len(requests_data)} requests and {len(predictions_data)} predictions, expected ratio of 4 times"

        correct_reqs = []

        # loop through the requests, select only the correct ones
        for req in requests_data:
            if req["idx"] != req["label"]:
                continue
            correct_reqs.append(req)

        # assert that the number of correct requests matches the number of predictions
        assert len(correct_reqs) == len(predictions_data), f"Found {len(correct_reqs)} correct requests and {len(predictions_data)} predictions, expected them to match"

        for req, pred in zip(correct_reqs, predictions_data):
            assert req['doc_id'] == pred['doc_id'], f"Request doc_id {req['doc_id']} does not match prediction doc_id {pred['doc_id']}"
            prompts += [req["request"]["context"] + req["request"]["continuation"]]
            correct += [1 if pred["metrics"]["acc_raw"] > 0 else 0]

    return prompts, correct


def launch_logits(args_dict):
    print("yay!")

    # load the model
    tokenizer = AutoTokenizer.from_pretrained(args_dict["model"])
    model = AutoModelForCausalLM.from_pretrained(args_dict["model"], device_map="auto", torch_dtype="auto")

    # we load the data here
    for eval_dataset_name in args_dict["task"]:
        print("evaluating dataset ", eval_dataset_name)
        prompts, correct = get_prompt_sequences_for_evaluation(eval_dataset_name, args_dict["eval_dir"])

        out_fn = os.path.join(args_dict["eval_dir"], f"{eval_dataset_name}-router.jsonl")

        out_file = open(out_fn, 'w')

        # initialize storage for summed router probabilities
        breakpoint()
        summed_router_probabilities = None

        # loop over dataset in batches
        for i in tqdm(range(0, len(prompts), args_dict["batch_size"])):
            batch_prompts = prompts[i:i+args_dict["batch_size"]]
            batch_correct = correct[i:i+args_dict["batch_size"]]

            # we perform forward pass on prompts
            inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, return_offsets_mapping=True).to(model.device)

            with torch.no_grad():
                out = model(input_ids = inputs["input_ids"].to(model.device), attention_mask=inputs["attention_mask"].to(model.device), output_router_logits=True)
                router_logits = [x.cpu() for x in out["router_logits"]]
                router_logits = torch.stack(router_logits) # this has dimension (layers, batch * sequence_length, num_experts)

            del out
            torch.cuda.empty_cache()

            # reshape router_logits
            router_logits = router_logits.view(router_logits.shape[0], inputs.input_ids.shape[0], inputs.input_ids.shape[1], router_logits.shape[-1]) # (layers, batch, sequence_length, num_experts)

            # select only the correct sequences in the batch if specified
            if args_dict["use_correct_only"]:
                correct_indices = [j for j, val in enumerate(batch_correct) if val == 1]
                if len(correct_indices) == 0:
                    continue
                router_logits = router_logits[:, correct_indices, :, :]
                batch_prompts = [batch_prompts[j] for j in correct_indices]

            # aggregate router probabilities across batch and sequence length
            router_probabilities = F.softmax(router_logits, dim=-1)

            # zero out all the padding tokens
            attention_mask_expanded = inputs.attention_mask.cpu().unsqueeze(0).unsqueeze(-1).expand(router_probabilities.shape[0], router_probabilities.shape[1], router_probabilities.shape[2], router_probabilities.shape[3]) # (layers, batch, sequence_length, num_experts)
            router_probabilities = router_probabilities * attention_mask_expanded

            summed_router_probabilities = router_probabilities.sum(dim=(1,2)) # (layers, num_experts)


            # # store the logits
            # record = {
            #     "token_index": token_index,
            #     "router_logits": prompt_router_logits
            # }

            # out_file.write(json.dumps(record) + "\n")
            # out_file.flush()

        out_file.close()


def main():
    args = _parser.parse_args()
    args_dict = vars(args)
    maybe_rc = launch_logits(args_dict)
    try:
        rc = int(maybe_rc)
        sys.exit(rc)
    except Exception:
        # not a return code
        pass

if __name__ == "__main__":
    main()
