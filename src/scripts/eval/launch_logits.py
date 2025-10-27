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

from tqdm import tqdm

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

## This is the main launching script for running evaluations on logits.

_parser = argparse.ArgumentParser()
_parser.add_argument("--model", type=str, help="Name of model from model library")
_parser.add_argument(
    "--task", type=str, nargs="+", required=False, help="Task spec(s) from library or jsonl file"
)
_parser.add_argument("--output-dir", type=str, default=None, help="Directory for output files")
_parser.add_argument("--batch-size", type=str, default=None, help="Override batch size")
_parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs to use")


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

    assert len(requests_files) == 1, f"Found {len(requests_files)} request files for gsm8k in {eval_folder}, expected 1"
    assert len(predictions_files) == 1, f"Found {len(predictions_files)} prediction files for gsm8k in {eval_folder}, expected 1"

    requests_file = requests_files[0]
    predictions_file = predictions_files[0]

    # load the jsonl file
    requests_data = load_jsonl_file(requests_file)
    predictions_data = load_jsonl_file(predictions_file)
    assert (len(requests_data) == len(predictions_data)), f"Found {len(requests_data)} requests and {len(predictions_data)} predictions, expected same number"

    # we now create the prompt sequences
    prompts = [] # records the full forward pass
    index = [] # records when we switch to model answers
    breakpoint()

    for req, pred in zip(requests_data, predictions_data):
        assert req['native_id'] == pred['native_id'], f"Request id {req['id']} does not match prediction id {pred['id']}"
        assert len(pred["model_output"]) == 1, f"Found {len(pred['model_output'])} model outputs for prediction id {pred['id']}, expected 1"
        prompts += [req["request"]["context"] + pred["model_output"][0]["continuation"]]
        index += [len(req["request"]["context"])]

    return prompts, index


def launch_logits(args_dict):
    print("yay!")

    breakpoint()

    # load the model
    tokenizer = AutoTokenizer.from_pretrained(args_dict["model"])
    model = AutoModelForCausalLM.from_pretrained(args_dict["model"], device_map="auto", torch_dtype="auto")

    # we load the data here
    for eval_dataset_name in args_dict["task"]:
        print("evaluating dataset ", eval_dataset_name)
        prompts, index = get_prompt_sequences_for_evaluation(eval_dataset_name, args_dict["output_dir"])

        out_fn = os.path.join(args_dict["output_dir"], f"{eval_dataset_name}-router.jsonl")

        out_file = open(out_fn, 'w')

        # loop over dataset in batches

        for i in tqdm(range(0, len(prompts), args_dict["batch_size"])):
            batch_prompts = prompts[i:i+args_dict["batch_size"]]
            batch_index = index[i:i+args_dict["batch_size"]]

            # we perform forward pass on prompts
            inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, return_offsets_mapping=True).to(model.device)

            # helper function to get the deliminator for input_ids
            def get_token_delimitor(offsets, char_index):
                for i, (start, end) in enumerate(offsets):
                    if start <= char_index < end:
                        return i
                return len(offsets) - 1

            # we record the token indexes that represent transition from input to output
            batch_token_index = []
            for j, char_index in enumerate(batch_index):
                offsets = inputs['offset_mapping'][j].tolist()
                token_index = get_token_delimitor(offsets, char_index)
                assert offsets[token_index][0] == char_index, f"char_index {char_index} does not match token start {offsets[token_index][0]}"
                batch_token_index += [token_index]

            with torch.no_grad():
                out = model(input_ids = inputs["input_ids"].to(model.device), attention_mask=inputs["attention_mask"].to(model.device), output_router_logits=True)
                router_logits = [x.cpu() for x in out["router_logits"]]
                router_logits = torch.stack(router_logits) # this has dimension (layers, batch * sequence_length, num_experts)

            del out
            torch.cuda.empty_cache()

            # reshape router_logits
            router_logits = router_logits.view(router_logits.shape[0], inputs.input_ids.shape[0], inputs.input_ids.shape[1], router_logits.shape[-1]) # (layers, batch, sequence_length, num_experts)

            # we now extract all router logits and save them
            for j in range(len(batch_prompts)):
                prompt = batch_prompts[j]
                token_index = batch_token_index[j]
                prompt_router_logits = router_logits[:, j, token_index:, :].cpu().numpy().tolist()

                # store the logits
                record = {
                    "prompt": prompt,
                    "token_index": token_index,
                    "router_logits": prompt_router_logits
                }

                out_file.write(json.dumps(record) + "\n")
                out_file.flush()

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
