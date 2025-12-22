import argparse
import copy
import inspect
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from typing import List
import torch.nn.functional as F
from oe_eval.utilities.remote_utils import cache_s3_folder, upload_directory

from tqdm import tqdm

import torch

from src.offline_evals.eval_utils import find_file, load_jsonl_file, get_eval_filename
from transformers import AutoTokenizer, AutoModelForCausalLM

## This is the main launching script for running evaluations on logits.

_parser = argparse.ArgumentParser()
_parser.add_argument("--model", type=str, help="Name of model from model library")
_parser.add_argument(
    "--task", type=str, nargs="+", required=False, help="Task spec(s) from library or jsonl file"
)
_parser.add_argument("--eval-dir", type=str, default=None, help="Directory corresponding to eval directory")
_parser.add_argument("--output-dir", type=str, default=None, help="Directory to save outputs. Should be model dependent")
_parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
_parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs to use")
_parser.add_argument("--use_correct_only", action='store_true', help="Use only correct sequences for evaluation")


logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger()



def get_prompt_sequences_for_evaluation(eval_dataset_name, eval_folder):
    # general matching rule
    requests_file = os.path.join(eval_folder, f"{eval_dataset_name}-requests.jsonl")
    predictions_file = os.path.join(eval_folder, f"{eval_dataset_name}-predictions.jsonl")

    # load the jsonl file
    requests_data = load_jsonl_file(requests_file)
    predictions_data = load_jsonl_file(predictions_file)

    prompts, correct = [], []

    if requests_data[0]["request_type"] == "loglikelihood":
        correct_reqs = []

        # loop through the requests, select only the correct ones and ones with correct context (rc tasks contain requests that exclude question)
        for req in requests_data:
            if "gsm8k" not in eval_dataset_name:
                # do filtering if not gsm8k (for gsm8k it's label is "null" and we don't have to filter it since everything is gold label
                if req["idx"] != req["label"]:
                    continue
                if req["request"]["context"].startswith("Answer:"):
                    continue
            correct_reqs.append(req)

        # assert that the number of correct requests matches the number of predictions
        assert len(correct_reqs) == len(predictions_data), f"Found {len(correct_reqs)} correct requests and {len(predictions_data)} predictions, expected them to match"

        for req, pred in zip(correct_reqs, predictions_data):
            assert req['doc_id'] == pred['doc_id'], f"Request doc_id {req['doc_id']} does not match prediction doc_id {pred['doc_id']}"
            prompts += [req["request"]["context"] + req["request"]["continuation"]]
            if "gsm8k" not in eval_dataset_name:
                correct += [1 if pred["metrics"]["acc_raw"] > 0 else 0]
            else:
                correct += [1] # gsm8k is always correct
    elif requests_data[0]["request_type"] == "generate_until":
        # assert that the number of correct requests matches the number of predictions. Note that for generate_until metrics, there is no incorrect metric
        assert len(requests_data) == len(
            predictions_data), f"Found {len(requests_data)} correct requests and {len(predictions_data)} predictions, expected them to match"

        for req, pred in zip(requests_data, predictions_data):
            assert req['doc_id'] == pred[
                'doc_id'], f"Request doc_id {req['doc_id']} does not match prediction doc_id {pred['doc_id']}"
            prompts += [req["request"]["context"] + req["doc"]["choices"][0]]
            correct += [1 if pred["metrics"]["f1"] > 0 else 0]
    else:
        raise NotImplementedError(f"Dataset {eval_dataset_name} not implemented in get_prompt_sequences_for_evaluation")

    return prompts, correct

def resolve_tasks(task_list):
    from src.scripts.eval.task_suites import get_task_suite_configs
    TASK_SUITE_CONFIGS = get_task_suite_configs()

    resolved_tasks = []
    for task in task_list:
        if task in TASK_SUITE_CONFIGS:
            resolved_tasks.extend(TASK_SUITE_CONFIGS[task]["tasks"])
        else:
            resolved_tasks.append(task)
    return resolved_tasks

def launch_logits(args_dict):
    print("yay!")

    # load the model
    print("loading model and tokenizer: ", args_dict["model"])
    tokenizer = AutoTokenizer.from_pretrained(args_dict["model"])
    model = AutoModelForCausalLM.from_pretrained(args_dict["model"], device_map="auto", torch_dtype="auto")

    # resolve task suites if needed
    args_dict["task"] = resolve_tasks(args_dict["task"])

    # we load the data here
    for eval_dataset_name in args_dict["task"]:
        print("evaluating dataset ", eval_dataset_name)

        # retrieve eval-dir if eval-dir is s3
        if args_dict["eval_dir"].startswith("s3://"):
            eval_dir = cache_s3_folder(args_dict["eval_dir"])["local_dir"]
        else:
            eval_dir = args_dict["eval_dir"]

        # get the request file for the corresponding task
        eval_dataset_name = get_eval_filename(eval_dataset_name)

        print("TEST DATASET NAME after get_eval_filename: ", eval_dataset_name)

        prompts, correct = get_prompt_sequences_for_evaluation(eval_dataset_name, eval_dir)

        if args_dict["output_dir"].startswith("s3://"):
            output_dir = tempfile.mkdtemp()
        else:
            output_dir = args_dict["output_dir"]
            os.makedirs(output_dir, exist_ok=True)

        out_fn = os.path.join(output_dir, f"{eval_dataset_name}-router.jsonl")

        if os.path.exists(out_fn):
            print(f"Output file {out_fn} already exists, skipping...")
            continue

        out_file = open(out_fn, 'w')

        # initialize storage for summed router probabilities
        num_layers = model.config.num_hidden_layers
        num_experts = model.config.num_experts
        tot_router_probabilities = torch.zeros((num_layers, num_experts))
        tot_tokens = 0

        print(f"Processing {len(prompts)} sequences...")
        # select only the correct sequences in the batch if specified
        if args_dict["use_correct_only"]:
            prompts = [prompts[j] for j, val in enumerate(correct) if val == 1]

        print(f"Use correct only is {args_dict["use_correct_only"]}, {len(prompts)} sequences remain.")

        # loop over dataset in batches
        for i in tqdm(range(0, len(prompts), args_dict["batch_size"])):
            batch_prompts = prompts[i:i+args_dict["batch_size"]]

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

            # aggregate router probabilities across batch and sequence length
            router_probabilities = F.softmax(router_logits, dim=-1)

            # zero out all the padding tokens
            attention_mask_expanded = inputs.attention_mask.cpu().unsqueeze(0).unsqueeze(-1).expand(router_probabilities.shape[0], router_probabilities.shape[1], router_probabilities.shape[2], router_probabilities.shape[3]) # (layers, batch, sequence_length, num_experts)
            router_probabilities = router_probabilities * attention_mask_expanded

            summed_router_probabilities = router_probabilities.sum(dim=(1,2)) # (layers, num_experts)
            # accumulate the summed router probabilities
            tot_router_probabilities += summed_router_probabilities

            tot_tokens += inputs.attention_mask.sum().item()

        # after processing all batches, we compute average router probabilities
        save_router_probabilities = tot_router_probabilities / tot_tokens
        out_file.write(json.dumps({"avg_router_probabilities": save_router_probabilities.tolist()}) + "\n")
        out_file.close()

        if args_dict["output_dir"].startswith("s3://"):
            upload_directory(output_dir, args_dict["output_dir"])


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
