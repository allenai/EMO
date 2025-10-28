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

from src.offline_evals.eval_utils import find_file, load_jsonl_file, find_task_substring
from transformers import AutoTokenizer, AutoModelForCausalLM

## This is the main launching script for running evaluations on logits.

_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--task", type=str, nargs="+", required=False, help="Task spec(s) from library or jsonl file"
)
_parser.add_argument("--input-dir", type=str, default=None, help="Directory corresponding to outputted requests")
_parser.add_argument("--output-dir", type=str, default=None, help="Directory corresponding to outputted processed jsonl")


logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger()

def get_correct_training_data(eval_dataset_name, eval_folder):
    # general matching rule
    requests_files = find_file(eval_folder, f"{eval_dataset_name}-requests")
    assert len(requests_files) == 1, f"Found {len(requests_files)} request files for {eval_dataset_name} in {eval_folder}, expected 1"

    requests_file = requests_files[0]

    # load the jsonl file
    requests_data = load_jsonl_file(requests_file)

    data = []

    if requests_data[0]["request_type"] == "loglikelihood":
        correct_reqs = []

        # loop through the requests, select only the correct ones
        for req in requests_data:
            if req["idx"] != req["label"]:
                continue
            correct_reqs.append(req)

        for req in correct_reqs:
            data += [req["request"]["context"] + req["request"]["continuation"]]
    else:
        raise NotImplementedError(f"Dataset {eval_dataset_name} not implemented in get_prompt_sequences_for_evaluation")

    return data


def extract_finetuning_examples(args_dict):
    print("yay!")

    # we load the data here
    for eval_dataset_name in args_dict["task"]:
        print("evaluating dataset ", eval_dataset_name)

        # convert dataset name to substring
        eval_dataset_name = find_task_substring(eval_dataset_name)

        data = get_correct_training_data(eval_dataset_name, args_dict["input_dir"])

        os.makedirs(args_dict["output_dir"], exist_ok=True)

        out_fn = os.path.join(args_dict["output_dir"], "out.jsonl")

        out_file = open(out_fn, 'w')

        # loop over dataset in batches
        for i in tqdm(data):
            out_file.write(json.dumps({"text": i, "id": str(hash(i)), "source": eval_dataset_name}) + "\n")
        out_file.close()

def main():
    args = _parser.parse_args()
    args_dict = vars(args)
    maybe_rc = extract_finetuning_examples(args_dict)
    try:
        rc = int(maybe_rc)
        sys.exit(rc)
    except Exception:
        # not a return code
        pass

if __name__ == "__main__":
    main()
