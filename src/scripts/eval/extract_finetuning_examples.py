import argparse
import json
import logging
import os
import sys

from tqdm import tqdm

from src.offline_evals.eval_utils import get_eval_filename, load_jsonl_file

## This is the main launching script for running evaluations on logits.

_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--task", type=str, nargs="+", required=False, help="Task spec(s) from library or jsonl file"
)
_parser.add_argument(
    "--eval-dir", type=str, default=None, help="Directory corresponding to outputted requests"
)
_parser.add_argument(
    "--token-dir",
    type=str,
    default=None,
    help="Directory corresponding to outputted processed jsonl",
)


logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger()


def get_correct_training_data(eval_dataset_name, eval_folder):
    # general matching rule
    requests_file = os.path.join(eval_folder, f"{eval_dataset_name}-requests.jsonl")

    # load the jsonl file
    requests_data = load_jsonl_file(requests_file)

    data = []

    if requests_data[0]["request_type"] == "loglikelihood":
        correct_reqs = []

        # loop through the requests, select only the correct ones
        for req in requests_data:
            if "gsm8k" not in eval_dataset_name:
                if req["idx"] != req["label"]:
                    continue
                if req["request"]["context"].startswith("Answer:"):
                    continue
            correct_reqs.append(req)

        for req in correct_reqs:
            data += [req["request"]["context"] + req["request"]["continuation"]]
    elif requests_data[0]["request_type"] == "generate_until":
        for req in requests_data:
            # for some tasks (e.g coqa), by default there is no space between context and choice, so we add it here
            if req["request"]["context"][-1] != " " and req["doc"]["choices"][0][0] != " ":
                data += [req["request"]["context"] + " " + req["doc"]["choices"][0]]
            else:
                data += [req["request"]["context"] + req["doc"]["choices"][0]]
    else:
        raise NotImplementedError(
            f"Dataset {eval_dataset_name} with request type {requests_data[0]['request_type']} not implemented in get_prompt_sequences_for_evaluation"
        )

    return data


def extract_finetuning_examples(args_dict):
    print("yay!")

    if not any(args_dict["task"]):
        print("No tasks specified, exiting...")
        return 0

    # we load the data here
    for eval_dataset_name in args_dict["task"]:
        print("evaluating dataset ", eval_dataset_name)

        # get the request file for the corresponding task
        eval_dataset_name = get_eval_filename(eval_dataset_name)

        out_fn = os.path.join(args_dict["eval_dir"], f"{eval_dataset_name}-processed.jsonl")
        if os.path.exists(out_fn):
            print(f"Output file {out_fn} already exists, skipping...")
            continue

        data = get_correct_training_data(eval_dataset_name, args_dict["eval_dir"])

        out_file = open(out_fn, "w")

        # loop over dataset in batches
        for i in tqdm(data):
            out_file.write(
                json.dumps({"text": i, "id": str(hash(i)), "source": eval_dataset_name}) + "\n"
            )
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
