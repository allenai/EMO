import numpy as np
import argparse

from transformers import AutoTokenizer

## This is the main launching script for creating loss masks for finetuning

_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--token_file_paths", type=str, nargs="+", required=False, help="Task spec(s) from library or jsonl file"
)
_parser.add_argument(
    "--tokenizer", type=str, default=None, help="Directory corresponding to outputted requests"
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
            if req["idx"] != req["label"]:
                continue
            if req["request"]["context"].startswith("Answer:"):
                continue
            correct_reqs.append(req)

        for req in correct_reqs:
            data += [req["request"]["context"] + req["request"]["continuation"]]
    else:
        raise NotImplementedError(f"Dataset {eval_dataset_name} not implemented in get_prompt_sequences_for_evaluation")

    return data


def prepare_finetuning_masks(args_dict):
    print("yay!")

    # load up the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args_dict["tokenizer"])
    # the string that represents the delimiter
    delimiter_str = "\nAnswer:"
    delimiter_token_ids = tokenizer(delimiter_str).input_ids

    for token_path in args_dict["token_file_paths"]:
        # Load your token file to get the shape
        tokens = np.load(token_path)
        num_tokens = len(tokens)

        # Create the label mask array
        # Initialize all to True (include all tokens by default)
        label_mask = np.ones(num_tokens, dtype=np.bool_)

        breakpoint()





        # Or mask based on token values (e.g., mask padding tokens):
        # pad_token_id = 0
        # label_mask = (tokens != pad_token_id)

        # Save the label mask file
        mask_path = str(token_path).replace('.npy', '_mask.npy')

        # Save as memory-mapped file (more efficient for large files):
        mmap_mask = np.memmap(mask_path, mode='w+', dtype=np.bool_, shape=(num_tokens,))
        mmap_mask[:] = label_mask
        mmap_mask.flush()

    #
    # # we load the data here
    # for eval_dataset_name in args_dict["task"]:
    #     print("evaluating dataset ", eval_dataset_name)
    #
    #     # get the request file for the corresponding task
    #     eval_dataset_name = get_eval_filename(eval_dataset_name)
    #
    #     out_fn = os.path.join(args_dict["eval_dir"], f"{eval_dataset_name}-processed.jsonl")
    #     if os.path.exists(out_fn):
    #         print(f"Output file {out_fn} already exists, skipping...")
    #         continue
    #
    #     data = get_correct_training_data(eval_dataset_name, args_dict["eval_dir"])
    #
    #     out_file = open(out_fn, 'w')
    #
    #     # loop over dataset in batches
    #     for i in tqdm(data):
    #         out_file.write(json.dumps({"text": i, "id": str(hash(i)), "source": eval_dataset_name}) + "\n")
    #     out_file.close()

def main():
    args = _parser.parse_args()
    args_dict = vars(args)
    maybe_rc = prepare_finetuning_masks(args_dict)
    try:
        rc = int(maybe_rc)
        sys.exit(rc)
    except Exception:
        # not a return code
        pass

if __name__ == "__main__":
    main()
