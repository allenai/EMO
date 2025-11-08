import numpy as np
import argparse

from transformers import AutoTokenizer
import logging
import sys
import os


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

def get_file_size(path):
    """Get file size in bytes"""
    return os.path.getsize(path)

def load_token_file(token_path, dtype=np.uint32):
    """
    Load a token file created by dolma.
    These are raw binary files (not standard .npy files with headers).
    """
    file_size = get_file_size(token_path)
    num_tokens = file_size // dtype().itemsize

    # Load as memory-mapped array in read mode
    tokens = np.memmap(token_path, mode='r', dtype=dtype, shape=(num_tokens,))
    return tokens

def prepare_finetuning_masks(args_dict):
    print("yay!")

    # load up the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args_dict["tokenizer"])
    # the string that represents the delimiter
    delimiter_str = "Answer:"
    delimiter_ids = tokenizer(delimiter_str).input_ids

    for token_path in args_dict["token_file_paths"]:
        # Load your token file to get the shape
        tokens = load_token_file(token_path)
        num_tokens = len(tokens)

        # Save the label mask file
        mask_path = str(token_path).replace('.npy', '_mask.npy')

        # Save as memory-mapped file (more efficient for large files):
        mmap_mask = np.memmap(mask_path, mode='w+', dtype=np.bool_, shape=(num_tokens,))


        prev_document = []
        # we now extract individual documents and mask accordingly
        for i in range(num_tokens):
            prev_document.append(tokens[i])
            if tokens[i] == 100257: # we hit the end of a document
                # find the delimiter in the previous document by searching for it
                delimiter_pos = []
                for i in range(len(prev_document)):
                    if prev_document[i:i+len(delimiter_ids)] == delimiter_ids:
                        delimiter_pos.append(i)
                breakpoint()
                assert len(delimiter_pos) == 1, f"Delimiter not found or found multiple times in document with length {len(prev_document)}"

                # create the label mask for the previous document
                label_mask = np.ones(len(prev_document), dtype=np.bool_)
                # mask out everything before the delimiter
                label_mask[:delimiter_pos[0]+len(delimiter_ids)] = False

                # write into mmap_mask
                mmap_mask[i - len(prev_document) + 1:i + 1] = label_mask
                # reset prev_document
                prev_document = []

            else:
                continue

        breakpoint()

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
