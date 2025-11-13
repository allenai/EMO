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

        masks = load_token_file(mask_path, dtype=np.bool_)
        breakpoint()

        # Save as memory-mapped file (more efficient for large files):
        # Initialize to False (masked out) by default
        mmap_mask = np.memmap(mask_path, mode='w+', dtype=np.bool_, shape=(num_tokens,))
        mmap_mask[:] = False  # Explicitly initialize to False

        if "hellaswag" in token_path.lower() or "winogrande" in token_path.lower():
            # special case: hellaswag has no delimiters, we just train on all tokens
            mmap_mask[:] = True
            mmap_mask.flush()
            continue

        prev_document = []
        document_start_idx = 0  # Track where the current document started

        # we now extract individual documents and mask accordingly
        for i in range(num_tokens):
            prev_document.append(tokens[i])

            if tokens[i] == 100257:  # we hit the end of a document
                # find the delimiter in the previous document by searching for it
                delimiter_pos = []
                for j in range(len(prev_document) - len(delimiter_ids) + 1):  # Fixed: don't go past end
                    if prev_document[j:j + len(delimiter_ids)] == delimiter_ids:
                        delimiter_pos.append(j)

                assert len(delimiter_pos) == 1, f"Delimiter not found or found multiple times in document with length {len(prev_document)}"

                # create the label mask for the previous document
                label_mask = np.ones(len(prev_document), dtype=np.bool_)
                # mask out everything before and including the delimiter
                label_mask[:delimiter_pos[0] + len(delimiter_ids)] = False

                # write into mmap_mask - use document_start_idx for correct indexing
                mmap_mask[document_start_idx:document_start_idx + len(prev_document)] = label_mask

                # reset for next document
                document_start_idx = i + 1
                prev_document = []

        # Process the last document if file doesn't end with 100257
        if len(prev_document) > 0:
            logger.warning(
                f"File does not end with document separator (100257). "
                f"Processing last document starting at index {document_start_idx}. Token path: {token_path}"
            )

            # find the delimiter in the last document
            delimiter_pos = []
            for j in range(len(prev_document) - len(delimiter_ids) + 1):
                if prev_document[j:j + len(delimiter_ids)] == delimiter_ids:
                    delimiter_pos.append(j)

            if len(delimiter_pos) == 0:
                raise ValueError(
                    f"Delimiter '{delimiter_str}' not found in last document starting at index {document_start_idx}, "
                    f"length {len(prev_document)}. Token path: {token_path}"
                )
            elif len(delimiter_pos) > 1:
                logger.warning(
                    f"Multiple delimiters found in last document starting at index {document_start_idx}. "
                    f"Using the first one at position {delimiter_pos[0]}. Token path: {token_path}"
                )

            # create the label mask for the last document
            label_mask = np.ones(len(prev_document), dtype=np.bool_)
            # mask out everything before and including the delimiter
            label_mask[:delimiter_pos[0] + len(delimiter_ids)] = False

            # write into mmap_mask
            mmap_mask[document_start_idx:document_start_idx + len(prev_document)] = label_mask

        mmap_mask.flush()

    # check if there are more than one file. If there are, the finetuning script breaks, so we throw an error to remind user
    if len(args_dict["token_file_paths"]) > 1:
        raise ValueError(f"There are a total of {len(args_dict['token_file_paths'])} token files provided for mask generation. The finetuning script currently only supports one file at a time. Be careful (masking script successfully completed)")

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
