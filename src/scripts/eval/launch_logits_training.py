import argparse
import json
import logging
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from oe_eval.utilities.remote_utils import upload_directory
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from src.olmo_core.data.mixes import DataMix

## This script computes average router probabilities on training data mixes.
## Unlike launch_logits.py (which works on eval task data), this loads
## pre-tokenized .npy files from a DataMix (e.g. code_mix).

_parser = argparse.ArgumentParser()
_parser.add_argument("--model", type=str, required=True, help="Name or path of model")
_parser.add_argument("--mix", type=str, required=True, help="DataMix name (e.g. code_mix)")
_parser.add_argument(
    "--mix-base-dir",
    type=str,
    default="/weka/oe-training-default/ai2-llm",
    help="Base directory where mix data is stored",
)
_parser.add_argument(
    "--tokenizer",
    type=str,
    default="allenai/dolma2-tokenizer",
    help="Tokenizer identifier used when building the mix paths",
)
_parser.add_argument(
    "--source-label",
    type=str,
    default=None,
    help="If set, only process paths with this source label (e.g. code_fim, swallowcode)",
)
_parser.add_argument(
    "--output-dir",
    type=str,
    required=True,
    help="Directory to save outputs",
)
_parser.add_argument("--batch-size", type=int, default=4, help="Number of sequences per batch")
_parser.add_argument("--seq-length", type=int, default=4096, help="Sequence length for chunking")
_parser.add_argument(
    "--max-tokens",
    type=int,
    default=None,
    help="Max tokens to process (deterministic: always takes the first N tokens from sorted paths)",
)
_parser.add_argument(
    "--seed", type=int, default=42, help="Random seed (unused currently, reserved)"
)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger()


def load_token_ids_from_paths(paths, seq_length, max_tokens=None, seed=42):
    """
    Load pre-tokenized .npy files and chunk into fixed-length sequences.

    Paths are sorted first then shuffled with the given seed, so the same
    seed + max_tokens always selects the exact same set of tokens.

    Returns a list of np.ndarray, each of shape (seq_length,).
    """
    # Sort first for a canonical order, then shuffle deterministically with seed
    rng = np.random.RandomState(seed)
    shuffled_paths = sorted(paths)
    rng.shuffle(shuffled_paths)

    sequences = []
    leftover = np.array([], dtype=np.uint32)
    total_tokens_loaded = 0

    for path in tqdm(shuffled_paths, desc="Loading .npy files"):
        tokens = np.fromfile(path, dtype=np.uint32)

        # If we have a max_tokens cap, figure out how many more we need
        if max_tokens is not None:
            remaining = max_tokens - total_tokens_loaded
            if remaining <= 0:
                break
            if len(tokens) > remaining:
                tokens = tokens[:remaining]

        total_tokens_loaded += len(tokens)

        # Prepend any leftover from previous file
        if len(leftover) > 0:
            tokens = np.concatenate([leftover, tokens])
            leftover = np.array([], dtype=np.uint32)

        # Chunk into sequences of seq_length
        num_full_sequences = len(tokens) // seq_length
        for i in range(num_full_sequences):
            sequences.append(tokens[i * seq_length : (i + 1) * seq_length])

        # Save remainder for next file
        leftover = tokens[num_full_sequences * seq_length :]

    # Discard final leftover (partial sequence)
    logger.info(
        f"Loaded {total_tokens_loaded:,} tokens -> {len(sequences)} sequences "
        f"of length {seq_length} (discarded {len(leftover)} leftover tokens)"
    )
    return sequences


def launch_logits_training(args_dict):
    # Build data mix paths
    mix = DataMix(args_dict["mix"])
    paths, labels = mix.build(base_dir=args_dict["mix_base_dir"], tokenizer=args_dict["tokenizer"])

    logger.info(f"Mix '{args_dict['mix']}' has {len(paths)} paths")

    # Group paths by label
    label_to_paths: dict[str, list[str]] = {}
    for path, label in zip(paths, labels):
        label_to_paths.setdefault(label, []).append(path)

    unique_labels = sorted(label_to_paths.keys())
    logger.info(f"Source labels: {unique_labels}")

    # Filter to requested source label if specified
    if args_dict["source_label"] is not None:
        if args_dict["source_label"] not in label_to_paths:
            logger.error(
                f"Source label '{args_dict['source_label']}' not found. "
                f"Available: {unique_labels}"
            )
            sys.exit(1)
        label_to_paths = {args_dict["source_label"]: label_to_paths[args_dict["source_label"]]}
        logger.info(
            f"Filtered to source label '{args_dict['source_label']}': "
            f"{len(label_to_paths[args_dict['source_label']])} paths"
        )

    # Load model
    logger.info(f"Loading model: {args_dict['model']}")
    model = AutoModelForCausalLM.from_pretrained(
        args_dict["model"], device_map="auto", torch_dtype="auto"
    )

    num_layers = model.config.num_hidden_layers
    num_experts = model.config.num_experts

    # Process each source label
    for label, label_paths in sorted(label_to_paths.items()):
        logger.info(f"Processing source label '{label}' ({len(label_paths)} files)")

        if args_dict["output_dir"].startswith("s3://"):
            output_dir = tempfile.mkdtemp()
        else:
            output_dir = args_dict["output_dir"]
            os.makedirs(output_dir, exist_ok=True)

        out_fn = os.path.join(output_dir, f"{args_dict['mix']}-{label}-router.jsonl")

        if os.path.exists(out_fn):
            logger.info(f"Output file {out_fn} already exists, skipping...")
            continue

        sequences = load_token_ids_from_paths(
            label_paths,
            seq_length=args_dict["seq_length"],
            max_tokens=args_dict["max_tokens"],
            seed=args_dict["seed"],
        )

        if len(sequences) == 0:
            logger.warning(f"No sequences loaded for label '{label}', skipping.")
            continue

        # Initialize storage for summed router probabilities
        tot_router_probabilities = torch.zeros((num_layers, num_experts))
        tot_tokens = 0

        logger.info(f"Processing {len(sequences)} sequences for label '{label}'...")

        batch_size = args_dict["batch_size"]
        for i in tqdm(range(0, len(sequences), batch_size), desc=f"Batches ({label})"):
            batch_seqs = sequences[i : i + batch_size]
            input_ids = torch.tensor(np.stack(batch_seqs), dtype=torch.long).to(model.device)
            # No padding needed — all sequences are exactly seq_length
            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_router_logits=True,
                )
                router_logits_list = [x.cpu() for x in out["router_logits"]]
                router_logits = torch.stack(
                    router_logits_list
                )  # (layers, batch * seq_length, num_experts)

            del out
            torch.cuda.empty_cache()

            # Reshape: (layers, batch, seq_length, num_experts)
            router_logits = router_logits.view(
                router_logits.shape[0],
                input_ids.shape[0],
                input_ids.shape[1],
                router_logits.shape[-1],
            )

            router_probabilities = F.softmax(router_logits, dim=-1)

            summed_router_probabilities = router_probabilities.sum(
                dim=(1, 2)
            )  # (layers, num_experts)
            tot_router_probabilities += summed_router_probabilities
            tot_tokens += input_ids.numel()

        # Compute average router probabilities
        save_router_probabilities = tot_router_probabilities / tot_tokens
        with open(out_fn, "w") as out_file:
            out_file.write(
                json.dumps({"avg_router_probabilities": save_router_probabilities.tolist()}) + "\n"
            )
        logger.info(f"Saved router probabilities to {out_fn} ({tot_tokens:,} tokens processed)")

        if args_dict["output_dir"].startswith("s3://"):
            upload_directory(output_dir, args_dict["output_dir"])


def main():
    args = _parser.parse_args()
    args_dict = vars(args)
    launch_logits_training(args_dict)


if __name__ == "__main__":
    main()
