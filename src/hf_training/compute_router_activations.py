"""
Compute router activations on validation set for expert pruning.

This script computes average router probabilities across validation examples,
which are used to determine which experts to keep when pruning.

Usage:
    python -m src.hf_training.compute_router_activations \
        --model allenai/OLMoE-1B-7B-0924 \
        --task gsm8k \
        --split validation \
        --output-file activations.jsonl \
        --batch-size 4
"""

import argparse
import json
import logging
import os
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.hf_training.data_utils import get_formatted_prompts
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_router_activations(
    model_name: str,
    task_name: str,
    split: str,
    output_file: str,
    batch_size: int = 4,
    num_calibration: Optional[int] = None,
    device: Optional[str] = None,
    use_correct_only: bool = False,
    num_shots_override: Optional[int] = None,
    prune_seed: int = 0,
) -> dict:
    """
    Compute average router probabilities across validation examples.

    Args:
        model_name: HuggingFace model name or path
        task_name: Name of the task (gsm8k, mmlu, squad, coqa)
        split: Dataset split (typically 'validation' or 'test')
        output_file: Path to save the activation file
        batch_size: Batch size for inference
        device: Device to use (auto if None)
        use_correct_only: If True, only use correct predictions (not implemented for HF datasets)

    Returns:
        Dict containing avg_router_probabilities
    """
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device is None else device,
        torch_dtype="auto",
    )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get formatted prompts from the dataset
    logger.info(
        f"Loading dataset: {task_name} ({split})"
        + (f" [num_shots={num_shots_override}]" if num_shots_override is not None else "")
    )
    prompts, _ = get_formatted_prompts(task_name, split, num_shots_override=num_shots_override)
    if num_calibration is None:
        logger.info(f"Loaded {len(prompts)} prompts, using all (no subsampling)")
    else:
        n_keep = min(num_calibration, len(prompts))
        logger.info(f"Loaded {len(prompts)} prompts, subsampling to {n_keep} (seed={prune_seed})")
        g = torch.Generator().manual_seed(prune_seed)
        perm = torch.randperm(len(prompts), generator=g).tolist()
        prompts = [prompts[i] for i in perm[:n_keep]]

    # Get model config
    num_layers = model.config.num_hidden_layers
    num_experts = (
        model.config.num_local_experts
        if hasattr(model.config, "num_local_experts")
        else model.config.num_experts
    )

    logger.info(f"Model has {num_layers} layers and {num_experts} experts")

    # Initialize storage for summed router probabilities
    tot_router_probabilities = torch.zeros((num_layers, num_experts))
    tot_tokens = 0

    # Process in batches
    logger.info(f"Processing {len(prompts)} sequences with batch size {batch_size}...")

    for i in tqdm(range(0, len(prompts), batch_size), desc="Computing activations"):
        batch_prompts = prompts[i : i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(model.device)

        # Forward pass with router logits
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_router_logits=True,
            )

            # Get router logits - shape: tuple of (batch * seq_len, num_experts) per layer
            router_logits = outputs.router_logits

            # Stack and reshape: (num_layers, batch * seq_len, num_experts)
            router_logits_stacked = torch.stack([r.cpu() for r in router_logits])

            # Reshape to (num_layers, batch, seq_len, num_experts)
            batch_size_actual = inputs.input_ids.shape[0]
            seq_len = inputs.input_ids.shape[1]
            router_logits_reshaped = router_logits_stacked.view(
                num_layers, batch_size_actual, seq_len, num_experts
            )

            # we now take out the shared experts if num_shared_experts > 0
            if model.config.num_shared_experts > 0:
                router_logits_reshaped_standard = router_logits_reshaped[
                    :, :, :, : num_experts - model.config.num_shared_experts
                ]
                router_logits_reshaped_shared = router_logits_reshaped[
                    :, :, :, num_experts - model.config.num_shared_experts :
                ]

                router_probabilities_standard = F.softmax(router_logits_reshaped_standard, dim=-1)
                router_probabilities_shared = F.softmax(router_logits_reshaped_shared, dim=-1)

                # Mask out padding tokens
                attention_mask_expanded_standard = (
                    inputs.attention_mask.cpu()
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .expand(
                        num_layers,
                        batch_size_actual,
                        seq_len,
                        num_experts - model.config.num_shared_experts,
                    )
                )
                attention_mask_expanded_shared = (
                    inputs.attention_mask.cpu()
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .expand(num_layers, batch_size_actual, seq_len, model.config.num_shared_experts)
                )

                router_probabilities_standard = (
                    router_probabilities_standard * attention_mask_expanded_standard
                )
                router_probabilities_shared = (
                    router_probabilities_shared * attention_mask_expanded_shared
                )

                summed_router_probabilities_standard = router_probabilities_standard.sum(dim=(1, 2))
                summed_router_probabilities_shared = router_probabilities_shared.sum(dim=(1, 2))

                tot_router_probabilities[
                    :, : num_experts - model.config.num_shared_experts
                ] += summed_router_probabilities_standard
                tot_router_probabilities[
                    :, num_experts - model.config.num_shared_experts :
                ] += summed_router_probabilities_shared

                tot_tokens += inputs.attention_mask.sum().item()
            else:
                # Convert to probabilities
                router_probabilities = F.softmax(router_logits_reshaped, dim=-1)

                # Mask out padding tokens
                attention_mask_expanded = (
                    inputs.attention_mask.cpu()
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .expand(num_layers, batch_size_actual, seq_len, num_experts)
                )
                router_probabilities = router_probabilities * attention_mask_expanded

                # Sum across batch and sequence length: (num_layers, num_experts)
                summed_router_probabilities = router_probabilities.sum(dim=(1, 2))
                tot_router_probabilities += summed_router_probabilities

                tot_tokens += inputs.attention_mask.sum().item()

        # Clean up
        del outputs
        torch.cuda.empty_cache()

    # Compute average
    avg_router_probabilities = tot_router_probabilities / tot_tokens
    logger.info(f"Processed {tot_tokens} total tokens")

    # Save to file
    result = {"avg_router_probabilities": avg_router_probabilities.tolist()}

    os.makedirs(
        os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True
    )
    with open(output_file, "w") as f:
        f.write(json.dumps(result) + "\n")

    logger.info(f"Saved activations to {output_file}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Compute router activations for expert pruning")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name (gsm8k, mmlu, squad, coqa)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split (default: validation)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save activation file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference (default: 4)",
    )
    parser.add_argument(
        "--num-calibration",
        type=int,
        default=None,
        help="Subsample calibration set to this many prompts (default: use all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=None,
        help="Override task config's num_shots (default: use config value)",
    )
    parser.add_argument(
        "--prune-seed",
        type=int,
        default=0,
        help="Seed for the calibration-subsample permutation (default: 0)",
    )

    args = parser.parse_args()

    compute_router_activations(
        model_name=args.model,
        task_name=args.task,
        split=args.split,
        output_file=args.output_file,
        batch_size=args.batch_size,
        num_calibration=args.num_calibration,
        device=args.device,
        num_shots_override=args.num_shots,
        prune_seed=args.prune_seed,
    )


if __name__ == "__main__":
    main()
