"""
Main finetuning script using HuggingFace Trainer.

Features:
- Load pruned HF model
- Load train split from HF Hub
- FSDP multi-GPU training
- bf16 training
- Gradient checkpointing
- Masked loss (labels=-100 ignored)
- Save N checkpoints evenly spaced

Usage:
    torchrun --nproc_per_node=4 -m src.hf_training.finetune \
        --model ./pruned_model \
        --task gsm8k \
        --output-dir ./finetuned_model \
        --num-epochs 3 \
        --num-checkpoints 5
"""

import argparse
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.hf_training.data_utils import load_finetuning_dataset

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FinetuneConfig:
    """Configuration for finetuning."""

    model_path: str
    task_name: str
    split: str
    output_dir: str
    num_epochs: int = 3
    num_checkpoints: int = 5
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 4096
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 42
    use_fsdp: bool = True
    gradient_checkpointing: bool = True
    bf16: bool = True
    logging_steps: int = 10
    report_to: str = "wandb"
    run_name: Optional[str] = None


class MaskedLossDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator that preserves the masked labels.

    Unlike the default DataCollatorForLanguageModeling which creates labels
    by copying input_ids, this collator uses the pre-computed labels with
    -100 for masked positions.
    """

    def __init__(self, tokenizer, pad_to_multiple_of=None):
        super().__init__(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=pad_to_multiple_of)

    def __call__(self, features):
        # Separate labels from features
        labels = [f.pop("labels") for f in features]

        # Pad input_ids and attention_mask
        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Pad labels separately (pad with -100 to ignore in loss)
        max_length = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            padding_length = max_length - len(label)
            padded_label = label + [-100] * padding_length
            padded_labels.append(padded_label)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def compute_save_steps(total_steps: int, num_checkpoints: int) -> int:
    """Compute save interval to get approximately num_checkpoints saves."""
    if num_checkpoints <= 0:
        return total_steps + 1  # Never save
    return max(1, total_steps // num_checkpoints)


def finetune(config: FinetuneConfig):
    """Run finetuning with the given configuration."""
    breakpoint()
    logger.info(f"Loading model from {config.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        attn_implementation="flash_attention_2",
    )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Load dataset
    logger.info(f"Loading dataset: {config.task_name} (train)")
    train_dataset = load_finetuning_dataset(
        task_name=config.task_name,
        split="train",
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
    )

    breakpoint()

    logger.info(f"Train dataset size: {len(train_dataset)}")

    # Calculate training steps
    effective_batch_size = (
        config.per_device_train_batch_size
        * config.gradient_accumulation_steps
        * int(os.environ.get("WORLD_SIZE", 1))
    )
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch_size)
    total_steps = steps_per_epoch * config.num_epochs
    save_steps = compute_save_steps(total_steps, config.num_checkpoints)

    logger.info(f"Effective batch size: {effective_batch_size}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Save every {save_steps} steps ({config.num_checkpoints} checkpoints)")

    # Setup FSDP config
    fsdp_config = None
    if config.use_fsdp:
        fsdp_config = {
            "fsdp_transformer_layer_cls_to_wrap": ["OlmoeDecoderLayer", "MixtralDecoderLayer", "LlamaDecoderLayer"],
        }

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_steps=save_steps,
        save_total_limit=config.num_checkpoints + 1,
        seed=config.seed,
        report_to=config.report_to,
        run_name=config.run_name,
        fsdp="full_shard auto_wrap" if config.use_fsdp else "",
        fsdp_config=fsdp_config if config.use_fsdp else None,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.gradient_checkpointing else None,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    # Data collator that preserves masked labels
    data_collator = MaskedLossDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save final model
    logger.info(f"Saving final model to {config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("Training complete!")
    return train_result


def main():
    parser = argparse.ArgumentParser(description="Finetune HuggingFace model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model (can be pruned model or HF model name)",
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
        required=True,
        help="one of train, validation, test",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--num-checkpoints",
        type=int,
        default=5,
        help="Number of checkpoints to save (default: 5)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=2,
        help="Per-device batch size (default: 2)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Maximum sequence length (default: 4096)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio (default: 0.1)",
    )
    parser.add_argument(
        "--no-fsdp",
        action="store_true",
        help="Disable FSDP (use for single GPU)",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for logging",
    )
    parser.add_argument(
        "--report-to",
        type=str,
        default="wandb",
        help="Where to report metrics (wandb, tensorboard, none)",
    )

    args = parser.parse_args()

    config = FinetuneConfig(
        model_path=args.model,
        task_name=args.task,
        split=args.split,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        num_checkpoints=args.num_checkpoints,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
        warmup_ratio=args.warmup_ratio,
        use_fsdp=not args.no_fsdp,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        run_name=args.run_name,
        report_to=args.report_to,
    )

    finetune(config)


if __name__ == "__main__":
    main()
