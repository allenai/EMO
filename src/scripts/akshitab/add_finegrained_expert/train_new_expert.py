"""
Example of how to train a transformer language model.

Launch this with torchrun:

    torchrun --nproc-per-node=4 src/examples/llm/train.py run_name [OVERRIDES...]
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, cast

import rich
import torch

from olmo_core.config import Config, DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.mixes import DataMix
from olmo_core.data.numpy_dataset import NumpyDatasetConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.transformer import PARTIAL_FREEZE_FN_REGISTRY, TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    BeakerCallback,
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    GradientMonitorCallback,
    HFConverterCallback,
    ProfilerCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all, setup_logging

log = logging.getLogger(__name__)

DATA_ROOT = "/weka/oe-training-default/ai2-llm"

SEQUENCE_LENGTH = 4096
# GLOBAL_BATCH_SIZE = 16 * SEQUENCE_LENGTH
GLOBAL_BATCH_SIZE = 1024 * SEQUENCE_LENGTH


# docs: start-define-config
@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    """Model config."""
    dataset: NumpyDatasetConfig
    """Dataset config."""
    data_loader: NumpyDataLoaderConfig
    """Data loader config."""
    trainer: TrainerConfig
    """Trainer config."""
    train_module: TransformerTrainModuleConfig
    """Train module config. Contains settings for optimizer."""
    init_seed: int = 12536
    """Random seed to initialize model weights."""
    load_path: Optional[str] = None
    """Path to load checkpoint from if no checkpoint is found in the save folder.
    Mainly used when you want to fine-tune from a pretrained model."""
    load_trainer_state: bool = False
    """Whether to load the trainer state (including data loader state) when loading from `load_path`.
    This only makes sense when trainer state is available in the checkpoint and you're resuming
    on the same dataset."""
    # docs: end-define-config


def partial_freeze_router_and_experts(
    model_config: TransformerConfig,
    name: str,
    param: torch.nn.Parameter,
    num_experts_to_train: int = 1,
) -> Optional[torch.Tensor]:
    if "experts" in name or "router" in name:
        mask = torch.ones_like(param, dtype=torch.bool)
        assert model_config.block.feed_forward_moe is not None
        num_experts = model_config.block.feed_forward_moe.num_experts
        expert_size = param.shape[0] // num_experts
        # Freeze all but the last `num_experts_to_train` experts.
        mask = freeze_indices(
            dim=0, indices=list(range(0, expert_size * (num_experts - num_experts_to_train)))
        )(name, param)
        return mask.float()
    return None


def freeze_indices(dim: int, indices: list[int]) -> Callable:
    def mask_fn(name: str, param: torch.Tensor) -> torch.Tensor:
        mask = torch.ones_like(param, dtype=torch.bool, device="cpu")
        idx = [slice(None)] * param.ndim
        idx[dim] = indices  # type: ignore
        mask[tuple(idx)] = False
        return mask.float()

    return mask_fn


def train(config: ExperimentConfig, eval_only: bool = False):
    if get_rank() == 0:
        rich.print(config)

    # Set RNG states on all devices.
    seed_all(config.init_seed)

    # docs: start-build-components
    # Build components.
    model = config.model.build(init_device="meta")

    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)
    # docs: end-build-components

    # Save config to W&B and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # docs: start-load-path
    # If we have a load path set and there is no checkpoint in the save folder, load the
    # checkpoint from the load path.
    checkpoint_loaded = False
    if not trainer.no_checkpoints:
        checkpoint_loaded = trainer.maybe_load_checkpoint()
        if not checkpoint_loaded and config.load_path:
            log.info(
                f"Loading checkpoint from {config.load_path} since no checkpoints were found in the save folder..."
            )
            trainer.load_checkpoint(config.load_path, load_trainer_state=config.load_trainer_state)
            checkpoint_loaded = True

    # In eval-only mode, ensure a checkpoint was loaded
    if eval_only and not checkpoint_loaded:
        raise RuntimeError(
            "Cannot run eval-only mode: no checkpoint found in save folder and no load_path provided. "
            "Please ensure checkpoints exist or provide --load-path."
        )

    if eval_only:
        log.info("Running in eval-only mode: will evaluate checkpoint and exit without training.")
    # docs: end-load-path

    # Train.
    trainer.fit()


def build_config(opts, overrides: List[str]) -> ExperimentConfig:
    save_folder = opts.save_folder
    if not save_folder:
        save_folder = f"/tmp/{opts.run_name}"

    work_dir = opts.work_dir
    if not work_dir:
        work_dir = "/tmp/dataset-cache"

    tokenizer_config = TokenizerConfig.dolma2()

    # docs: start-model-config

    global PARTIAL_FREEZE_FN_REGISTRY
    PARTIAL_FREEZE_FN_REGISTRY[
        "partial_freeze_router_and_experts"
    ] = partial_freeze_router_and_experts

    # if opts.num_experts_to_train != opts.added_experts:
    #     log.warning(
    #         f"num_experts_to_train ({opts.num_experts_to_train}) != added_experts ({opts.added_experts})."
    #         f" This will train weights for the last {opts.num_experts_to_train} existing expert(s)."
    #     )

    model_config = TransformerConfig.olmoe_1B_7B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        n_layers=16,
        d_model=2048,
        n_heads=16,
        num_experts=128 + opts.num_experts_to_train,
        top_k=8,
        freeze_params=[
            "embeddings.*",
            "blocks.*.attention*",
            "blocks.*.feed_forward_norm.*",
            "lm_head.*",
        ],
        partial_freeze_params_mask_fn_name="partial_freeze_router_and_experts",
        partial_freeze_params_mask_fn_kwargs={"num_experts_to_train": opts.num_experts_to_train},
    )

    print(model_config)
    # docs: end-model-config

    log.info(f"Using data root: {DATA_ROOT}")

    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMoE_mix_0824,
        tokenizer=tokenizer_config,
        mix_base_dir=DATA_ROOT,
        sequence_length=SEQUENCE_LENGTH,
        max_target_sequence_length=max(8192, SEQUENCE_LENGTH),
        work_dir=work_dir,
        generate_doc_lengths=False,
        instance_filter_config=None,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,  # NOTE: this is specified in tokens, not instances
        seed=0,
        num_workers=4,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=4
        * SEQUENCE_LENGTH,  # NOTE: this is specified in tokens, not instances
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(
            lr=opts.lr,
            # AdamW adds wd to partially frozen params, which causes weights to drift.
            # Ideally, the experts being trained would have 0.1, but due to partial freezing implementation limitations,
            # all experts including the trainable ones have to have 0.0 wd.
            weight_decay=0.0,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0)),
                # OptimGroupOverride(params=["blocks.*.feed_forward_moe.experts.*"], opts=dict(weight_decay=0.0)),
                # OptimGroupOverride(params=["blocks.*.feed_forward_moe.router.*"], opts=dict(weight_decay=0.0)),
            ],
            fused=True,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_fraction=0.1),
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=5,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=5000,
                ephemeral_save_interval=100,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=opts.run_name,
                cancel_check_interval=10,
                enabled=False,  # change to true to enable
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.run_name,
                cancel_check_interval=10,
                enabled=True,  # change to true to enable
            ),
        )
        .with_callback("beaker", BeakerCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("profiler", ProfilerCallback(enabled=False))
        .with_callback(
            "downstream_evaluator",
            # https://github.com/allenai/OLMo-in-loop-evals/blob/main/src/olmo_eval/tasks.py#L1752
            DownstreamEvaluatorCallbackConfig(
                tasks=[
                    "hellaswag",
                    "arc_challenge",
                    "piqa",
                    "copa",
                    "mmlu_stem",
                    "mmlu_humanities",
                    "mmlu_social_sciences",
                    "mmlu_other",
                    "basic_skills_common_knowledge_rc_5shot",
                    "basic_skills_logical_reasoning_rc_5shot",
                    "basic_skills_pattern_rc_5shot",
                    "basic_skills_string_operations_rc_5shot",
                    # math
                    "basic_skills_arithmetic_rc_5shot",
                    "gsm8k_gold_bpb_5shot",
                    "minerva_math_algebra_gold_bpb_0shot",
                    "minerva_math_counting_and_probability_gold_bpb_0shot",
                    "minerva_math_geometry_gold_bpb_0shot",
                    "minerva_math_intermediate_algebra_gold_bpb_0shot",
                    "minerva_math_number_theory_gold_bpb_0shot",
                    "minerva_math_prealgebra_gold_bpb_0shot",
                    "minerva_math_precalculus_gold_bpb_0shot",
                    "minerva_math_500_gold_bpb_0shot",
                    # code
                    "basic_skills_coding_rc_5shot",
                    "codex_humaneval_gold_bpb_0shot",
                    "codex_humaneval_gold_bpb_3shot",
                    "codex_mbpp_gold_bpb_0shot",
                    "codex_mbpp_gold_bpb_3shot",
                ]
                + [
                    f"mt_mbpp_{lang}_gold_bpb_3shot"
                    for lang in [
                        "haskell",
                        "go",
                        "python",
                        "cpp",
                        "javascript",
                        "swift",
                        "scala",
                        "bash",
                        "typescript",
                        "c",
                        "php",
                        "rust",
                        "csharp",
                        "r",
                        "ruby",
                        "java",
                        "matlab",
                    ]
                ],
                tokenizer=tokenizer_config,
                eval_interval=250,
                eval_on_startup=opts.eval_only,
                cancel_after_first_eval=opts.eval_only,
            ),
        )
        .with_callback(
            "gradient_monitor",
            GradientMonitorCallback(
                layer_names=["expert.mlp", "router"],
                max_steps_to_monitor=10,
                log_all_params=True,
            ),
        )
        # .with_callback(
        #     "hf_converter",  # TODO: fix bug that causes it to stall.
        #     HFConverterCallback(
        #         enabled=True,
        #         dtype=DType.float32,
        #         max_sequence_length=SEQUENCE_LENGTH,
        #         device="cpu",
        #     ),
        # )
    )

    config = ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    )

    # Apply overrides.
    # docs: start-config-merge
    config = config.merge(overrides)
    # docs: end-config-merge

    # # DEBUG / TEST
    # config.model.build(init_device="meta")

    return config


def parser_args():
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        usage=f"python {sys.argv[0]} RUN_NAME [OPTIONS...] [CONFIG_OVERRIDES...]",
        description="Train a transformer language model on c4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_name", type=str, help="""The name of the run.""")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=2048,
        help="""The sequence length to train and eval on.""",
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        help="""A local or remote directory to save checkpoints to.
        Defaults to a temporary directory if not provided.""",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        help="""A local working directory for dataset preprocessing.
        Defaults to a temporary directory if not provided.""",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="""Print the config and exit.""",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=4e-4,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--num-experts-to-train",
        type=int,
        default=1,
        help="Number of experts to train (last n experts). Remaining are frozen.",
    )
    parser.add_argument(
        "--added-experts",
        type=int,
        default=1,
        help="Number of experts actually added.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="""Run evaluations only on existing checkpoint without training.
        This will load the latest checkpoint and run downstream evals.""",
    )
    opts, overrides = parser.parse_known_args()
    return opts, overrides


def main():
    opts, overrides = parser_args()
    setup_logging()
    config = build_config(opts, overrides)

    if opts.dry_run:
        rich.print(config)
        return

    prepare_training_environment()
    train(config, eval_only=opts.eval_only)
    teardown_training_environment()


if __name__ == "__main__":
    main()
