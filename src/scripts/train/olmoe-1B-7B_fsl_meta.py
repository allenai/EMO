"""
PARENT: "src/scripts/train/olmoe-1B-7B_fsl.py"

Entry point for the meta_learning experiment (FOMAML-style EMO pretraining: selective 32-expert
inner pass -> pseudo-step on expert weights -> full-model outer pass; see
olmo_core.train.train_module.transformer.meta_learning).

Differences vs the parent:

- Only the ``two-level_lb-batch_reduce-dp_sharedexp_randpool`` model type is supported (the meta
  train module requires the randpool router).
- The train module is :class:`MetaLearningTransformerTrainModuleConfig`; meta knobs are set via
  dotted overrides, e.g. ``--train_module.meta_mode=same_tokens --train_module.inner_lr=3e-2
  --train_module.inner_pool_size=32``.
- Two pool-pinned LM (ppl) evaluators are attached on the v3-small ppl validation mix:
  ``lm-full`` (model-default eval pool) and ``lm-pool32`` (pool pinned to 32). Their difference is
  the selective-vs-full CE gap, the experiment's headline metric.

Launch through scripts/meta_learning/*.sh (never ad-hoc).
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import List, Optional, cast

import rich

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.mixes import DataMix
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.moe.twolevel_batchlb_reducedp_sharedexp_randpool_router import (
    MoETwoLevelBatchLBReduceDPSharedExpRandPoolRouterConfig,
)
from olmo_core.nn.transformer import TransformerBlockConfig, TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.optim.scheduler import WSD, SchedulerUnits
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
    ConsoleLoggerCallback,
    DownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    PoolPinnedLMEvaluatorCallbackConfig,
    ProfilerCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    MetaLearningTransformerTrainModuleConfig,
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)

SEQUENCE_LENGTH = 4096


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
    train_module: MetaLearningTransformerTrainModuleConfig
    """Train module config (meta-learning variant). Contains settings for optimizer."""
    init_seed: int = 12536
    """Random seed to initialize model weights."""
    load_path: Optional[str] = None
    """Path to load checkpoint from if no checkpoint is found in the save folder."""
    load_trainer_state: bool = False
    """Whether to load the trainer state (including data loader state) when loading from `load_path`."""
    load_optim_state: bool = True
    """Whether to load the optimizer state when loading from `load_path`."""


def train(opts, config: ExperimentConfig):
    if get_rank() == 0:
        rich.print(config)

    seed_all(config.init_seed)

    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    if not trainer.no_checkpoints and not trainer.maybe_load_checkpoint() and config.load_path:
        log.info(
            f"Loading checkpoint from {config.load_path} since no checkpoints were found in the save folder..."
        )
        trainer.load_checkpoint(
            config.load_path,
            load_trainer_state=config.load_trainer_state,
            load_optim_state=config.load_optim_state,
        )

    trainer.fit()


def build_config(opts, overrides: List[str]) -> ExperimentConfig:
    save_folder = opts.save_folder
    if not save_folder:
        save_folder = f"/tmp/{opts.run_name}"

    work_dir = opts.work_dir
    if not work_dir:
        work_dir = "/tmp/dataset-cache"

    tokenizer_config = TokenizerConfig.dolma2()

    model_config = TransformerConfig.olmoe_1B_7B(
        vocab_size=tokenizer_config.padded_vocab_size(),
    )

    assert isinstance(model_config.block, TransformerBlockConfig)
    assert model_config.block.feed_forward_moe is not None
    if opts.model_type != "two-level_lb-batch_reduce-dp_sharedexp_randpool":
        raise ValueError(
            "The meta_learning entry script only supports "
            f"--model-type=two-level_lb-batch_reduce-dp_sharedexp_randpool, got: {opts.model_type}"
        )
    if opts.min_document_expert_pool is None or opts.max_document_expert_pool is None:
        raise ValueError(
            "Both min_document_expert_pool and max_document_expert_pool must be specified."
        )
    if opts.num_shared_experts is None:
        raise ValueError("num_shared_experts must be specified.")

    router_kwargs = model_config.block.feed_forward_moe.router.as_dict(
        exclude_none=True, recurse=False
    )
    router_kwargs.pop("name")
    router_kwargs.update(
        min_document_expert_pool=opts.min_document_expert_pool,
        max_document_expert_pool=opts.max_document_expert_pool,
        eos_token_id=tokenizer_config.eos_token_id,
        num_shared_experts=opts.num_shared_experts,
    )
    if opts.eval_document_expert_pool is not None:
        router_kwargs["eval_document_expert_pool"] = opts.eval_document_expert_pool

    model_config.block.feed_forward_moe.router = (
        MoETwoLevelBatchLBReduceDPSharedExpRandPoolRouterConfig(**router_kwargs)
    )

    log.info(f"Using data root: {opts.data_root}")

    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_mix_0625,
        tokenizer=tokenizer_config,
        mix_base_dir=opts.data_root,
        sequence_length=SEQUENCE_LENGTH,
        max_target_sequence_length=max(8192, SEQUENCE_LENGTH),
        work_dir=work_dir,
        generate_doc_lengths=False,
        instance_filter_config=None,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=opts.global_batch_size * SEQUENCE_LENGTH,  # in tokens
        seed=0,
        num_workers=4,
    )

    if opts.scheduler == "wsd":
        scheduler = WSD(
            units=SchedulerUnits.steps,
            warmup=opts.warmup_steps,
            warmup_fraction=None,
            decay=opts.decay_steps,
            decay_fraction=None,
            decay_min_lr=0.0,
        )
    else:
        scheduler = CosWithWarmup(warmup=opts.warmup_steps)

    train_module_config = MetaLearningTransformerTrainModuleConfig(
        rank_microbatch_size=4 * SEQUENCE_LENGTH,  # in tokens
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(
            lr=opts.lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
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
        scheduler=scheduler,
        # Meta knobs (defaults; override with --train_module.<knob>=...).
        meta_mode="same_tokens",
        inner_lr=0.0,
        inner_pool_size=32,
        lambda_inner=0.0,
        lb_on_inner=False,
        inner_grad_clip=1.0,
        log_grad_cosine=True,
    )

    ppl_eval_dataset = NumpyPaddedFSLDatasetConfig.from_data_mix(
        DataMix.v3_small_ppl_validation,
        mix_base_dir=opts.data_root,
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=tokenizer_config,
        work_dir=work_dir,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=5,
        )
        # The trainer's default console logger doesn't include the meta diagnostics; register our
        # own (the trainer only adds a default when none is present).
        .with_callback(
            "console_logger",
            ConsoleLoggerCallback(
                metrics_log_interval=5,
                metrics=[
                    "train/CE loss",
                    "train/PPL",
                    "train/Z loss",
                    "train/load balancing loss",
                    "train/router Z loss",
                    "train/block */load imbalance",
                    "train/meta *",
                    "gpu_memory/*",
                    "optim/total grad norm",
                    "optim/step skipped",
                    "optim/LR*",
                    "throughput/*",
                ],
            ),
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
            CometCallback(name=opts.run_name, cancel_check_interval=10, enabled=False),
        )
        .with_callback(
            "wandb",
            WandBCallback(name=opts.run_name, cancel_check_interval=10, enabled=True),
        )
        .with_callback("beaker", BeakerCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("profiler", ProfilerCallback(enabled=False))
        .with_callback(
            "downstream_evaluator",
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
                ],
                tokenizer=tokenizer_config,
                eval_interval=250,
            ),
        )
        # Selective-vs-full CE gap: same ppl validation mix evaluated at the model-default eval
        # pool (full, per the recipe's eval_document_expert_pool) and pinned to 32.
        .with_callback(
            "lm_eval_full",
            PoolPinnedLMEvaluatorCallbackConfig(
                eval_dataset=ppl_eval_dataset,
                eval_interval=500,
                eval_pool=None,
                name="lm-full",
            ),
        )
        .with_callback(
            "lm_eval_pool32",
            PoolPinnedLMEvaluatorCallbackConfig(
                eval_dataset=ppl_eval_dataset,
                eval_interval=500,
                eval_pool=32,
                name="lm-pool32",
            ),
        )
    )

    config = ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    )

    config = config.merge(overrides)

    return config


def parser_args():
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        usage=f"python {sys.argv[0]} RUN_NAME [OPTIONS...] [CONFIG_OVERRIDES...]",
        description="Meta-learning (FOMAML) EMO pretraining.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_name", type=str, help="""The name of the run.""")
    parser.add_argument("--save-folder", type=str, help="Directory to save checkpoints to.")
    parser.add_argument("--work-dir", type=str, help="Working directory for dataset preprocessing.")
    parser.add_argument("--dry-run", action="store_true", help="Print the config and exit.")
    parser.add_argument(
        "--model-type",
        type=str,
        default="two-level_lb-batch_reduce-dp_sharedexp_randpool",
        help="Type of MoE model to use (only the randpool router is supported here).",
    )
    parser.add_argument(
        "--num_shared_experts", type=int, help="Number of shared experts that are always activated."
    )
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate for the optimizer.")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cos",
        choices=["cos", "wsd"],
        help="LR scheduler: 'cos' (cosine-with-warmup) or 'wsd' (warmup-stable-decay).",
    )
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Linear warmup steps.")
    parser.add_argument(
        "--decay_steps", type=int, default=1192, help="Final linear decay steps (WSD only)."
    )
    parser.add_argument(
        "--global_batch_size", type=int, default=1024, help="Global batch size in instances."
    )
    parser.add_argument(
        "--min_document_expert_pool", type=int, help="Minimum per-document expert pool size."
    )
    parser.add_argument(
        "--max_document_expert_pool", type=int, help="Maximum per-document expert pool size."
    )
    parser.add_argument(
        "--eval_document_expert_pool",
        type=int,
        help="Fixed pool size during evaluation. Defaults to midpoint of min/max.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/weka/oe-training-default/ai2-llm",
        help="Root directory for the data mix (mix_base_dir).",
    )
    opts, overrides = parser.parse_known_args()
    return opts, overrides


def main():
    opts, overrides = parser_args()
    config = build_config(opts, overrides)

    if opts.dry_run:
        rich.print(config)
        return

    prepare_training_environment()
    try:
        train(opts, config)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
