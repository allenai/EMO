"""
Example of how to train a transformer language model.

Launch this with torchrun:

    torchrun --nproc-per-node=4 src/examples/llm/train.py run_name [OVERRIDES...]
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
    TokenizerConfig,
)
from olmo_core.data.mixes import DataMix
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.moe.mutualinfo_router import MoEMutualInfoRouterConfig
from olmo_core.nn.moe.twolevel_batchlb_fullzloss_router import (
    MoETwoLevelBatchLBFullZLossRouterConfig,
)
from olmo_core.nn.moe.twolevel_batchlb_nomaskaux_router import (
    MoETwoLevelBatchLBNoMaskAuxRouterConfig,
)
from olmo_core.nn.moe.router_lbreducedp import MoELinearLBReduceDPRouterConfig
from olmo_core.nn.moe.twolevel_batchlb_router import MoETwoLevelBatchLBRouterConfig
from olmo_core.nn.moe.twolevel_pbatchlb_router import MoETwoLevelPBatchLBRouterConfig
from olmo_core.nn.moe.twolevel_router import MoETwoLevelRouterConfig
from olmo_core.nn.moe.twolevel_batchlb_reducedp_router import MoETwoLevelBatchLBReduceDPRouterConfig
from olmo_core.nn.moe.twolevel_batchlb_reducedp_sharedexp_router import MoETwoLevelBatchLBReduceDPSharedExpRouterConfig
from olmo_core.nn.moe.twolevel_sampling_nolb_router import (
    MoETwoLevelSamplingNoLBRouterConfig,
)
from olmo_core.nn.moe.twolevel_topp_batchlb_router import (
    MoETwoLevelTopPBatchLBRouterConfig,
)
from olmo_core.nn.transformer import TransformerConfig
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
    ProfilerCallback,
    WandBCallback,
)
from olmo_core.train.callbacks.expert_pool_scheduler import ExpertPoolSchedulerCallback
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

# from data_mixes import CustomDataMix

log = logging.getLogger(__name__)

# HACK
# DATA_ROOT = "/weka/oe-training-default/ai2-llm"
DATA_ROOT = "/root/ryanwang"

SEQUENCE_LENGTH = 4096
# GLOBAL_BATCH_SIZE = 4 * SEQUENCE_LENGTH
# GLOBAL_BATCH_SIZE = 16 * SEQUENCE_LENGTH
GLOBAL_BATCH_SIZE = 1024 * SEQUENCE_LENGTH


def parse_poolsched(spec: str):
    if spec is None:
        return None
    s = spec.strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    pairs = [p.strip() for p in s.split(",") if p.strip()]
    out = {}
    for p in pairs:
        k, v = p.split(":", 1)
        out[k.strip()] = int(v.strip())
    return out


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


def train(opts, config: ExperimentConfig):
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
    if not trainer.no_checkpoints and not trainer.maybe_load_checkpoint() and config.load_path:
        log.info(
            f"Loading checkpoint from {config.load_path} since no checkpoints were found in the save folder..."
        )
        trainer.load_checkpoint(config.load_path, load_trainer_state=config.load_trainer_state)
    # docs: end-load-path

    # Train.
    trainer.fit()


#
# def apply_twolevel_routers(model, config, document_expert_pool: int):
#     """
#     Replace each MoE layer's router with MoETwoLevelRouter,
#     passing the correct per-layer index and preserving original settings.
#     """
#     # we first define the new kwargs for the PruningMoERouter
#     model_config = config.model
#     kwargs = model_config.block.feed_forward_moe.router.as_dict(exclude_none=True, recurse=False)
#     kwargs.pop("name")
#     kwargs.update(
#         document_expert_pool=document_expert_pool,
#         eos_token_id=config.dataset.tokenizer.eos_token_id,
#     )
#
#     for i, (k, block) in enumerate(model.blocks.items()):
#         # Only touch MoE layers
#         if not getattr(block, "is_moe", False):
#             continue
#
#         old_router = block.router  # MoERouter
#
#         new_router = MoETwoLevelRouterConfig(**kwargs).build(
#             d_model=old_router.d_model,
#             num_experts=old_router.num_experts,
#             init_device=old_router.weight.device.type if hasattr(old_router, 'weight') else "cpu",
#             lb_loss_weight=getattr(old_router, 'lb_loss_weight', None),
#             lb_loss_granularity=getattr(old_router, 'lb_loss_granularity', None),
#             z_loss_weight=getattr(old_router, 'z_loss_weight', None),
#         )
#
#         # Swap in
#         block.feed_forward_moe.router = new_router


def build_config(opts, overrides: List[str]) -> ExperimentConfig:
    save_folder = opts.save_folder
    if not save_folder:
        save_folder = f"/tmp/{opts.run_name}"

    work_dir = opts.work_dir
    if not work_dir:
        work_dir = "/tmp/dataset-cache"

    tokenizer_config = TokenizerConfig.dolma2()

    model_config = TransformerConfig.olmoe_1B_7B(
        vocab_size=tokenizer_config.padded_vocab_size(),  # a little bigger than actual vocab size to make it a multiple of 128
    )

    # Apply special routers or other modifications to the model here if needed.
    if opts.model_type == "dense" or opts.model_type == "moe":
        log.info("Using default routers; no modifications applied.")
        pass
    elif opts.model_type == "moe_lbreducedp":
        log.info("Applying standard moe routers with data parallel reduced load balancing to the model...")
        router_kwargs = model_config.block.feed_forward_moe.router.as_dict(
            exclude_none=True, recurse=False
        )
        router_kwargs.pop("name")

        # Replace router config
        model_config.block.feed_forward_moe.router = MoELinearLBReduceDPRouterConfig(**router_kwargs)
    elif opts.model_type == "two-level":
        log.info("Applying two-level routers to the model...")
        if opts.document_expert_pool is None:
            raise ValueError("document_expert_pool must be specified for two-level model type.")
        # Get existing router config parameters
        router_kwargs = model_config.block.feed_forward_moe.router.as_dict(
            exclude_none=True, recurse=False
        )
        router_kwargs.pop("name")
        router_kwargs.update(
            document_expert_pool=opts.document_expert_pool,
            eos_token_id=tokenizer_config.eos_token_id,
        )

        # Replace router config
        model_config.block.feed_forward_moe.router = MoETwoLevelRouterConfig(**router_kwargs)
    elif opts.model_type == "two-level_lb-batch":
        log.info(
            "Applying two-level with batch-leve load balancing (olmoe lb) routers to the model..."
        )
        if opts.document_expert_pool is None:
            raise ValueError("document_expert_pool must be specified for two-level model type.")
        # Get existing router config parameters
        router_kwargs = model_config.block.feed_forward_moe.router.as_dict(
            exclude_none=True, recurse=False
        )
        router_kwargs.pop("name")
        router_kwargs.update(
            document_expert_pool=opts.document_expert_pool,
            eos_token_id=tokenizer_config.eos_token_id,
        )

        # Replace router config
        model_config.block.feed_forward_moe.router = MoETwoLevelBatchLBRouterConfig(**router_kwargs)
    elif opts.model_type == "two-level_lb-batch_reduce-dp":
        log.info(
            "Applying two-level with batch-leve load balancing (olmoe lb) routers that are reduced across dp ranks to the model..."
        )
        if opts.document_expert_pool is None:
            raise ValueError("document_expert_pool must be specified for two-level model type.")
        # Get existing router config parameters
        router_kwargs = model_config.block.feed_forward_moe.router.as_dict(
            exclude_none=True, recurse=False
        )
        router_kwargs.pop("name")
        router_kwargs.update(
            document_expert_pool=opts.document_expert_pool,
            eos_token_id=tokenizer_config.eos_token_id,
        )

        # Replace router config
        model_config.block.feed_forward_moe.router = MoETwoLevelBatchLBReduceDPRouterConfig(**router_kwargs)
    elif opts.model_type == "two-level_lb-batch_reduce-dp_sharedexp":
        log.info(
            "Applying two-level with batch-leve load balancing (olmoe lb) routers that are reduced across dp ranks and have shared routers to the model..."
        )
        if opts.document_expert_pool is None:
            raise ValueError("document_expert_pool must be specified for two-level model type.")
        if opts.num_shared_experts is None:
            raise ValueError(
                "num_shared_experts must be specified for two-level_lb-batch_reduce-dp_sharedexp model type."
            )
        # Get existing router config parameters
        router_kwargs = model_config.block.feed_forward_moe.router.as_dict(
            exclude_none=True, recurse=False
        )
        router_kwargs.pop("name")
        router_kwargs.update(
            document_expert_pool=opts.document_expert_pool,
            eos_token_id=tokenizer_config.eos_token_id,
            num_shared_experts=opts.num_shared_experts,
        )

        # Replace router config
        model_config.block.feed_forward_moe.router = MoETwoLevelBatchLBReduceDPSharedExpRouterConfig(**router_kwargs)
    elif opts.model_type == "two-level_p_lb-batch":
        log.info(
            "Applying two-level with batch-level load balancing using probabilities to the model..."
        )
        if opts.document_expert_pool is None:
            raise ValueError("document_expert_pool must be specified for two-level model type.")
        if opts.expert_uncond_entropy_bias is None:
            raise ValueError(
                "expert_uncond_entropy_bias must be specified for two-level_p_lb-batch model type."
            )
        if opts.expert_uncond_lb_prob_bias is None:
            raise ValueError(
                "expert_uncond_lb_prob_bias must be specified for two-level_p_lb-batch model type."
            )
        # Get existing router config parameters
        router_kwargs = model_config.block.feed_forward_moe.router.as_dict(
            exclude_none=True, recurse=False
        )
        router_kwargs.pop("name")
        router_kwargs.update(
            document_expert_pool=opts.document_expert_pool,
            eos_token_id=tokenizer_config.eos_token_id,
            expert_uncond_entropy_bias=opts.expert_uncond_entropy_bias,
            expert_uncond_lb_prob_bias=opts.expert_uncond_lb_prob_bias,
        )

        # Replace router config
        model_config.block.feed_forward_moe.router = MoETwoLevelPBatchLBRouterConfig(
            **router_kwargs
        )
    elif opts.model_type == "two-level_topp_lb-batch":
        log.info(
            "Applying two-level with batch-leve load balancing (olmoe lb) routers and top-p selection to the model..."
        )
        if opts.top_p is None:
            raise ValueError("top_p must be specified for two-level topp batchlb loss model type.")
        if opts.max_document_expert_pool is None or opts.min_document_expert_pool is None:
            raise ValueError(
                "Both max_document_expert_pool and min_document_expert_pool must be specified for two-level_topp_lb-batch model type."
            )
        # Get existing router config parameters
        router_kwargs = model_config.block.feed_forward_moe.router.as_dict(
            exclude_none=True, recurse=False
        )
        router_kwargs.pop("name")
        router_kwargs.update(
            top_p=opts.top_p,
            max_document_expert_pool=opts.max_document_expert_pool,
            min_document_expert_pool=opts.min_document_expert_pool,
            eos_token_id=tokenizer_config.eos_token_id,
        )

        # Replace router config
        model_config.block.feed_forward_moe.router = MoETwoLevelTopPBatchLBRouterConfig(
            **router_kwargs
        )
    elif opts.model_type == "two-level_lb-batch_nomaskaux":
        log.info(
            "Applying two-level with batch-leve load balancing with no masking on aux loss routers to the model..."
        )
        if opts.document_expert_pool is None:
            raise ValueError("document_expert_pool must be specified for two-level model type.")
        # Get existing router config parameters
        router_kwargs = model_config.block.feed_forward_moe.router.as_dict(
            exclude_none=True, recurse=False
        )
        router_kwargs.pop("name")
        router_kwargs.update(
            document_expert_pool=opts.document_expert_pool,
            eos_token_id=tokenizer_config.eos_token_id,
        )

        # Replace router config
        model_config.block.feed_forward_moe.router = MoETwoLevelBatchLBNoMaskAuxRouterConfig(
            **router_kwargs
        )
    elif opts.model_type == "two-level_lb-batch_fullzloss":
        log.info(
            "Applying two-level with batch-leve load balancing (olmoe lb) and full zloss (even for unactivated experts, standard for olmoe) routers to the model..."
        )
        if opts.document_expert_pool is None:
            raise ValueError("document_expert_pool must be specified for two-level model type.")
        # Get existing router config parameters
        router_kwargs = model_config.block.feed_forward_moe.router.as_dict(
            exclude_none=True, recurse=False
        )
        router_kwargs.pop("name")
        router_kwargs.update(
            document_expert_pool=opts.document_expert_pool,
            eos_token_id=tokenizer_config.eos_token_id,
        )

        # Replace router config
        model_config.block.feed_forward_moe.router = MoETwoLevelBatchLBFullZLossRouterConfig(
            **router_kwargs
        )
    elif opts.model_type == "two-level_sampling_nolb":
        log.info("Applying two-level with sampling and no load balancing routers to the model...")
        if opts.document_expert_pool is None:
            raise ValueError("document_expert_pool must be specified for two-level model type.")
        # Get existing router config parameters
        router_kwargs = model_config.block.feed_forward_moe.router.as_dict(
            exclude_none=True, recurse=False
        )
        router_kwargs.pop("name")
        router_kwargs.update(
            document_expert_pool=opts.document_expert_pool,
            eos_token_id=tokenizer_config.eos_token_id,
        )

        # Replace router config
        model_config.block.feed_forward_moe.router = MoETwoLevelSamplingNoLBRouterConfig(
            **router_kwargs
        )
    elif opts.model_type == "mutual-info":
        log.info("Applying mutual info router to the model...")
        if opts.expert_cond_token_entropy_bias is None or opts.expert_uncond_entropy_bias is None:
            raise ValueError(
                "Both expert_cond_token_entropy_bias and expert_uncond_entropy_bias must be specified for mutual-info model type."
            )
        # Get existing router config parameters
        router_kwargs = model_config.block.feed_forward_moe.router.as_dict(
            exclude_none=True, recurse=False
        )
        router_kwargs.pop("name")
        router_kwargs.update(
            expert_cond_token_entropy_bias=opts.expert_cond_token_entropy_bias,
            expert_uncond_entropy_bias=opts.expert_uncond_entropy_bias,
        )

        # Replace router config
        model_config.block.feed_forward_moe.router = MoEMutualInfoRouterConfig(**router_kwargs)
    else:
        raise ValueError(f"Unknown model type: {opts.model_type}")

    # docs: end-model-config

    log.info(f"Using data root: {DATA_ROOT}")

    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_mix_0625,
        tokenizer=tokenizer_config,
        mix_base_dir=DATA_ROOT,
        sequence_length=SEQUENCE_LENGTH,
        max_target_sequence_length=max(8192, SEQUENCE_LENGTH),
        work_dir=work_dir,
        generate_doc_lengths=False,
        instance_filter_config=None,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=opts.global_batch_size
        * SEQUENCE_LENGTH,  # NOTE: this is specified in tokens, not instances
        seed=0,
        num_workers=4,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=2
        * SEQUENCE_LENGTH,  # NOTE: this is specified in tokens, not instances
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
        scheduler=CosWithWarmup(warmup_steps=2000),
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
        # .with_callback(
        #     "downstream_evaluator",
        #     # https://github.com/allenai/OLMo-in-loop-evals/blob/main/src/olmo_eval/tasks.py#L1752
        #     DownstreamEvaluatorCallbackConfig(
        #         tasks=[
        #             "hellaswag",
        #             "arc_challenge",
        #             "piqa",
        #             "copa",
        #             "mmlu_stem",
        #             "mmlu_humanities",
        #             "mmlu_social_sciences",
        #             "mmlu_other",
        #         ],
        #         tokenizer=tokenizer_config,
        #         eval_interval=250,
        #     ),
        # )
        .with_callback(
            "expert_pool_scheduler",
            ExpertPoolSchedulerCallback(**parse_poolsched(opts.poolsched)),
        )
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
        "--model-type",
        type=str,
        help="Type of MoE model to use.",
    )
    parser.add_argument(
        "--document-expert-pool",
        type=int,
        help="Number of experts for a specific document to choose top-k from",
    )
    parser.add_argument(
        "--num_shared_experts",
        type=int,
        help="Number of shared experts that are always activated",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=4e-4,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--expert_cond_token_entropy_bias",
        type=float,
        help="Bias term for expert conditional token entropy in mutual info router.",
    )
    parser.add_argument(
        "--expert_uncond_entropy_bias",
        type=float,
        help="Bias term for expert unconditional entropy in mutual info router.",
    )
    parser.add_argument(
        "--expert_uncond_lb_prob_bias",
        type=float,
        help="Bias term for expert unconditional lb in twolevel pbatchlb router.",
    )
    parser.add_argument(
        "--poolsched",
        type=str,
        default="{min_pool: -1, decay_steps: -1}",
        help="Type of pool scheduling to use. Only applies for twolevelbatchlb for now.",
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        help="Global batch size to use.",
        default=1024,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        help="Top-p value for expert selection in two-level_topp_lb-batch router.",
    )
    parser.add_argument(
        "--max_document_expert_pool",
        type=int,
        help="Maximum number of experts for a specific document to choose top-p from",
    )
    parser.add_argument(
        "--min_document_expert_pool",
        type=int,
        help="Minimum number of experts for a specific document to choose top-p from",
    )
    opts, overrides = parser.parse_known_args()
    return opts, overrides


def main():
    opts, overrides = parser_args()
    # note that this function basically initializes all the classes but does not actually call build on them
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
