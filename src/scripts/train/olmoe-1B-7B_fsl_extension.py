"""
Extension / continual-pretraining training script.

Clone of olmoe-1B-7B_fsl_anneal.py with the anneal-specific pieces replaced:
    - LR is a CLI arg (default 4e-4), not auto-extracted from the checkpoint.
    - Scheduler is CosWithWarmup(warmup_fraction=0.1), not WSD.
    - Duration comes from --num-tokens (Duration.tokens(N)).
    - Checkpoint is loaded with load_trainer_state=False (fresh step/data state).
    - Downstream eval uses the broad 55-task suite from train_selected_experts.py.
Everything else (router branching, FSDP, compile, callbacks, expert_pool_scheduler,
densefirst block_overrides) matches the annealing script. All weights train; no
expert gradient masking, no freeze_params.

Launch with torchrun (local) or python -m olmo_core.launch.beaker (cluster):

    torchrun --nproc-per-node=4 src/scripts/train/olmoe-1B-7B_fsl_extension.py run_name \
        --load-path <CKPT>/model_and_optim --num-tokens 10000000000 [OVERRIDES...]
"""

import argparse
import logging
import sys
from dataclasses import dataclass, replace
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
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.moe.mutualinfo_router import MoEMutualInfoRouterConfig
from olmo_core.nn.moe.router_lbreducedp import MoELinearLBReduceDPRouterConfig
from olmo_core.nn.moe.router_lbreducedp_sharedexp import (
    MoELinearLBReduceDPSharedExpRouterConfig,
)
from olmo_core.nn.moe.twolevel_batchlb_fullzloss_router import (
    MoETwoLevelBatchLBFullZLossRouterConfig,
)
from olmo_core.nn.moe.twolevel_batchlb_nomaskaux_router import (
    MoETwoLevelBatchLBNoMaskAuxRouterConfig,
)
from olmo_core.nn.moe.twolevel_batchlb_reducedp_router import (
    MoETwoLevelBatchLBReduceDPRouterConfig,
)
from olmo_core.nn.moe.twolevel_batchlb_reducedp_sharedexp_randpool_router import (
    MoETwoLevelBatchLBReduceDPSharedExpRandPoolRouterConfig,
)
from olmo_core.nn.moe.twolevel_batchlb_reducedp_sharedexp_router import (
    MoETwoLevelBatchLBReduceDPSharedExpRouterConfig,
)
from olmo_core.nn.moe.twolevel_batchlb_reducedp_sharedexppool_router import (
    MoETwoLevelBatchLBReduceDPSharedExpPoolRouterConfig,
)
from olmo_core.nn.moe.twolevel_batchlb_router import MoETwoLevelBatchLBRouterConfig
from olmo_core.nn.moe.twolevel_pbatchlb_router import MoETwoLevelPBatchLBRouterConfig
from olmo_core.nn.moe.twolevel_router import MoETwoLevelRouterConfig
from olmo_core.nn.moe.twolevel_sampling_nolb_router import (
    MoETwoLevelSamplingNoLBRouterConfig,
)
from olmo_core.nn.moe.twolevel_topp_batchlb_router import (
    MoETwoLevelTopPBatchLBRouterConfig,
)
from olmo_core.nn.transformer import (
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    Duration,
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

log = logging.getLogger(__name__)

DATA_ROOT = "/weka/oe-training-default/ai2-llm"

SEQUENCE_LENGTH = 4096


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


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    trainer: TrainerConfig
    train_module: TransformerTrainModuleConfig
    init_seed: int = 12536
    load_path: Optional[str] = None
    load_trainer_state: bool = False


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
        trainer.load_checkpoint(config.load_path, load_trainer_state=config.load_trainer_state)

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

    # Router branching — identical to olmoe-1B-7B_fsl_anneal.py.
    assert isinstance(model_config.block, TransformerBlockConfig)
    assert model_config.block.feed_forward_moe is not None
    if opts.model_type == "dense" or opts.model_type == "moe":
        log.info("Using default routers; no modifications applied.")
        pass
    elif opts.model_type == "moe_lbreducedp":
        log.info(
            "Applying standard moe routers with data parallel reduced load balancing to the model..."
        )
        router_kwargs = model_config.block.feed_forward_moe.router.as_dict(
            exclude_none=True, recurse=False
        )
        router_kwargs.pop("name")
        model_config.block.feed_forward_moe.router = MoELinearLBReduceDPRouterConfig(
            **router_kwargs
        )
    elif opts.model_type == "moe_lbreducedp_sharedexp":
        log.info(
            "Applying standard moe routers with data parallel reduced load balancing and shared experts to the model..."
        )
        if opts.num_shared_experts is None:
            raise ValueError(
                "num_shared_experts must be specified for moe_lbreducedp_sharedexp model type."
            )
        router_kwargs = model_config.block.feed_forward_moe.router.as_dict(
            exclude_none=True, recurse=False
        )
        router_kwargs.pop("name")
        router_kwargs.update(num_shared_experts=opts.num_shared_experts)
        model_config.block.feed_forward_moe.router = MoELinearLBReduceDPSharedExpRouterConfig(
            **router_kwargs
        )
    elif opts.model_type == "two-level":
        log.info("Applying two-level routers to the model...")
        if opts.document_expert_pool is None:
            raise ValueError("document_expert_pool must be specified for two-level model type.")
        router_kwargs = model_config.block.feed_forward_moe.router.as_dict(
            exclude_none=True, recurse=False
        )
        router_kwargs.pop("name")
        router_kwargs.update(
            document_expert_pool=opts.document_expert_pool,
            eos_token_id=tokenizer_config.eos_token_id,
        )
        model_config.block.feed_forward_moe.router = MoETwoLevelRouterConfig(**router_kwargs)
    elif opts.model_type == "two-level_lb-batch":
        log.info(
            "Applying two-level with batch-leve load balancing (olmoe lb) routers to the model..."
        )
        if opts.document_expert_pool is None:
            raise ValueError("document_expert_pool must be specified for two-level model type.")
        router_kwargs = model_config.block.feed_forward_moe.router.as_dict(
            exclude_none=True, recurse=False
        )
        router_kwargs.pop("name")
        router_kwargs.update(
            document_expert_pool=opts.document_expert_pool,
            eos_token_id=tokenizer_config.eos_token_id,
        )
        model_config.block.feed_forward_moe.router = MoETwoLevelBatchLBRouterConfig(**router_kwargs)
    elif opts.model_type == "two-level_lb-batch_reduce-dp":
        log.info(
            "Applying two-level with batch-leve load balancing (olmoe lb) routers that are reduced across dp ranks to the model..."
        )
        if opts.document_expert_pool is None:
            raise ValueError("document_expert_pool must be specified for two-level model type.")
        router_kwargs = model_config.block.feed_forward_moe.router.as_dict(
            exclude_none=True, recurse=False
        )
        router_kwargs.pop("name")
        router_kwargs.update(
            document_expert_pool=opts.document_expert_pool,
            eos_token_id=tokenizer_config.eos_token_id,
        )
        model_config.block.feed_forward_moe.router = MoETwoLevelBatchLBReduceDPRouterConfig(
            **router_kwargs
        )
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
        router_kwargs = model_config.block.feed_forward_moe.router.as_dict(
            exclude_none=True, recurse=False
        )
        router_kwargs.pop("name")
        router_kwargs.update(
            document_expert_pool=opts.document_expert_pool,
            eos_token_id=tokenizer_config.eos_token_id,
            num_shared_experts=opts.num_shared_experts,
        )
        model_config.block.feed_forward_moe.router = (
            MoETwoLevelBatchLBReduceDPSharedExpRouterConfig(**router_kwargs)
        )
    elif opts.model_type == "two-level_lb-batch_reduce-dp_sharedexp_densefirst":
        log.info(
            "Applying two-level with batch-level load balancing (olmoe lb) routers that are reduced across dp ranks and have shared routers to the model, with dense first two layers..."
        )
        if opts.document_expert_pool is None:
            raise ValueError("document_expert_pool must be specified for two-level model type.")
        if opts.num_shared_experts is None:
            raise ValueError(
                "num_shared_experts must be specified for two-level_lb-batch_reduce-dp_sharedexp_densefirst model type."
            )
        router_kwargs = model_config.block.feed_forward_moe.router.as_dict(
            exclude_none=True, recurse=False
        )
        router_kwargs.pop("name")
        router_kwargs.update(
            document_expert_pool=opts.document_expert_pool,
            eos_token_id=tokenizer_config.eos_token_id,
            num_shared_experts=opts.num_shared_experts,
        )
        model_config.block.feed_forward_moe.router = (
            MoETwoLevelBatchLBReduceDPSharedExpRouterConfig(**router_kwargs)
        )
    elif opts.model_type == "two-level_lb-batch_reduce-dp_sharedexp_randpool":
        log.info(
            "Applying two-level with batch-level load balancing (olmoe lb) routers that are reduced across dp ranks, have shared routers, and random document expert pool sizes..."
        )
        if opts.min_document_expert_pool is None or opts.max_document_expert_pool is None:
            raise ValueError(
                "Both min_document_expert_pool and max_document_expert_pool must be specified for two-level_lb-batch_reduce-dp_sharedexp_randpool model type."
            )
        if opts.num_shared_experts is None:
            raise ValueError(
                "num_shared_experts must be specified for two-level_lb-batch_reduce-dp_sharedexp_randpool model type."
            )
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
    elif opts.model_type == "two-level_lb-batch_reduce-dp_sharedexppool":
        log.info(
            "Applying two-level with batch-leve load balancing (olmoe lb) routers that are reduced across dp ranks and have shared routers (that is a pool) to the model..."
        )
        if opts.document_expert_pool is None:
            raise ValueError("document_expert_pool must be specified for two-level model type.")
        if opts.num_shared_experts is None:
            raise ValueError(
                "num_shared_experts must be specified for two-level_lb-batch_reduce-dp_sharedexp model type."
            )
        if opts.num_shared_experts_pool is None:
            raise ValueError(
                "num_shared_experts_pool must be specified for two-level_lb-batch_reduce-dp_sharedexpchoice model type."
            )
        if opts.shared_exp_lb_loss is None:
            raise ValueError(
                "shared_exp_lb_loss must be specified for two-level_lb-batch_reduce-dp_sharedexpchoice model type."
            )
        shared_exp_lb_loss = opts.shared_exp_lb_loss / model_config.n_layers
        router_kwargs = model_config.block.feed_forward_moe.router.as_dict(
            exclude_none=True, recurse=False
        )
        router_kwargs.pop("name")
        router_kwargs.update(
            document_expert_pool=opts.document_expert_pool,
            eos_token_id=tokenizer_config.eos_token_id,
            num_shared_experts=opts.num_shared_experts,
            num_shared_experts_pool=opts.num_shared_experts_pool,
            shared_exp_lb_loss=shared_exp_lb_loss,
        )
        model_config.block.feed_forward_moe.router = (
            MoETwoLevelBatchLBReduceDPSharedExpPoolRouterConfig(**router_kwargs)
        )
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
        model_config.block.feed_forward_moe.router = MoETwoLevelTopPBatchLBRouterConfig(
            **router_kwargs
        )
    elif opts.model_type == "two-level_lb-batch_nomaskaux":
        log.info(
            "Applying two-level with batch-leve load balancing with no masking on aux loss routers to the model..."
        )
        if opts.document_expert_pool is None:
            raise ValueError("document_expert_pool must be specified for two-level model type.")
        router_kwargs = model_config.block.feed_forward_moe.router.as_dict(
            exclude_none=True, recurse=False
        )
        router_kwargs.pop("name")
        router_kwargs.update(
            document_expert_pool=opts.document_expert_pool,
            eos_token_id=tokenizer_config.eos_token_id,
        )
        model_config.block.feed_forward_moe.router = MoETwoLevelBatchLBNoMaskAuxRouterConfig(
            **router_kwargs
        )
    elif opts.model_type == "two-level_lb-batch_fullzloss":
        log.info(
            "Applying two-level with batch-leve load balancing (olmoe lb) and full zloss (even for unactivated experts, standard for olmoe) routers to the model..."
        )
        if opts.document_expert_pool is None:
            raise ValueError("document_expert_pool must be specified for two-level model type.")
        router_kwargs = model_config.block.feed_forward_moe.router.as_dict(
            exclude_none=True, recurse=False
        )
        router_kwargs.pop("name")
        router_kwargs.update(
            document_expert_pool=opts.document_expert_pool,
            eos_token_id=tokenizer_config.eos_token_id,
        )
        model_config.block.feed_forward_moe.router = MoETwoLevelBatchLBFullZLossRouterConfig(
            **router_kwargs
        )
    elif opts.model_type == "two-level_sampling_nolb":
        log.info("Applying two-level with sampling and no load balancing routers to the model...")
        if opts.document_expert_pool is None:
            raise ValueError("document_expert_pool must be specified for two-level model type.")
        router_kwargs = model_config.block.feed_forward_moe.router.as_dict(
            exclude_none=True, recurse=False
        )
        router_kwargs.pop("name")
        router_kwargs.update(
            document_expert_pool=opts.document_expert_pool,
            eos_token_id=tokenizer_config.eos_token_id,
        )
        model_config.block.feed_forward_moe.router = MoETwoLevelSamplingNoLBRouterConfig(
            **router_kwargs
        )
    elif opts.model_type == "mutual-info":
        log.info("Applying mutual info router to the model...")
        if opts.expert_cond_token_entropy_bias is None or opts.expert_uncond_entropy_bias is None:
            raise ValueError(
                "Both expert_cond_token_entropy_bias and expert_uncond_entropy_bias must be specified for mutual-info model type."
            )
        router_kwargs = model_config.block.feed_forward_moe.router.as_dict(
            exclude_none=True, recurse=False
        )
        router_kwargs.pop("name")
        router_kwargs.update(
            expert_cond_token_entropy_bias=opts.expert_cond_token_entropy_bias,
            expert_uncond_entropy_bias=opts.expert_uncond_entropy_bias,
        )
        model_config.block.feed_forward_moe.router = MoEMutualInfoRouterConfig(**router_kwargs)
    else:
        raise ValueError(f"Unknown model type: {opts.model_type}")

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
        global_batch_size=opts.global_batch_size * SEQUENCE_LENGTH,
        seed=0,
        num_workers=4,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=4 * SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(
            lr=opts.lr,
            weight_decay=0.0,
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
                enabled=False,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.run_name,
                cancel_check_interval=10,
                enabled=True,
            ),
        )
        .with_callback("beaker", BeakerCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("profiler", ProfilerCallback(enabled=False))
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=[
                    "arc_challenge_test_rc_5shot",
                    "arc_easy_test_rc_5shot",
                    "hellaswag_rc_5shot",
                    "winogrande_val_rc_5shot",
                    "csqa_val_rc_5shot",
                    "piqa_val_rc_5shot",
                    "socialiqa_val_rc_5shot",
                    "mmlu_stem_val_rc_5shot",
                    "mmlu_humanities_val_rc_5shot",
                    "mmlu_social_sciences_val_rc_5shot",
                    "mmlu_other_val_rc_5shot",
                    "mmlu_stem_test_rc_5shot",
                    "mmlu_humanities_test_rc_5shot",
                    "mmlu_social_sciences_test_rc_5shot",
                    "mmlu_other_test_rc_5shot",
                    "basic_skills_common_knowledge_rc_5shot",
                    "basic_skills_logical_reasoning_rc_5shot",
                    "basic_skills_pattern_rc_5shot",
                    "basic_skills_string_operations_rc_5shot",
                    "copycolors_10way_fast",
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
            ),
        )
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
        load_path=opts.load_path,
        load_trainer_state=False,
    )

    config = config.merge(overrides)

    # Set max_duration from --num-tokens (after merge so CLI --trainer.max_duration overrides win if present).
    config.trainer.max_duration = Duration.tokens(opts.num_tokens)

    # Apply dense first layer overrides AFTER merge so CLI overrides
    # (backend, qk_norm, etc.) are inherited by the dense blocks.
    if opts.model_type == "two-level_lb-batch_reduce-dp_sharedexp_densefirst":
        assert isinstance(config.model.block, TransformerBlockConfig)
        moe_cfg = config.model.block.feed_forward_moe
        assert moe_cfg is not None
        dense_hidden = moe_cfg.router.top_k * moe_cfg.hidden_size
        dense_block = replace(
            config.model.block,
            name=TransformerBlockType.default,
            feed_forward=FeedForwardConfig(hidden_size=dense_hidden),
            feed_forward_moe=None,
        )
        config.model.block_overrides = {0: dense_block, 1: dense_block}

    return config


def parser_args():
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        usage=f"python {sys.argv[0]} RUN_NAME [OPTIONS...] [CONFIG_OVERRIDES...]",
        description="Continual-pretraining / extension training script for MoE models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_name", type=str, help="""The name of the run.""")
    parser.add_argument(
        "--save-folder", type=str, help="A local or remote directory to save checkpoints to."
    )
    parser.add_argument(
        "--work-dir", type=str, help="A local working directory for dataset preprocessing."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the config and exit.")
    parser.add_argument("--model-type", type=str, help="Type of MoE model to use.")
    parser.add_argument(
        "--document-expert-pool",
        type=int,
        help="Number of experts for a specific document to choose top-k from",
    )
    parser.add_argument(
        "--num_shared_experts", type=int, help="Number of shared experts that are always activated"
    )
    parser.add_argument(
        "--num_shared_experts_pool", type=int, help="Number of shared experts to keep in the pool"
    )
    parser.add_argument("--expert_cond_token_entropy_bias", type=float)
    parser.add_argument("--expert_uncond_entropy_bias", type=float)
    parser.add_argument("--expert_uncond_lb_prob_bias", type=float)
    parser.add_argument("--poolsched", type=str, default="{min_pool: -1, decay_steps: -1}")
    parser.add_argument("--global_batch_size", type=int, default=1024)
    parser.add_argument("--top_p", type=float)
    parser.add_argument("--max_document_expert_pool", type=int)
    parser.add_argument("--min_document_expert_pool", type=int)
    parser.add_argument("--shared_exp_lb_loss", type=float)
    parser.add_argument("--eval_document_expert_pool", type=int)
    parser.add_argument(
        "--lr", type=float, default=4e-4, help="Peak learning rate for CosWithWarmup."
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        required=True,
        help="Total training token budget (e.g. 10_000_000_000).",
    )
    parser.add_argument(
        "--load-path",
        type=str,
        required=True,
        help="Path to base checkpoint (e.g. <ckpt>/model_and_optim). Model+optim weights are loaded; trainer state is NOT loaded.",
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
