"""OLMoE3 scaling-ladder 275M rung, runnable for a fixed token budget.

Model + recipe taken from allenai/scaling-ladders ``ladders/olmoe3/workloads/{arch,common,
pretraining}.py`` (branch codex/olmoe3-integration-run). Only the 275M rung is kept; the
builders and guards are otherwise verbatim. It needs the OLMo-core the ladder was
qualified against, which is the ``external/OLMo-core`` submodule pinned to 0e5c2d44
(``PYTHONPATH=external/OLMo-core/src``); this repo's own ``olmo_core`` (v2.3.0) lacks
the fused MoE-v2 / LatentMoE / KDA stack.

The ladder trained this rung on 4 B300s (FA4 + CuTe KDA). Here the defaults follow the other
scripts in model_scripts/: ai2/jupiter H100s, 4 nodes x 8 GPUs, allocated, so the H100-capable
kernels (flash-attn 3, FLA Triton KDA) and a CUDA-12.8 image are selected instead.

Usage (from the repo root; see scripts/sparse_experts/model_scripts/olmoe3_275m_10b.sh):
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_275m.py \
        <launch|dry_run|train> <run_name> <cluster> [--config.overrides ...]

Env knobs (all forwarded to the Beaker worker, which rebuilds the config):
    OLMOE3_TOKENS         token budget (required)
    OLMOE3_LR             peak LR (default 1.2e-3; the ladder's WSD sweep gave 1.6e-3 @ 8.5B
                          tokens and 8e-4 @ 17B tokens)
    OLMOE3_NUM_NODES      nodes (default 4), OLMOE3_NUM_GPUS GPUs per node (default 8)
    OLMOE3_RANK_MB        sequences per rank per micro-batch (default 2 -> 32 ranks x 2 = the fixed
                          64-sequence global batch, no grad accumulation)
    OLMOE3_EP_SIZE        expert-parallel degree (default 1)
    OLMOE3_ATTN_BACKEND   flash_3 (default, Hopper) | flash_2 | flash_4 (Blackwell only, the
                          ladder's own setting)
    OLMOE3_USE_CUTE_KDA   0 (default, FLA Triton KDA kernel, any GPU) | 1 (Blackwell CuTe kernel)
    OLMOE3_IMAGE          Beaker image (default: the team's torch 2.10 / cu128 H100 image; the
                          ladder's own cu130 B300 image needs a CUDA-13 driver, which jupiter lacks)
    OLMOE3_PREEMPTIBLE    0 (default, allocated) | 1
    OLMOE3_FOLLOW         1 (default) streams logs and blocks; 0 submits and returns
    OLMOE3_EMO            1 -> EMO document-pool routing on the same model (the ladder's own EMO
                          setting: per-document pool drawn uniformly from [top_k=16, 512] experts,
                          eval pool 512, local-batch LB loss with global load balancing). Default 0.
    OLMOE3_SAVE_ROOT / OLMOE3_WORK_DIR / OLMOE3_DATA_ROOT / OLMOE3_WANDB_TAGS
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
from copy import deepcopy
from functools import partial

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    InstanceFilterConfig,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.experiment import (
    CommonComponents,
    DataComponents,
    build_common_components as build_default_common_components,
    build_config,
    main,
)
from olmo_core.launch.beaker import BeakerEnvSecret, BeakerEnvVar, BeakerWekaBucket
from olmo_core.launch.beaker_presets import get_preset
from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    GateConfig,
    GateGranularity,
    KimiDeltaAttentionConfig,
)
from olmo_core.nn.ddp import OLMoDDPTransformerBlockConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe import (
    EmoRouterConfig,
    LatentMoEConfig,
    MoELoadBalancingLossGranularity,
    MoERouterGatingFunction,
)
from olmo_core.nn.moe.v2.ep_config import ExpertParallelConfig, ExpertParallelPath
from olmo_core.nn.moe.v2.fp8 import MoERowwiseFP8Config
from olmo_core.nn.moe.v2.routed_experts import RoutedExpertsConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.moe.v2.shared_experts import SharedExpertsConfig
from olmo_core.nn.transformer import (
    OLMoDDPModelConfig,
    TransformerBlockType,
    TransformerType,
)
from olmo_core.optim import OLMoDDPOptimizerConfig, OptimGroupOverride, SchedulerUnits
from olmo_core.optim.scheduler import (
    ComposableScheduler,
    ComposableSchedulerStage,
    ComposableSchedulerStageType,
)
from olmo_core.train import Duration, LoadStrategy, TrainerConfig
from olmo_core.train.callbacks import (
    BeakerCallback,
    CheckpointerCallback,
    CheckpointRemovalStrategy,
    SpeedMonitorCallback,
    WandBCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.train_module import (
    OLMoDDPTrainModuleConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    if raw.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if raw.strip().lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


# ---------------------------------------------------------------------------------------------
# Model: the ladder's "275m" geometry (dense-geometry-kda-nope-gated-latent-moe-l2).
# ---------------------------------------------------------------------------------------------
VOCAB_SIZE = 100_352  # dolma2 tokenizer, padded
HEAD_DIM = 128
TOP_K = 16
LATENT_COMPRESSION = 2
INIT_SEED = 0
INIT_STD = 0.02

D_MODEL = 640
N_LAYERS = 10
N_HEADS = 8
N_KV_HEADS = 8
EXPERT_HIDDEN_SIZE = 544  # multiple of 32 that active-matches the dense 275M rung
NUM_ROUTED_EXPERTS = 512
LATENT_DIM = D_MODEL // LATENT_COMPRESSION
FULL_ATTENTION_LAYERS = tuple(range(4, N_LAYERS, 5))  # (4, 9); layer 0 is dense KDA
KDA_LAYERS = tuple(i for i in range(N_LAYERS) if i not in FULL_ATTENTION_LAYERS)
EXPECTED_ACTIVE_PARAMS = 276_669_264
EXPECTED_ACTIVE_NON_EMBEDDING_PARAMS = 212_443_984
EXPECTED_TOTAL_PARAMS = 2_607_948_624

KDA_USE_CUTE_KERNEL = _env_bool("OLMOE3_USE_CUTE_KDA", False)
EMO_ENABLED = _env_bool("OLMOE3_EMO", False)
EOS_TOKEN_ID = 100_257  # dolma2 tokenizer; EMO derives document segments from EOS positions
EMO_MIN_POOL = int(os.environ.get("OLMOE3_EMO_MIN_POOL", str(TOP_K)))
EMO_MAX_POOL = int(os.environ.get("OLMOE3_EMO_MAX_POOL", str(NUM_ROUTED_EXPERTS)))
EMO_EVAL_POOL = int(os.environ.get("OLMOE3_EMO_EVAL_POOL", str(NUM_ROUTED_EXPERTS)))
ATTN_BACKEND = AttentionBackendName(os.environ.get("OLMOE3_ATTN_BACKEND", "flash_3"))


def _layer_norm() -> LayerNormConfig:
    return LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False, dtype=DType.float32)


def _kda() -> KimiDeltaAttentionConfig:
    return KimiDeltaAttentionConfig(
        n_heads=N_HEADS,
        n_v_heads=N_HEADS,
        head_dim=HEAD_DIM,
        expand_v=2.0,
        allow_neg_eigval=True,
        conv_size=4,
        conv_bias=False,
        norm_eps=1e-5,
        use_cute_kernel=KDA_USE_CUTE_KERNEL,
        dtype=DType.float32,
    )


def _full_attention(layer_norm: LayerNormConfig) -> AttentionConfig:
    return AttentionConfig(
        name=AttentionType.default,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        bias=False,
        gate=GateConfig(granularity=GateGranularity.elementwise, full_precision=True),
        rope=None,  # NoPE
        qk_norm=deepcopy(layer_norm),
        backend=ATTN_BACKEND,
        scalable_softmax=True,
        dtype=DType.float32,
        use_head_qk_norm=True,
    )


def _shared_expert(hidden_size: int = EXPERT_HIDDEN_SIZE) -> SharedExpertsConfig:
    return SharedExpertsConfig(
        d_model=D_MODEL, hidden_size=hidden_size, num_experts=1, bias=False, dtype=DType.float32
    )


def _moe_block(layer_norm: LayerNormConfig, sequence_mixer) -> OLMoDDPTransformerBlockConfig:
    return OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        sequence_mixer=sequence_mixer,
        layer_norm=deepcopy(layer_norm),
        shared_experts=_shared_expert(),
        routed_experts=RoutedExpertsConfig(
            d_model=LATENT_DIM,
            hidden_size=EXPERT_HIDDEN_SIZE,
            num_experts=NUM_ROUTED_EXPERTS,
            bias=False,
            dtype=DType.float32,
            rowwise_fp8=MoERowwiseFP8Config(enabled=False),
        ),
        # Routing sees the full-width token; only the routed payload is projected.
        routed_experts_router=MoERouterConfigV2(
            d_model=D_MODEL,
            num_experts=NUM_ROUTED_EXPERTS,
            top_k=TOP_K,
            bias=False,
            normalize_expert_weights=1.0,
            gating_function=MoERouterGatingFunction.softmax,
            dtype=DType.float32,
            lb_loss_weight=0.01,
            # As in the ladder's EMO branch: EMO needs global (DP-wide) load balancing, which
            # requires local-batch granularity; the plain model keeps the ladder's instance LB.
            lb_loss_granularity=(
                MoELoadBalancingLossGranularity.local_batch
                if EMO_ENABLED
                else MoELoadBalancingLossGranularity.instance
            ),
            global_load_balancing=EMO_ENABLED,
            z_loss_weight=1e-5,
            restore_weight_scale=True,
            use_recompute_fp32_cast=False,
            emo=(
                EmoRouterConfig(
                    eos_token_id=EOS_TOKEN_ID,
                    min_document_expert_pool=EMO_MIN_POOL,
                    max_document_expert_pool=EMO_MAX_POOL,
                    eval_document_expert_pool=EMO_EVAL_POOL,
                )
                if EMO_ENABLED
                else None
            ),
        ),
        latent_moe=LatentMoEConfig(latent_dim=LATENT_DIM, up_proj_input_norm_enabled=False),
        use_peri_norm=True,
        use_pre_norm=False,
        checkpoint_attn=False,
        checkpoint_permute_moe_unpermute=False,
        checkpoint_second_unpermute=False,
        ep=ExpertParallelConfig(path=ExpertParallelPath.rowwise_nvshmem),
        rowwise_fp8=MoERowwiseFP8Config(enabled=False),
    )


def _dense_first_block(layer_norm: LayerNormConfig) -> OLMoDDPTransformerBlockConfig:
    # Matches the dense mainline ladder's 8*d_model SwiGLU FFN exactly.
    return OLMoDDPTransformerBlockConfig(
        name=TransformerBlockType.moe_fused_v2,
        sequence_mixer=_kda(),
        layer_norm=deepcopy(layer_norm),
        shared_experts=_shared_expert(hidden_size=8 * D_MODEL),
        use_peri_norm=True,
        use_pre_norm=False,
        checkpoint_attn=False,
        checkpoint_permute_moe_unpermute=False,
        checkpoint_second_unpermute=False,
    )


def build_model_config(common: CommonComponents) -> OLMoDDPModelConfig:
    layer_norm = _layer_norm()
    vocab_size = common.tokenizer.padded_vocab_size()
    full_attention = _moe_block(layer_norm, _full_attention(layer_norm))
    model = OLMoDDPModelConfig(
        name=TransformerType.moe_fused_v2,
        d_model=D_MODEL,
        vocab_size=vocab_size,
        n_layers=N_LAYERS,
        block=_moe_block(layer_norm, _kda()),
        block_overrides={
            0: _dense_first_block(layer_norm),
            **{i: deepcopy(full_attention) for i in FULL_ATTENTION_LAYERS},
        },
        lm_head=LMHeadConfig(layer_norm=deepcopy(layer_norm), bias=False, dtype=DType.float32),
        embedding_norm=deepcopy(layer_norm),
        dtype=DType.float32,
        init_method="normal",
        init_seed=INIT_SEED,
        init_std=INIT_STD,
        embed_scale=math.sqrt(D_MODEL),
        tie_word_embeddings=False,
        two_batch_overlap=False,
        recompute_all_blocks_by_chunk=False,
        recompute_each_block=False,
    )
    model.validate()

    resolved = model.resolved_block_configs
    kda = tuple(i for i, b in enumerate(resolved) if isinstance(b.sequence_mixer, KimiDeltaAttentionConfig))
    attn = tuple(i for i, b in enumerate(resolved) if isinstance(b.sequence_mixer, AttentionConfig))
    if kda != KDA_LAYERS or attn != FULL_ATTENTION_LAYERS:
        raise ValueError(f"mixer layout drifted: kda={kda} attn={attn}")
    if resolved[0].routed_experts is not None or resolved[0].latent_moe is not None:
        raise ValueError("layer 0 must remain dense")
    if vocab_size == VOCAB_SIZE:
        actual = (model.num_active_params, model.num_active_non_embedding_params, model.num_params)
        expected = (EXPECTED_ACTIVE_PARAMS, EXPECTED_ACTIVE_NON_EMBEDDING_PARAMS, EXPECTED_TOTAL_PARAMS)
        if actual != expected:
            raise ValueError(f"parameter-count drift: expected {expected}, found {actual}")
    return model


# ---------------------------------------------------------------------------------------------
# Recipe (ladder PT defaults) + this repo's Beaker / output conventions.
# ---------------------------------------------------------------------------------------------
SEQUENCE_LENGTH = 8192
GLOBAL_BATCH_SIZE = 64 * SEQUENCE_LENGTH  # dense-mainline fixed 275M batch = 524,288 tokens
DATA_MIX = DataMix.Dolma3p5_14t
LOADER_SEED = 928_543_231  # dense-mainline PT loader seed

TOKENS = int(float(os.environ["OLMOE3_TOKENS"])) if "OLMOE3_TOKENS" in os.environ else None
LR = float(os.environ.get("OLMOE3_LR", "1.2e-3"))
NUM_NODES = int(os.environ.get("OLMOE3_NUM_NODES", "4"))
NUM_GPUS = int(os.environ.get("OLMOE3_NUM_GPUS", "8"))
RANK_MICROBATCH_SEQUENCES = int(os.environ.get("OLMOE3_RANK_MB", "2"))
EP_SIZE = int(os.environ.get("OLMOE3_EP_SIZE", "1"))
PREEMPTIBLE = _env_bool("OLMOE3_PREEMPTIBLE", False)
DATA_ROOT = os.environ.get("OLMOE3_DATA_ROOT", "s3://ai2-llm")
SAVE_ROOT = os.environ.get("OLMOE3_SAVE_ROOT", "/weka/oe-training-default/ryanwang/EMO/sparse_experts")
WORK_DIR = os.environ.get("OLMOE3_WORK_DIR", "/weka/oe-training-default/ryanwang/dataset-cache")
EXTRA_WANDB_TAGS = [t for t in os.environ.get("OLMOE3_WANDB_TAGS", "").split(",") if t]

OLMO_CORE_SUBMODULE = "external/OLMo-core"
BEAKER_WORKSPACE = os.environ.get("BEAKER_WORKSPACE", "ai2/flex2")
BEAKER_PRIORITY = os.environ.get("BEAKER_PRIORITY", "urgent")
OLMO_DDP_PRESET = get_preset("olmo-ddp")  # the team's B300 image (torch 2.11 / cu130 / FA4 / NVSHMEM)
# jupiter's driver is CUDA 12.8, so the olmo-ddp preset's CUDA-13 B300 image cannot run there.
# This is the team's torch 2.10 / cu128 image (flash-attn 2+3, grouped_gemm, TransformerEngine).
H100_IMAGE = "petew/olmo-core-tch2100cu128-2026-01-23"
BEAKER_IMAGE = os.environ.get("OLMOE3_IMAGE") or H100_IMAGE
WANDB_PROJECT, WANDB_ENTITY = "emo-extension", "ryanyxw"

FORWARDED_ENV = (
    "OLMOE3_TOKENS", "OLMOE3_LR", "OLMOE3_NUM_NODES", "OLMOE3_NUM_GPUS", "OLMOE3_RANK_MB", "OLMOE3_EP_SIZE",
    "OLMOE3_ATTN_BACKEND", "OLMOE3_USE_CUTE_KDA", "OLMOE3_PREEMPTIBLE", "OLMOE3_DATA_ROOT",
    "OLMOE3_SAVE_ROOT", "OLMOE3_WORK_DIR", "OLMOE3_WANDB_TAGS", "OLMOE3_IMAGE",
    "OLMOE3_EMO", "OLMOE3_EMO_MIN_POOL", "OLMOE3_EMO_MAX_POOL", "OLMOE3_EMO_EVAL_POOL",
)


def _beaker_env_vars() -> list[BeakerEnvVar]:
    values = dict(OLMO_DDP_PRESET.env_vars)
    values.update(
        {
            "OLMO_SYMM_VDEV2D_AUTO_BUILD": "1" if EP_SIZE > 1 else "0",
            "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST", "9.0"),  # H100; any JIT build targets only this
            # S3 credentials: the pinned launcher injects S3_PROFILE=S3 and the pinned olmo_core hands
            # that name straight to boto3 (a named profile disables env-var credentials), so the
            # worker writes an [S3] profile from the AWS env secrets in post_setup (see below).
            "PYTHONPATH": f"{OLMO_CORE_SUBMODULE}/src",
            # gantry would otherwise `uv pip install` THIS repo (pyproject pins torch==2.8.0) into
            # the image. Install the pinned submodule instead, exactly like scaling-ladders does.
            "GANTRY_INSTALL_CMD": (
                "uv pip install --system --break-system-packages "
                f"-e '{OLMO_CORE_SUBMODULE}[beaker,wandb,fla]'"
            ),
        }
    )
    values.update({k: v for k in FORWARDED_ENV if (v := os.environ.get(k)) is not None})
    return [BeakerEnvVar(name=k, value=v) for k, v in values.items()]


def build_common_components(cli_context, **kwargs) -> CommonComponents:
    common = build_default_common_components(cli_context, **kwargs)
    if (launch := common.launch) is not None:
        launch.workspace = BEAKER_WORKSPACE
        launch.priority = BEAKER_PRIORITY
        launch.preemptible = PREEMPTIBLE
        launch.min_runtime = None
        launch.num_gpus = NUM_GPUS
        launch.torchrun = True  # 1-GPU jobs still need torchrun (LOCAL_RANK etc.)
        launch.launch_timeout = 6 * 3600  # allocated multi-node jobs can queue for hours; default 5 min
        launch.follow = _env_bool("OLMOE3_FOLLOW", True)  # 0 -> fire-and-forget; watch with beaker/gantry CLIs
        launch.allow_dirty = True  # untracked checkpoint dirs live in the tree; push discipline is manual
        launch.beaker_image = BEAKER_IMAGE
        launch.gh_token_secret = "RYAN_GITHUB_TOKEN"
        launch.env_vars = _beaker_env_vars()
        # The preset's symm-mem/NVSHMEM extension prebuild only serves the rowwise-EP transport;
        # at EP=1 it is dead weight and its nvcc build fails on jupiter (no GPU arch detection ->
        # builds sm_50..sm_120). Skip it and disable the import-time auto-build too.
        write_aws_profile = (
            'mkdir -p ~/.aws && printf "[S3]\\naws_access_key_id=%s\\naws_secret_access_key=%s\\n" '
            '"$AWS_ACCESS_KEY_ID" "$AWS_SECRET_ACCESS_KEY" > ~/.aws/credentials'
        )
        launch.post_setup = (
            f"{write_aws_profile} && {OLMO_DDP_PRESET.post_setup}" if EP_SIZE > 1 else write_aws_profile
        )
        launch.env_secrets = [
            BeakerEnvSecret(name="BEAKER_TOKEN", secret="RYAN_BEAKER_TOKEN"),
            BeakerEnvSecret(name="WANDB_API_KEY", secret="RYAN_WANDB_API_KEY"),
            BeakerEnvSecret(name="AWS_ACCESS_KEY_ID", secret="RYAN_AWS_ACCESS_KEY_ID"),
            BeakerEnvSecret(name="AWS_SECRET_ACCESS_KEY", secret="RYAN_AWS_SECRET_ACCESS_KEY"),
            BeakerEnvSecret(name="HF_TOKEN", secret="RYAN_HF_TOKEN", required=False),
        ]
        launch.google_credentials_secret = None
        launch.aws_config_secret = None
        launch.aws_credentials_secret = None
        launch.shared_filesystem = True
        if not any(b.bucket == "oe-training-default" for b in launch.weka_buckets):
            launch.weka_buckets.append(BeakerWekaBucket("oe-training-default", "/weka/oe-training-default"))
    return dataclasses.replace(common, save_folder=f"{SAVE_ROOT}/{common.run_name}", work_dir=WORK_DIR)


def build_data_components(common: CommonComponents) -> DataComponents:
    dataset = NumpyFSLDatasetConfig.from_data_mix(
        DATA_MIX,
        tokenizer=common.tokenizer,
        mix_base_dir=DATA_ROOT,
        work_dir=common.work_dir,
        sequence_length=common.max_sequence_length,
        max_target_sequence_length=SEQUENCE_LENGTH,
        generate_doc_lengths=False,
        instance_filter_config=InstanceFilterConfig(
            repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
        ),
    )
    return DataComponents(
        dataset=dataset,
        data_loader=NumpyDataLoaderConfig(
            global_batch_size=common.global_batch_size,
            seed=LOADER_SEED,
            num_workers=8,
            prefetch_factor=8,
            num_threads=4,
        ),
    )


def _scheduler(tokens: int) -> ComposableScheduler:
    """Ladder PT schedule: 10% of the budget linear warmup, then cosine to 10% of peak."""
    warmup = max(GLOBAL_BATCH_SIZE, int((tokens * 0.1 // GLOBAL_BATCH_SIZE) * GLOBAL_BATCH_SIZE))
    return ComposableScheduler(
        units=SchedulerUnits.tokens,
        stages=[
            ComposableSchedulerStage(
                duration=warmup,
                shape=ComposableSchedulerStageType.linear,
                start_lr_fraction=0.0,
                end_lr_fraction=1.0,
            ),
            ComposableSchedulerStage(
                duration=max(tokens - warmup, GLOBAL_BATCH_SIZE),
                shape=ComposableSchedulerStageType.cosine,
                end_lr_fraction=0.1,
            ),
        ],
    )


def build_train_module_config(common: CommonComponents) -> OLMoDDPTrainModuleConfig:
    assert TOKENS is not None, "set OLMOE3_TOKENS"
    print(
        f"[olmoe3_275m] tokens={TOKENS:,} steps={TOKENS // GLOBAL_BATCH_SIZE:,} lr={LR:g} "
        f"nodes={NUM_NODES} gpus/node={NUM_GPUS} rank_mb={RANK_MICROBATCH_SEQUENCES} ep={EP_SIZE} attn={ATTN_BACKEND} "
        f"cute_kda={KDA_USE_CUTE_KERNEL} emo={EMO_ENABLED}"
        + (f" pool=[{EMO_MIN_POOL},{EMO_MAX_POOL}] eval_pool={EMO_EVAL_POOL}" if EMO_ENABLED else "")
    )
    return OLMoDDPTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_SEQUENCES * common.max_sequence_length,
        max_sequence_length=common.max_sequence_length,
        optim=OLMoDDPOptimizerConfig(
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                # Dense mainline: only token embeddings are exempt from weight decay.
                OptimGroupOverride(params=["embeddings.weight"], opts={"weight_decay": 0.0}),
                # Routed experts get their own group for OLMoDDP's distributed expert handling
                # but inherit the optimizer-level LR.
                OptimGroupOverride(params=["*routed_experts.w_up_gate", "*routed_experts.w_down"], opts={}),
            ],
            compile=True,
            dtype=DType.float32,
            sigma_factor=6,  # SkipStepAdamW: skip loss/grad-norm spikes beyond 6 rolling sigma
            max_grad_norm=1.0,
            use_distributed=True,
        ),
        scheduler=_scheduler(TOKENS),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp, reduce_grads_in_fp32=True, accumulate_grads_in_fp32=True
        ),
        ep_config=TransformerExpertParallelConfig(degree=EP_SIZE) if EP_SIZE > 1 else None,
        pp_config=None,
        tp_config=None,
        cp_config=None,
        ac_config=None,
        float8_config=None,
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )


def build_trainer_config(common: CommonComponents, cluster: str) -> TrainerConfig:
    assert TOKENS is not None, "set OLMOE3_TOKENS"
    cancel_check_interval = 1000
    trainer = TrainerConfig(
        load_strategy=LoadStrategy.if_available,
        save_folder=common.save_folder,
        work_dir=common.work_dir,
        save_overwrite=False,
        checkpointer=CheckpointerConfig(save_thread_count=3, load_thread_count=8, throttle_uploads=True),
        metrics_collect_interval=10,
        cancel_check_interval=cancel_check_interval,
        async_bookkeeping=False,
        max_duration=Duration.tokens(TOKENS),
    )
    return (
        trainer.with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=500,
                save_async=False,
                pre_train_checkpoint=False,
                remove=CheckpointRemovalStrategy.ephemeral_only,
            ),
        )
        .with_callback("speed_monitor", SpeedMonitorCallback())
        .with_callback("beaker", BeakerCallback())
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                group=common.run_name,
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                cancel_check_interval=cancel_check_interval,
                enabled=True,
                tags=["pretraining", "sparse_experts", "olmoe3_275m", "emo" if EMO_ENABLED else "noemo",
                      cluster.rsplit("/", 1)[-1], *EXTRA_WANDB_TAGS],
            ),
        )
    )


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(f"Usage: {sys.argv[0]} <launch|dry_run|train> <run_name> <cluster> [overrides...]")
    cluster = sys.argv[3]
    main(
        config_builder=partial(
            build_config,
            global_batch_size=GLOBAL_BATCH_SIZE,
            max_sequence_length=SEQUENCE_LENGTH,
            num_nodes=NUM_NODES,
            common_config_builder=build_common_components,
            data_config_builder=build_data_components,
            model_config_builder=build_model_config,
            train_module_config_builder=build_train_module_config,
            trainer_config_builder=partial(build_trainer_config, cluster=cluster),
            include_default_evals=False,
            beaker_workspace=BEAKER_WORKSPACE,
            num_execution_units=1,
        )
    )
