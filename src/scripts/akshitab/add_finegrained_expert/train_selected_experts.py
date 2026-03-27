"""
Train selected experts by index.

Unlike train_new_expert.py which only unfreezes the last N experts,
this script allows specifying arbitrary expert indices to unfreeze and train.

Launch this with torchrun:

    torchrun --nproc-per-node=4 src/scripts/akshitab/add_finegrained_expert/train_selected_experts.py run_name \
        --experts-to-train 5,10,42 [OVERRIDES...]
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, cast

import rich
import torch
from torch.distributed.tensor import DTensor

from olmo_core.config import Config, DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.mixes import DataMix
from olmo_core.data.numpy_dataset import NumpyDatasetConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_local_tensor, get_rank
from olmo_core.nn.moe.mlp import DroplessMoEMLP, SplitExpertDroplessMoEMLP
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
    GradientMonitorCallback,
    HFConverterCallback,
    ProfilerCallback,
    SplitExpertConverterCallback,
    WandBCallback,
)
from olmo_core.train.callbacks.callback import Callback
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all, setup_logging

log = logging.getLogger(__name__)

DATA_ROOT = "/weka/oe-training-default/ai2-llm"

SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_SIZE = 1024 * SEQUENCE_LENGTH


# ---------------------------------------------------------------------------
# Gradient mask callback that works with arbitrary expert indices
# ---------------------------------------------------------------------------

def _create_expert_mask_by_indices(
    size: int,
    num_experts: int,
    experts_to_train: List[int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Create a 1D mask that is 1.0 for experts in `experts_to_train` and 0.0 elsewhere.
    Assumes experts are stacked along dimension 0.
    """
    mask = torch.zeros(size, dtype=dtype, device=device)
    expert_size = size // num_experts
    for idx in experts_to_train:
        start = idx * expert_size
        end = start + expert_size
        mask[start:end] = 1.0
    return mask


@dataclass
class SelectedExpertGradientMaskCallback(Callback):
    """
    Masks gradients so that only selected experts (by index) are trained.

    Unlike FrozenExpertGradientMaskCallback which freezes the first N experts,
    this callback allows training an arbitrary set of expert indices.

    :param num_experts: Total number of experts in the model.
    :param experts_to_train: List of expert indices to train. All others are frozen.
    :param layer_patterns: Parameter name patterns to apply masking to.
    """

    priority: ClassVar[int] = 100

    num_experts: int = 128
    experts_to_train: List[int] = field(default_factory=list)
    layer_patterns: List[str] = field(default_factory=lambda: ["experts", "router"])

    _mask_cache: dict = field(default_factory=dict, repr=False)
    _logged_params: bool = field(default=False, repr=False)

    def _should_mask(self, name: str) -> bool:
        return any(pattern in name for pattern in self.layer_patterns)

    def _get_or_create_mask(
        self, name: str, grad: torch.Tensor, full_shape: torch.Size
    ) -> torch.Tensor:
        local_grad = get_local_tensor(grad)
        cache_key = f"{name}_{local_grad.shape}_{local_grad.device}"

        if cache_key not in self._mask_cache:
            if isinstance(grad, DTensor):
                full_mask = _create_expert_mask_by_indices(
                    size=full_shape[0],
                    num_experts=self.num_experts,
                    experts_to_train=self.experts_to_train,
                    dtype=local_grad.dtype,
                    device="cpu",
                )
                if len(full_shape) > 1:
                    full_mask = full_mask.view(-1, *([1] * (len(full_shape) - 1)))
                    full_mask = full_mask.expand(full_shape).contiguous()

                from torch.distributed.tensor import distribute_tensor

                mask_dtensor = distribute_tensor(
                    full_mask.to(local_grad.device),
                    grad.device_mesh,
                    grad.placements,
                )
                mask = get_local_tensor(mask_dtensor)
            else:
                mask = _create_expert_mask_by_indices(
                    size=full_shape[0],
                    num_experts=self.num_experts,
                    experts_to_train=self.experts_to_train,
                    dtype=local_grad.dtype,
                    device=local_grad.device,
                )
                if len(full_shape) > 1:
                    mask = mask.view(-1, *([1] * (len(full_shape) - 1)))
                    mask = mask.expand(full_shape)

            self._mask_cache[cache_key] = mask

        return self._mask_cache[cache_key]

    def pre_optim_step(self):
        masked_count = 0

        for name, param in self.trainer.train_module.model.named_parameters():
            if not self._should_mask(name):
                continue
            if param.grad is None:
                continue

            if isinstance(param, DTensor):
                full_shape = param.shape
            else:
                full_shape = param.shape

            mask = self._get_or_create_mask(name, param.grad, full_shape)
            local_grad = get_local_tensor(param.grad)
            local_grad.mul_(mask)
            masked_count += 1

        if not self._logged_params:
            frozen = self.num_experts - len(self.experts_to_train)
            log.info(
                f"SelectedExpertGradientMask: Masking gradients for {masked_count} parameters "
                f"(training experts {self.experts_to_train}, freezing {frozen}/{self.num_experts} experts)"
            )
            self._logged_params = True


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

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


def split_expert_mlps(model, experts_to_train: List[int]):
    """
    Replace each :class:`DroplessMoEMLP` in the model with a
    :class:`SplitExpertDroplessMoEMLP` that has separate frozen/trainable parameters.
    This must be called after ``model.build()`` but before FSDP wrapping.
    """
    for block in model.blocks:
        if block.feed_forward_moe is None:
            continue
        old_mlp = block.feed_forward_moe.experts.mlp
        if not isinstance(old_mlp, DroplessMoEMLP):
            log.warning(f"Skipping non-DroplessMoEMLP: {type(old_mlp)}")
            continue
        new_mlp = SplitExpertDroplessMoEMLP(
            d_model=old_mlp.d_model,
            hidden_size=old_mlp.hidden_size,
            num_experts=old_mlp.num_experts,
            experts_to_train=experts_to_train,
            dtype=torch.float32,
            init_device="meta",
        )
        block.feed_forward_moe.experts.mlp = new_mlp
        num_experts = old_mlp.num_experts
    log.info(
        f"Split expert MLPs: {len(experts_to_train)} trainable, "
        f"{num_experts - len(experts_to_train)} frozen"
    )


def train(config: ExperimentConfig, experts_to_train: Optional[List[int]] = None,
          split_params: bool = False, eval_only: bool = False):
    if get_rank() == 0:
        rich.print(config)

    seed_all(config.init_seed)

    model = config.model.build(init_device="meta")
    if split_params and experts_to_train is not None:
        split_expert_mlps(model, experts_to_train)
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    checkpoint_loaded = False
    if not trainer.no_checkpoints:
        checkpoint_loaded = trainer.maybe_load_checkpoint()
        if not checkpoint_loaded and config.load_path:
            log.info(
                f"Loading checkpoint from {config.load_path} since no checkpoints were found in the save folder..."
            )
            trainer.load_checkpoint(config.load_path, load_trainer_state=config.load_trainer_state)
            checkpoint_loaded = True

    if eval_only and not checkpoint_loaded:
        raise RuntimeError(
            "Cannot run eval-only mode: no checkpoint found in save folder and no load_path provided."
        )

    if eval_only:
        log.info("Running in eval-only mode: will evaluate checkpoint and exit without training.")

    trainer.fit()


def build_config(opts, overrides: List[str]) -> ExperimentConfig:
    save_folder = opts.save_folder
    if not save_folder:
        save_folder = f"/tmp/{opts.run_name}"

    work_dir = opts.work_dir
    if not work_dir:
        work_dir = "/tmp/dataset-cache"

    tokenizer_config = TokenizerConfig.dolma2()

    # SplitExpertDroplessMoEMLP → frozen expert params are requires_grad=False by construction
    # so we can use the same config for both SplitExpertDroplessMoEMLP and regular DroplessMoEMLP;
    # no need to specify separate freeze_params in the config. We don't do this for router
    # since the router is typically a small fraction of the total parameters, so zeroing out its gradients
    # is cheaper and doesn't require architectural changes.

    if opts.base_model_config:
        config_path = os.path.join(opts.base_model_config, "config.json")
        log.info(f"Loading model config from {config_path}")
        with open(config_path, "r") as f:
            base_config = json.load(f)
        model_config = TransformerConfig.from_dict(base_config["model"])
        model_config.freeze_params = [
            "embeddings.*",
            "blocks.*.attention*",
            "blocks.*.feed_forward_norm.*",
            "lm_head.*",
        ]
    else:
        model_config = TransformerConfig.olmoe_1B_7B(
            vocab_size=tokenizer_config.padded_vocab_size(),
            n_layers=16,
            d_model=2048,
            n_heads=16,
            num_experts=opts.num_experts,
            top_k=8,
            freeze_params=[
                "embeddings.*",
                "blocks.*.attention*",
                "blocks.*.feed_forward_norm.*",
                "lm_head.*",
            ],
        )

    print(model_config)

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
        global_batch_size=GLOBAL_BATCH_SIZE,
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
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0)),
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

    experts_to_train = [int(x) for x in opts.experts_to_train.split(",")]
    assert model_config.block.feed_forward_moe is not None
    num_experts = model_config.block.feed_forward_moe.num_experts
    log.info(f"Experts to train: {experts_to_train} (out of {num_experts} total)")

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
                        "haskell", "go", "python", "cpp", "javascript", "swift",
                        "scala", "bash", "typescript", "c", "php", "rust",
                        "csharp", "r", "ruby", "java", "matlab",
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
        .with_callback(
            "selected_expert_gradient_mask",
            SelectedExpertGradientMaskCallback(
                num_experts=num_experts,
                experts_to_train=experts_to_train,
                layer_patterns=["experts", "router"],
            ),
        )
        .with_callback(
            "hf_converter",
            HFConverterCallback(
                enabled=True,
                dtype=DType.float32,
                max_sequence_length=SEQUENCE_LENGTH,
                device="cpu",
            ),
        )
        .with_callback(
            "split_expert_converter",
            SplitExpertConverterCallback(
                enabled=opts.split_expert_params,
                experts_to_train=experts_to_train,
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
    print("Final merged config:")
    print(config)

    return config


def parser_args():
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        usage=f"python {sys.argv[0]} RUN_NAME [OPTIONS...] [CONFIG_OVERRIDES...]",
        description="Train selected experts (by index) in an MoE model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_name", type=str, help="The name of the run.")
    parser.add_argument(
        "--experts-to-train",
        type=str,
        required=True,
        help="Comma-separated list of expert indices to unfreeze and train (e.g. '5,10,42').",
    )
    parser.add_argument(
        "--base-model-config",
        type=str,
        help="Path to checkpoint directory containing config.json for the base model. "
        "When provided, the model config (including router type) is loaded from this file "
        "instead of using the default olmoe_1B_7B config.",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=128,
        help="Total number of experts in the model (only used when --base-model-config is not provided).",
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        help="A local or remote directory to save checkpoints to.",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        help="A local working directory for dataset preprocessing.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=4e-4,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the config and exit.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluations only on existing checkpoint without training.",
    )
    parser.add_argument(
        "--split-expert-params",
        action="store_true",
        help="Split expert MLP parameters into separate frozen/trainable nn.Parameters. "
        "Saves optimizer memory by not allocating Adam states for frozen experts. "
        "Incompatible with expert parallelism.",
    )
    opts, overrides = parser.parse_known_args()
    return opts, overrides


def main():
    opts, overrides = parser_args()
    setup_logging()

    # Validate expert indices
    if opts.base_model_config:
        config_path = os.path.join(opts.base_model_config, "config.json")
        with open(config_path, "r") as f:
            _cfg = json.load(f)
        num_experts = _cfg["model"]["block"]["feed_forward_moe"]["num_experts"]
    else:
        num_experts = opts.num_experts
    experts_to_train = [int(x) for x in opts.experts_to_train.split(",")]
    for idx in experts_to_train:
        if idx < 0 or idx >= num_experts:
            raise ValueError(
                f"Expert index {idx} is out of range [0, {num_experts}). "
                f"Got --experts-to-train={opts.experts_to_train}"
            )

    config = build_config(opts, overrides)

    if opts.dry_run:
        rich.print(config)
        return

    prepare_training_environment()
    train(
        config,
        experts_to_train=experts_to_train,
        split_params=opts.split_expert_params,
        eval_only=opts.eval_only,
    )
    teardown_training_environment()


if __name__ == "__main__":
    main()
