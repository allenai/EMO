import copy
import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from torch.distributed import DeviceMesh
from torch.distributed.pipelining import PipelineStage

from olmo_core.config import Config, DType, StrEnum
from olmo_core.distributed.parallel import (
    ContextParallelConfig,
    DataParallelConfig,
    DataParallelType,
    ExpertParallelConfig,
    PipelineParallelConfig,
    TensorParallelConfig,
)
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention.ring import (
    RingAttentionLoadBalancerType,
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)
from olmo_core.nn.transformer import (
    Transformer,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelWrappingStrategy,
)
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.train_module.config import TrainModuleConfig

if TYPE_CHECKING:
    from .pipeline_train_module import TransformerPipelineTrainModule
    from .train_module import TransformerTrainModule

log = logging.getLogger(__name__)


class BatchSimulationMethod(StrEnum):
    """Opt-in methods for making a global batch optimize like a smaller batch."""

    none = "none"
    structured_noise = "structured_noise"
    local_sgd = "local_sgd"
    diloco = "diloco"


@dataclass
class BatchSimulationConfig(Config):
    """
    Configure an opt-in smaller-batch optimization simulation.

    The default :data:`method` is ``none`` and exactly preserves normal OLMo-core training.
    Batch sizes are expressed in tokens, consistent with the trainer and data loader.
    """

    method: BatchSimulationMethod = BatchSimulationMethod.none
    global_batch_size: Optional[int] = None
    simulated_batch_size: Optional[int] = None
    local_sgd_sync_interval: int = 1
    diloco_inner_steps: int = 500
    """Number of independent AdamW steps per DiLoCo outer round (``H`` in the paper)."""
    diloco_outer_steps: Optional[List[int]] = None
    """
    Optional exact global optimizer steps at which to apply DiLoCo outer updates.

    When set, this schedule replaces :data:`diloco_inner_steps`. This is useful for schedules
    whose round lengths are not constant, such as synchronizing at exact integer-epoch
    pre-decay frontiers after token-to-step rounding.
    """
    diloco_replica_checkpoint_steps: List[int] = field(default_factory=list)
    """
    Exact outer-update steps at which to save every unaggregated replica model.

    Each replica snapshot is written immediately before its DiLoCo pseudo-gradient is formed and
    before the outer optimizer changes the model. These steps must be a subset of
    :data:`diloco_outer_steps`.
    """
    diloco_outer_lr: float = 0.7
    """Learning rate for the DiLoCo outer Nesterov optimizer."""
    diloco_outer_momentum: float = 0.9
    """Momentum for the DiLoCo outer Nesterov optimizer."""
    recalibrate_second_moment_on_start: bool = False
    """
    Recalibrate AdamW's loaded second moment when a local-update method starts from a
    conventional checkpoint.

    This keeps the bias-corrected first-moment signal estimate fixed and scales only the
    estimated stochastic-gradient variance by ``global_batch_size / simulated_batch_size``.
    Enable this exactly once for a conventional-to-LocalSGD or conventional-to-DiLoCo
    transition. Disable it when resuming a native local-update checkpoint.
    """
    diloco_recalibrate_second_moment_on_start: Optional[bool] = None
    """Deprecated alias for :data:`recalibrate_second_moment_on_start`."""
    seed: int = 0

    def __post_init__(self):
        if self.method == BatchSimulationMethod.none:
            if (
                self.diloco_outer_steps is not None
                or self.diloco_replica_checkpoint_steps
                or self.recalibrate_second_moment_on_start_enabled
            ):
                raise OLMoConfigurationError(
                    "Local-update configuration is only valid for LocalSGD or DiLoCo"
                )
            return
        if self.global_batch_size is None or self.global_batch_size <= 0:
            raise OLMoConfigurationError(
                "'global_batch_size' must be a positive token count when batch simulation is enabled"
            )
        if self.simulated_batch_size is None or self.simulated_batch_size <= 0:
            raise OLMoConfigurationError(
                "'simulated_batch_size' must be a positive token count when batch simulation is enabled"
            )
        if self.simulated_batch_size > self.global_batch_size:
            raise OLMoConfigurationError("'simulated_batch_size' cannot exceed 'global_batch_size'")
        if self.global_batch_size % self.simulated_batch_size != 0:
            raise OLMoConfigurationError(
                "'global_batch_size' must be divisible by 'simulated_batch_size'"
            )
        if self.local_sgd_sync_interval < 1:
            raise OLMoConfigurationError("'local_sgd_sync_interval' must be at least 1")
        if self.method == BatchSimulationMethod.diloco:
            if self.diloco_inner_steps < 1:
                raise OLMoConfigurationError("'diloco_inner_steps' must be at least 1")
            if self.diloco_outer_lr <= 0:
                raise OLMoConfigurationError("'diloco_outer_lr' must be greater than zero")
            if not 0 <= self.diloco_outer_momentum < 1:
                raise OLMoConfigurationError(
                    "'diloco_outer_momentum' must be between zero (inclusive) and one (exclusive)"
                )
            if self.diloco_outer_steps is not None:
                self._validate_step_schedule("diloco_outer_steps", self.diloco_outer_steps)
            if self.diloco_replica_checkpoint_steps:
                self._validate_step_schedule(
                    "diloco_replica_checkpoint_steps",
                    self.diloco_replica_checkpoint_steps,
                )
                if self.diloco_outer_steps is None:
                    raise OLMoConfigurationError(
                        "'diloco_replica_checkpoint_steps' requires 'diloco_outer_steps'"
                    )
                missing_outer_steps = sorted(
                    set(self.diloco_replica_checkpoint_steps) - set(self.diloco_outer_steps)
                )
                if missing_outer_steps:
                    raise OLMoConfigurationError(
                        "'diloco_replica_checkpoint_steps' must be a subset of "
                        f"'diloco_outer_steps'; missing {missing_outer_steps}"
                    )
        elif self.diloco_outer_steps is not None or self.diloco_replica_checkpoint_steps:
            raise OLMoConfigurationError(
                "DiLoCo-specific configuration is only valid when method='diloco'"
            )

        if self.recalibrate_second_moment_on_start_enabled and not self.uses_local_updates:
            raise OLMoConfigurationError(
                "Second-moment recalibration is only valid for LocalSGD or DiLoCo"
            )

    @property
    def recalibrate_second_moment_on_start_enabled(self) -> bool:
        """Resolve the generic option while preserving old DiLoCo launch configurations."""
        if self.diloco_recalibrate_second_moment_on_start is not None:
            return self.diloco_recalibrate_second_moment_on_start
        return self.recalibrate_second_moment_on_start

    @staticmethod
    def _validate_step_schedule(name: str, steps: List[int]) -> None:
        if not steps:
            raise OLMoConfigurationError(f"'{name}' cannot be empty")
        if any(not isinstance(step, int) or isinstance(step, bool) or step < 1 for step in steps):
            raise OLMoConfigurationError(f"'{name}' must contain only positive integer steps")
        if steps != sorted(set(steps)):
            raise OLMoConfigurationError(f"'{name}' must be strictly increasing and unique")

    @property
    def num_ghost_batches(self) -> int:
        if self.method == BatchSimulationMethod.none:
            return 1
        assert self.global_batch_size is not None
        assert self.simulated_batch_size is not None
        return self.global_batch_size // self.simulated_batch_size

    @property
    def enabled(self) -> bool:
        return self.method != BatchSimulationMethod.none

    @property
    def uses_local_updates(self) -> bool:
        """Whether each data-parallel replica performs independent optimizer updates."""
        return self.method in (BatchSimulationMethod.local_sgd, BatchSimulationMethod.diloco)

    @property
    def local_update_sync_interval(self) -> int:
        """Number of independent inner steps between replica synchronizations."""
        if self.method == BatchSimulationMethod.diloco:
            return self.diloco_inner_steps
        return self.local_sgd_sync_interval

    @property
    def uses_exact_diloco_outer_steps(self) -> bool:
        """Whether DiLoCo outer updates follow exact global steps instead of fixed ``H``."""
        return self.method == BatchSimulationMethod.diloco and self.diloco_outer_steps is not None

    def is_diloco_outer_step(self, step: int) -> bool:
        """Return whether ``step`` is an explicitly scheduled DiLoCo outer update."""
        return self.diloco_outer_steps is not None and step in self.diloco_outer_steps

    def should_save_diloco_replicas(self, step: int) -> bool:
        """Return whether raw replica models should be saved before the outer update at ``step``."""
        return step in self.diloco_replica_checkpoint_steps


@beta_feature
@dataclass
class TransformerPipelineParallelConfig(PipelineParallelConfig):
    """
    Transformer-specific pipeline parallel config.
    """

    split_points: Optional[List[int]] = None
    """
    A list of unique, increasing block indices that define how to split the model into stages.

    For example, ``split_points = [0, 2]`` with a 4-layer model means the model will be split into
    3 stages, with the first containing just the embedding, the second containing blocks 0 and 1,
    and the third containing blocks 2 and 3 and the language modeling head.

    If not specified the split points are determined automatically based on the schedule type.
    """

    def get_split_points(self, n_layers: int) -> List[int]:
        if self.split_points is not None:
            return self.split_points

        # Multi-stage schedules support more than 2 stages per rank, but this is the default if
        # no pipeline split is specified.
        num_stages_per_rank = 1 if self.schedule.is_single_stage else 2
        total_stages = self.degree * num_stages_per_rank
        num_layers = n_layers
        if total_stages > num_layers:
            raise OLMoConfigurationError("Total stages cannot be greater than the number of layers")

        base_interval = num_layers // total_stages
        extra_layers = num_layers % total_stages

        splits: List[int] = []
        current_layer = 0
        for i in range(total_stages - 1):
            if i == 0:
                current_layer += base_interval
            else:
                # Middle stages get an extra layer if there are any remaining
                if extra_layers > 0:
                    current_layer += base_interval + 1
                    extra_layers -= 1
                else:
                    current_layer += base_interval
            splits.append(current_layer)
        log.info(f"Auto generated pipeline split points will be {splits}")
        return splits

    def split_model(
        self, model: Transformer, *, pp_mesh: DeviceMesh, device: torch.device
    ) -> Tuple[List[PipelineStage], List[Transformer]]:
        split_points = self.get_split_points(model.n_layers)
        pp_rank = pp_mesh.get_local_rank()

        def build_stage(
            stage_idx: int,
            start_layer: Optional[int],
            stop_layer: Optional[int],
            is_first: bool = False,
            is_last: bool = False,
        ) -> Tuple[PipelineStage, Transformer]:
            model_chunk = copy.deepcopy(model)
            if not is_first:
                model_chunk.embeddings = None  # type: ignore

            drop_layers = start_layer is not None
            for block_idx in range(model.n_layers):
                # we keep layers in a contiguous region between start (inclusive) and stop (exclusive)
                if block_idx == start_layer:
                    drop_layers = False
                if block_idx == stop_layer:
                    drop_layers = True
                if drop_layers:
                    del model_chunk.blocks[str(block_idx)]

            if not is_last:
                model_chunk.lm_head = None  # type: ignore

            stage = PipelineStage(
                model_chunk,
                stage_idx,
                num_stages,
                device,
                group=pp_mesh.get_group("pp"),
            )
            return stage, model_chunk

        num_stages = len(split_points) + 1
        stage_idx = pp_rank

        stages = []
        models = []
        for stage_idx in self.stage_ids_this_rank(pp_rank, num_stages):
            start_layer = split_points[stage_idx - 1] if stage_idx > 0 else None
            stop_layer = split_points[stage_idx] if stage_idx < num_stages - 1 else None
            stage, model_chunk = build_stage(
                stage_idx,
                start_layer,
                stop_layer,
                is_first=stage_idx == 0,
                is_last=stage_idx == num_stages - 1,
            )
            log.info(
                f"PP rank {pp_rank} is building stage {stage_idx} with start layer "
                f"{start_layer}, stop layer {stop_layer}: {model_chunk}"
            )
            stages.append(stage)
            models.append(model_chunk)

        return stages, models


@dataclass
class TransformerDataParallelConfig(DataParallelConfig):
    """
    Transformer-specific data parallel config.
    """

    wrapping_strategy: TransformerDataParallelWrappingStrategy = (
        TransformerDataParallelWrappingStrategy.full
    )
    """
    The wrapping strategy.
    """

    prefetch_factor: int = 0


@dataclass
class TransformerTensorParallelConfig(TensorParallelConfig):
    """
    Transformer-specific tensor parallel config.
    """


@dataclass
class TransformerContextParallelConfig(ContextParallelConfig):
    """
    Transformer-specific context parallel config.
    """

    ring: RingContextParallelStyle | None = None
    uly: UlyssesContextParallelStyle | None = None

    def __post_init__(self):
        if self.ring is not None and self.uly is not None:
            raise NotImplementedError(
                "Only one of ring or ulysses can be specified. While not technically "
                "mutually exclusive, a combined context parallel style is not yet supported."
            )
        elif self.ring is None and self.uly is None:
            raise OLMoConfigurationError("One of ring or uly must be specified")

    @classmethod
    def zig_zag(cls, degree: int, head_stride: int = 1) -> "TransformerContextParallelConfig":
        return cls(
            degree=degree,
            ring=RingContextParallelStyle(
                load_balancer=RingAttentionLoadBalancerType.zig_zag,
                head_stride=head_stride,
            ),
        )

    @classmethod
    def llama3(cls, degree: int, head_stride: int = 1) -> "TransformerContextParallelConfig":
        return cls(
            degree=degree,
            ring=RingContextParallelStyle(
                load_balancer=RingAttentionLoadBalancerType.llama3,
                head_stride=head_stride,
            ),
        )

    @classmethod
    def ulysses(cls, degree: int) -> "TransformerContextParallelConfig":
        return cls(
            degree=degree,
            uly=UlyssesContextParallelStyle(),
        )


@dataclass
class TransformerExpertParallelConfig(ExpertParallelConfig):
    """
    Transformer-specific expert parallel config.
    """


@beta_feature
@dataclass
class TransformerActivationCheckpointingConfig(Config):
    """
    Defines the activation checkpointing strategy for a transformer model.
    """

    mode: TransformerActivationCheckpointingMode = TransformerActivationCheckpointingMode.full
    """
    The activation checkpointing mode.
    """

    block_interval: Optional[int] = None
    """
    Required when :data:`mode` is "selected_blocks". Determines which blocks are wrapped.
    """

    modules: Optional[List[str]] = None
    """
    Required when :data:`mode` is "selected_modules". A list of modules names to wrap for
    activation checkpointing. Globs are supported.
    """

    activation_memory_budget: Optional[float] = None
    """
    Required when :data:`mode` is "budget". Memory budget for activation checkpointing in range [0, 1].
    0 = recompute all activations, 1 = recompute none (default). Requires compilation to be enabled.

    See https://pytorch.org/blog/activation-checkpointing-techniques/ for more details.
    """

    def __post_init__(self):
        if (
            self.mode == TransformerActivationCheckpointingMode.selected_blocks
            and self.block_interval is None
        ):
            raise OLMoConfigurationError(
                "'block_interval' is required for 'selected_blocks' activation checkpointing"
            )
        elif (
            self.mode == TransformerActivationCheckpointingMode.selected_modules
            and self.modules is None
        ):
            raise OLMoConfigurationError(
                "'modules' is required for 'selected_modules' activation checkpointing"
            )


@dataclass
class TransformerTrainModuleConfig(TrainModuleConfig):
    """
    A configuration class for building :class:`TransformerTrainModule` or
    :class:`TransformerPipelineTrainModule` instances.

    .. seealso::
        See the :class:`TransformerTrainModule` and :class:`TransformerPipelineTrainModule`
        documentation for a description of the fields.
    """

    rank_microbatch_size: int
    max_sequence_length: int

    # Optimizer settings.

    optim: OptimConfig
    max_grad_norm: Optional[float] = None
    scheduler: Optional[Scheduler] = None

    # Model settings.

    compile_model: bool = False
    float8_config: Optional[Float8Config] = None
    pp_config: Optional[TransformerPipelineParallelConfig] = None
    dp_config: Optional[TransformerDataParallelConfig] = None
    tp_config: Optional[TransformerTensorParallelConfig] = None
    cp_config: Optional[TransformerContextParallelConfig] = None
    ep_config: Optional[TransformerExpertParallelConfig] = None
    ac_config: Optional[TransformerActivationCheckpointingConfig] = None

    # Loss function settings.

    z_loss_multiplier: Optional[float] = None

    # Checkpoint settings.

    state_dict_save_opts: Optional[Dict[str, Any]] = None
    state_dict_load_opts: Optional[Dict[str, Any]] = None
    load_key_mapping: Optional[Dict[str, str]] = None
    validate_optimizer_hyperparameters_on_load: bool = False

    # Other train settings.

    autocast_precision: Optional[DType] = None
    label_ignore_index: int = -100
    batch_simulation: BatchSimulationConfig = field(default_factory=BatchSimulationConfig)

    def build(
        self,
        model: Transformer,
        device: Optional[torch.device] = None,
    ) -> Union["TransformerTrainModule", "TransformerPipelineTrainModule"]:
        """
        Build the corresponding :class:`TransformerTrainModule` or :class:`TransformerPipelineTrainModule.

        :param model: The :class:`~olmo_core.nn.transformer.Transformer` model to train.
        :param device: The device to train on.
        """
        from .pipeline_train_module import TransformerPipelineTrainModule
        from .train_module import TransformerTrainModule

        kwargs = self.as_dict(exclude_none=True, recurse=False)
        batch_simulation = kwargs.pop("batch_simulation")
        if (autocast_precision := kwargs.pop("autocast_precision", None)) is not None:
            kwargs["autocast_precision"] = cast(DType, autocast_precision).as_pt()
        if (state_dict_save_opts := kwargs.pop("state_dict_save_opts", None)) is not None:
            kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(**state_dict_save_opts)
        if (state_dict_load_opts := kwargs.pop("state_dict_load_opts", None)) is not None:
            kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(**state_dict_load_opts)

        if self.pp_config is not None:
            if batch_simulation.enabled:
                raise OLMoConfigurationError(
                    "batch simulation is not currently compatible with pipeline parallelism"
                )
            return TransformerPipelineTrainModule(
                model=model,
                device=device,
                **kwargs,
            )
        else:
            if batch_simulation.uses_local_updates:
                if (
                    self.tp_config is not None
                    or self.cp_config is not None
                    or self.ep_config is not None
                ):
                    raise OLMoConfigurationError(
                        "local-update batch simulation currently requires pure data parallelism"
                    )
                if self.dp_config is None or self.dp_config.name not in (
                    DataParallelType.fsdp,
                    DataParallelType.hsdp,
                ):
                    raise OLMoConfigurationError(
                        "local-update batch simulation requires FSDP or HSDP data parallelism"
                    )

                # Each HSDP replica consumes one simulated batch while parameters remain sharded
                # within a replica if there are more ranks than replicas.
                effective_dp_config = replace(
                    self.dp_config,
                    name=DataParallelType.hsdp,
                    num_replicas=batch_simulation.num_ghost_batches,
                    shard_degree=None,
                )
                self.dp_config = effective_dp_config
                kwargs["dp_config"] = effective_dp_config

            return TransformerTrainModule(
                model=model,
                device=device,
                batch_simulation=batch_simulation,
                **kwargs,
            )


@beta_feature
@dataclass
class TransformerPipelineTrainModuleConfig(TransformerTrainModuleConfig):
    """
    Kept for backwards compatibility, but please use :class:`TransformerTrainModuleConfig` instead.
    """

    def __post_init__(self):
        if self.pp_config is None:
            raise OLMoConfigurationError("'pp_config' is required")
