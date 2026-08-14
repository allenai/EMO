import contextlib
import json
import logging
from dataclasses import replace
from functools import cached_property, lru_cache
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

from olmo_core.data.utils import get_labels, split_batch
from olmo_core.distributed.checkpoint import (
    merge_state_dicts,
    prune_state_dict,
    save_state_dict,
    swap_param_keys,
)
from olmo_core.distributed.parallel import (
    DataParallelType,
    build_world_mesh,
    get_dp_process_group,
    get_dp_replicate_mesh,
    get_dp_shard_mesh,
)
from olmo_core.distributed.utils import (
    barrier,
    get_local_tensor,
    get_rank,
    get_reduce_divide_factor,
    get_world_size,
    is_distributed,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.io import join_path
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.nn.transformer import Transformer
from olmo_core.nn.transformer.config import TransformerActivationCheckpointingMode
from olmo_core.optim import OptimConfig, SkipStepOptimizer
from olmo_core.optim.scheduler import Scheduler
from olmo_core.utils import (
    gc_cuda,
    get_default_device,
    log_once,
    move_to_device,
    warn_once,
)

from ...common import ReduceType
from ..train_module import EvalBatchSpec, TrainModule, assert_optimizer_lrs_nonzero
from .batch_simulation import (
    average_module_and_optimizer_state,
    clone_local_parameter_tensors,
    diloco_outer_step,
    recalibrate_adam_second_moment_for_batch_size,
    structured_noise_loss_scales,
)
from .common import parallelize_model
from .config import (
    BatchSimulationConfig,
    BatchSimulationMethod,
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
    TransformerTensorParallelConfig,
)

log = logging.getLogger(__name__)

_DILOCO_INNER_OPTIM_PREFIX = "diloco_inner_optim_replica_"


def optimizer_hyperparameters(optim: Optimizer) -> List[Dict[str, Any]]:
    """Snapshot every optimizer parameter-group field except parameter references."""

    def normalize(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            return value.item() if value.numel() == 1 else value.tolist()
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(normalize(item) for item in value)
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    return [
        {key: normalize(value) for key, value in group.items() if key != "params"}
        for group in optim.param_groups
    ]


def assert_optimizer_hyperparameters_match(
    expected: List[Dict[str, Any]], actual: List[Dict[str, Any]]
) -> None:
    """Assert that checkpoint loading did not change any optimizer hyperparameter."""
    if expected == actual:
        return
    details = []
    missing = object()
    for group_idx in range(max(len(expected), len(actual))):
        expected_group = expected[group_idx] if group_idx < len(expected) else {}
        actual_group = actual[group_idx] if group_idx < len(actual) else {}
        for key in sorted(set(expected_group) | set(actual_group)):
            expected_value = expected_group.get(key, missing)
            actual_value = actual_group.get(key, missing)
            if expected_value != actual_value:
                details.append(
                    f"group {group_idx} {key}: checkpoint "
                    f"{('<missing>' if actual_value is missing else repr(actual_value))} "
                    f"!= command "
                    f"{('<missing>' if expected_value is missing else repr(expected_value))}"
                )
    raise OLMoConfigurationError(
        "Checkpoint optimizer hyperparameters do not exactly match the command config:\n - "
        + "\n - ".join(details)
    )


class TransformerTrainModule(TrainModule):
    """
    A :class:`TrainModule` for any :class:`~olmo_core.nn.transformer.Transformer` model
    implementation provided by this library.

    .. tip::
        Use the :class:`TransformerTrainModuleConfig` to easily configure and build
        :class:`TransformerTrainModule` instances.

    :param model: The :class:`~olmo_core.nn.transformer.Transformer` model to train.
    :param optim: The corresponding optimizer config.
    :param rank_microbatch_size: The microbatch size *in tokens* per rank,
        i.e. the number of tokens to process at a time from each rank.

        .. note:: This must evenly divide into the global batch size by a factor of the data
            parallel world size. If this is less than the global batch divided by the data
            parallel world size then gradient accumulation is used.
    :param max_sequence_length: The maximum expected sequence length during training and evaluation.
    :param compile_model: Whether to compile to the model.
    :param float8_config: Float8 configuration for the model.
    :param dp_config: Data parallel configuration for the model.
    :param tp_config: Tensor parallel configuration for the model.
    :param cp_config: Context parallel configuration for the model.
    :param ac_config: Activation checkpointing configuration for the model.
    :param z_loss_multiplier: Use Z-loss with this multiplier.
    :param autocast_precision: Enable AMP with this data type.
    :param max_grad_norm: Clip gradient norms to this value.
    :param scheduler: Optional learning rate scheduler for the optimizer.
    :param device: The device to train on.
    :param state_dict_save_opts: Can be used to override the state dict options used
        when saving a checkpoint.
    :param state_dict_load_opts: Can be used to override the state dict options used
        when loading a checkpoint.
    :param load_key_mapping: Can be used to load a checkpoint where certain parameter have different names.
        This dictionary should map current keys to keys in the checkpoint to be loaded.
    :param validate_optimizer_hyperparameters_on_load: Abort a checkpoint load if any optimizer
        parameter-group setting differs from the optimizer built from the command config.
    :param batch_simulation: Optional structured-noise, local-SGD, or DiLoCo smaller-batch
        simulation. The default configuration is disabled and preserves normal optimization.
    """

    def __init__(
        self,
        model: Transformer,
        optim: OptimConfig,
        rank_microbatch_size: int,
        max_sequence_length: int,
        compile_model: bool = False,
        float8_config: Optional[Float8Config] = None,
        dp_config: Optional[TransformerDataParallelConfig] = None,
        tp_config: Optional[TransformerTensorParallelConfig] = None,
        cp_config: Optional[TransformerContextParallelConfig] = None,
        ep_config: Optional[TransformerExpertParallelConfig] = None,
        ac_config: Optional[TransformerActivationCheckpointingConfig] = None,
        z_loss_multiplier: Optional[float] = None,
        autocast_precision: Optional[torch.dtype] = None,
        max_grad_norm: Optional[float] = None,
        scheduler: Optional[Scheduler] = None,
        device: Optional[torch.device] = None,
        state_dict_save_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        state_dict_load_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        load_key_mapping: Optional[Dict[str, str]] = None,
        label_ignore_index: int = -100,
        batch_simulation: Optional[BatchSimulationConfig] = None,
        validate_optimizer_hyperparameters_on_load: bool = False,
    ):
        super().__init__()
        batch_simulation = batch_simulation or BatchSimulationConfig()

        # Validate some options.
        if rank_microbatch_size % max_sequence_length != 0:
            raise OLMoConfigurationError(
                f"'rank_microbatch_size' ({rank_microbatch_size:,d} tokens) must be divisible by "
                f"'max_sequence_length' ({max_sequence_length:,d} tokens)"
            )

        # Build world mesh.
        self.device = device or get_default_device()
        self.world_mesh: Optional[DeviceMesh] = None
        if is_distributed():
            self.world_mesh = build_world_mesh(
                dp=dp_config, tp=tp_config, cp=cp_config, ep=ep_config, device_type=self.device.type
            )
            log.info(f"Data parallel world size = {get_world_size(self.dp_process_group):,d}")
        elif (
            dp_config is not None
            or tp_config is not None
            or ep_config is not None
            or cp_config is not None
        ):
            raise OLMoConfigurationError(
                "Training parallelism configs are only valid for distributed training"
            )

        if (
            ac_config is not None
            and ac_config.mode == TransformerActivationCheckpointingMode.budget
            and not compile_model
        ):
            raise OLMoConfigurationError(
                "Activation checkpointing with 'budget' mode requires compilation to be enabled"
            )

        # Parallelize model. Local-update methods use the full HSDP-shaped world mesh for data
        # assignment and replica groups, but each replica must expose its own local
        # gradients to its optimizer. Native HSDP's ``set_requires_all_reduce(False)``
        # is an accumulation mode: it retains gradients in a private partial-reduction
        # buffer until a later all-reduce, so an optimizer step in between sees no grads.
        # Wrapping on the shard sub-mesh gives every replica an independent FSDP model;
        # ``_maybe_sync_local_sgd`` remains the only cross-replica synchronization.
        dp_model_mesh: Optional[DeviceMesh] = None
        if batch_simulation.uses_local_updates and self.world_mesh is not None:
            dp_model_mesh = get_dp_shard_mesh(self.world_mesh)

        self.model = parallelize_model(
            model,
            world_mesh=self.world_mesh,
            dp_model_mesh=dp_model_mesh,
            device=self.device,
            max_sequence_length=max_sequence_length,
            rank_microbatch_size=rank_microbatch_size,
            compile_model=compile_model,
            float8_config=float8_config,
            dp_config=dp_config,
            tp_config=tp_config,
            cp_config=cp_config,
            ep_config=ep_config,
            ac_config=ac_config,
        )
        self._model_mode: Optional[Literal["train", "eval"]] = None

        self._dp_config = dp_config
        self._cp_config = cp_config
        self._tp_config = tp_config
        self._ep_config = ep_config
        self.label_ignore_index = label_ignore_index
        self.z_loss_multiplier = z_loss_multiplier
        self.rank_microbatch_size = rank_microbatch_size
        self.max_sequence_length = max_sequence_length
        self.autocast_precision = autocast_precision
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler
        self.batch_simulation = batch_simulation
        self._local_sgd_steps_since_sync = 0
        self.state_dict_save_opts = state_dict_save_opts or dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, cpu_offload=True
        )
        self.state_dict_load_opts = state_dict_load_opts or dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, strict=True
        )
        self.load_key_mapping = load_key_mapping
        self.validate_optimizer_hyperparameters_on_load = validate_optimizer_hyperparameters_on_load

        # Build optimizer(s).
        log.info("Building optimizer...")
        self.optim: Optimizer = optim.build(self.model, strict=True)
        self._diloco_outer_optim: Optional[Optimizer] = None
        self._diloco_outer_parameters: list[torch.Tensor] = []
        if self.batch_simulation.method == BatchSimulationMethod.diloco:
            self._diloco_outer_optim = torch.optim.SGD(
                (parameter for parameter in self.model.parameters() if parameter.requires_grad),
                lr=self.batch_simulation.diloco_outer_lr,
                momentum=self.batch_simulation.diloco_outer_momentum,
                weight_decay=0.0,
                nesterov=self.batch_simulation.diloco_outer_momentum > 0,
            )
            self._diloco_outer_parameters = clone_local_parameter_tensors(self.model)

    @property
    def dp_process_group(self) -> Optional[dist.ProcessGroup]:
        return None if self.world_mesh is None else get_dp_process_group(self.world_mesh)

    @property
    def eval_batch_spec(self) -> EvalBatchSpec:
        return EvalBatchSpec(
            self.rank_microbatch_size,
            max_sequence_length=self.max_sequence_length,
            #  fixed_sequence_length=self.tp_enabled,
        )

    @property
    def dp_config(self) -> Optional[TransformerDataParallelConfig]:
        return self._dp_config

    @property
    def tp_enabled(self) -> bool:
        return self._tp_config is not None

    @property
    def cp_enabled(self) -> bool:
        return self._cp_config is not None

    @property
    def ep_enabled(self) -> bool:
        return self._ep_config is not None

    @cached_property
    def world_size(self) -> int:
        return get_world_size()

    @cached_property
    def _reduce_divide_factor(self) -> float:
        return get_reduce_divide_factor(self.world_size)

    def pre_train(self):
        # Validate batch size.
        # NOTE: we run this in `pre_train()` instead of, say, `on_attach()` because callbacks
        # like `BatchSizeScheduler` may change the global batch size after the module is attached.
        dp_ws = get_world_size(self.trainer.dp_process_group)
        if self.trainer.global_batch_size % (self.rank_microbatch_size * dp_ws) != 0:
            raise OLMoConfigurationError(
                f"global batch size ({self.trainer.global_batch_size:,d}) must be divisible by "
                f"micro-batch size ({self.rank_microbatch_size:,d}) x DP world size ({dp_ws})"
            )

        if self.batch_simulation.enabled:
            configured_global_batch_size = self.batch_simulation.global_batch_size
            assert configured_global_batch_size is not None
            if configured_global_batch_size != self.trainer.global_batch_size:
                raise OLMoConfigurationError(
                    "batch simulation 'global_batch_size' "
                    f"({configured_global_batch_size:,d}) does not match the trainer's global "
                    f"batch size ({self.trainer.global_batch_size:,d})"
                )
            assert self.batch_simulation.simulated_batch_size is not None
            if self.batch_simulation.simulated_batch_size % self.max_sequence_length != 0:
                raise OLMoConfigurationError(
                    "batch simulation 'simulated_batch_size' must be divisible by "
                    "'max_sequence_length'"
                )

        if self.batch_simulation.method == BatchSimulationMethod.structured_noise:
            rank_batch_size = self.trainer.global_batch_size // dp_ws
            if rank_batch_size % self.batch_simulation.num_ghost_batches != 0:
                raise OLMoConfigurationError(
                    "each rank's batch must divide evenly into structured-noise ghost batches"
                )
            rank_ghost_batch_size = rank_batch_size // self.batch_simulation.num_ghost_batches
            if rank_ghost_batch_size % self.max_sequence_length != 0:
                raise OLMoConfigurationError(
                    "each rank's structured-noise ghost batch must contain a whole number of sequences"
                )

        if self.batch_simulation.uses_local_updates:
            if self.world_mesh is None or self.dp_config is None:
                raise OLMoConfigurationError(
                    "local-update batch simulation requires distributed HSDP"
                )
            if self.dp_config.name != DataParallelType.hsdp:
                raise OLMoConfigurationError("local-update batch simulation requires HSDP")
            replica_count = get_dp_replicate_mesh(self.world_mesh).size()
            if replica_count != self.batch_simulation.num_ghost_batches:
                raise OLMoConfigurationError(
                    f"local-update method expected "
                    f"{self.batch_simulation.num_ghost_batches} replicas, "
                    f"but the world mesh contains {replica_count}"
                )

    def state_dict(self, *, optim: Optional[bool] = None) -> Dict[str, Any]:
        self._maybe_sync_local_sgd(force=True)
        if optim is None:
            optim = True
        return self._get_state_dict(self.state_dict_save_opts, optim=optim)

    def state_dict_to_load(
        self, metadata: Metadata, *, optim: Optional[bool] = None
    ) -> Dict[str, Any]:
        local_inner_optim_key = self._inner_optimizer_state_key()
        has_common_optim_state: bool = False
        has_local_inner_optim_state: bool = False
        has_replica_inner_optim_state: bool = False
        has_diloco_outer_optim_state: bool = False
        for key in metadata.state_dict_metadata.keys():
            if key.startswith("optim."):
                has_common_optim_state = True
            elif key.startswith(f"{local_inner_optim_key}."):
                has_local_inner_optim_state = True
                has_replica_inner_optim_state = True
            elif key.startswith(_DILOCO_INNER_OPTIM_PREFIX):
                has_replica_inner_optim_state = True
            elif key.startswith("diloco_outer_optim."):
                has_diloco_outer_optim_state = True

        if (
            self.batch_simulation.method == BatchSimulationMethod.diloco
            and has_replica_inner_optim_state
            and not has_local_inner_optim_state
            and optim is not False
        ):
            raise RuntimeError(
                f"Checkpoint does not contain inner optimizer state for this DiLoCo replica "
                f"('{local_inner_optim_key}')"
            )

        inner_optim_key = local_inner_optim_key if has_local_inner_optim_state else "optim"
        has_optim_state = has_local_inner_optim_state or has_common_optim_state

        if optim is None:
            if not has_optim_state:
                log.warning("No optimizer state found in checkpoint")
                optim = False
            else:
                optim = True

        load_opts = self.state_dict_load_opts
        if optim:
            if not has_optim_state:
                raise RuntimeError(
                    "Checkpoint does not contain optimizer state, but 'optim=True' was requested"
                )

            if f"{inner_optim_key}.param_groups.0.params" in metadata.state_dict_metadata:
                # unflattened optimizer state
                if load_opts.flatten_optimizer_state_dict:
                    log.warning(
                        "Loading checkpoint with an unflattened optimizer state even though "
                        "'flatten_optimizer_state_dict=True' in train module's 'state_dict_load_opts', "
                        "automatically switching to 'flatten_optimizer_state_dict=False'."
                    )
                    load_opts = replace(load_opts, flatten_optimizer_state_dict=False)
            else:
                # flattened optimizer state
                if not load_opts.flatten_optimizer_state_dict:
                    log.warning(
                        "Loading checkpoint with a flattened optimizer state even though "
                        "'flatten_optimizer_state_dict=False' in train module's 'state_dict_load_opts', "
                        "automatically switching to 'flatten_optimizer_state_dict=True'."
                    )
                    load_opts = replace(load_opts, flatten_optimizer_state_dict=True)

        state_dict = self._get_state_dict(
            load_opts,
            optim=optim,
            inner_optimizer_key=inner_optim_key,
            include_diloco_outer_optim=bool(
                optim and has_diloco_outer_optim_state and self._diloco_outer_optim is not None
            ),
        )
        if self.load_key_mapping is not None:
            swap_param_keys(
                state_dict,
                self.load_key_mapping,
                metadata=metadata,
                optimizer_keys=(inner_optim_key, "diloco_outer_optim"),
            )

        if not load_opts.strict:
            # Remove any keys in the 'state_dict' that are not present in the checkpoint.
            pruned_keys = prune_state_dict(state_dict, set(metadata.state_dict_metadata.keys()))
            if pruned_keys:
                log.warning(f"Checkpoint is missing the following keys: {pruned_keys}")

        return state_dict

    def state_dict_to_save(self, *, optim: Optional[bool] = None) -> Dict[str, Any]:
        self._maybe_sync_local_sgd(force=True)
        if optim is None:
            optim = True
        return self._get_state_dict(self.state_dict_save_opts, optim=optim)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        local_inner_optim_key = self._inner_optimizer_state_key()
        inner_optim_key = local_inner_optim_key if local_inner_optim_key in state_dict else "optim"
        load_optim = inner_optim_key in state_dict
        load_diloco_outer_optim = "diloco_outer_optim" in state_dict
        expected_optim_hyperparameters = (
            optimizer_hyperparameters(self.optim)
            if load_optim and self.validate_optimizer_hyperparameters_on_load
            else None
        )
        expected_diloco_outer_hyperparameters = (
            optimizer_hyperparameters(self._diloco_outer_optim)
            if load_diloco_outer_optim
            and self._diloco_outer_optim is not None
            and self.validate_optimizer_hyperparameters_on_load
            else None
        )

        if self.load_key_mapping is not None:
            swap_param_keys(
                state_dict,
                self.load_key_mapping,
                reverse=True,
                quiet=True,
                optimizer_keys=(inner_optim_key, "diloco_outer_optim"),
            )

        # NOTE: `dist_cp_sd.set_(model|optimizer)_state_dict()` doesn't respect `strict=False`
        # option with missing keys, so we have to handle that on our own.
        if not self.state_dict_load_opts.strict:
            flatten_optimizer_state_dict = (
                False if not load_optim else ("state" not in state_dict[inner_optim_key])
            )
            load_opts = replace(
                self.state_dict_load_opts, flatten_optimizer_state_dict=flatten_optimizer_state_dict
            )
            full_state_dict = self._get_state_dict(
                load_opts,
                optim=load_optim,
                inner_optimizer_key=inner_optim_key,
                include_diloco_outer_optim=load_diloco_outer_optim,
            )
            merge_state_dicts(state_dict, full_state_dict)

        dist_cp_sd.set_model_state_dict(
            self.model,
            state_dict["model"],
            options=self.state_dict_load_opts,
        )
        gc_cuda()
        if load_optim:
            dist_cp_sd.set_optimizer_state_dict(
                self.model,
                self.optim,
                state_dict[inner_optim_key],
                options=self.state_dict_load_opts,
            )
            if expected_optim_hyperparameters is not None:
                assert_optimizer_hyperparameters_match(
                    expected_optim_hyperparameters,
                    optimizer_hyperparameters(self.optim),
                )
            if (
                self.batch_simulation.uses_local_updates
                and self.batch_simulation.recalibrate_second_moment_on_start_enabled
                and inner_optim_key == "optim"
            ):
                adjusted = recalibrate_adam_second_moment_for_batch_size(
                    self.optim,
                    batch_size_ratio=float(self.batch_simulation.num_ghost_batches),
                )
                log.info(
                    "Recalibrated Adam second moments for %d parameter states at batch-size "
                    "ratio %.3f while starting %s from a conventional checkpoint",
                    adjusted,
                    float(self.batch_simulation.num_ghost_batches),
                    self.batch_simulation.method.value,
                )
            gc_cuda()
        if load_diloco_outer_optim:
            if self._diloco_outer_optim is None:
                raise RuntimeError(
                    "Checkpoint contains DiLoCo outer optimizer state, but DiLoCo is disabled"
                )
            dist_cp_sd.set_optimizer_state_dict(
                self.model,
                self._diloco_outer_optim,
                state_dict["diloco_outer_optim"],
                options=self.state_dict_load_opts,
            )
            if expected_diloco_outer_hyperparameters is not None:
                assert_optimizer_hyperparameters_match(
                    expected_diloco_outer_hyperparameters,
                    optimizer_hyperparameters(self._diloco_outer_optim),
                )
            gc_cuda()
        elif self._diloco_outer_optim is not None:
            # Starting DiLoCo from a conventional checkpoint begins with zero outer momentum.
            self._diloco_outer_optim.state.clear()
        self._local_sgd_steps_since_sync = 0
        if self._diloco_outer_optim is not None:
            self._diloco_outer_parameters = clone_local_parameter_tensors(self.model)

    def validate_load_path_checkpoint(self) -> None:
        assert_optimizer_lrs_nonzero(self.optim)
        if self._diloco_outer_optim is not None and self._diloco_outer_optim.state:
            assert_optimizer_lrs_nonzero(self._diloco_outer_optim)

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        # Set model to train mode if it isn't already.
        self._set_model_mode("train")

        # Generate labels.
        if "labels" not in batch:
            batch["labels"] = get_labels(batch, label_ignore_index=self.label_ignore_index)

        # Calculate how many tokens will be used in the loss.
        batch_num_tokens = batch["labels"].numel()
        batch_num_tokens_per_instance = batch["labels"].shape[1]
        batch_num_tokens_for_loss = move_to_device(
            (batch["labels"] != self.label_ignore_index).sum(), self.device
        )

        # Record percentage of masked labels.
        self.record_metric(
            "train/masked labels (%)",  # just a proportion, not a percentage
            (batch_num_tokens - batch_num_tokens_for_loss) / batch_num_tokens,
            ReduceType.mean,
        )

        # Record percentage of masked instances.
        if (instance_mask := batch.get("instance_mask")) is not None:
            self.record_metric(
                "train/masked instances (%)",  # just a proportion, not a percentage
                (~instance_mask).float().mean(),
                ReduceType.mean,
            )

            # WARN: When we mask out instances with the instance filter, we count those tokens
            # for the loss anyways. They will count as tokens with a zero loss. This means we
            # get an artificially *low* loss for these batches. But it is really hard (and slow)
            # to do this properly in a distributed setup. We add back in the full number of tokens
            # for the loss so that each rank contributes to the loss calculation fairly.
            batch_num_tokens_for_loss += (~instance_mask).sum() * batch_num_tokens_per_instance

        # Batch losses to record.
        ce_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        z_batch_loss: Optional[torch.Tensor] = None
        if self.z_loss_multiplier is not None:
            z_batch_loss = move_to_device(torch.tensor(0.0), self.device)

        # Split into micro-batches.
        if self.rank_microbatch_size < (seq_len := batch["input_ids"].shape[1]):
            raise RuntimeError(
                f"Microbatch size ({self.rank_microbatch_size}) is too small relative to sequence length ({seq_len})"
            )
        microbatch_instances = self.rank_microbatch_size // seq_len
        if self.batch_simulation.method == BatchSimulationMethod.structured_noise:
            rank_ghost_batch_tokens = batch_num_tokens // self.batch_simulation.num_ghost_batches
            microbatch_instances = min(microbatch_instances, rank_ghost_batch_tokens // seq_len)
        micro_batches = split_batch(batch, microbatch_instances)
        num_micro_batches = len(micro_batches)

        structured_noise_scales: Optional[Tuple[float, ...]] = None
        micro_batches_per_ghost = 0
        if self.batch_simulation.method == BatchSimulationMethod.structured_noise:
            ghost_count = self.batch_simulation.num_ghost_batches
            if num_micro_batches % ghost_count != 0:
                raise OLMoConfigurationError(
                    "the number of rank microbatches must be divisible by the number of "
                    "structured-noise ghost batches"
                )
            micro_batches_per_ghost = num_micro_batches // ghost_count
            structured_noise_scales = structured_noise_loss_scales(
                ghost_count,
                seed=self.batch_simulation.seed,
                step=self.trainer.global_step,
            )

        # Train one micro-batch at a time.
        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                input_ids, labels, model_kwargs = self._prepare_batch(micro_batch)

                # Run forward pass, get losses.
                _, loss, ce_loss, z_loss = self.model_forward(
                    input_ids,
                    labels=labels,
                    ignore_index=self.label_ignore_index,
                    loss_reduction="sum",
                    z_loss_multiplier=self.z_loss_multiplier,
                    loss_div_factor=batch_num_tokens_for_loss,
                    return_logits=False,
                    **model_kwargs,
                )

                # Update total batch CE and Z loss.
                ce_batch_loss += get_local_tensor(ce_loss.detach())
                del ce_loss
                if z_batch_loss is not None:
                    assert z_loss is not None
                    z_batch_loss += get_local_tensor(z_loss.detach())
                    del z_loss

                # Run backward pass.
                if structured_noise_scales is not None:
                    ghost_batch_idx = micro_batch_idx // micro_batches_per_ghost
                    loss = loss * structured_noise_scales[ghost_batch_idx]
                loss.backward()

        del batch  # In case this helps with memory utilization.

        self.model.post_batch(dry_run=dry_run)

        if dry_run:
            self.model.reset_auxiliary_metrics()
            return

        if structured_noise_scales is not None:
            self.record_metric(
                "simulated batch size",
                float(self.batch_simulation.simulated_batch_size or 0),
                namespace="batch simulation",
            )
            self.record_metric(
                "structured noise RMS multiplier",
                float((self.batch_simulation.num_ghost_batches - 1) ** 0.5),
                namespace="batch simulation",
            )

        # Record loss metrics.
        if isinstance(self.optim, SkipStepOptimizer):
            # Need to reduce the loss right away for the SkipStepOptimizer.
            if is_distributed():
                ce_batch_loss.div_(self._reduce_divide_factor)
                dist.all_reduce(ce_batch_loss)
                ce_batch_loss.div_(self.world_size)
                ce_batch_loss.mul_(self._reduce_divide_factor)
            self.record_ce_loss(ce_batch_loss)
            self.optim.latest_loss = ce_batch_loss
        else:
            self.record_ce_loss(ce_batch_loss, ReduceType.mean)
        if z_batch_loss is not None:
            assert self.z_loss_multiplier is not None
            self.record_metric(
                "Z loss",
                z_batch_loss,
                ReduceType.mean,
                namespace="train",
            )
            self.record_metric(
                "Z loss unscaled",
                z_batch_loss / self.z_loss_multiplier,
                ReduceType.mean,
                namespace="train",
            )

        # And additional metrics.
        for metric_name, (metric_val, reduction) in self.model.compute_auxiliary_metrics(
            reset=True
        ).items():
            self.record_metric(
                metric_name,
                metric_val,
                reduction,
                namespace="train",
            )

    def eval_batch(
        self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        # Evaluation and checkpointing should always observe one coherent synchronized model.
        self._maybe_sync_local_sgd(force=True)
        # CP and TP are supported for PPL evals (LMEvaluator) since they only need per-token
        # CE loss. Downstream evals that require full logits will fail naturally if attempted
        # with CP or TP.

        input_ids, labels, model_kwargs = self._prepare_batch(batch, labels)

        # When using CP/TP, shard the label_mask along the sequence dimension to match the
        # sharded ce_loss output shape. CP shards first (S -> S/CP), then TP shards further
        # (S/CP -> S/(CP*TP)).
        if self.cp_enabled and "label_mask" in model_kwargs:
            assert self.model._cp_load_balancer is not None
            (label_mask,) = self.model._cp_load_balancer.batch_shard(
                inputs=[model_kwargs["label_mask"]],
                seq_dims=[1],
                pad_values=[0],
            )
            model_kwargs["label_mask"] = label_mask.to(torch.bool)

        if self.tp_enabled and "label_mask" in model_kwargs:
            tp_mesh = self.model._tp_mesh
            assert tp_mesh is not None
            chunks = model_kwargs["label_mask"].chunk(tp_mesh.size(), dim=1)
            model_kwargs["label_mask"] = chunks[tp_mesh.get_local_rank()]

        self._set_model_mode("eval")

        with self._eval_batch_context():
            output = self.model_forward(
                input_ids,
                labels=labels,
                ignore_index=self.label_ignore_index,
                loss_reduction="none",
                return_logits=False if (self.cp_enabled or self.tp_enabled) else None,
                **model_kwargs,
            )

        if self.tp_enabled and isinstance(output, LMOutputWithLoss):
            output = output._replace(ce_loss=get_local_tensor(output.ce_loss))

        return output

    def optim_step(self):
        if self.batch_simulation.uses_local_updates:
            self._ensure_local_sgd_gradients_materialized()

        # Maybe clip gradients.
        if self.max_grad_norm is not None:
            if self.batch_simulation.uses_local_updates:
                grad_norm = self._clip_local_sgd_grad_norm(self.max_grad_norm)
                grad_norm_reduce_type: Optional[ReduceType] = ReduceType.mean
            else:
                grad_norm = self._clip_grad_norm(self.max_grad_norm)
                # Normal DP grad norm is already reduced over ranks.
                grad_norm_reduce_type = None
            self.trainer.record_metric(
                "total grad norm",
                grad_norm,
                reduce_type=grad_norm_reduce_type,
                namespace="optim",
            )
            if isinstance(self.optim, SkipStepOptimizer):
                self.optim.latest_grad_norm = grad_norm

        # Maybe adjust learning rate.
        if self.scheduler is not None:
            for group_idx, group in enumerate(self.optim.param_groups):
                new_lr = self.scheduler.set_lr(group, self.trainer)
                self.trainer.record_metric(f"LR (group {group_idx})", new_lr, namespace="optim")

        # Step optimizer.
        self.optim.step()
        if isinstance(self.optim, SkipStepOptimizer):
            self.record_metric("step skipped", self.optim.step_skipped, namespace="optim")

        self.model.post_optim_step()

        if self.batch_simulation.uses_local_updates:
            self._local_sgd_steps_since_sync += 1
            synced = self._maybe_sync_local_sgd()
            self.record_metric(
                "replica synchronization",
                float(synced),
                namespace=(
                    "DiLoCo"
                    if self.batch_simulation.method == BatchSimulationMethod.diloco
                    else "local SGD"
                ),
            )

    def zero_grads(self):
        self.optim.zero_grad(set_to_none=True)

    def model_forward(
        self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        """
        Run a forward pass on a micro-batch, returning the logits.
        """
        with self._model_forward_context():
            return self.model(input_ids, labels=labels, **kwargs)

    @lru_cache
    def num_flops_per_token(self, seq_len: int) -> Optional[int]:
        try:
            return self.model.num_flops_per_token(seq_len)
        except NotImplementedError as ex:
            warn_once(f"Unable to estimate num flops per token: {ex}")
        return None

    def _inner_optimizer_state_key(self) -> str:
        """Return the checkpoint key for this rank's inner optimizer state."""
        if self.batch_simulation.method != BatchSimulationMethod.diloco:
            return "optim"
        if self.world_mesh is None:
            raise RuntimeError("DiLoCo optimizer checkpointing requires a distributed world mesh")
        replica_idx = get_dp_replicate_mesh(self.world_mesh).get_local_rank()
        return f"{_DILOCO_INNER_OPTIM_PREFIX}{replica_idx}"

    def global_num_flops_in_batch(self, batch: Dict[str, Any]) -> Optional[int]:
        global_num_tokens = self.trainer.data_loader.global_num_tokens_in_batch(batch)
        if global_num_tokens is None:
            return None
        flops_per_token = self.num_flops_per_token(seq_len=batch["input_ids"].shape[1])
        return flops_per_token * global_num_tokens if flops_per_token is not None else None

    @contextlib.contextmanager
    def _train_microbatch_context(
        self, micro_batch_idx: int, num_micro_batches: int
    ) -> Generator[None, None, None]:
        is_last_mb = micro_batch_idx == num_micro_batches - 1
        with contextlib.ExitStack() as stack:
            if isinstance(self.model, FSDPModule):
                assert self.dp_config is not None
                # On the last backward FSDP waits on pending gradient reduction and clears internal data
                # data structures for backward prefetching.
                self.model.set_is_last_backward(is_last_mb)
                # For native HSDP we can delay the gradients all-reduce until the final
                # micro-batch. Local SGD is wrapped on each replica's shard sub-mesh, so
                # it must leave gradient materialization enabled on every micro-batch.
                if (
                    self.dp_config.name == DataParallelType.hsdp
                    and not self.batch_simulation.uses_local_updates
                ):
                    self.model.set_requires_all_reduce(is_last_mb)
            elif isinstance(self.model, DDP):
                # For DDP, only sync gradients on the final micro-batch.
                if not is_last_mb:
                    stack.enter_context(self.model.no_sync())

            yield

    @contextlib.contextmanager
    def _eval_batch_context(self) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            yield

    @contextlib.contextmanager
    def _model_forward_context(self) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            if self.autocast_precision is not None:
                stack.enter_context(torch.autocast(self.device.type, dtype=self.autocast_precision))
            yield

    def _get_state_dict(
        self,
        sd_options: dist_cp_sd.StateDictOptions,
        optim: bool = True,
        inner_optimizer_key: Optional[str] = None,
        include_diloco_outer_optim: Optional[bool] = None,
    ) -> Dict[str, Any]:
        state_dict: Dict[str, Any] = {
            "model": dist_cp_sd.get_model_state_dict(self.model, options=sd_options),
        }
        if optim:
            if inner_optimizer_key is None:
                inner_optimizer_key = self._inner_optimizer_state_key()
            state_dict[inner_optimizer_key] = dist_cp_sd.get_optimizer_state_dict(
                self.model, self.optim, options=sd_options
            )
        if include_diloco_outer_optim is None:
            include_diloco_outer_optim = optim and self._diloco_outer_optim is not None
        if include_diloco_outer_optim:
            if self._diloco_outer_optim is None:
                raise RuntimeError(
                    "Cannot save DiLoCo outer optimizer state when DiLoCo is disabled"
                )
            state_dict["diloco_outer_optim"] = dist_cp_sd.get_optimizer_state_dict(
                self.model,
                self._diloco_outer_optim,
                options=sd_options,
            )
        return state_dict

    def _clip_grad_norm(
        self, max_grad_norm: float, norm_type: float = 2.0, foreach: Optional[bool] = None
    ) -> torch.Tensor:
        if isinstance(self.model, FSDP):
            return self.model.clip_grad_norm_(max_grad_norm)

        # Adapted from https://github.com/pytorch/torchtitan/blob/2a4437014e66bcf88a3f0419b816266e6326d539/torchtitan/utils.py#L348

        parameters = [p for p in self.model.parameters()]
        grads = [p.grad for p in parameters if p.grad is not None]

        total_norm = nn.utils.get_total_norm(
            grads, norm_type=norm_type, error_if_nonfinite=False, foreach=foreach
        )

        # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
        # We can simply reduce the DTensor to get the total norm in this tensor's process group
        # and then convert it to a local tensor.
        # NOTE: It has two purposes:
        #       1. to make sure the total norm is computed correctly when PP is used (see below)
        #       2. to return a reduced total_norm tensor whose .item() would return the correct value
        if isinstance(total_norm, DTensor):
            # Will reach here if any non-PP parallelism is used.
            # If only using PP, total_norm will be a local tensor.
            total_norm = total_norm.full_tensor()

        torch.nn.utils.clip_grads_with_norm_(parameters, max_grad_norm, total_norm, foreach=foreach)
        return total_norm

    def _clip_local_sgd_grad_norm(self, max_grad_norm: float) -> torch.Tensor:
        """Clip independently within each local-update replica (across its HSDP shards)."""
        assert self.world_mesh is not None
        local_norm_squared = torch.zeros((), device=self.device, dtype=torch.float32)
        gradients = []
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                gradient = get_local_tensor(parameter.grad)
                gradients.append(gradient)
                local_norm_squared.add_(gradient.detach().float().square().sum())

        shard_group = get_dp_shard_mesh(self.world_mesh).get_group()
        if dist.get_world_size(shard_group) > 1:
            dist.all_reduce(local_norm_squared, op=dist.ReduceOp.SUM, group=shard_group)
        total_norm = local_norm_squared.sqrt()
        clip_coefficient = torch.clamp(max_grad_norm / (total_norm + 1e-6), max=1.0)
        for gradient in gradients:
            gradient.mul_(clip_coefficient.to(gradient.device, dtype=gradient.dtype))
        return total_norm

    def _ensure_local_sgd_gradients_materialized(self) -> None:
        if any(parameter.grad is not None for parameter in self.model.parameters()):
            return
        raise RuntimeError(
            "local-update optimizer step has no materialized gradients; the model must be "
            "wrapped independently within each replica rather than using HSDP's deferred "
            "all-reduce accumulation mode"
        )

    def _maybe_sync_local_sgd(self, *, force: bool = False) -> bool:
        if not self.batch_simulation.uses_local_updates:
            return False
        if self._local_sgd_steps_since_sync == 0:
            return False

        current_step: Optional[int] = None
        if self.batch_simulation.uses_exact_diloco_outer_steps:
            current_step = self.trainer.global_step
            if not self.batch_simulation.is_diloco_outer_step(current_step):
                if force:
                    raise RuntimeError(
                        "Refusing to aggregate a partial DiLoCo round at unscheduled step "
                        f"{current_step}. Add this step to 'diloco_outer_steps', or move the "
                        "checkpoint/evaluation to an explicitly scheduled outer step."
                    )
                return False
        elif (
            not force
            and self._local_sgd_steps_since_sync < self.batch_simulation.local_update_sync_interval
        ):
            return False

        assert self.world_mesh is not None
        replica_group = get_dp_replicate_mesh(self.world_mesh).get_group()
        if self.batch_simulation.method == BatchSimulationMethod.local_sgd:
            average_module_and_optimizer_state(self.model, self.optim, replica_group)
        else:
            assert self.batch_simulation.method == BatchSimulationMethod.diloco
            assert self._diloco_outer_optim is not None
            if current_step is not None and self.batch_simulation.should_save_diloco_replicas(
                current_step
            ):
                self._save_diloco_replicas_before_outer_step(current_step)
            outer_metrics = diloco_outer_step(
                self.model,
                self._diloco_outer_optim,
                self._diloco_outer_parameters,
                replica_group,
            )
            for name, value in outer_metrics.items():
                self.record_metric(name, value, namespace="DiLoCo outer")
        self._local_sgd_steps_since_sync = 0
        return True

    def _save_diloco_replicas_before_outer_step(self, step: int) -> None:
        """Save each raw local replica model immediately before its DiLoCo outer update."""
        if self.batch_simulation.method != BatchSimulationMethod.diloco:
            raise RuntimeError("Raw DiLoCo replica checkpoints require method='diloco'")
        if not self.batch_simulation.is_diloco_outer_step(step):
            raise RuntimeError(f"Step {step} is not a configured DiLoCo outer step")
        assert self.world_mesh is not None

        replica_mesh = get_dp_replicate_mesh(self.world_mesh)
        shard_mesh = get_dp_shard_mesh(self.world_mesh)
        replica_idx = replica_mesh.get_local_rank()
        shard_group = shard_mesh.get_group()
        root = join_path(self.trainer.save_folder, f"step{step}-diloco-replicas")
        replica_dir = join_path(root, f"replica{replica_idx}")
        model_dir = join_path(replica_dir, "model")

        # Deliberately omit both optimizers. The raw snapshots are a compact, immutable record of
        # the eight local model endpoints for offline mean merging and evaluation. Saving through
        # state_dict_to_save() here would force an aggregation before the snapshot.
        replica_state = self._get_state_dict(
            self.state_dict_save_opts,
            optim=False,
            include_diloco_outer_optim=False,
        )
        save_state_dict(
            model_dir,
            replica_state,
            process_group=shard_group,
            thread_count=self.trainer.checkpointer.save_thread_count,
            throttle_uploads=self.trainer.checkpointer.throttle_uploads,
        )

        if get_rank(shard_group) == 0:
            self.trainer.checkpointer.write_file(
                replica_dir,
                "metadata.json",
                json.dumps(
                    {
                        "format": "olmo-core-diloco-replica-model-v1",
                        "globalStep": step,
                        "replica": replica_idx,
                        "replicaCount": replica_mesh.size(),
                        "innerStepsSinceOuterUpdate": self._local_sgd_steps_since_sync,
                        "aggregated": False,
                        "modelCheckpoint": "model",
                    },
                    indent=2,
                )
                + "\n",
            )
        barrier()
        if get_rank() == 0:
            self.trainer.checkpointer.write_file(
                root,
                "manifest.json",
                json.dumps(
                    {
                        "format": "olmo-core-diloco-replica-set-v1",
                        "globalStep": step,
                        "replicaCount": replica_mesh.size(),
                        "replicas": [f"replica{idx}/model" for idx in range(replica_mesh.size())],
                        "savedBeforeOuterUpdate": True,
                    },
                    indent=2,
                )
                + "\n",
            )
        barrier()

    def _prepare_batch(
        self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        input_ids = batch.pop("input_ids")
        labels = labels if labels is not None else batch.pop("labels", None)
        if "doc_lens" in batch and "max_doc_lens" in batch:
            log_once(log, "intra-document masking enabled")
        return input_ids, labels, batch

    def _set_model_mode(self, mode: Literal["train", "eval"]):
        if self._model_mode != mode:
            if mode == "train":
                self.model.train()
            elif mode == "eval":
                self.model.eval()
            else:
                raise ValueError(f"Invalid model mode: {mode}")
            self._model_mode = mode
