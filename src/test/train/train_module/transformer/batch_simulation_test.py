import math
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from olmo_core.distributed.checkpoint import (
    get_checkpoint_metadata,
    load_state_dict,
    save_state_dict,
    swap_param_keys,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.optim import NoOpConfig
from olmo_core.testing.distributed import run_distributed_test
from olmo_core.train.train_module.transformer import (
    BatchSimulationConfig,
    BatchSimulationMethod,
    TransformerDataParallelConfig,
    TransformerTrainModule,
    TransformerTrainModuleConfig,
)
from olmo_core.train.train_module.transformer.batch_simulation import (
    average_module_and_optimizer_state,
    clone_local_parameter_tensors,
    diloco_outer_step,
    structured_noise_loss_scales,
)


def test_batch_simulation_is_opt_in():
    config = BatchSimulationConfig()
    assert config.method == BatchSimulationMethod.none
    assert not config.enabled
    assert config.num_ghost_batches == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"global_batch_size": None, "simulated_batch_size": 64},
        {"global_batch_size": 512, "simulated_batch_size": None},
        {"global_batch_size": 64, "simulated_batch_size": 128},
        {"global_batch_size": 500, "simulated_batch_size": 64},
        {
            "global_batch_size": 512,
            "simulated_batch_size": 64,
            "local_sgd_sync_interval": 0,
        },
    ],
)
def test_batch_simulation_rejects_invalid_configuration(kwargs):
    with pytest.raises(OLMoConfigurationError):
        BatchSimulationConfig(method=BatchSimulationMethod.structured_noise, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"diloco_inner_steps": 0},
        {"diloco_outer_lr": 0.0},
        {"diloco_outer_lr": -0.1},
        {"diloco_outer_momentum": 0.0},
        {"diloco_outer_momentum": 1.0},
    ],
)
def test_diloco_rejects_invalid_outer_optimizer_configuration(kwargs):
    with pytest.raises(OLMoConfigurationError):
        BatchSimulationConfig(
            method=BatchSimulationMethod.diloco,
            global_batch_size=512,
            simulated_batch_size=64,
            **kwargs,
        )


def test_structured_noise_scales_have_target_moments_and_are_reproducible():
    ghost_count = 8
    scales = structured_noise_loss_scales(ghost_count, seed=17, step=23)

    assert scales == structured_noise_loss_scales(ghost_count, seed=17, step=23)
    assert scales != structured_noise_loss_scales(ghost_count, seed=17, step=24)
    assert sum(scales) == pytest.approx(float(ghost_count))
    assert sum(value * value for value in scales) == pytest.approx(float(ghost_count**2))


def test_structured_noise_is_applied_to_raw_gradient():
    ghost_gradients = torch.tensor([-2.0, -0.5, 0.25, 1.0, 2.0, 3.0, 4.0, 6.0])
    scales = structured_noise_loss_scales(len(ghost_gradients), seed=7, step=11)
    parameter = torch.tensor(0.0, requires_grad=True)

    # Each linear loss contributes one ghost gradient. This is exactly how the train module
    # weights ghost-batch losses before calling backward and before Adam observes the gradient.
    loss = sum(
        scale * parameter * gradient / len(ghost_gradients)
        for scale, gradient in zip(scales, ghost_gradients)
    )
    loss.backward()

    expected = sum(
        scale * gradient.item() / len(ghost_gradients)
        for scale, gradient in zip(scales, ghost_gradients)
    )
    assert parameter.grad is not None
    assert parameter.grad.item() == pytest.approx(expected)
    assert not math.isclose(parameter.grad.item(), ghost_gradients.mean().item())


def test_local_sgd_config_builds_one_hsdp_replica_per_simulated_batch():
    config = TransformerTrainModuleConfig(
        rank_microbatch_size=16,
        max_sequence_length=8,
        optim=NoOpConfig(),
        dp_config=TransformerDataParallelConfig(name=DataParallelType.fsdp),
        batch_simulation=BatchSimulationConfig(
            method=BatchSimulationMethod.local_sgd,
            global_batch_size=512,
            simulated_batch_size=64,
            local_sgd_sync_interval=4,
        ),
    )

    with patch(
        "olmo_core.train.train_module.transformer.train_module.TransformerTrainModule"
    ) as train_module_cls:
        config.build(MagicMock())

    kwargs = train_module_cls.call_args.kwargs
    assert kwargs["batch_simulation"].method == BatchSimulationMethod.local_sgd
    assert kwargs["dp_config"].name == DataParallelType.hsdp
    assert kwargs["dp_config"].num_replicas == 8
    assert kwargs["dp_config"].shard_degree is None


def test_diloco_config_builds_one_hsdp_replica_per_simulated_batch():
    config = TransformerTrainModuleConfig(
        rank_microbatch_size=16,
        max_sequence_length=8,
        optim=NoOpConfig(),
        dp_config=TransformerDataParallelConfig(name=DataParallelType.fsdp),
        batch_simulation=BatchSimulationConfig(
            method=BatchSimulationMethod.diloco,
            global_batch_size=512,
            simulated_batch_size=64,
            diloco_inner_steps=500,
            diloco_outer_lr=0.7,
            diloco_outer_momentum=0.9,
        ),
    )

    with patch(
        "olmo_core.train.train_module.transformer.train_module.TransformerTrainModule"
    ) as train_module_cls:
        config.build(MagicMock())

    kwargs = train_module_cls.call_args.kwargs
    assert kwargs["batch_simulation"].method == BatchSimulationMethod.diloco
    assert kwargs["dp_config"].name == DataParallelType.hsdp
    assert kwargs["dp_config"].num_replicas == 8
    assert kwargs["dp_config"].shard_degree is None


def test_default_train_module_config_preserves_normal_data_parallelism():
    config = TransformerTrainModuleConfig(
        rank_microbatch_size=16,
        max_sequence_length=8,
        optim=NoOpConfig(),
        dp_config=TransformerDataParallelConfig(name=DataParallelType.fsdp),
    )

    with patch(
        "olmo_core.train.train_module.transformer.train_module.TransformerTrainModule"
    ) as train_module_cls:
        config.build(MagicMock())

    kwargs = train_module_cls.call_args.kwargs
    assert kwargs["batch_simulation"].method == BatchSimulationMethod.none
    assert kwargs["dp_config"].name == DataParallelType.fsdp
    assert kwargs["dp_config"].num_replicas is None


def test_batch_simulation_can_be_enabled_with_config_overrides():
    config = TransformerTrainModuleConfig(
        rank_microbatch_size=16,
        max_sequence_length=8,
        optim=NoOpConfig(),
        dp_config=TransformerDataParallelConfig(name=DataParallelType.fsdp),
    ).merge(
        [
            "batch_simulation.method=structured_noise",
            "batch_simulation.global_batch_size=512",
            "batch_simulation.simulated_batch_size=64",
            "batch_simulation.seed=42",
        ]
    )

    assert config.batch_simulation.method == BatchSimulationMethod.structured_noise
    assert config.batch_simulation.global_batch_size == 512
    assert config.batch_simulation.simulated_batch_size == 64
    assert config.batch_simulation.num_ghost_batches == 8
    assert config.batch_simulation.seed == 42


def test_local_sgd_sync_interval_gate():
    train_module = object.__new__(TransformerTrainModule)
    train_module.batch_simulation = BatchSimulationConfig(
        method=BatchSimulationMethod.local_sgd,
        global_batch_size=512,
        simulated_batch_size=64,
        local_sgd_sync_interval=2,
    )
    train_module._local_sgd_steps_since_sync = 1
    train_module.world_mesh = MagicMock()
    train_module.model = nn.Linear(1, 1)
    train_module.optim = torch.optim.AdamW(train_module.model.parameters())
    replicate_mesh = MagicMock()
    process_group = MagicMock()
    replicate_mesh.get_group.return_value = process_group

    with (
        patch(
            "olmo_core.train.train_module.transformer.train_module.get_dp_replicate_mesh",
            return_value=replicate_mesh,
        ),
        patch(
            "olmo_core.train.train_module.transformer.train_module."
            "average_module_and_optimizer_state"
        ) as average_state,
    ):
        assert not train_module._maybe_sync_local_sgd()
        average_state.assert_not_called()

        train_module._local_sgd_steps_since_sync = 2
        assert train_module._maybe_sync_local_sgd()
        average_state.assert_called_once_with(train_module.model, train_module.optim, process_group)
        assert train_module._local_sgd_steps_since_sync == 0


def test_diloco_sync_interval_gate_uses_outer_optimizer():
    train_module = object.__new__(TransformerTrainModule)
    train_module.batch_simulation = BatchSimulationConfig(
        method=BatchSimulationMethod.diloco,
        global_batch_size=512,
        simulated_batch_size=64,
        diloco_inner_steps=2,
    )
    train_module._local_sgd_steps_since_sync = 2
    train_module.world_mesh = MagicMock()
    train_module.model = nn.Linear(1, 1)
    train_module.optim = torch.optim.AdamW(train_module.model.parameters())
    train_module._diloco_outer_optim = torch.optim.SGD(
        train_module.model.parameters(), lr=0.7, momentum=0.9, nesterov=True
    )
    train_module._diloco_outer_parameters = clone_local_parameter_tensors(train_module.model)
    replicate_mesh = MagicMock()
    process_group = MagicMock()
    replicate_mesh.get_group.return_value = process_group

    with (
        patch(
            "olmo_core.train.train_module.transformer.train_module.get_dp_replicate_mesh",
            return_value=replicate_mesh,
        ),
        patch(
            "olmo_core.train.train_module.transformer.train_module.diloco_outer_step"
        ) as outer_step,
        patch(
            "olmo_core.train.train_module.transformer.train_module."
            "average_module_and_optimizer_state"
        ) as average_state,
    ):
        assert train_module._maybe_sync_local_sgd()
        outer_step.assert_called_once_with(
            train_module.model,
            train_module._diloco_outer_optim,
            train_module._diloco_outer_parameters,
            process_group,
        )
        average_state.assert_not_called()
        assert train_module._local_sgd_steps_since_sync == 0


def test_checkpoint_key_mapping_updates_both_diloco_optimizers():
    state_dict = {
        "model": {"current": torch.tensor(1.0)},
        "diloco_inner_optim_replica_0": {
            "state.current.exp_avg": torch.tensor(2.0),
            "param_groups.current.lr": 0.01,
        },
        "diloco_outer_optim": {
            "state.current.momentum_buffer": torch.tensor(3.0),
            "param_groups.current.lr": 0.7,
        },
    }

    swap_param_keys(
        state_dict,
        {"current": "checkpoint"},
        optimizer_keys=("diloco_inner_optim_replica_0", "diloco_outer_optim"),
    )

    assert set(state_dict["model"]) == {"checkpoint"}
    assert "state.checkpoint.exp_avg" in state_dict["diloco_inner_optim_replica_0"]
    assert "param_groups.checkpoint.lr" in state_dict["diloco_inner_optim_replica_0"]
    assert "state.checkpoint.momentum_buffer" in state_dict["diloco_outer_optim"]
    assert "param_groups.checkpoint.lr" in state_dict["diloco_outer_optim"]


def test_local_sgd_rejects_optimizer_step_without_materialized_gradients():
    train_module = object.__new__(TransformerTrainModule)
    train_module.model = nn.Linear(1, 1)

    with pytest.raises(RuntimeError, match="no materialized gradients"):
        train_module._ensure_local_sgd_gradients_materialized()

    train_module.model(torch.ones(1, 1)).sum().backward()
    train_module._ensure_local_sgd_gradients_materialized()


def _check_local_sgd_state_average():
    rank = dist.get_rank()
    module = nn.Linear(1, 1, bias=False)
    parameter = next(module.parameters())
    parameter.data.fill_(float(rank + 1))
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.1)
    optimizer.state[parameter] = {
        "step": torch.tensor(float(rank + 1)),
        "exp_avg": torch.full_like(parameter, float(10 * (rank + 1))),
        "exp_avg_sq": torch.full_like(parameter, float(100 * (rank + 1))),
    }

    average_module_and_optimizer_state(module, optimizer, dist.group.WORLD)

    assert parameter.item() == pytest.approx(1.5)
    assert optimizer.state[parameter]["step"].item() == pytest.approx(1.5)
    assert optimizer.state[parameter]["exp_avg"].item() == pytest.approx(15.0)
    assert optimizer.state[parameter]["exp_avg_sq"].item() == pytest.approx(150.0)


def test_local_sgd_averages_parameters_and_adam_state():
    run_distributed_test(
        _check_local_sgd_state_average,
        backend="gloo",
        world_size=2,
    )


def _check_diloco_outer_step_preserves_local_adam_state():
    rank = dist.get_rank()
    world_mesh = init_device_mesh(
        "cpu",
        (2, 1),
        mesh_dim_names=("dp_replicate", "dp_shard"),
    )
    module = nn.Linear(1, 1, bias=False)
    module.weight.data.fill_(1.0)
    fully_shard(module, mesh=world_mesh["dp_shard"])
    parameter = next(module.parameters())

    inner_optimizer = torch.optim.AdamW(module.parameters(), lr=0.1)
    inner_optimizer.state[parameter] = {
        "step": torch.tensor(float(rank + 1)),
        "exp_avg": torch.full_like(parameter, float(10 * (rank + 1))),
        "exp_avg_sq": torch.full_like(parameter, float(100 * (rank + 1))),
    }
    expected_exp_avg = get_local_tensor(inner_optimizer.state[parameter]["exp_avg"]).clone()

    outer_optimizer = torch.optim.SGD(
        module.parameters(), lr=1.0, momentum=0.5, weight_decay=0.0, nesterov=True
    )
    outer_parameters = clone_local_parameter_tensors(module)
    with torch.no_grad():
        get_local_tensor(parameter).fill_(0.8 if rank == 0 else 0.6)
    replica_group = world_mesh["dp_replicate"].get_group()

    diloco_outer_step(module, outer_optimizer, outer_parameters, replica_group)

    # Average pseudo-gradient = 1 - mean(0.8, 0.6) = 0.3. The first PyTorch Nesterov
    # update is (1 + momentum) * gradient, hence 1 - 1.5 * 0.3 = 0.55.
    assert get_local_tensor(parameter).item() == pytest.approx(0.55)
    assert outer_parameters[0].item() == pytest.approx(0.55)
    assert torch.equal(
        get_local_tensor(inner_optimizer.state[parameter]["exp_avg"]),
        expected_exp_avg,
    )

    with torch.no_grad():
        get_local_tensor(parameter).fill_(0.45 if rank == 0 else 0.35)
    diloco_outer_step(module, outer_optimizer, outer_parameters, replica_group)

    # The second average pseudo-gradient is 0.15. Momentum remains 0.3 and the Nesterov
    # update is 0.15 + 0.5 * 0.3 = 0.3, producing 0.25.
    assert get_local_tensor(parameter).item() == pytest.approx(0.25)
    assert outer_parameters[0].item() == pytest.approx(0.25)
    assert torch.equal(
        get_local_tensor(inner_optimizer.state[parameter]["exp_avg"]),
        expected_exp_avg,
    )


def test_diloco_applies_nesterov_to_average_pseudo_gradient():
    run_distributed_test(
        _check_diloco_outer_step_preserves_local_adam_state,
        backend="gloo",
        world_size=2,
        start_method="spawn",
    )


def _build_diloco_checkpoint_test_module(
    world_mesh, *, rank_value: float, flatten_optimizer_state: bool
):
    train_module = object.__new__(TransformerTrainModule)
    train_module.model = nn.Linear(1, 1, bias=False)
    train_module.model.weight.data.fill_(1.0)
    train_module.world_mesh = world_mesh
    train_module.batch_simulation = BatchSimulationConfig(
        method=BatchSimulationMethod.diloco,
        global_batch_size=2,
        simulated_batch_size=1,
    )
    train_module.optim = torch.optim.AdamW(train_module.model.parameters(), lr=0.01)
    parameter = next(train_module.model.parameters())
    train_module.optim.state[parameter] = {
        "step": torch.tensor(rank_value),
        "exp_avg": torch.full_like(parameter, rank_value),
        "exp_avg_sq": torch.full_like(parameter, rank_value * 10),
    }
    train_module._diloco_outer_optim = torch.optim.SGD(
        train_module.model.parameters(), lr=0.7, momentum=0.9, nesterov=True
    )
    train_module._diloco_outer_optim.state[parameter] = {
        "momentum_buffer": torch.full_like(parameter, 0.25)
    }
    train_module._diloco_outer_parameters = clone_local_parameter_tensors(train_module.model)
    train_module._local_sgd_steps_since_sync = 0
    train_module.state_dict_save_opts = dist_cp_sd.StateDictOptions(
        flatten_optimizer_state_dict=flatten_optimizer_state,
        cpu_offload=True,
    )
    train_module.state_dict_load_opts = dist_cp_sd.StateDictOptions(
        flatten_optimizer_state_dict=flatten_optimizer_state,
        strict=True,
    )
    train_module.load_key_mapping = None
    train_module.validate_optimizer_hyperparameters_on_load = False
    return train_module


def _check_diloco_checkpoint_preserves_replica_optimizer_state(
    checkpoint_dir: str, flatten_optimizer_state: bool
):
    rank = dist.get_rank()
    world_mesh = init_device_mesh(
        "cpu",
        (2, 1),
        mesh_dim_names=("dp_replicate", "dp_shard"),
    )
    source = _build_diloco_checkpoint_test_module(
        world_mesh,
        rank_value=float(rank + 1),
        flatten_optimizer_state=flatten_optimizer_state,
    )
    save_state_dict(checkpoint_dir, source.state_dict_to_save())
    dist.barrier()

    target = _build_diloco_checkpoint_test_module(
        world_mesh,
        rank_value=0.0,
        flatten_optimizer_state=flatten_optimizer_state,
    )
    metadata = get_checkpoint_metadata(checkpoint_dir)
    state_dict = target.state_dict_to_load(metadata)
    load_state_dict(checkpoint_dir, state_dict)
    target.load_state_dict(state_dict)

    parameter = next(target.model.parameters())
    assert target.optim.state[parameter]["exp_avg"].item() == pytest.approx(rank + 1)
    assert target.optim.state[parameter]["exp_avg_sq"].item() == pytest.approx((rank + 1) * 10)
    assert target._diloco_outer_optim is not None
    assert target._diloco_outer_optim.state[parameter]["momentum_buffer"].item() == pytest.approx(
        0.25
    )

    # A conventional checkpoint has one shared inner optimizer and no outer optimizer.
    # DiLoCo should accept it as an initialization point and start outer momentum from zero.
    conventional_dir = f"{checkpoint_dir}-conventional"
    conventional = _build_diloco_checkpoint_test_module(
        world_mesh,
        rank_value=4.0,
        flatten_optimizer_state=flatten_optimizer_state,
    )
    conventional.batch_simulation = BatchSimulationConfig()
    conventional._diloco_outer_optim = None
    conventional._diloco_outer_parameters = []
    save_state_dict(conventional_dir, conventional.state_dict_to_save())
    dist.barrier()

    initialized_from_conventional = _build_diloco_checkpoint_test_module(
        world_mesh,
        rank_value=0.0,
        flatten_optimizer_state=flatten_optimizer_state,
    )
    metadata = get_checkpoint_metadata(conventional_dir)
    state_dict = initialized_from_conventional.state_dict_to_load(metadata)
    load_state_dict(conventional_dir, state_dict)
    initialized_from_conventional.load_state_dict(state_dict)

    parameter = next(initialized_from_conventional.model.parameters())
    assert initialized_from_conventional.optim.state[parameter]["exp_avg"].item() == pytest.approx(
        4.0
    )
    assert initialized_from_conventional._diloco_outer_optim is not None
    assert initialized_from_conventional._diloco_outer_optim.state == {}


@pytest.mark.parametrize("flatten_optimizer_state", [True, False])
def test_diloco_checkpoint_preserves_replica_optimizer_state(tmp_path, flatten_optimizer_state):
    run_distributed_test(
        _check_diloco_checkpoint_preserves_replica_optimizer_state,
        backend="gloo",
        world_size=2,
        start_method="spawn",
        func_args=(str(tmp_path / "diloco-checkpoint"), flatten_optimizer_state),
    )


def _check_local_sgd_replica_gradients_materialize_and_step():
    rank = dist.get_rank()
    world_mesh = init_device_mesh(
        "cpu",
        (2, 1),
        mesh_dim_names=("dp_replicate", "dp_shard"),
    )

    module = nn.Linear(2, 1, bias=False)
    module.weight.data.fill_(1.0)
    # The model is sharded only within a replica. With a shard degree of one this
    # is a one-rank FSDP group, while the replicate dimension remains available
    # for explicit periodic parameter/state averaging.
    fully_shard(module, mesh=world_mesh["dp_shard"])
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.1, weight_decay=0.0)

    input_ = torch.tensor([[float(rank + 1), 1.0]])
    second_input = torch.tensor([[float(rank + 2), 1.0]])
    target = torch.tensor([[0.0]])
    before = next(module.parameters()).detach().to_local().clone()
    loss = (module(input_) - target).square().sum()
    loss.backward()

    local_grad = next(module.parameters()).grad
    assert local_grad is not None
    assert torch.linalg.vector_norm(local_grad.to_local()).item() > 0.0
    first_grad = local_grad.to_local().clone()

    # A rank can still split its simulated local batch into multiple microbatches.
    # The per-replica FSDP wrapper must accumulate those gradients before the one
    # local optimizer step, instead of replacing or hiding the first backward.
    second_loss = (module(second_input) - target).square().sum()
    second_loss.backward()
    accumulated_grad = next(module.parameters()).grad
    assert accumulated_grad is not None
    assert not torch.equal(first_grad, accumulated_grad.to_local())

    optimizer.step()
    after_local_step = next(module.parameters()).detach().to_local().clone()
    assert not torch.equal(before, after_local_step)

    replica_group = world_mesh["dp_replicate"].get_group()
    average_module_and_optimizer_state(module, optimizer, replica_group)
    after_average = next(module.parameters()).detach().to_local().clone()
    gathered = [torch.empty_like(after_average) for _ in range(2)]
    dist.all_gather(gathered, after_average, group=replica_group)
    assert torch.equal(gathered[0], gathered[1])


def test_local_sgd_replica_fsdp_materializes_gradients_and_updates_parameters():
    run_distributed_test(
        _check_local_sgd_replica_gradients_materialize_and_step,
        backend="gloo",
        world_size=2,
        start_method="spawn",
    )


def _check_local_sgd_tiny_training_reduces_loss():
    rank = dist.get_rank()
    world_mesh = init_device_mesh(
        "cpu",
        (2, 1),
        mesh_dim_names=("dp_replicate", "dp_shard"),
    )
    module = nn.Linear(2, 1, bias=False)
    module.weight.data.fill_(1.0)
    fully_shard(module, mesh=world_mesh["dp_shard"])
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.05, weight_decay=0.0)
    replica_group = world_mesh["dp_replicate"].get_group()

    input_ = torch.tensor([[float(rank + 1), 1.0]])
    target = torch.tensor([[0.0]])
    initial_loss = (module(input_) - target).square().detach()

    for step in range(6):
        optimizer.zero_grad(set_to_none=True)
        loss = (module(input_) - target).square().sum()
        loss.backward()
        assert next(module.parameters()).grad is not None
        optimizer.step()
        if (step + 1) % 2 == 0:
            average_module_and_optimizer_state(module, optimizer, replica_group)

    final_loss = (module(input_) - target).square().detach()
    assert final_loss.item() < initial_loss.item()


def test_local_sgd_tiny_training_reduces_loss():
    run_distributed_test(
        _check_local_sgd_tiny_training_reduces_loss,
        backend="gloo",
        world_size=2,
        start_method="spawn",
    )
