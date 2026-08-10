import math
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
from torch import nn

from olmo_core.distributed.parallel import DataParallelType
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
