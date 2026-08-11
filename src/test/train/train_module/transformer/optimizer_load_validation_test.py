import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.train.train_module.transformer.train_module import (
    assert_optimizer_hyperparameters_match,
    optimizer_hyperparameters,
)


def test_optimizer_hyperparameters_captures_every_group_setting():
    model = torch.nn.Linear(4, 2)
    optim = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8
    )

    snapshot = optimizer_hyperparameters(optim)

    assert len(snapshot) == 1
    assert "params" not in snapshot[0]
    for key in ("lr", "weight_decay", "betas", "eps", "maximize", "foreach", "capturable"):
        assert key in snapshot[0]


def test_optimizer_hyperparameter_assertion_rejects_checkpoint_override():
    model = torch.nn.Linear(4, 2)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    expected = optimizer_hyperparameters(optim)
    optim.param_groups[0]["weight_decay"] = 0.033

    with pytest.raises(
        OLMoConfigurationError,
        match=r"group 0 weight_decay: checkpoint 0\.033 != command 0\.1",
    ):
        assert_optimizer_hyperparameters_match(expected, optimizer_hyperparameters(optim))


def test_optimizer_hyperparameter_assertion_rejects_missing_fields():
    with pytest.raises(OLMoConfigurationError, match="checkpoint <missing> != command 0.1"):
        assert_optimizer_hyperparameters_match([{"weight_decay": 0.1}], [{}])
