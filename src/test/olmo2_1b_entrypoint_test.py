import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_entrypoint_module():
    path = Path(__file__).parents[2] / "src" / "scripts" / "train" / "olmo2-1B.py"
    spec = importlib.util.spec_from_file_location("olmo2_1b_entrypoint", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prefer_explicit_load_path_enables_entrypoint_force_exact_policy():
    module = _load_entrypoint_module()
    config = SimpleNamespace(force_exact_trainer_load_path=False)
    trainer = SimpleNamespace(prefer_explicit_load_path=True)

    module._apply_explicit_load_path_policy(config, trainer)

    assert config.force_exact_trainer_load_path is True


def test_entrypoint_force_exact_policy_defaults_to_false():
    module = _load_entrypoint_module()
    config = SimpleNamespace(force_exact_trainer_load_path=False)
    trainer = SimpleNamespace(prefer_explicit_load_path=False)

    module._apply_explicit_load_path_policy(config, trainer)

    assert config.force_exact_trainer_load_path is False


def test_embedding_weight_decay_defaults_to_zero_override():
    module = _load_entrypoint_module()

    overrides = module._embedding_group_overrides(decay_embeddings=False)

    assert len(overrides) == 1
    assert overrides[0].params == ["embeddings.weight"]
    assert overrides[0].opts == {"weight_decay": 0.0}


def test_embedding_weight_decay_opt_in_uses_global_optimizer_wd():
    module = _load_entrypoint_module()

    assert module._embedding_group_overrides(decay_embeddings=True) == []


def test_mlp_weight_decay_all_layers_targets_all_three_matrices():
    module = _load_entrypoint_module()

    override = module._mlp_weight_decay_group_override(0.1, "all", n_layers=16)

    assert override is not None
    assert override.params == [
        "blocks.*.feed_forward.w1.weight",
        "blocks.*.feed_forward.w2.weight",
        "blocks.*.feed_forward.w3.weight",
    ]
    assert override.opts == {"weight_decay": 0.1}


def test_mlp_weight_decay_upper_half_targets_all_three_matrices():
    module = _load_entrypoint_module()

    override = module._mlp_weight_decay_group_override(0.2, "upper-half", n_layers=8)

    assert override is not None
    assert override.params == [
        f"blocks.{layer}.feed_forward.w{projection}.weight"
        for layer in range(4, 8)
        for projection in (1, 2, 3)
    ]
    assert override.opts == {"weight_decay": 0.2}


def test_mlp_weight_decay_upper_half_w2_targets_only_w2():
    module = _load_entrypoint_module()

    override = module._mlp_weight_decay_group_override(0.3, "upper-half-w2", n_layers=8)

    assert override is not None
    assert override.params == [
        f"blocks.{layer}.feed_forward.w2.weight" for layer in range(4, 8)
    ]
    assert override.opts == {"weight_decay": 0.3}


def test_mlp_weight_decay_disabled_adds_no_group():
    module = _load_entrypoint_module()

    overrides = module._optimizer_group_overrides(
        decay_embeddings=False,
        mlp_weight_decay=None,
        mlp_weight_decay_scope="all",
        n_layers=16,
    )

    assert len(overrides) == 1
    assert overrides[0].params == ["embeddings.weight"]


def test_mlp_weight_decay_rejects_negative_value():
    module = _load_entrypoint_module()

    try:
        module._mlp_weight_decay_group_override(-0.1, "all", n_layers=16)
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("negative MLP weight decay should be rejected")
