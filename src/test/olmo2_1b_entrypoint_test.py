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
