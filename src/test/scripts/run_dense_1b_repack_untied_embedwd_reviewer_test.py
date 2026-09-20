import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/models/run_dense_1b_repack_untied_embedwd_reviewer.py"
SPEC = importlib.util.spec_from_file_location("reviewer_repack_untied", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def load(batch: int):
    path = ROOT / (
        "scripts/models/manifests/"
        f"dense-1b-repack-untied-embedwd-reviewer-bs{batch}.json"
    )
    return json.loads(path.read_text())


def test_manifests_and_recipe_are_exact():
    for batch, wd, ga in ((64, "0.3", 1), (512, "1.0", 8)):
        config = load(batch)
        MODULE.validate_config(config)
        assert config["gradientAccumulation"] == ga
        args = MODULE.base_arguments(config, "1e-3", wd)
        assert "--model.tie_embeddings=false" in args
        assert "--decay-embeddings" in args
        assert "--model.tie_embeddings=true" not in args


def test_predecay_saves_every_epoch_and_repacks():
    config = load(64)
    args = MODULE.predecay_training_arguments(
        config, "1e-3", "0.3", 8, None, "test", None
    )
    expected = [MODULE.stable_step(epoch, 64) for epoch in range(1, 9)]
    assert f"--trainer.callbacks.checkpointer.fixed_steps=[{','.join(map(str, expected))}]" in args
    assert "--dynamic-repacking" in args
    assert config["hardStopAtMaxEpoch"] is True
