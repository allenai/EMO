from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "models"
sys.path.insert(0, str(SCRIPTS))

import run_dense_1b_pool3b_bs128_e32_continuation as continuation

MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "scripts/models/manifests/dense-1b-pool3b-bs128-e32-continuation-v1.json"
)


def argument(arguments: list[str], prefix: str) -> str:
    values = [value for value in arguments if value.startswith(prefix)]
    assert len(values) == 1
    return values[0]


def make_complete_checkpoint(path: Path) -> None:
    (path / "model_and_optim").mkdir(parents=True)
    (path / "model_and_optim" / ".metadata").touch()
    (path / "model_and_optim" / "rank0.distcp").touch()
    (path / "train").mkdir()
    (path / "config.json").write_text("{}")
    for rank in range(8):
        (path / "train" / f"rank{rank}.pt").touch()


def make_post_result(item: dict, epoch: int, value: float) -> None:
    path = continuation.post_result_path(item, epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "comparisonGroup": "post_decay",
                "epoch": epoch,
                "validationExact": value,
                "sourcePreDecayCheckpoint": str(
                    continuation.EXPECTED_OUTPUT / f"step{continuation.checkpoint_step(epoch)}"
                ),
            }
        )
    )


def test_manifest_pins_exact_e32_lineage_and_baseline() -> None:
    config, item = continuation.load(MANIFEST)
    continuation.validate(config, item, check_filesystem=False)
    assert continuation.checkpoint_step(32) == 164794
    assert Path(item["sourceCheckpoint"]).name == "step164794"
    assert config["baselinePostDecay"]["validationExact"] == 2.78441


def test_producer_checkpoints_every_epoch_to_e48() -> None:
    _, item = continuation.load(MANIFEST)
    pending = list(range(33, 49))
    arguments = continuation.producer_arguments(item, 48, Path(item["sourceCheckpoint"]), pending)
    fixed = json.loads(
        argument(arguments, "--trainer.callbacks.checkpointer.fixed_steps=").split("=", 1)[1]
    )
    assert fixed == [continuation.checkpoint_step(epoch) for epoch in pending]
    assert argument(arguments, "--trainer.load_path=").endswith("/step164794")
    assert "ConstantScheduler" in argument(arguments, "--train_module.scheduler=")


def test_e48_and_e64_gates_match_requested_policy() -> None:
    config, item = continuation.load(MANIFEST)
    baseline = {"validationExact": config["baselinePostDecay"]["validationExact"]}
    worse = {"validationExact": config["baselinePostDecay"]["validationExact"] + 0.001}
    assert continuation.evaluator_decision(config, item, 48, baseline)["status"] == "improving"
    assert continuation.evaluator_decision(config, item, 48, worse)["status"] == "worse"
    original_loader = continuation.load_post_result
    try:
        continuation.load_post_result = lambda _item, _epoch: {"validationExact": 2.78}
        assert (
            continuation.evaluator_decision(config, item, 64, {"validationExact": 2.78})["status"]
            == "saturated"
        )
        assert (
            continuation.evaluator_decision(config, item, 64, {"validationExact": 2.77})["status"]
            == "improving"
        )
    finally:
        continuation.load_post_result = original_loader


def test_cleanup_deletes_only_recovery_checkpoints() -> None:
    config, item = continuation.load(MANIFEST)
    with tempfile.TemporaryDirectory() as raw:
        output = Path(raw)
        original_output = continuation.EXPECTED_OUTPUT
        original_state_name = continuation.STATE_NAME
        original_evaluator_state = continuation.evaluator.state_dir
        try:
            continuation.EXPECTED_OUTPUT = output
            continuation.STATE_NAME = ".state"
            continuation.evaluator.state_dir = lambda _item: output / ".eval"
            for epoch in range(33, 49):
                make_complete_checkpoint(output / f"step{continuation.checkpoint_step(epoch)}")
            make_post_result(item, 48, 2.78)
            removed = continuation.cleanup_nonessential_checkpoints(item, 48)
        finally:
            continuation.EXPECTED_OUTPUT = original_output
            continuation.STATE_NAME = original_state_name
            continuation.evaluator.state_dir = original_evaluator_state
        assert removed == list(range(33, 48))
        assert not (output / f"step{continuation.checkpoint_step(33)}").exists()
        assert (output / f"step{continuation.checkpoint_step(48)}").is_dir()
