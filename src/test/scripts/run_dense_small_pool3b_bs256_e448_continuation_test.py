from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "models"
sys.path.insert(0, str(SCRIPTS))

import run_dense_small_pool3b_bs256_e448_continuation as continuation  # noqa: E402
import submit_dense_small_pool3b_bs256_e448_continuation as submission  # noqa: E402


def test_manifest_and_exact_steps() -> None:
    config, item = continuation.load(continuation.DEFAULT_MANIFEST)
    continuation.validate(config, item, filesystem=False)
    assert continuation.checkpoint_step(384) == 988769
    assert continuation.checkpoint_step(448) == 1153564
    assert continuation.CHECKPOINT_EPOCHS == tuple(range(388, 449, 4))
    assert item["nodeCount"] == 2
    assert item["gpuCount"] == 16
    assert item["gradientAccumulationSteps"] == 1


def test_arguments_preserve_global_batch_and_optimizer_state() -> None:
    config, item = continuation.load(continuation.DEFAULT_MANIFEST)
    args = continuation.producer_arguments(config, item, Path(item["sourceCheckpoint"]), list(continuation.CHECKPOINT_EPOCHS))
    assert "--data_loader.global_batch_size=1048576" in args
    assert "--train_module.rank_microbatch_size=65536" in args
    assert "--trainer.load_trainer_state=true" in args
    assert "--trainer.load_optim_state=true" in args
    assert "--trainer.reset_data_loader_state_on_load_path=true" in args
    assert "--model.tie_embeddings=true" in args
    assert "--decay-embeddings" in args
    assert not any("WSD" in value for value in args)
    assert json.loads(next(x for x in args if x.startswith("--trainer.callbacks.checkpointer.fixed_steps=")).split("=", 1)[1])[-1] == 1153564


def test_submission_is_preemptible_and_resumable() -> None:
    _, item = continuation.load(continuation.DEFAULT_MANIFEST)
    revision = "41555d0bdeb43a773c87d6ed68ff59c9b9a387d4"
    spec = submission.build_spec(item, revision, "urgent")
    submission.validate_spec(spec, revision)
    context = spec["tasks"][0]["context"]
    assert "minRuntime" not in context
    assert context["autoResume"] is True
    assert spec["retry"]["allowedTaskRetries"] == 8
