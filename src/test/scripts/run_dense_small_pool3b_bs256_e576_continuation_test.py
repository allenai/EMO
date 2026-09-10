from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "models"
sys.path.insert(0, str(SCRIPTS))

import run_dense_small_pool3b_bs256_e576_continuation as continuation  # noqa: E402
import submit_dense_small_pool3b_bs256_e576_continuation as submission  # noqa: E402


def test_decay_recovery_steps_cover_quarters_and_endpoint() -> None:
    assert continuation.POST_START_STEP == 1483154
    assert continuation.POST_ENDPOINT_STEP == 1647950
    assert continuation.POST_RECOVERY_STEPS == (1524353, 1565552, 1606751, 1647950)


def test_postdecay_resume_uses_latest_checkpoint_and_remaining_steps() -> None:
    continuation.configure_base()
    config, item = continuation.base.load(continuation.DEFAULT_MANIFEST)
    output = Path("/tmp/post-e576")
    resume_step = continuation.POST_RECOVERY_STEPS[1]
    resume = output / f"step{resume_step}"
    args = continuation.postdecay_arguments(
        config, item, resume, output, "post-e576-test", resume_step
    )
    assert f"--trainer.load_path={resume}" in args
    fixed = next(
        value for value in args
        if value.startswith("--trainer.callbacks.checkpointer.fixed_steps=")
    )
    assert json.loads(fixed.split("=", 1)[1]) == list(
        continuation.POST_RECOVERY_STEPS[2:]
    )
    assert "--trainer.load_trainer_state=true" in args
    assert "--trainer.load_optim_state=true" in args


def test_submission_requests_maximum_protection() -> None:
    continuation.configure_base()
    _, item = continuation.base.load(continuation.DEFAULT_MANIFEST)
    revision = "41555d0bdeb43a773c87d6ed68ff59c9b9a387d4"
    spec = submission.build_spec(item, revision, "urgent")
    submission.validate_spec(spec, revision)
    context = spec["tasks"][0]["context"]
    assert context["minRuntime"] == "8h"
    assert context["autoResume"] is True
    assert spec["retry"]["allowedTaskRetries"] == 8
