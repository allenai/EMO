from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "models"
sys.path.insert(0, str(SCRIPTS))

import run_dense_small_pool3b_bs256_e640_continuation as continuation  # noqa: E402


def test_decay_recovery_steps_cover_quarters_and_endpoint() -> None:
    assert continuation.POST_START_STEP == 1647948
    assert continuation.POST_ENDPOINT_STEP == 1831055
    assert continuation.POST_RECOVERY_STEPS == (1693724, 1739501, 1785278, 1831055)


def test_postdecay_resume_accepts_decayed_optimizer_state() -> None:
    continuation.configure_base()
    config, item = continuation.base.load(continuation.DEFAULT_MANIFEST)
    output = Path("/tmp/post-e640")
    resume_step = continuation.POST_RECOVERY_STEPS[0]
    resume = output / f"step{resume_step}"
    args = continuation.postdecay_arguments(
        config, item, resume, output, "post-e640-test", resume_step
    )
    assert f"--trainer.load_path={resume}" in args
    fixed = next(
        value for value in args
        if value.startswith("--trainer.callbacks.checkpointer.fixed_steps=")
    )
    assert json.loads(fixed.split("=", 1)[1]) == list(
        continuation.POST_RECOVERY_STEPS[1:]
    )
    assert "--trainer.load_trainer_state=true" in args
    assert "--trainer.load_optim_state=true" in args
    assert "--train_module.validate_optimizer_hyperparameters_on_load=false" in args

