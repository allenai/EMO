import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_submitter_module():
    models = Path(__file__).parents[3] / "scripts" / "models"
    sys.path.insert(0, str(models))
    path = models / "submit_dense_1b_weight_tying_sequential.py"
    spec = importlib.util.spec_from_file_location("dense_wt_sequential_submitter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stage_log_parent_is_created_before_tee():
    module = _load_submitter_module()
    shell, log_path = module.with_stage_log_capture(
        "preflight\ntorchrun train.py\npostflight", "/weka/canonical", 2
    )

    assert log_path == "/weka/canonical/.embwd_e2.log"
    assert shell.index("mkdir -p /weka/canonical") < shell.index(" | tee ")


def test_failed_retry_resumes_contiguous_completed_frontier(tmp_path):
    module = _load_submitter_module()
    report = {
        "weightTyingEmbeddingDecaySequentialRuns": [
            {
                "id": "wtseq-embwd-bs512-wd0.333",
                "status": "failed",
                "stages": [
                    {"epoch": 1, "status": "complete", "validationExact": 3.971},
                    {"epoch": 2, "status": "planned"},
                ],
            }
        ]
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    module.REPORT_PATH = report_path
    args = SimpleNamespace(
        retry_failed=True,
        decay_embeddings=True,
        global_sequences=512,
        weight_decay="0.333",
    )

    module.configure_failed_retry(args)

    assert args.resume_after_epoch == 1
    assert args.resume_validation == 3.971
