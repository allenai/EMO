from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts/models/run_dense_1b_dr_wt_embwd_redecay.py"
SUBMITTER = ROOT / "scripts/models/submit_dense_1b_dr_wt_embwd_redecay.py"
MANIFEST = ROOT / "scripts/models/manifests/dense-1b-dr-wt-embwd-redecay-retry01.json"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_manifest_uses_exact_sources_and_isolated_retry_outputs() -> None:
    runner = load(RUNNER, "dense1b_redecay_runner")
    manifest = json.loads(MANIFEST.read_text())
    assert len(manifest["runs"]) == 3
    for item in manifest["runs"]:
        runner.validate_item(item)
        assert item["retryOutput"].endswith(f"e{item['epoch']}-retry01")
        assert item["sourceCheckpoint"].endswith(f"step{item['sourceStep']}")
        assert item["endpointStep"] > item["sourceStep"]
        assert item["expectedRuntimeSeconds"] > 0


def test_submit_spec_omits_min_runtime(monkeypatch) -> None:
    submitter = load(SUBMITTER, "dense1b_redecay_submitter")
    manifest = json.loads(MANIFEST.read_text())
    item = manifest["runs"][0]
    base = {
        "tasks": [{"envVars": [{"name": "GIT_REF", "value": "old"}], "datasets": []}]
    }
    monkeypatch.setattr(submitter, "command", lambda *args, **kwargs: json.dumps(base))
    spec = submitter.build_spec(item, "a" * 40, "urgent")
    context = spec["tasks"][0]["context"]
    assert context == {"priority": "urgent", "autoResume": True}
    assert "minRuntime" not in context
    assert spec["retry"] == {"allowedTaskRetries": 8}


def test_monitor_parses_compact_trainer_eta() -> None:
    monitor = load(
        ROOT / "scripts/models/monitor_dense_1b_dr_wt_embwd_redecay.py",
        "dense1b_redecay_monitor",
    )
    assert monitor.duration_seconds("2h24m") == 8640
    assert monitor.duration_seconds("1d6h9m") == 108540
