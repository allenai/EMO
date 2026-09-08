#!/usr/bin/env python3
"""Refresh active exact-WD 474M Pool-3B BS256 probes and continuations."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPORT = Path("reports/0802/data/wsd_checkpoint_producer_grid.json")
REPORT_JS = REPORT.with_suffix(".js")
POLICY = "dense_474m_pool3b_bs256_wd_probes_v1"
E64_POLICY = "dense_474m_pool3b_bs256_wd03_e64_two_node_v1"
EXPECTED_AUTHOR = "sewonm"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
RESULT = re.compile(
    r"DENSE_SMALL_CHECKPOINT_EVALUATOR_RESULT id=([^ ]+) epoch=([0-9]+) json=(\{.*\})$",
    re.MULTILINE,
)
POST_START = re.compile(
    r"DENSE_SMALL_CHECKPOINT_EVALUATOR_START id=([^ ]+) epoch=([0-9]+)",
    re.MULTILINE,
)
COMPLETE = re.compile(r"DENSE474M_POOL3B_WD_PROBE_COMPLETE json=(\{.*\})$", re.MULTILINE)
E64_PD_START = re.compile(r"DENSE474M_POOL3B_WD03_E64_PD_START id=([^ ]+) source_epoch=([0-9]+)")
E64_PD_RETAINED = re.compile(r"DENSE474M_POOL3B_WD03_E64_PD_RETAINED id=([^ ]+) epoch=([0-9]+)")
E64_RESULT = re.compile(
    r"DENSE474M_POOL3B_WD03_E64_POST_RESULT id=([^ ]+) epoch=([0-9]+) json=(\{.*\})$",
    re.MULTILINE,
)
E64_DECISION = re.compile(
    r"DENSE474M_POOL3B_WD03_E64_DECISION id=([^ ]+) epoch=([0-9]+) json=(\{.*\})$",
    re.MULTILINE,
)
E64_COMPLETE = re.compile(r"DENSE474M_POOL3B_WD03_E64_JOB_COMPLETE id=([^ ]+) terminal_epoch=64")
NONFINITE = re.compile(r"(?:loss[^\n]{0,40}(?:nan|inf)|(?:nan|inf)[^\n]{0,40}loss)", re.IGNORECASE)


def command(arguments: list[str]) -> str:
    return subprocess.run(arguments, check=True, capture_output=True, text=True).stdout


def one_job(job_id: str) -> dict[str, Any]:
    payload = json.loads(command(["beaker", "job", "inspect", job_id, "--format", "json"]))
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError(f"expected one exact job for {job_id}")
    job = payload[0]
    if (job.get("author") or {}).get("name") != EXPECTED_AUTHOR:
        raise RuntimeError(f"refusing job {job_id} with unexpected owner")
    return job


def live_state(job: dict[str, Any]) -> str:
    status = job.get("status") or {}
    if "canceled" in status or "cancelled" in status:
        return "canceled"
    if "finalized" in status or "exited" in status:
        return "complete" if status.get("exitCode") == 0 else "failed"
    if "started" in status:
        return "running"
    if "scheduled" in status:
        return "scheduled"
    return "submitted"


def verify(job: dict[str, Any], record: dict[str, Any]) -> None:
    execution = job.get("execution") or {}
    spec = execution.get("spec") or {}
    if str(execution.get("experiment")) != str(record["experiment"]):
        raise RuntimeError(f"job {record['job']} belongs to the wrong experiment")
    if (spec.get("resources") or {}).get("gpuCount") != 8:
        raise RuntimeError(f"job {record['job']} is not an eight-GPU replica")
    if "minRuntime" in (spec.get("context") or {}):
        raise RuntimeError(f"job {record['job']} has unauthorized minRuntime")
    env = {item.get("name"): item.get("value") for item in spec.get("envVars", [])}
    if env.get("GIT_REF") != record.get("revision"):
        raise RuntimeError(f"job {record['job']} has the wrong revision")
    if env.get("PYTORCH_CUDA_ALLOC_CONF") != "expandable_segments:True":
        raise RuntimeError(f"job {record['job']} lacks expandable CUDA segments")
    arguments = spec.get("arguments") or []
    if record.get("policy") == E64_POLICY:
        if env.get("NUM_NODES") != "2" or job.get("hostNetworking") is not True:
            raise RuntimeError(f"job {record['job']} lacks two-node topology")
        if "scripts/models/run_dense_474m_pool3b_bs256_wd03_e64_two_node.py" not in arguments:
            raise RuntimeError(f"job {record['job']} has the wrong E64 runner")
    elif str(record["id"]) not in arguments:
        raise RuntimeError(f"job {record['job']} has the wrong coordinate")


def main() -> None:
    report = json.loads(REPORT.read_text())
    group = report.get("weightDecayProbes474m") or {}
    if group.get("policy") != POLICY:
        raise RuntimeError("474M WD-probe registry is missing or has the wrong policy")
    records = group.get("coordinates") or []
    if len(records) != 2 or {str(item.get("weightDecay")) for item in records} != {"0.033", "0.3"}:
        raise RuntimeError("expected the exact WD0.033 and WD0.3 coordinates")
    for record in records:
        if record.get("status") in {"complete", "canceled", "failed", "terminal_e64"}:
            continue
        if not record.get("job"):
            raise RuntimeError(f"active coordinate {record.get('id')} has no exact job ID")
        job = one_job(str(record["job"]))
        verify(job, record)
        state = live_state(job)
        record["beakerStatus"] = state
        record["status"] = state
        expected_epochs = (16, 32, 64) if record.get("policy") == E64_POLICY else (16, 32)
        record["currentEpoch"] = next(
            (epoch for epoch in expected_epochs if epoch not in set(record.get("resolvedPostEpochs", []))),
            expected_epochs[-1],
        )
        record["currentPhase"] = "producer" if state in {"submitted", "scheduled", "running"} else "terminal"
        logs = ""
        try:
            since = "24h" if state in {"complete", "failed", "canceled"} else "70m"
            logs = ANSI.sub("", command(["beaker", "job", "logs", str(record["job"]), "--since", since]))
        except subprocess.CalledProcessError:
            pass
        results = {
            int(epoch): json.loads(raw)
            for identifier, epoch, raw in RESULT.findall(logs)
            if identifier == record["id"]
        }
        results.update(
            {
                int(epoch): json.loads(raw)
                for identifier, epoch, raw in E64_RESULT.findall(logs)
                if identifier == record["id"]
            }
        )
        starts = [
            int(epoch)
            for identifier, epoch in POST_START.findall(logs)
            if identifier == record["id"]
        ]
        post = record.setdefault("postDecayResults", {})
        for epoch, result in results.items():
            if epoch not in set(expected_epochs):
                raise RuntimeError(f"unexpected POST epoch E{epoch}")
            if not isinstance(result.get("validationExact"), (int, float)):
                raise RuntimeError(f"E{epoch} POST lacks validationExact")
            post[str(epoch)] = result
        resolved = sorted(int(epoch) for epoch in post)
        record["resolvedPostEpochs"] = resolved
        retained = set(record.get("resolvedCheckpointEpochs", [])) | set(resolved) | set(starts)
        retained.update(
            int(epoch)
            for identifier, epoch in E64_PD_RETAINED.findall(logs)
            if identifier == record["id"]
        )
        record["resolvedCheckpointEpochs"] = sorted(retained)
        record["currentEpoch"] = next(
            (epoch for epoch in expected_epochs if epoch not in resolved), expected_epochs[-1]
        )
        pd_starts = [
            int(epoch)
            for identifier, epoch in E64_PD_START.findall(logs)
            if identifier == record["id"]
        ]
        if starts and starts[-1] not in resolved:
            record.update({"currentEpoch": starts[-1], "currentPhase": "post"})
        elif pd_starts:
            record.update({"currentEpoch": 64, "currentPhase": "producer"})
        elif resolved and resolved[-1] == 16:
            record["currentPhase"] = "producer"
        complete_markers = [json.loads(raw) for raw in COMPLETE.findall(logs)]
        matched_complete = any(marker.get("coordinate") == record["id"] for marker in complete_markers)
        signals = ["nonfinite-training-loss"] if NONFINITE.search(logs) else []
        record["wandbHealth"] = {
            "status": "critical" if signals else ("healthy" if state == "running" else state),
            "checkedAt": datetime.now(tz=UTC).isoformat(),
            "beakerState": state,
            "criticalSignals": signals,
        }
        e64_complete = record["id"] in E64_COMPLETE.findall(logs)
        decisions = [
            json.loads(raw)
            for identifier, _epoch, raw in E64_DECISION.findall(logs)
            if identifier == record["id"]
        ]
        if decisions:
            record["decision"] = decisions[-1]
        if ((matched_complete and resolved == [16, 32]) or (e64_complete and resolved == [16, 32, 64])) and state == "complete":
            final_epoch = expected_epochs[-1]
            record.update({"status": "terminal_e64" if final_epoch == 64 else "complete", "currentPhase": "complete", "currentEpoch": final_epoch})
        elif state in {"failed", "canceled"}:
            record.update({"status": state, "currentPhase": "terminal", "needsAttention": True})
        elif state == "complete":
            record.update(
                {
                    "status": "failed_validation",
                    "currentPhase": "terminal",
                    "needsAttention": True,
                    "reason": f"Terminal without all healthy matched POST markers {list(expected_epochs)}.",
                }
            )
        elif signals:
            record["needsAttention"] = True
    states = {record["status"] for record in records}
    terminal_failures = {"failed", "canceled", "failed_validation"}
    successful_terminal = {"complete", "terminal_e64"}
    group["status"] = (
        "complete"
        if states <= (successful_terminal | terminal_failures) and bool(states & successful_terminal)
        else "failed"
        if states <= terminal_failures
        else "running"
    )
    report["updatedAt"] = datetime.now(tz=UTC).isoformat()
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    REPORT_JS.write_text(
        "window.ICSL_CHECKPOINT_PRODUCER_GRID=" + json.dumps(report, separators=(",", ":")) + ";\n"
    )
    print(json.dumps([{"id": item["id"], "status": item["status"]} for item in records], indent=2))


if __name__ == "__main__":
    main()
