#!/usr/bin/env python3
"""Pack (model, task, keep-k) items of the finetune-free selective-expert eval into Beaker jobs.

Each job builds one Python environment (Torch 2.10 / vLLM 0.19.1 / Transformers 5.16.1 / FLA
0.5.2 plus OLMES and this repo's eval dependencies, frozen in ``requirements-emo-venv.txt``),
then runs ``run_item.py`` for every item of its plan in sequence, each in its own process.
The repository is used as mounted on Weka (no push needed); OLMES (``oe_eval``) is read from
the vendored copy next to the requirements file.  A plan item that already has metrics is
skipped, so a retried or relaunched job only does the missing work.

    python scripts/selective_hf/olmo35/launch.py --models sft decay4t-emo --keep-k 64 128 full \\
        --pack 10 --cluster ai2/jupiter,ai2/ceres,ai2/saturn [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
# local -> job path spelling (this host mounts the Weka buckets at short paths)
MOUNTS = {"/root": "/weka/oe-training-default/ryanwang", "/ckpts": "/weka/olmo-3p5-checkpoints"}

MODELS = {
    "sft": "/ckpts/scratch/olmo35-14t600b-mt100b-lc-sft-20261014/olmo35-14t600b-mt100b-lc-sft-20261014-sft/emo/step5402/hf",
    "mt20": "/ckpts/scratch/olmo35-small-14t600b-mt20-20261012/emo/step5961/hf",
    "stable14t": "/ckpts/scratch/trillion-14t-20261005/7to1-emo-stable-834466/hf",
    "decay4t-emo": "/ckpts/scratch/olmo35-small-decay4t-20260916/emo/step240000/hf",
    "decay4t-non-emo": "/ckpts/scratch/olmo35-small-decay4t-20260916/non-emo/step240000/hf",
}

# Task groups of launch_selective_hf.sh, in its order.
TASKS = [
    "arc_easy_merged", "arc_challenge_merged", "boolq_merged", "hellaswag_merged", "csqa_merged",
    "openbookqa_merged", "piqa_merged", "socialiqa_merged", "winogrande_merged",
    "gsm8k_generation_8shot_merged",
    "squad_merged", "coqa_merged", "naturalqs_merged", "triviaqa_merged", "drop_merged",
] + [f"mmlu_merged_{c}" for c in (
    "biology", "business", "chemistry", "computer_science", "culture", "economics", "engineering", "geography", "health",
    "history", "law", "math", "other", "philosophy_cat", "physics", "politics", "psychology")
] + [f"mmlu_pro_merged_{c}" for c in (
    "math", "health", "physics", "business", "biology", "chemistry", "computer_science", "economics", "engineering",
    "philosophy", "other", "history", "psychology", "law")]

# Selection cost: one forward pass over the whole validation pool per MoE layer.  Measured on
# the SFT export: ~120K prompt characters / second / layer (ARC-Easy: 2.4M chars, 20 s per
# layer).  validation_pool_sizes.json (written by the pool-size pass) holds each pool's size.
POOL_SIZES_FILE = Path(__file__).resolve().parents[3].parent / "scaling-ladders/output/emo_selective/validation_pool_sizes.json"
CHARS_PER_SEC_PER_LAYER = 120_000
MOE_LAYERS = 15
EVAL_HOURS = 0.25  # vLLM load + test-split eval, per item (generous)
ENV_HOURS = 0.3  # per job: venv build


# Measured item durations (item_summary.json of finished items, aggregated per task / keep-k by
# the launcher's --durations file) take precedence over the character-count estimate.
DURATIONS_FILE = POOL_SIZES_FILE.parent / "item_durations.json"
MEASURED_SAFETY = 1.15


def est_hours(task: str, keep_k: str, pools: dict, durations: dict | None = None) -> float:
    m = (durations or {}).get(f"{task}|{keep_k}")
    if m and m.get("total_min"):
        return m["total_min"] / 60 * MEASURED_SAFETY
    if keep_k == "full":
        return EVAL_HOURS
    chars = pools.get(task, {}).get("validation_chars")
    if chars is None:
        return 1.0 + EVAL_HOURS
    return chars / CHARS_PER_SEC_PER_LAYER * MOE_LAYERS / 3600 + EVAL_HOURS

EVAL_IMAGE = "akshitab/olmo-core-tch2110cu130-fa4-rma-2026-07-24"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
JOB_VENV = "/tmp/emo-eval-env"
WEKA_BUCKETS = ("oe-training-default", "olmo-3p5-checkpoints")


def to_job(p: str | Path) -> str:
    p = str(Path(p).resolve()) if not str(p).startswith("/weka/") else str(p)
    for local, job in sorted(MOUNTS.items(), key=lambda kv: -len(kv[0])):
        if p == local or p.startswith(local + "/"):
            return job + p[len(local):]
    raise ValueError(f"{p} is not on a Weka mount")


def q(s: str) -> str:
    return s if re.fullmatch(r"[A-Za-z0-9_./:=,@+-]+", s) else "'" + s.replace("'", "'\"'\"'") + "'"


def job_script(repo_job: str, pylib_job: str, ladders_job: str, plan_job: str, n_items: int, hf_home_job: str, base_dir_job: str) -> str:
    req = f"{pylib_job}/requirements-emo-venv.txt"
    return "\n".join([
        "set -euo pipefail",
        "command -v uv >/dev/null 2>&1 || python -m pip install --quiet --no-input uv",
        f"uv venv --python 3.12 {JOB_VENV}",
        f"export PATH={JOB_VENV}/bin:$PATH",
        f"uv pip install --python {JOB_VENV}/bin/python --index-url {TORCH_INDEX} torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0",
        # the frozen set (vllm, transformers, fla, lm_eval, OLMES deps, ...) then the ladder plugins from copies
        f"uv pip install --python {JOB_VENV}/bin/python -r {req}",
        "mkdir -p /tmp/ladders-src",
        f"for t in olmo-core ladders/olmoe3/vllm_plugin ladders/olmoe3/transformers_plugin; do mkdir -p /tmp/ladders-src/$t && tar -C {ladders_job}/$t --exclude=.git --exclude='*.egg-info' --exclude=__pycache__ -cf - . | tar -C /tmp/ladders-src/$t -xf -; done",
        f"uv pip install --python {JOB_VENV}/bin/python /tmp/ladders-src/olmo-core /tmp/ladders-src/ladders/olmoe3/vllm_plugin /tmp/ladders-src/ladders/olmoe3/transformers_plugin",
        f"export TRITON_PTXAS_PATH={JOB_VENV}/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas",
        "python -c 'import torch, vllm, transformers, lm_eval, ray, olmo_core.nn.moe.v2; assert torch.cuda.is_available(); print(torch.__version__, vllm.__version__, transformers.__version__)'",
        # (run_item.py adds the src/ packages it needs; src/ itself would shadow the installed olmo_core)
        f"export HF_HOME={hf_home_job} PYTHONPATH={pylib_job}:{repo_job} TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 OLMO_VLLM_TORCH_GROUPED_MOE=1 OLMO_VLLM_FLA_KDA=1",
        f"cd {repo_job}",
        "failed=0",
        f"for i in $(seq 0 {n_items - 1}); do",
        f"  echo \"== plan item $i / {n_items}\"",
        f"  python scripts/selective_hf/olmo35/run_plan_item.py --plan {q(plan_job)} --index $i --base-dir {q(base_dir_job)} || failed=1",
        "  pkill -f 'VLLM::[E]ngineCore' >/dev/null 2>&1 || true",
        "  sleep 5",
        "done",
        "exit $failed",
    ]) + "\n"


def build_spec(name: str, script: str, cluster: str, budget: str, priority: str, retries: int, timeout: str, description: str) -> dict:
    env = {"HF_HUB_DISABLE_PROGRESS_BARS": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "OMP_NUM_THREADS": "12",
           "UV_CACHE_DIR": "/tmp/emo-uv-cache", "UV_LINK_MODE": "hardlink"}
    return {
        "version": "v2", "budget": budget, "description": description, "retry": {"allowedTaskRetries": retries},
        "tasks": [{
            "name": "emo-selective", "image": {"beaker": EVAL_IMAGE}, "command": ["bash", "-c", script],
            "envVars": [{"name": k, "value": v} for k, v in env.items()] + [{"name": "HF_TOKEN", "secret": "HF_TOKEN"}],
            "datasets": [{"mountPath": f"/weka/{b}", "source": {"weka": b}} for b in WEKA_BUCKETS],
            "resources": {"gpuCount": 1, "cpuCount": 12, "memory": "96GiB", "sharedMemory": "16GiB"},
            "context": {"priority": priority, "preemptible": False},
            "constraints": {"cluster": [c.strip() for c in cluster.split(",") if c.strip()]},
            "timeout": timeout,
        }],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, choices=sorted(MODELS))
    ap.add_argument("--keep-k", nargs="+", default=["64", "128", "full"])
    ap.add_argument("--tasks", nargs="*", default=None, help="subset of the 46 merged tasks (default all)")
    ap.add_argument("--base-dir", default="/root/scaling-ladders/output/emo_selective/selective_evals_final")
    ap.add_argument("--pylib", default="/root/scaling-ladders/output/emo_selective/pylib")
    ap.add_argument("--ladders-repo", default="/root/scaling-ladders")
    ap.add_argument("--pack", type=float, default=7.0, help="target estimated hours of work per job (an item above it gets its own job)")
    ap.add_argument("--pool-sizes", default=str(POOL_SIZES_FILE))
    ap.add_argument("--durations", default=str(DURATIONS_FILE), help="measured per task|keep_k durations (see est_hours)")
    ap.add_argument("--cluster", default="ai2/jupiter,ai2/ceres,ai2/saturn")
    ap.add_argument("--workspace", default="ai2/flex2")
    ap.add_argument("--budget", default="ai2/oe-other")
    ap.add_argument("--priority", default="urgent")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--timeout", default=None, help="Beaker task timeout (default: 3x the job's estimated hours, at least 12h)")
    ap.add_argument("--job-prefix", default="emosel")
    ap.add_argument("--from-jobs", nargs="*", default=None, help="only relaunch the unfinished items of these earlier jobs (names or launch.json paths)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tasks = args.tasks or TASKS
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        raise SystemExit(f"unknown tasks: {unknown}")
    from run_item import metrics_done, relative_dir  # noqa: E402  (same directory)

    pools = json.loads(Path(args.pool_sizes).read_text()) if Path(args.pool_sizes).is_file() else {}
    durations = json.loads(Path(args.durations).read_text()) if Path(args.durations).is_file() else {}
    wanted = None  # (model_key, task, keep_k) triples when --from-jobs restricts the relaunch
    if args.from_jobs:
        wanted = set()
        for ref in args.from_jobs:
            path = Path(ref) if ref.endswith(".json") else Path(args.base_dir) / "__jobs__" / ref / "launch.json"
            for it in json.loads(path.read_text())["items"]:
                wanted.add((it["model_key"], it["task"], str(it["keep_k"])))
    items = []
    for m in args.models:
        for k in args.keep_k:
            for t in tasks:
                if wanted is not None and (m, t, str(k)) not in wanted:
                    continue
                out = Path(args.base_dir) / relative_dir(MODELS[m], t, k)
                if metrics_done(out / "results" / "checkpoint-0", t):
                    continue
                items.append({"model_key": m, "model": to_job(MODELS[m]), "task": t, "keep_k": k, "weight": round(est_hours(t, k, pools, durations), 2)})
    if not items:
        print("nothing to launch (all items have metrics)")
        return
    # size-balanced packing: largest first into the lightest open bin, bins capped at --pack weight
    items.sort(key=lambda it: -it["weight"])
    bins: list[list[dict]] = []
    for it in items:
        target = None
        for b in bins:
            if sum(x["weight"] for x in b) + it["weight"] <= args.pack:
                target = b
                break
        if target is None:
            bins.append([it])
        else:
            target.append(it)
    stamp = dt.datetime.now().strftime("%m%d%H%M")
    jobs_dir = Path(args.base_dir) / "__jobs__"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(items)} items -> {len(bins)} jobs; estimated hours per job (selection + eval): "
          + ", ".join(f"{sum(x['weight'] for x in b):.1f}" for b in bins))
    for j, chunk in enumerate(bins):
        name = f"{args.job_prefix}-{'-'.join(args.models)[:30]}-{j:02d}-{stamp}"[:100]
        job_dir = jobs_dir / name
        job_dir.mkdir(parents=True, exist_ok=True)
        plan = job_dir / "plan.json"
        plan.write_text(json.dumps({"name": name, "items": chunk}, indent=1))
        script = job_script(to_job(REPO), to_job(args.pylib), to_job(args.ladders_repo), to_job(plan), len(chunk),
                            to_job(Path(args.base_dir).parent / "hf-cache"), to_job(args.base_dir))
        timeout = args.timeout or f"{max(12, int(3 * sum(x['weight'] for x in chunk)) + 1)}h"
        spec = build_spec(name, script, args.cluster, args.budget, args.priority, args.retries, timeout,
                          json.dumps({"items": [(it["model_key"], it["task"], it["keep_k"]) for it in chunk]}))
        (job_dir / "beaker_spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
        summary = ", ".join(f"{it['model_key']}/{it['task']}/{it['keep_k']}" for it in chunk)
        if args.dry_run:
            print(f"DRY-RUN {name}: {len(chunk)} items (~{sum(x['weight'] for x in chunk):.1f} h): {summary[:200]}")
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(yaml.safe_dump(spec, sort_keys=False))
            path = f.name
        try:
            proc = subprocess.run(["beaker", "experiment", "create", "-w", args.workspace, "-n", name, path], capture_output=True, text=True)
        finally:
            os.unlink(path)
        out = proc.stdout + proc.stderr
        if proc.returncode != 0:
            raise SystemExit(f"beaker experiment create failed for {name}: {out[-2000:]}")
        m = re.search(r"\b(01[A-Z0-9]{24})\b", out)
        eid = m.group(1) if m else None
        (job_dir / "launch.json").write_text(json.dumps({"name": name, "experiment_id": eid, "url": f"https://beaker.org/ex/{eid}",
                                                        "items": chunk, "launched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}, indent=1))
        print(f"LAUNCHED {name}  https://beaker.org/ex/{eid}  ({len(chunk)} items: {summary[:160]})")


if __name__ == "__main__":
    main()
