#!/usr/bin/env python3
"""Run one (model, task, keep-k) item of the selective-expert evaluation, no finetuning.

Mirrors steps 1+2 and 4 of ``scripts/selective_hf/hf_finetune_with_selective_layerwise.sh``:

1. greedy layerwise expert selection on the task's merged *validation* split (full pool,
   the task's default shots) via ``src.hf_training.greedy_prune_layerwise``;
2. evaluation of the selected-expert model on the merged *test* split via
   ``src.scripts.eval.launch_eval --task <task>-pruned --pruned_split test``.

The finetuning step is skipped, so the result corresponds to the pipeline's
``checkpoint-0`` ("inference" / no-finetune) numbers.  Output layout matches the pipeline so
``scripts/plotting/get_table_scores_selective_evals_final.py`` can read it::

    <base-dir>/<sanitized_model>/<task>_keepk_<K>_bs-32_lr-5e-5_epoch-1_selectivemode-layerwise/
        selected_model/pruning_metadata.json      (weights deleted after the eval unless --keep-weights)
        results/checkpoint-0/task-*-metrics.json

``--keep-k full`` evaluates the unpruned model on the same test split (no selection), under
``<task>_keepk_full_...``.

Evaluation runs through OLMES's vLLM backend by default: these exports' KDA layers reject
padded batches in HF, and HF generation is very slow for them, while OLMES scores
log-likelihood requests from vLLM prompt logprobs and generation from vLLM sampling.  The
task definitions, prompts and metrics are the same ones the pipeline uses.  ``--eval-backend
hf`` keeps the pipeline's HF path (use ``--eval-batch-size 1`` for these exports).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
GENERATION_TASKS = {
    "squad_merged", "coqa_merged", "naturalqs_merged", "triviaqa_merged", "drop_merged",
    "gsm8k_generation_8shot_merged",
}
BATCH_SIZE, LR, EPOCHS = 32, "5e-5", 1  # pipeline constants, kept in the directory name


def sanitized(model: str) -> str:
    import re

    return re.sub(r"[^a-zA-Z0-9_-]", "", model)


# Beaker jobs see the Weka buckets under /weka/...; this host mounts them at short paths.
# Output directories are named after the short spelling so both sides agree.
CANONICAL_MOUNTS = {"/weka/olmo-3p5-checkpoints": "/ckpts", "/weka/oe-training-default/ryanwang": "/root"}


def canonical(model: str) -> str:
    for job, local in CANONICAL_MOUNTS.items():
        if model == job or model.startswith(job + "/"):
            return local + model[len(job):]
    return model


def relative_dir(model: str, task: str, keep_k: str) -> str:
    return f"{sanitized(canonical(model))}/{task}_keepk_{keep_k}_bs-{BATCH_SIZE}_lr-{LR}_epoch-{EPOCHS}_selectivemode-layerwise"


def metrics_done(results_dir: Path, task: str) -> bool:
    return any(results_dir.glob("task-*-metrics.json"))


def src_shim() -> Path:
    """A directory exposing only the ``src/`` packages the eval path needs.

    ``src/`` also holds this repository's vendored ``olmo_core`` fork; putting ``src/`` itself
    on ``PYTHONPATH`` would shadow the installed OLMo-core that the OLMo 3.5 transformers /
    vLLM plugins import (``olmo_core.nn.moe.v2``).
    """
    shim = Path(os.environ.get("TMPDIR", "/tmp")) / f"emo-src-shim-{os.getuid()}"
    shim.mkdir(parents=True, exist_ok=True)
    for name in ("hf_training", "offline_evals", "scripts"):
        link = shim / name
        target = REPO / "src" / name
        if link.is_symlink() and link.resolve() == target.resolve():
            continue
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)
    return shim


def run(cmd: list[str], env: dict, log: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    with log.open("a") as f:
        f.write("+ " + " ".join(cmd) + "\n")
    proc = subprocess.run(cmd, cwd=str(REPO), env=env)
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd[:4])} ...")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF export directory (never modified)")
    ap.add_argument("--task", required=True, help="merged task name, e.g. arc_easy_merged")
    ap.add_argument("--keep-k", required=True, help="64, 128, ... or 'full' (no selection)")
    ap.add_argument("--base-dir", required=True, help="output root (selective_evals_final equivalent)")
    ap.add_argument("--num-shared-experts", type=int, default=0)
    ap.add_argument("--select-batch-size", type=int, default=32)
    ap.add_argument("--eval-backend", choices=["hf", "vllm", "auto"], default="vllm", help="auto: vllm for generation tasks, hf otherwise")
    ap.add_argument("--eval-batch-size", type=int, default=None, help="default: 32, 16 for gsm8k, 4 for *history*")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--keep-weights", action="store_true", help="do not delete selected_model weights after the eval")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.base_dir) / relative_dir(args.model, args.task, args.keep_k)
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results" / "checkpoint-0"
    log = out / "run_item.log"
    if metrics_done(results, args.task) and not args.force:
        print(f"ALREADY_DONE {out}", flush=True)
        return

    env = dict(os.environ)
    # launch_eval shells out to a bare ``python``; make sure it is this interpreter
    env["PATH"] = str(Path(sys.executable).parent) + ":" + env.get("PATH", "")
    env["PYTHONPATH"] = f"{src_shim()}:{REPO}" + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    t0 = time.time()

    # ---- 1+2. selection -------------------------------------------------------------
    if args.keep_k == "full":
        eval_model = args.model
    else:
        selected = out / "selected_model"
        if not (selected / "pruning_metadata.json").is_file() or args.force:
            run(
                [sys.executable, "-m", "src.hf_training.greedy_prune_layerwise", "--model", args.model, "--task", args.task,
                 "--split", "validation", "--prune-keep-k", str(int(args.keep_k)), "--num-shared-experts", str(args.num_shared_experts),
                 "--save-path", str(selected), "--batch-size", str(args.select_batch_size), "--trust-remote-code"],
                env, log,
            )
        if not (selected / "config.json").is_file():
            raise SystemExit(f"selected model weights missing (deleted after a previous eval?): {selected}; use --force to redo the selection")
        eval_model = str(selected)
    t_sel = time.time() - t0

    # ---- 4. eval on the merged test split -------------------------------------------
    backend = args.eval_backend
    if backend == "auto":
        backend = "vllm" if args.task in GENERATION_TASKS else "hf"
    bs = args.eval_batch_size or (16 if "gsm8k_generation_8shot" in args.task else 4 if "history" in args.task else 32)
    model_args = "trust_remote_code=true"
    if backend == "vllm":
        # max_length=2048 mirrors the HF path's default context (OLMES MODEL_DEFAULTS); the rest
        # are the vLLM engine options the OLMo 3.5 eval recipe uses for these exports.
        model_args += (",dtype=bfloat16,max_length=2048,gpu_memory_utilization=0.75,enforce_eager=true"
                       ",max_num_seqs=32,max_num_batched_tokens=4096,mamba_ssm_cache_dtype=float32,attention_backend=FLASH_ATTN")
    env_eval = dict(env)
    env_eval.setdefault("HF_DATASETS_OFFLINE", "0")
    env_eval.update(OLMO_VLLM_TORCH_GROUPED_MOE="1", OLMO_VLLM_FLA_KDA="1")
    run(
        [sys.executable, "-m", "src.scripts.eval.launch_eval", "--model", eval_model, "--model-type", backend,
         "--task", f"{args.task}-pruned", "--pruned_split", "test", "--output-dir", str(results),
         "--batch-size", str(bs), "--gpus", str(args.gpus), "--model-args", model_args],
        env_eval, log,
    )
    if not metrics_done(results, args.task):
        raise SystemExit(f"eval produced no metrics file under {results}")

    # ---- bookkeeping ----------------------------------------------------------------
    summary = {
        "model": args.model, "task": args.task, "keep_k": args.keep_k, "eval_backend": backend, "eval_batch_size": bs,
        "selection_seconds": round(t_sel), "total_seconds": round(time.time() - t0),
        "metrics_files": sorted(p.name for p in results.glob("task-*-metrics.json")),
    }
    (out / "item_summary.json").write_text(json.dumps(summary, indent=1))
    if args.keep_k != "full" and not args.keep_weights:
        selected = out / "selected_model"
        for p in selected.iterdir():
            if p.name != "pruning_metadata.json" and (p.suffix in {".safetensors", ".bin", ".pt"} or p.name.startswith("model")):
                p.unlink() if p.is_file() else shutil.rmtree(p)
        print(f"deleted selected-model weights under {selected} (kept pruning_metadata.json)", flush=True)
    print("ITEM_DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
