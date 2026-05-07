"""
Zero-shot variant of ``analyze_nprune_keep_sets``.

Identical sweep (model × task × keep_k × N) but forces ``num_shots=0`` when loading
each task's validation prompts, so the calibration set contains only the raw question
stems (no 5-/8-shot demonstrations). The rest of the pipeline — subsample seed,
tokenizer kwargs, pruning algorithm, snapshot/restore, output JSON schema — is
byte-identical to ``analyze_nprune_keep_sets``.

Implementation note: we do NOT modify ``data_utils.get_formatted_prompts`` (which
has a fixed task-config read path). Instead we replicate its body inline, making a
shallow copy of the TASK_CONFIGS entry and setting ``num_shots = 0`` before handing
it to ``load_task``. Nothing in the rest of the codebase is affected.

Default output root: ``claude_outputs/prune_plots/nprune_analysis_0shot/`` so results
do not collide with the 5-/8-shot ``nprune_analysis/`` tree.
"""

import argparse
import copy
import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Local imports mirror analyze_nprune_keep_sets.py.
from offline_evals.run_eval import load_task
from scripts.eval.tasks import get_task_configs
from src.hf_training.data_utils import get_oe_task_name
from src.hf_training.greedy_prune_layerwise import (
    compute_layerwise_keep_sets,
    restore_model_state,
    snapshot_model_state,
)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_formatted_prompts_0shot(task_name: str, split: str) -> Tuple[List[str], str]:
    """Replica of ``data_utils.get_formatted_prompts`` that overrides
    ``num_shots`` to 0 on a local copy of the task config.

    Keep this function in lock-step with ``get_formatted_prompts`` if that ever
    changes — we duplicate the body deliberately to avoid editing shared code.
    """
    oe_task_name = get_oe_task_name(task_name, split)
    TASK_CONFIGS = get_task_configs()
    task_config = copy.deepcopy(TASK_CONFIGS[oe_task_name])
    task_config["num_shots"] = 0
    task = load_task(task_config, "tmp")
    task.download()
    task.build_all_requests()

    dataset = []
    request_type = task._instances[0].request_type

    if request_type == "loglikelihood":
        for instance in task._instances:
            if instance.idx == instance.label and not instance.request.context.startswith(
                "Answer:"
            ):
                dataset.append(instance.request.context + instance.request.continuation)
            elif "gsm8k" in task_name:
                dataset.append(instance.request.context + instance.request.continuation)

    elif request_type == "generate_until":
        for instance in task._instances:
            choice = instance.doc["choices"][0]
            if isinstance(choice, tuple):
                choice = choice[0]
            if choice and instance.request.context[-1] != " " and choice[0] != " ":
                dataset.append(instance.request.context + " " + choice)
            else:
                dataset.append(instance.request.context + choice)

    return dataset, request_type


def _stringify_model(model_path: str) -> str:
    p = Path(model_path)
    rel = f"{p.parent.name}/{p.name}" if p.parent.name else p.name
    return re.sub(r"[^a-zA-Z0-9_-]", "", rel)


def _parse_nprune(v: str) -> Optional[int]:
    if v.lower() == "all":
        return None
    return int(v)


def _nprune_tag(n: Optional[int]) -> str:
    return "all" if n is None else str(n)


def _first_moe_layer_idx(model) -> Optional[int]:
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            return i
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to HF model")
    parser.add_argument("--num-shared-experts", type=int, default=0)
    parser.add_argument("--tasks", type=str, nargs="+", required=True)
    parser.add_argument("--prune-keep-k-values", type=int, nargs="+", required=True)
    parser.add_argument(
        "--num-prune-examples-values",
        type=str,
        nargs="+",
        required=True,
        help="List of calibration sizes; accepts integers and the literal 'all'",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="claude_outputs/prune_plots/nprune_analysis_0shot",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if not (0 <= args.shard_idx < args.num_shards):
        raise ValueError(
            f"--shard-idx must be in [0, --num-shards); got {args.shard_idx} / {args.num_shards}"
        )

    nprune_values: List[Optional[int]] = [_parse_nprune(v) for v in args.num_prune_examples_values]
    stringified_model = _stringify_model(args.model)
    model_out_root = Path(args.output_dir) / stringified_model
    model_out_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model: {args.model}")
    config = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto" if args.device is None else args.device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    first_moe_idx = _first_moe_layer_idx(model)
    if first_moe_idx is None:
        raise RuntimeError("No MoE layers found in model")
    logger.info(f"First MoE layer index: {first_moe_idx}")

    pristine = snapshot_model_state(model)
    logger.info(
        f"Snapshotted pristine state for "
        f"{sum(1 for s in pristine['layers'] if s is not None)} MoE layers"
    )

    my_tasks = [(i, t) for i, t in enumerate(args.tasks) if i % args.num_shards == args.shard_idx]
    logger.info(
        f"Shard {args.shard_idx}/{args.num_shards}: handling {len(my_tasks)} of "
        f"{len(args.tasks)} tasks: {[t for _, t in my_tasks]}"
    )

    total_combos = len(my_tasks) * len(nprune_values) * len(args.prune_keep_k_values)
    combo_idx = 0

    for _, task in my_tasks:
        logger.info(f"=== Task: {task} (num_shots=0) ===")
        prompts, _ = _get_formatted_prompts_0shot(task, args.split)
        logger.info(f"Loaded {len(prompts)} validation prompts")
        perm = torch.randperm(len(prompts), generator=torch.Generator().manual_seed(0)).tolist()

        for n_cal in nprune_values:
            if n_cal is None:
                sub_prompts = prompts
            else:
                n_keep = min(n_cal, len(prompts))
                sub_prompts = [prompts[i] for i in perm[:n_keep]]

            for keep_k in args.prune_keep_k_values:
                combo_idx += 1
                out_dir = model_out_root / task / f"keepk-{keep_k}"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"nprune-{_nprune_tag(n_cal)}.json"

                if args.skip_existing and out_path.exists():
                    logger.info(
                        f"[{combo_idx}/{total_combos}] SKIP (exists): "
                        f"{task} keepk={keep_k} N={_nprune_tag(n_cal)}"
                    )
                    continue

                logger.info(
                    f"[{combo_idx}/{total_combos}] task={task} "
                    f"keepk={keep_k} N={_nprune_tag(n_cal)} "
                    f"(using {len(sub_prompts)} prompts, 0-shot)"
                )

                restore_model_state(model, pristine)

                pristine_gate = pristine["layers"][first_moe_idx]["gate_weight"]
                current_gate = model.model.layers[first_moe_idx].mlp.gate.weight.data
                if not torch.equal(pristine_gate.to(current_gate.device), current_gate):
                    raise RuntimeError(
                        f"restore_model_state did not produce a byte-identical gate "
                        f"weight at layer {first_moe_idx}"
                    )

                experts_kept, avg_probs = compute_layerwise_keep_sets(
                    model=model,
                    tokenizer=tokenizer,
                    task_name=task,
                    split=args.split,
                    prune_keep_k=keep_k,
                    num_shared_experts=args.num_shared_experts,
                    batch_size=args.batch_size,
                    num_calibration=n_cal,
                    prompts=sub_prompts,
                )

                payload = {
                    "model": args.model,
                    "task": task,
                    "split": args.split,
                    "num_shots": 0,
                    "prune_keep_k": keep_k,
                    "num_shared_experts": args.num_shared_experts,
                    "num_calibration": n_cal,
                    "num_calibration_effective": len(sub_prompts),
                    "experts_kept_per_layer": experts_kept,
                    "avg_probs_per_layer": avg_probs,
                }
                with open(out_path, "w") as f:
                    json.dump(payload, f)
                logger.info(f"Wrote {out_path}")

    logger.info(f"Done. {combo_idx} combos processed.")


if __name__ == "__main__":
    main()
