#!/usr/bin/env python3
"""Submit one guarded dense-1B BS1024 batch-simulation EP4 run.

The experiment uses the sealed 1B-token Step 1 pool, fixed LR 1e-3, WSD with
10% terminal decay, and simulated batch size 64 sequences.  Structured-noise
runs use one 8-GPU node.  Local-SGD runs use two 8-GPU nodes so that the HSDP
replica degree is exactly 1024 / 64 = 16.
"""

from __future__ import annotations

import argparse
import copy
import json
import secrets
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

SEQUENCE_LENGTH = 4096
GLOBAL_SEQUENCES = 1024
SIMULATED_SEQUENCES = 64
GLOBAL_BATCH_TOKENS = GLOBAL_SEQUENCES * SEQUENCE_LENGTH
SIMULATED_BATCH_TOKENS = SIMULATED_SEQUENCES * SEQUENCE_LENGTH
TARGET_TOKENS = 4_000_000_000
TARGET_STEP = 954
PRE_DECAY_STEP = 858
LEARNING_RATE = "1e-3"
WEIGHT_DECAYS = ("0.1", "0.333")
METHODS = ("structured_noise", "local_sgd_h4", "local_sgd_h16")
SUBSET_MANIFEST = "src/olmo_core/data/subsets/0802/dclm_0802_repeated_train_1b.json"
VALIDATION_MANIFEST = "src/olmo_core/data/subsets/0802/dclm_0802_validation.json"
OUTPUT_ROOT = "/weka/oe-training-default/sewonm/icsl/models"
TRAIN_SCRIPT = "src/scripts/train/olmo2-1B.py"
REPORT = Path("reports/0802/data/wsd_batch_simulation_1b.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-single-node-experiment", required=True)
    parser.add_argument("--base-multi-node-experiment", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--weight-decay", choices=WEIGHT_DECAYS, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--workspace", default="ai2/flex2")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def get_spec(experiment: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["beaker", "experiment", "spec", experiment, "--format", "json"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def method_config(method: str) -> tuple[str, int | None, int]:
    if method == "structured_noise":
        return "structured_noise", None, 1
    if method == "local_sgd_h4":
        return "local_sgd", 4, 2
    if method == "local_sgd_h16":
        return "local_sgd", 16, 2
    raise ValueError(method)


def run_name(method: str, weight_decay: str) -> str:
    return (
        "dense_1b_step1_0802_repeated_dclm1b_wsd_bs1024_simbs64_"
        f"{method}_e4_lr{LEARNING_RATE}_wd{weight_decay}_warmup24"
    )


def training_arguments(method: str, weight_decay: str) -> list[str]:
    config_method, sync_interval, _ = method_config(method)
    name = run_name(method, weight_decay)
    output = f"{OUTPUT_ROOT}/{name}"
    args = [
        TRAIN_SCRIPT,
        name,
        "--data-root=/weka/oe-training-default/ai2-llm",
        f"--save-folder={output}",
        "--dataset.mix=null",
        f"--dataset.subset_manifest={SUBSET_MANIFEST}",
        "--dataset.mix_base_dir=/weka/oe-training-default/ai2-llm",
        "--work-dir=/weka/oe-training-default/sewonm/dataset-cache",
        f"--trainer.max_duration={{value: {TARGET_TOKENS}, unit: tokens}}",
        "--trainer.callbacks.wandb.enabled=true",
        "--trainer.callbacks.wandb.entity=ai2-llm",
        "--trainer.callbacks.wandb.project=sewonm-icsl",
        f"--trainer.callbacks.wandb.name={name}",
        "--trainer.callbacks.wandb.tags="
        + f"[pretraining,step1,0802,repeated-data,dclm-train-only,wsd,bs1024,simbs64,{method},wd{weight_decay}]",
        "--trainer.callbacks.downstream_evaluator.tasks="
        + "[arc_easy,arc_challenge,boolq,csqa_val_rc_5shot,hellaswag,openbookqa_test_rc_5shot,piqa,socialiqa_val_rc_5shot,winogrande]",
        "--trainer.callbacks.downstream_evaluator.eval_interval=null",
        "--trainer.callbacks.downstream_evaluator.eval_on_finish=true",
        "--trainer.callbacks.heldout_evaluator="
        + "{_CLASS_: olmo_core.train.callbacks.evaluator_callback.LMEvaluatorCallbackConfig, "
        + "eval_dataset: {_CLASS_: olmo_core.data.numpy_dataset.NumpyFSLDatasetConfig, "
        + "tokenizer: {_CLASS_: olmo_core.data.tokenizer.TokenizerConfig, vocab_size: 100278, "
        + "eos_token_id: 100257, pad_token_id: 100277, identifier: allenai/dolma2-tokenizer}, "
        + "mix: null, mix_base_dir: /weka/oe-training-default/ai2-llm, "
        + f"subset_manifest: {VALIDATION_MANIFEST}, "
        + "metadata: [{label: dclm-validation-0802}], sequence_length: 4096, "
        + "work_dir: /weka/oe-training-default/sewonm/dataset-cache}, "
        + "eval_interval: null, eval_on_finish: true, name: heldout}",
        "--dataset.instance_filter_config={repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}",
        "--model.block.name=default",
        "--model.block.sequence_mixer.qk_norm=null",
        f"--data_loader.global_batch_size={GLOBAL_BATCH_TOKENS}",
        f"--train_module.rank_microbatch_size={4 * SEQUENCE_LENGTH}",
        "--train_module.scheduler={_CLASS_: olmo_core.optim.scheduler.WSD, units: steps, warmup: 24, decay_fraction: 0.1}",
        f"--trainer.callbacks.checkpointer.fixed_steps=[{PRE_DECAY_STEP}]",
        "--trainer.callbacks.checkpointer.save_interval=1000000000",
        "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999",
        "--init_seed=12536",
        "--data_loader.seed=0",
        f"--train_module.optim.weight_decay={weight_decay}",
        f"--lr={LEARNING_RATE}",
        f"--train_module.batch_simulation.method={config_method}",
        f"--train_module.batch_simulation.global_batch_size={GLOBAL_BATCH_TOKENS}",
        f"--train_module.batch_simulation.simulated_batch_size={SIMULATED_BATCH_TOKENS}",
        "--train_module.batch_simulation.seed=12536",
    ]
    if sync_interval is not None:
        args.append(f"--train_module.batch_simulation.local_sgd_sync_interval={sync_interval}")
    return args


def validate_registered_tuple(method: str, weight_decay: str) -> None:
    data = json.loads(REPORT.read_text())
    matches = [
        run
        for run in data["runs"]
        if run["method"] == method
        and run["lr"] == LEARNING_RATE
        and run["wd"] == weight_decay
        and run["targetEpoch"] == 4
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one registered tuple for {method}, WD {weight_decay}; "
            f"found {len(matches)}"
        )
    if matches[0]["status"] not in ("planned", "print-only-verified"):
        raise RuntimeError(
            f"registered tuple has status {matches[0]['status']!r}; refusing duplicate submission"
        )


def update_env(task: dict[str, Any], *, revision: str, nodes: int, rdzv_id: str) -> None:
    env = task.setdefault("envVars", [])
    updates = {
        "GIT_REF": revision,
        "GIT_BRANCH": "sewonm/icsl-noise",
        "NUM_NODES": str(nodes),
    }
    if nodes > 1:
        updates.update({"GANTRY_RDZV_ID": rdzv_id, "GANTRY_RDZV_PORT": "29400"})
    for key, value in updates.items():
        for item in env:
            if item.get("name") == key:
                item["value"] = value
                break
        else:
            env.append({"name": key, "value": value})


def build_spec(
    *,
    single_spec: dict[str, Any],
    multi_spec: dict[str, Any],
    revision: str,
    method: str,
    weight_decay: str,
    priority: str,
) -> dict[str, Any]:
    _, sync_interval, nodes = method_config(method)
    spec = copy.deepcopy(single_spec if nodes == 1 else multi_spec)
    if len(spec["tasks"]) != nodes:
        raise RuntimeError(f"base spec has {len(spec['tasks'])} tasks, expected {nodes}")

    name = run_name(method, weight_decay)
    output = f"{OUTPUT_ROOT}/{name}"
    args = training_arguments(method, weight_decay)
    dry_run = [*args[:2], "--dry-run", *args[2:]]
    rdzv_id = secrets.token_hex(6)

    for replica_rank, task in enumerate(spec["tasks"]):
        commands = ["set -euo pipefail"]
        if nodes == 1:
            commands.extend(
                (
                    shlex.join(["test", "-f", SUBSET_MANIFEST]),
                    shlex.join(["test", "-f", VALIDATION_MANIFEST]),
                    shlex.join(["test", "!", "-e", output]),
                    shlex.join(
                        [
                            "echo",
                            (
                                f"BATCH_SIM_PREFLIGHT method={method} global_bs=1024 "
                                f"simulated_bs=64 lr={LEARNING_RATE} wd={weight_decay} "
                                f"target=e4 pre_decay_step={PRE_DECAY_STEP}"
                            ),
                        ]
                    ),
                    shlex.join(["python", *dry_run]),
                    shlex.join(["torchrun", "--nproc-per-node=8", *args]),
                )
            )
        else:
            commands.extend(
                (
                    'if [ "${BEAKER_REPLICA_RANK:-0}" = 0 ]; then',
                    shlex.join(["test", "-f", SUBSET_MANIFEST]),
                    shlex.join(["test", "-f", VALIDATION_MANIFEST]),
                    shlex.join(["test", "!", "-e", output]),
                    shlex.join(
                        [
                            "echo",
                            (
                                f"BATCH_SIM_PREFLIGHT method={method} global_bs=1024 "
                                f"simulated_bs=64 lr={LEARNING_RATE} wd={weight_decay} "
                                f"H={sync_interval} replicas=16 target=e4 "
                                f"pre_decay_step={PRE_DECAY_STEP}"
                            ),
                        ]
                    ),
                    shlex.join(["python", *dry_run]),
                    "fi",
                    "torchrun "
                    + '--nnodes="$BEAKER_REPLICA_COUNT:$BEAKER_REPLICA_COUNT" '
                    + '--nproc-per-node="$BEAKER_ASSIGNED_GPU_COUNT" '
                    + '--rdzv-id="$GANTRY_RDZV_ID" --rdzv-backend=static '
                    + '--rdzv-endpoint="$BEAKER_LEADER_REPLICA_HOSTNAME:$GANTRY_RDZV_PORT" '
                    + '--node-rank="$BEAKER_REPLICA_RANK" --rdzv-conf="read_timeout=420" '
                    + shlex.join(args),
                )
            )
        task["name"] = "main" if nodes == 1 else f"main-replica-{replica_rank}"
        task["arguments"] = ["bash", "-lc", "\n".join(commands)]
        task["resources"] = {"gpuCount": 8, "sharedMemory": "10 GiB"}
        task["context"] = {"priority": priority, "minRuntime": 0, "autoResume": False}
        task["hostNetworking"] = nodes > 1
        task["propagateFailure"] = nodes > 1
        task["propagatePreemption"] = nodes > 1
        if nodes > 1:
            task["synchronizedStartTimeout"] = 5_400_000_000_000
        else:
            task.pop("synchronizedStartTimeout", None)
        update_env(task, revision=revision, nodes=nodes, rdzv_id=rdzv_id)

    spec.pop("description", None)
    return spec


def main() -> None:
    args = parse_args()
    validate_registered_tuple(args.method, args.weight_decay)
    spec = build_spec(
        single_spec=get_spec(args.base_single_node_experiment),
        multi_spec=get_spec(args.base_multi_node_experiment),
        revision=args.revision,
        method=args.method,
        weight_decay=args.weight_decay,
        priority=args.priority,
    )
    if args.print_only:
        json.dump(spec, sys.stdout, indent=2)
        print()
        return
    completed = subprocess.run(
        [
            "beaker",
            "experiment",
            "create",
            "-",
            "--name",
            args.name,
            "--workspace",
            args.workspace,
            "--priority",
            args.priority,
        ],
        check=True,
        input=json.dumps(spec),
        stdout=subprocess.PIPE,
        text=True,
    )
    print(completed.stdout, end="")


if __name__ == "__main__":
    main()
