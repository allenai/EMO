#!/usr/bin/env python3
"""Submit one guarded four-node BS1024 local-update chain through E16.

The launcher has exactly five authorized configurations: four LocalSGD H=4
chains (simBS64/256 x BS1024 E2/E4 initialization) and one DiLoCo H=32
outerLR0.5/momentum0.7 experiment containing both initialization paths.  Every
stage performs terminal WSD decay and evaluation, while the following stage
loads the preceding stage's exact *pre-decay* checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from submit_bs1024_local_update_frontier import (
    BRANCH,
    FRONTIERS,
    ROOT,
    TRAIN_MANIFEST,
    extract_training,
    mapped_frontier_steps,
    set_env,
    source_spec,
    training_arguments,
    upsert,
)

REPORT = Path("reports/0802/data/wsd_batch_simulation_1b.json")
TARGETS = (4, 8, 12, 16)
LOCAL_NAMES = {
    (64, 2): "bs1024-ls-h4-sim64-e2init-chain-e16-r02",
    (64, 4): "bs1024-ls-h4-sim64-e4init-chain-e16-r02",
    (256, 2): "bs1024-ls-h4-sim256-e2init-chain-e16-r02",
    (256, 4): "bs1024-ls-h4-sim256-e4init-chain-e16-r02",
}
DILOCO_NAME = "bs1024-diloco-h32-olr0p5-om0p7-sim64-e2e4-chain-e16-r02"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("local-sgd", "diloco"), required=True)
    parser.add_argument("--simulated-sequences", type=int, choices=(64, 256))
    parser.add_argument("--init-epoch", type=int, choices=(2, 4))
    parser.add_argument("--revision", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--workspace", default="ai2/flex2")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument(
        "--resume-wd1-e16-only",
        action="store_true",
        help=(
            "resume an authorized simBS256 WD1.0 branch from its "
            "healthy E12 pre-decay checkpoint after a chain-control failure"
        ),
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a full 40-character SHA")
    if args.kind == "local-sgd":
        if (args.simulated_sequences, args.init_epoch) not in LOCAL_NAMES:
            parser.error("LocalSGD requires one authorized simBS/init pair")
        if args.resume_wd1_e16_only and (
            args.simulated_sequences != 256 or args.init_epoch not in (2, 4)
        ):
            parser.error("--resume-wd1-e16-only is authorized only for simBS256")
        expected_name = LOCAL_NAMES[(args.simulated_sequences, args.init_epoch)]
    else:
        if args.resume_wd1_e16_only:
            parser.error("--resume-wd1-e16-only is a LocalSGD-only recovery")
        if args.simulated_sequences is not None or args.init_epoch is not None:
            parser.error("the DiLoCo job contains both init paths; omit simBS/init")
        expected_name = DILOCO_NAME
    # Retry attempt markers belong only to experiment/W&B/control metadata.
    # Scientific output paths are derived exclusively from method/sim/init/WD
    # below and therefore remain canonical across attempts.
    expected_stem = expected_name.rsplit("-r", 1)[0]
    if not re.fullmatch(re.escape(expected_stem) + r"-r[0-9]{2}", args.name):
        parser.error(f"authorized name pattern is {expected_stem}-rNN")
    return args


def output_path(method: str, sim: int, init: int, wd: str) -> str:
    if method == "local_sgd":
        stem = f"bs1024_dr_local_sgd_h4_simbs{sim}"
    else:
        stem = f"bs1024_dr_diloco_h32_olr0.5_om0.7_vrecal_simbs{sim}"
    return f"{ROOT}/{stem}_init=bs1024e{init}_lr1e-3_wd{wd}"


def conventional_source(init: int, wd: str) -> str:
    return f"{ROOT}/bs1024_dr_lr1e-3_wd{wd}/step{FRONTIERS[init][0]}"


def local_native_start(sim: int, init: int) -> tuple[int, str]:
    start = {(64, 2): 8, (64, 4): 8, (256, 2): 4, (256, 4): 8}[(sim, init)]
    return start, f"{output_path('local_sgd', sim, init, '0.333')}/step{FRONTIERS[start][0]}"


def stage_namespace(
    *, method: str, sim: int, init: int, wd: str, start: int, target: int,
    source: str, output: str, conventional: bool, revision: str, name: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        method=method,
        simulated_sequences=sim,
        global_sequences=1024,
        lr="1e-3",
        revision=revision,
        sync_interval=4 if method == "local_sgd" else 32,
        diloco_outer_lr=0.5,
        diloco_outer_momentum=0.7,
        wd=wd,
        start_epoch=start,
        target_epoch=target,
        source_checkpoint=source,
        output=output,
        name=name,
        conventional_transition=conventional,
        workspace="ai2/flex2",
        priority="urgent",
        print_only=False,
    )


def stage_command(
    script: str,
    source_arguments: list[str],
    *,
    method: str,
    sim: int,
    init: int,
    wd: str,
    start: int,
    target: int,
    source: str,
    output: str,
    conventional: bool,
    revision: str,
    chain_name: str,
    nodes: int,
    rdzv_id: str,
    rdzv_port: int,
    metric_dir: str,
) -> tuple[list[str], str]:
    ns = stage_namespace(
        method=method, sim=sim, init=init, wd=wd, start=start, target=target,
        source=source, output=output, conventional=conventional,
        revision=revision, name=chain_name,
    )
    pre_decay, endpoint = mapped_frontier_steps(ns)
    train_args = training_arguments(source_arguments, ns, pre_decay, endpoint)
    recipe = "ls-h4" if method == "local_sgd" else "diloco-h32-olr0p5-om0p7"
    attempt = chain_name.rsplit("-", 1)[-1]
    train_args = upsert(
        train_args,
        "--trainer.callbacks.wandb.name=",
        f"--trainer.callbacks.wandb.name={Path(output).name}_e{target}_{recipe}_chain-{attempt}",
    )
    run_name = Path(output).name
    leader = '${BEAKER_REPLICA_RANK:-0}' if nodes > 1 else "0"
    commands: list[str] = [f'echo "LOCAL_UPDATE_CHAIN_STAGE_START target=E{target} source={source}"']
    if nodes > 1:
        # Beaker normally exposes BEAKER_LEADER_REPLICA_HOSTNAME for replicated
        # tasks, but that variable is not guaranteed after a multi-task spec is
        # expanded into concrete replica tasks.  Publish rank 0's hostname on
        # the shared filesystem instead so rendezvous does not depend on an
        # optional injected variable.
        host_file = f"{metric_dir}/rdzv-{rdzv_id}.host"
        commands.extend(
            [
                f'if [ "{leader}" = 0 ]; then',
                f"mkdir -p {shlex.quote(metric_dir)}",
                'rdzv_host="$(hostname -f)"',
                'test -n "${rdzv_host}"',
                f"printf '%s\\n' \"${{rdzv_host}}\" > {shlex.quote(host_file + '.tmp')}",
                f"mv {shlex.quote(host_file + '.tmp')} {shlex.quote(host_file)}",
                "fi",
                f"for rdzv_wait in $(seq 1 180); do test -s {shlex.quote(host_file)} && break; sleep 1; done",
                shlex.join(["test", "-s", host_file]),
                f"rdzv_host=$(sed -n 1p {shlex.quote(host_file)})",
                'test -n "${rdzv_host}"',
            ]
        )
    commands.append(f'if [ "{leader}" = 0 ]; then')
    commands.extend(
        [
            shlex.join(["test", "-d", source]),
            shlex.join(["test", "-f", TRAIN_MANIFEST]),
            shlex.join(["test", "!", "-e", f"{output}/step{pre_decay}"]),
            shlex.join(["test", "!", "-e", f"{output}/step{endpoint}"]),
            f'test "$(git rev-parse HEAD)" = "{revision}"',
        ]
    )
    if conventional:
        commands.append(shlex.join(["test", "!", "-e", output]))
    commands.append(shlex.join(["python", script, run_name, "--dry-run", *train_args]))
    commands.append("fi")
    if nodes > 1:
        launch = (
            f'torchrun --nnodes="{nodes}:{nodes}" --nproc-per-node="$BEAKER_ASSIGNED_GPU_COUNT" '
            f"--rdzv-id={shlex.quote(rdzv_id)} --rdzv-backend=static "
            f"--rdzv-endpoint=\"${{rdzv_host}}:{rdzv_port}\" "
            '--node-rank="$BEAKER_REPLICA_RANK" --rdzv-conf="read_timeout=420" '
            + shlex.join([script, run_name, *train_args])
        )
    else:
        launch = 'torchrun --nproc-per-node="$BEAKER_ASSIGNED_GPU_COUNT" ' + shlex.join(
            [script, run_name, *train_args]
        )
    log = f"/tmp/{chain_name}-init{init}-wd{wd}-e{target}.log"
    commands.append(f'if [ "{leader}" = 0 ]; then')
    commands.append(f"{launch} 2>&1 | tee {shlex.quote(log)}")
    commands.append("else")
    commands.append(launch)
    commands.append("fi")
    commands.append(f'if [ "{leader}" = 0 ]; then')
    commands.extend(
        [
            shlex.join(["test", "-d", f"{output}/step{pre_decay}"]),
            shlex.join(["test", "-d", f"{output}/step{endpoint}"]),
            "stage_ce=$(sed -n "
            + shlex.quote(r"s/.*dclm-validation-0802\/CE loss=\([0-9][0-9.]*\).*/\1/p")
            + f" {shlex.quote(log)} | tail -n 1)",
            'test -n "${stage_ce}"',
            f"printf '%s\\n' \"${{stage_ce}}\" > {shlex.quote(metric_dir + f'/.e{target}-wd{wd}.tmp')}",
            f"mv {shlex.quote(metric_dir + f'/.e{target}-wd{wd}.tmp')} {shlex.quote(metric_dir + f'/e{target}-wd{wd}.val')}",
            f'echo "LOCAL_UPDATE_CHAIN_STAGE_COMPLETE target=E{target} validation=${{stage_ce}} pre_decay=step{pre_decay} endpoint=step{endpoint}"',
        ]
    )
    commands.append("fi")
    return commands, f"{output}/step{pre_decay}"


def branch_commands(
    script: str,
    source_arguments: list[str],
    *,
    method: str,
    sim: int,
    init: int,
    wd: str,
    revision: str,
    chain_name: str,
    metric_dir: str,
    nodes: int,
    gated: bool,
) -> list[str]:
    output = output_path(method, sim, init, wd)
    if method == "diloco" or wd == "1.0":
        start = init
        source = conventional_source(init, wd)
        conventional = True
    else:
        start, source = local_native_start(sim, init)
        conventional = False
    commands = ["set -euo pipefail"]
    leader = '${BEAKER_REPLICA_RANK:-0}' if nodes > 1 else "0"
    commands.extend(
        [
            f'if [ "{leader}" = 0 ]; then',
            f"mkdir -p {shlex.quote(metric_dir)}",
            (
                "python -m pytest -q src/test/train/train_module/transformer/"
                "batch_simulation_test.py -k 'local_sgd'"
                if method == "local_sgd"
                else "python -m pytest -q src/test/train/train_module/transformer/"
                "batch_simulation_test.py -k 'diloco_checkpoint_preserves_replica_optimizer_state or "
                "diloco_does_not_expose_raw_replica_checkpointing'"
            ),
            "fi",
        ]
    )
    targets = [target for target in TARGETS if target > start]
    for index, target in enumerate(targets):
        digest = hashlib.sha256(f"{chain_name}:{init}:{wd}:{target}".encode()).hexdigest()
        stage, next_source = stage_command(
            script,
            source_arguments,
            method=method,
            sim=sim,
            init=init,
            wd=wd,
            start=start,
            target=target,
            source=source,
            output=output,
            conventional=conventional,
            revision=revision,
            chain_name=chain_name,
            nodes=nodes,
            rdzv_id=digest[:12],
            rdzv_port=29000 + int(digest[:8], 16) % 1000,
            metric_dir=metric_dir,
        )
        commands.extend(stage)
        if gated and target < 16:
            peer = "1.0" if wd == "0.333" else "0.333"
            own_file = f"{metric_dir}/e{target}-wd{wd}.val"
            peer_file = f"{metric_dir}/e{target}-wd{peer}.val"
            commands.extend(
                [
                    f'if [ "{leader}" = 0 ]; then',
                    f"for gate_wait in $(seq 1 720); do test -s {shlex.quote(peer_file)} && break; sleep 30; done",
                    shlex.join(["test", "-s", own_file]),
                    shlex.join(["test", "-s", peer_file]),
                    f"own_ce=$(sed -n 1p {shlex.quote(own_file)})",
                    f"peer_ce=$(sed -n 1p {shlex.quote(peer_file)})",
                    f'echo "WD_GATE target=E{target} wd={wd} own=${{own_ce}} peer_wd={peer} peer=${{peer_ce}}"',
                ]
            )
            if wd == "0.333":
                commands.extend(
                    [
                        'if awk -v wd1="${peer_ce}" -v wd0333="${own_ce}" '
                        + shlex.quote("BEGIN { exit !(wd1 < wd0333) }")
                        + "; then",
                        f'echo "WD_GATE_PRUNE wd=0.333 after=E{target}"',
                        f"touch {shlex.quote(metric_dir + f'/prune-wd0.333-after-e{target}')}",
                        f"touch {shlex.quote(metric_dir + f'/done-init{init}-wd{wd}')}",
                        "exit 0",
                        "fi",
                    ]
                )
            commands.append("fi")
        start, source, conventional = target, next_source, False
    commands.extend(
        [
            f'if [ "{leader}" = 0 ]; then touch {shlex.quote(metric_dir + f"/done-init{init}-wd{wd}")}; fi',
            f'echo "LOCAL_UPDATE_CHAIN_COMPLETE init=E{init} wd={wd}"',
        ]
    )
    return commands


def completed_validation(*, method: str, wd: str, epoch: int) -> float:
    """Return one durable completed validation value for a WD gate seed."""
    report = json.loads(REPORT.read_text())
    values: list[float] = []
    for run in report.get("runs", []):
        if run.get("method") != method or str(run.get("wd")) != wd:
            continue
        result = (run.get("results") or {}).get(str(epoch)) or {}
        if result.get("status") == "complete" and result.get("validation") is not None:
            values.append(float(result["validation"]))
    if not values:
        raise RuntimeError(
            f"missing completed validation seed for method={method} wd={wd} epoch=E{epoch}"
        )
    if max(values) - min(values) > 1e-9:
        raise RuntimeError(
            f"conflicting validation seeds for method={method} wd={wd} epoch=E{epoch}: {values}"
        )
    return values[0]


def paired_local_wd_commands(
    script: str,
    source_arguments: list[str],
    *,
    sim: int,
    init: int,
    revision: str,
    chain_name: str,
    metric_dir: str,
    nodes: int,
) -> list[str]:
    """Run both simBS256 WDs stage-by-stage on the same four nodes.

    WD0.333 starts from its healthy native H4 checkpoint. Its validation at that
    source frontier seeds the first matched gate. WD1.0 starts conventionally,
    and every later WD pair is evaluated before either branch advances.
    """
    if sim != 256:
        raise ValueError("paired WD chains are only authorized for simBS256")
    leader = '${BEAKER_REPLICA_RANK:-0}' if nodes > 1 else "0"
    method_key = f"post_e{init}_local_sgd_h4_bs1024_dr_simbs{sim}"
    wd0333_start, wd0333_source = local_native_start(sim, init)
    baseline = completed_validation(
        method=method_key, wd="0.333", epoch=wd0333_start
    )
    prune_file = f"{metric_dir}/prune-wd0.333"
    commands = [
        "set -euo pipefail",
        f'if [ "{leader}" = 0 ]; then',
        f"mkdir -p {shlex.quote(metric_dir)}",
        (
            "python -m pytest -q src/test/train/train_module/transformer/"
            "batch_simulation_test.py -k 'local_sgd'"
        ),
        f"printf '%s\\n' {shlex.quote(str(baseline))} > "
        + shlex.quote(f"{metric_dir}/e{wd0333_start}-wd0.333.val"),
        "fi",
    ]
    states: dict[str, dict[str, Any]] = {
        "0.333": {
            "start": wd0333_start,
            "source": wd0333_source,
            "conventional": False,
        },
        "1.0": {
            "start": init,
            "source": conventional_source(init, "1.0"),
            "conventional": True,
        },
    }
    for target in TARGETS:
        for wd in ("0.333", "1.0"):
            state = states[wd]
            start = int(state["start"])
            if target <= start:
                continue
            digest = hashlib.sha256(
                f"{chain_name}:{init}:{wd}:{target}".encode()
            ).hexdigest()
            stage, next_source = stage_command(
                script,
                source_arguments,
                method="local_sgd",
                sim=sim,
                init=init,
                wd=wd,
                start=start,
                target=target,
                source=str(state["source"]),
                output=output_path("local_sgd", sim, init, wd),
                conventional=bool(state["conventional"]),
                revision=revision,
                chain_name=chain_name,
                nodes=nodes,
                rdzv_id=digest[:12],
                rdzv_port=29000 + int(digest[:8], 16) % 1000,
                metric_dir=metric_dir,
            )
            if wd == "0.333":
                commands.append(f"if [ ! -f {shlex.quote(prune_file)} ]; then")
            commands.extend(stage)
            if wd == "0.333":
                commands.append("fi")
            state.update(start=target, source=next_source, conventional=False)

        wd1_state = states["1.0"]
        if target < wd0333_start or int(wd1_state["start"]) < target:
            continue
        own_file = f"{metric_dir}/e{target}-wd0.333.val"
        peer_file = f"{metric_dir}/e{target}-wd1.0.val"
        decision_file = f"{metric_dir}/e{target}-wd-gate.decision"
        decision_tmp = f"{metric_dir}/.e{target}-wd-gate.decision.tmp"
        commands.extend(
            [
                f'if [ "{leader}" = 0 ]; then',
                f"if [ -f {shlex.quote(prune_file)} ]; then",
                f"printf '%s\\n' prune > {shlex.quote(decision_tmp)}",
                "else",
                shlex.join(["test", "-s", own_file]),
                shlex.join(["test", "-s", peer_file]),
                f"wd0333_ce=$(sed -n 1p {shlex.quote(own_file)})",
                f"wd1_ce=$(sed -n 1p {shlex.quote(peer_file)})",
                f'echo "WD_GATE target=E{target} wd0.333=${{wd0333_ce}} wd1.0=${{wd1_ce}}"',
                'if awk -v wd1="${wd1_ce}" -v wd0333="${wd0333_ce}" '
                + shlex.quote("BEGIN { exit !(wd1 < wd0333) }")
                + "; then",
                f'echo "WD_GATE_PRUNE wd=0.333 after=E{target}"',
                f"touch {shlex.quote(prune_file)}",
                f"printf '%s\\n' prune > {shlex.quote(decision_tmp)}",
                "else",
                f"printf '%s\\n' keep > {shlex.quote(decision_tmp)}",
                "fi",
                "fi",
                f"mv {shlex.quote(decision_tmp)} {shlex.quote(decision_file)}",
                "fi",
                f"for gate_wait in $(seq 1 720); do test -s {shlex.quote(decision_file)} && break; sleep 1; done",
                shlex.join(["test", "-s", decision_file]),
                f"gate_decision=$(sed -n 1p {shlex.quote(decision_file)})",
                "test \"${gate_decision}\" = keep || test \"${gate_decision}\" = prune",
                f'echo "WD_GATE_SHARED_DECISION target=E{target} decision=${{gate_decision}}"',
            ]
        )
    commands.extend(
        [
            f'if [ "{leader}" = 0 ]; then',
            f"touch {shlex.quote(metric_dir + f'/done-init{init}-wd0.333')}",
            f"touch {shlex.quote(metric_dir + f'/done-init{init}-wd1.0')}",
            "fi",
            f'echo "LOCAL_UPDATE_CHAIN_COMPLETE init=E{init} paired_wd=true"',
        ]
    )
    return commands


def clean_task(task: dict[str, Any], *, revision: str, priority: str) -> dict[str, Any]:
    task = copy.deepcopy(task)
    blocked = {"GANTRY_RDZV_ID", "GANTRY_RDZV_PORT", "NUM_NODES"}
    task["envVars"] = [
        env for env in task.get("envVars", [])
        if env.get("name") not in blocked
        and not (str(env.get("name", "")).startswith("BEAKER_") and env.get("name") != "BEAKER_TOKEN")
    ]
    set_env(task, "GIT_REF", revision)
    set_env(task, "GIT_BRANCH", BRANCH)
    task["resources"] = {"gpuCount": 8, "sharedMemory": "10 GiB"}
    task["context"] = {"priority": priority, "minRuntime": "0s", "autoResume": False}
    task["hostNetworking"] = True
    task["propagateFailure"] = True
    task["propagatePreemption"] = True
    return task


def training_task(
    base: dict[str, Any], *, name: str, commands: list[str], nodes: int,
    revision: str, priority: str,
) -> dict[str, Any]:
    task = clean_task(base, revision=revision, priority=priority)
    task["name"] = name
    task["arguments"] = ["bash", "-lc", "\n".join(commands)]
    if nodes > 1:
        task.update(replicas=nodes, leaderSelection=True, synchronizedStartTimeout="90m")
    else:
        for key in ("replicas", "leaderSelection", "synchronizedStartTimeout"):
            task.pop(key, None)
    return task


def idle_task(
    base: dict[str, Any], *, name: str, done_files: list[str], replicas: int,
    revision: str, priority: str,
) -> dict[str, Any]:
    task = clean_task(base, revision=revision, priority=priority)
    task["name"] = name
    tests = " && ".join(f"test -f {shlex.quote(path)}" for path in done_files)
    task["arguments"] = [
        "bash", "-lc",
        "set -euo pipefail\n"
        f"for idle_wait in $(seq 1 20160); do if {tests}; then exit 0; fi; sleep 30; done\n"
        "echo 'timed out waiting for chain completion' >&2\nexit 1",
    ]
    task.update(replicas=replicas, leaderSelection=True, synchronizedStartTimeout="90m")
    return task


def registered(name: str) -> None:
    report = json.loads(REPORT.read_text())
    matches = [run for run in report.get("runs", []) if run.get("chainName") == name]
    if not matches or any(run.get("status") != "planned" for run in matches):
        raise RuntimeError(f"{name} is not fully registered as planned")


def build_spec(args: argparse.Namespace) -> dict[str, Any]:
    method = "local_sgd" if args.kind == "local-sgd" else "diloco"
    sim = args.simulated_sequences or 64
    source = source_spec(method, sim)
    script, source_arguments = extract_training(source)
    base_task = source["tasks"][0]
    metric_dir = f"{ROOT}/.chain-control/{args.name}"
    tasks: list[dict[str, Any]] = []
    done_files: list[str] = []
    if args.kind == "local-sgd":
        init = args.init_epoch
        assert init is not None and args.simulated_sequences is not None
        if args.resume_wd1_e16_only:
            commands = ["set -euo pipefail"]
            commands.extend(
                stage_command(
                    script,
                    source_arguments,
                    method=method,
                    sim=sim,
                    init=init,
                    wd="1.0",
                    start=12,
                    target=16,
                    source=f"{output_path('local_sgd', sim, init, '1.0')}/step2575",
                    output=output_path("local_sgd", sim, init, "1.0"),
                    conventional=False,
                    revision=args.revision,
                    chain_name=args.name,
                    nodes=4,
                    rdzv_id=hashlib.sha256(
                        f"{args.name}:{init}:1.0:16".encode()
                    ).hexdigest()[:12],
                    rdzv_port=29641,
                    metric_dir=metric_dir,
                )[0]
            )
            commands.extend(
                [
                    f'if [ "${{BEAKER_REPLICA_RANK:-0}}" = 0 ]; then '
                    f"touch {shlex.quote(metric_dir + f'/done-init{init}-wd1.0')}; fi",
                    f'echo "LOCAL_UPDATE_CHAIN_COMPLETE init=E{init} wd=1.0 recovery=true"',
                ]
            )
            tasks.append(
                training_task(
                    base_task,
                    name="local-sgd-wd1p0-e16-recovery",
                    commands=commands,
                    nodes=4,
                    revision=args.revision,
                    priority=args.priority,
                )
            )
        elif sim == 64:
            commands = branch_commands(
                script, source_arguments, method=method, sim=sim, init=init, wd="0.333",
                revision=args.revision, chain_name=args.name, metric_dir=metric_dir,
                nodes=4, gated=False,
            )
            tasks.append(training_task(
                base_task, name="local-sgd-wd0p333", commands=commands, nodes=4,
                revision=args.revision, priority=args.priority,
            ))
        else:
            commands = paired_local_wd_commands(
                script,
                source_arguments,
                sim=sim,
                init=init,
                revision=args.revision,
                chain_name=args.name,
                metric_dir=metric_dir,
                nodes=4,
            )
            tasks.append(training_task(
                base_task, name="local-sgd-paired-wd", commands=commands, nodes=4,
                revision=args.revision, priority=args.priority,
            ))
    else:
        commands: list[str] = []
        for init in (2, 4):
            commands.extend(branch_commands(
                script, source_arguments, method=method, sim=64, init=init, wd="0.333",
                revision=args.revision, chain_name=args.name, metric_dir=metric_dir,
                nodes=4, gated=False,
            ))
        tasks.append(training_task(
            base_task, name="diloco-e2e4-init", commands=commands, nodes=4,
            revision=args.revision, priority=args.priority,
        ))
    spec = copy.deepcopy(source)
    spec["tasks"] = tasks
    spec.pop("description", None)
    shell = "\n".join(task["arguments"][2] for task in tasks)
    required = [args.revision, "step3432", "step3815", "dclm-validation-0802"]
    missing = [value for value in required if value not in shell]
    if missing:
        raise RuntimeError(f"built chain is missing required guards: {missing}")
    if args.kind == "diloco":
        for required_arg in ("diloco_outer_lr=0.5", "diloco_outer_momentum=0.7"):
            if required_arg not in shell:
                raise RuntimeError(f"missing {required_arg}")
    return spec


def main() -> None:
    args = parse_args()
    registered(args.name)
    spec = build_spec(args)
    if args.print_only:
        json.dump(spec, sys.stdout, indent=2)
        print()
        return
    result = subprocess.run(
        [
            "beaker", "experiment", "create", "-", "--name", args.name,
            "--workspace", args.workspace,
        ],
        input=json.dumps(spec), check=True, stdout=subprocess.PIPE, text=True,
    )
    print(result.stdout, end="")


if __name__ == "__main__":
    main()
