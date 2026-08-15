#!/usr/bin/env python3
"""Build or submit the audited E4-only retry for the canonical H4 LocalSGD run."""

import argparse
import json
import subprocess


SOURCE_EXPERIMENT = "01KZY8FYTWXG1PX88FEX7Y8MDY"
REVISION = "a1e6b71a5bbec5294ecf5498905f2ec850371ea7"
SOURCE_STEP = (
    "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm1b/"
    "bs512_dr_local_sgd_h4_init=bs512e1_lr1e-3_wd0.333/step858"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    source = json.loads(
        subprocess.run(
            ["beaker", "experiment", "spec", SOURCE_EXPERIMENT, "--format", "json"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout
    )
    task = source["tasks"][0]
    shell = task["arguments"][2].splitlines()
    marker = "LOCAL_UPDATE_PREFLIGHT method=post_e1_local_sgd_h4_dr epoch=4"
    marker_index = next(i for i, line in enumerate(shell) if marker in line)
    start = marker_index - 3
    end = next(
        i
        for i in range(marker_index, len(shell))
        if "LOCAL_UPDATE_STAGE_COMPLETE method=post_e1_local_sgd_h4_dr epoch=4" in shell[i]
    )
    continuation = shell[start : end + 1]
    if not any(line == f"test -d {SOURCE_STEP}" for line in continuation):
        raise RuntimeError("E4 continuation does not guard the exact step858 source")

    explicit = "--trainer.prefer_explicit_load_path=true"
    rewritten = []
    for line in continuation:
        if f"--trainer.load_path={SOURCE_STEP}" in line and explicit not in line:
            line = line.replace(
                " --trainer.reset_data_loader_state_on_load_path=false",
                f" {explicit} --trainer.reset_data_loader_state_on_load_path=false",
            )
        rewritten.append(line)
    training_lines = [line for line in rewritten if line.startswith(("python ", "torchrun "))]
    if len(training_lines) != 2 or any(explicit not in line for line in training_lines):
        raise RuntimeError("explicit-load preference missing from E4 dry-run or training command")
    for required in (
        "--train_module.batch_simulation.local_sgd_sync_interval=4",
        "--dynamic-repacking",
        "--train_module.optim.weight_decay=0.333",
        "--lr=1e-3",
        "--trainer.callbacks.checkpointer.fixed_steps=[1716]",
    ):
        if any(required not in line for line in training_lines):
            raise RuntimeError(f"missing required E4 setting: {required}")

    task["arguments"] = [
        "bash",
        "-lc",
        "\n".join(
            [
                "set -euo pipefail",
                f'test "$(git rev-parse HEAD)" = "{REVISION}"',
                *rewritten,
            ]
        ),
    ]
    task["context"] = {"priority": "urgent", "minRuntime": 0, "autoResume": False}
    for env in task.get("envVars", []):
        if env.get("name") == "GIT_REF":
            env["value"] = REVISION
        elif env.get("name") == "GIT_BRANCH":
            env["value"] = "sewonm/icsl"
    source["description"] = (
        "LocalSGD H4 BS512-E1-init DR WD0.333 E4-only continuation from exact pre-decay step858."
    )

    if not args.submit:
        print(json.dumps(source, indent=2))
        return
    completed = subprocess.run(
        [
            "beaker",
            "experiment",
            "create",
            "-",
            "--name",
            "bs512-dr-local-sgd-h4-init-bs512e1-wd0.333-e4",
            "--workspace",
            "ai2/flex2",
            "--priority",
            "urgent",
            "--format",
            "json",
        ],
        check=True,
        input=json.dumps(source),
        text=True,
    )


if __name__ == "__main__":
    main()
