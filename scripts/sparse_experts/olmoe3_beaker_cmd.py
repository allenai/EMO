#!/usr/bin/env python3
"""Launch an arbitrary command on Beaker with the olmoe3_275m launch settings (pinned submodule).

Same image / env / secrets / weka / install redirection as scripts/sparse_experts/olmoe3_275m.py,
but for one-off jobs (evals, extraction) on N GPUs of a single node, unallocated filler by default.

  PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py \
      --name routing-pilot --gpus 1 -- python scripts/sparse_experts/olmoe3_routing/extract_routing.py ...
Options: --gpus N (default 1), --cluster (ai2/jupiter), --allocated, --follow, --dry-run.
"""
import argparse, os, sys
from olmo_core.launch.beaker import BeakerEnvSecret, BeakerEnvVar, BeakerLaunchConfig, BeakerWekaBucket

H100_IMAGE = "petew/olmo-core-tch2100cu128-2026-01-23"
SUB = "external/OLMo-core"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--cluster", default="ai2/jupiter")
    ap.add_argument("--image", default=H100_IMAGE)
    ap.add_argument("--workspace", default="ai2/flex2")
    ap.add_argument("--priority", default="urgent")
    ap.add_argument("--allocated", action="store_true", help="allocated slot instead of unallocated filler")
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--env", action="append", default=[], help="extra KEY=VALUE for the worker")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    assert cmd, "give the command after --"
    env = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "OLMO_SYMM_VDEV2D_AUTO_BUILD": "0",
        "TORCH_CUDA_ARCH_LIST": "9.0",
        "PYTHONPATH": f"{SUB}/src:src",
        "GANTRY_INSTALL_CMD": f"uv pip install --system --break-system-packages -e '{SUB}[beaker,wandb,fla]'",
    }
    for kv in a.env:
        k, v = kv.split("=", 1); env[k] = v
    aws = ('mkdir -p ~/.aws && printf "[S3]\\naws_access_key_id=%s\\naws_secret_access_key=%s\\n" '
           '"$AWS_ACCESS_KEY_ID" "$AWS_SECRET_ACCESS_KEY" > ~/.aws/credentials')
    launch = BeakerLaunchConfig(
        name=a.name, cmd=cmd, task_name="run", budget="ai2/oe-other", workspace=a.workspace,
        clusters=[a.cluster], num_nodes=1, num_gpus=a.gpus, beaker_image=a.image, priority=a.priority,
        preemptible=False if a.allocated else None, min_runtime=None if a.allocated else "0s",
        shared_filesystem=True, weka_buckets=[BeakerWekaBucket("oe-training-default", "/weka/oe-training-default")],
        env_vars=[BeakerEnvVar(name=k, value=v) for k, v in env.items()],
        env_secrets=[BeakerEnvSecret(name="BEAKER_TOKEN", secret="RYAN_BEAKER_TOKEN"),
                     BeakerEnvSecret(name="WANDB_API_KEY", secret="RYAN_WANDB_API_KEY"),
                     BeakerEnvSecret(name="AWS_ACCESS_KEY_ID", secret="RYAN_AWS_ACCESS_KEY_ID"),
                     BeakerEnvSecret(name="AWS_SECRET_ACCESS_KEY", secret="RYAN_AWS_SECRET_ACCESS_KEY"),
                     BeakerEnvSecret(name="HF_TOKEN", secret="RYAN_HF_TOKEN", required=False)],
        gh_token_secret="RYAN_GITHUB_TOKEN", allow_dirty=True, torchrun=False, follow=a.follow,
        post_setup=aws, launch_timeout=6 * 3600, step_soft_timeout=None, step_timeout=None,
    )
    if a.dry_run:
        print(launch); return
    launch.launch()


if __name__ == "__main__":
    main()
