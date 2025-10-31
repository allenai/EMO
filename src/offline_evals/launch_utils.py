import logging
import os
import subprocess
import tempfile

import yaml  # type: ignore[import-untyped]
from oe_eval_internal.utilities.launch_utils import (
    add_gantry_arg,
    make_beaker_spec,
    make_gantry_command,
)

logger = logging.getLogger(__name__)


def launch_internal(args_dict, run_eval_command, internal_args, num_tasks):
    gantry_args = internal_args["gantry_args"]
    print(gantry_args)
    beaker_spec = make_beaker_spec(
        run_eval_command,
        gantry_args,
        needs_nfs=internal_args["needs_nfs"],
        needs_s3=internal_args["needs_s3"],
    )
    beaker_command = (
        f"beaker experiment create -n {gantry_args['name']} -w {gantry_args['workspace']}"
    )
    gantry_command = ""
    if args_dict["use_gantry"]:
        if gantry_args.get("beaker-retries"):
            raise ValueError("Cannot use gantry with beaker-retries!")
        # from git.repo import Repo

        # repo = Repo(ROOT_DIR)
        # branch_name = repo.active_branch.name

        branch_name = "main"

        # This is not wanted for non-gantry jobs
        add_gantry_arg(gantry_args, "env", f"GIT_BRANCH_NAME={branch_name}")
        gantry_command = make_gantry_command(gantry_args)

    if args_dict["dry_run"] or not internal_args["beaker_cluster_list"]:
        if args_dict["use_gantry"]:
            full_command = gantry_command + " -- " + run_eval_command
            logger.info(f"Gantry command: {full_command}")
        elif internal_args["beaker_cluster_list"] and not args_dict["run_local"]:
            full_command = beaker_spec
            logger.info(f"Beaker command: {beaker_command}")
            logger.info(f"Beaker spec: {yaml.dump(beaker_spec, sort_keys=False)}")
        else:
            full_command = run_eval_command
            logger.info(f"Command: {run_eval_command}")
        return {"command": full_command}

    if args_dict["run_local"]:
        logger.info(f"Running eval locally on {num_tasks} tasks!")
        logger.info(f"Command: {run_eval_command}")
        return subprocess.run(run_eval_command, shell=True).returncode

    if args_dict["use_gantry"]:
        logger.info(f"Launching eval through gantry on {num_tasks} tasks!")
        # Needs to be run in the root directory of the repo
        # os.chdir(ROOT_DIR)
        full_command = gantry_command + " -- " + run_eval_command
        subprocess.run(full_command, shell=True)
        return 0

    import pdb

    pdb.set_trace()
    logger.info(f"Launching eval through beaker on {num_tasks} tasks!")
    spec_file, spec_path = tempfile.mkstemp()
    try:
        with os.fdopen(spec_file, "w") as file:
            yaml.dump(beaker_spec, file, sort_keys=False, default_flow_style=True)
            subprocess.run(beaker_command + " " + spec_path, shell=True)
    finally:
        os.remove(spec_path)
    return 0
