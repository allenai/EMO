import argparse
import glob
import logging
import os
import shutil
import subprocess
import tempfile

from oe_eval.utils import load_json, load_jsonl, make_cli_command, save_json

## Script for creating in-loop OLMo data for a given oe-eval task
# Example:
#
# python scripts/create_in_loop_data.py --task arc_challenge:mc::olmes arc_easy:mc::olmes --task_sub_name rc_5shot --split validation --olmo-dir ~/gitroot/OLMo
#

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger()

_parser = argparse.ArgumentParser(description="Create in-loop OLMo data for a given oe-eval task")
_parser.add_argument(
    "--task", type=str, nargs="+", required=True, help="Task spec(s) from library or jsonl file"
)
_parser.add_argument("--olmo-dir", type=str, required=True, help="Root directory for OLMo codebase")
_parser.add_argument(
    "--limit",
    type=float,
    default=None,
    help="Override max number (or fraction) of instances for a task",
)
_parser.add_argument(
    "--num-shots", type=int, default=None, help="Override number of shots for each task"
)
_parser.add_argument("--split", default=None, type=str, help="split from which to pull eval docs")
_parser.add_argument(
    "--task_core_name_overrides",
    type=str,
    nargs="+",
    required=False,
    help="Task core name overrides",
)
_parser.add_argument(
    "--task_sub_names", type=str, nargs="+", required=False, help="Task sub name(s)"
)


def main():
    args = _parser.parse_args()
    task = args.task
    olmo_dir = args.olmo_dir
    olmo_data_dir = os.path.join(olmo_dir, "olmo_data", "oe_eval_tasks")
    if not os.path.exists(olmo_data_dir):
        raise ValueError(f"OLMo data directory {olmo_data_dir} does not exist")
    limit = args.limit
    num_shots = args.num_shots
    split = args.split
    task_core_name_overrides = args.task_core_name_overrides
    task_sub_names = args.task_sub_names
    oe_eval_args = {"task": task}
    if limit is not None:
        oe_eval_args["limit"] = limit
    if num_shots is not None:
        oe_eval_args["num_shots"] = num_shots
    if split is not None:
        oe_eval_args["split"] = split
    with tempfile.TemporaryDirectory() as tmp_dir:
        oe_eval_args["output_dir"] = tmp_dir
        oe_eval_args["save_raw_requests"] = True

        logger.info(f"Creating requests data in {tmp_dir}")
        run_oe_eval_command = make_cli_command("python -m offline_evals.run_eval", oe_eval_args)
        subprocess.run(run_oe_eval_command, shell=True)
        num_tasks = len(glob.glob(f"{tmp_dir}/task-*-requests.jsonl"))
        if task_core_name_overrides and len(task_core_name_overrides) != num_tasks:
            raise ValueError(
                f"Number of task core name overrides ({len(task_core_name_overrides)}) must be equal to the number of tasks ({num_tasks})"
            )
        if task_sub_names:
            num_sub_names = len(task_sub_names)
            if num_sub_names != 1 and num_sub_names != num_tasks:
                raise ValueError(
                    f"Number of task sub names ({num_sub_names}) must be 1 or equal to the number of tasks ({num_tasks})"
                )

        # First loop through all tasks and check that things look okay
        olmo_output_dirs = []
        for task_idx in range(num_tasks):
            request_file = glob.glob(f"{tmp_dir}/task-{task_idx:03d}*-requests.jsonl")
            if len(request_file) != 1:
                raise ValueError(
                    f"Expected 1 request file for task {task_idx}, found {len(request_file)}"
                )
            metrics_file = glob.glob(f"{tmp_dir}/task-{task_idx:03d}*-metrics.json")
            if len(metrics_file) != 1:
                raise ValueError(
                    f"Expected 1 metrics file for task {task_idx}, found {len(metrics_file)}"
                )
            metrics = load_json(metrics_file[0])
            primary_metric = metrics["task_config"]["primary_metric"]
            if primary_metric not in [
                "acc_raw",
                "acc_per_char",
                "acc_per_token",
                "acc_uncond",
                "logits_per_byte",
            ]:
                raise ValueError(f"Primary metric {primary_metric} not supported")
            task_core_name = metrics["task_config"]["task_core"]
            if task_core_name_overrides:
                task_core_name = task_core_name_overrides[task_idx]
            task_sub_name = None
            if task_sub_names:
                if len(task_sub_names) == 1:
                    task_sub_name = task_sub_names[0]
                else:
                    task_sub_name = task_sub_names[task_idx]
            olmo_output_dir = os.path.join(olmo_data_dir, task_core_name)
            if task_sub_name:
                olmo_output_dir = os.path.join(olmo_output_dir, task_sub_name)
            if os.path.exists(olmo_output_dir) or olmo_output_dir in olmo_output_dirs:
                raise ValueError(f"OLMo data directory {olmo_output_dir} already exists!")
            olmo_output_dirs.append(olmo_output_dir)

        for task_idx in range(num_tasks):
            request_file = glob.glob(f"{tmp_dir}/task-{task_idx:03d}*-requests.jsonl")
            metrics_file = glob.glob(f"{tmp_dir}/task-{task_idx:03d}*-metrics.json")
            metrics = load_json(metrics_file[0])
            primary_metric = metrics["task_config"]["primary_metric"]
            task_core_name = metrics["task_config"]["task_core"]
            if task_core_name_overrides:
                task_core_name = task_core_name_overrides[task_idx]
            task_sub_name = None
            if task_sub_names:
                if len(task_sub_names) == 1:
                    task_sub_name = task_sub_names[0]
                else:
                    task_sub_name = task_sub_names[task_idx]
            olmo_output_dir = os.path.join(olmo_data_dir, task_core_name)
            if task_sub_name:
                olmo_output_dir = os.path.join(olmo_output_dir, task_sub_name)
            os.makedirs(olmo_output_dir)
            requests = load_jsonl(request_file[0])
            last_doc_id = requests[-1]["doc_id"]
            if last_doc_id > 1000000:
                last_doc_id -= 1000000
            metrics["num_instances"] = last_doc_id + 1
            new_request_file = os.path.join(olmo_output_dir, "requests.jsonl")
            shutil.move(request_file[0], new_request_file)
            subprocess.run(f"gzip {new_request_file}", shell=True)
            if primary_metric in ["acc_per_token", "acc_uncond"]:
                logger.warning(
                    f"Primary metric {primary_metric} not supported by OLMo, using acc_per_char"
                )
                metrics["task_config"]["primary_metric"] = "acc_per_char"
            for key_delete in [
                "metrics",
                "beaker_info",
                "processing_time",
                "compute_config",
                "model_config",
                "model_hash",
            ]:
                if key_delete in metrics:
                    del metrics[key_delete]
            save_json(os.path.join(olmo_output_dir, "config.json"), metrics)
            logger.info(f"Created OLMo data directory {olmo_output_dir}")


if __name__ == "__main__":
    main()