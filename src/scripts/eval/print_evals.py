#!/usr/bin/env python3
"""
Evaluation Results Aggregator

This script collects and displays evaluation results from local or S3 model directories.
It expects task-*metrics.json files containing evaluation scores.

The --base-dir argument can be:
  1. A parent directory containing multiple model directories, where each model directory contains task-*metrics.json files
  2. A single model directory containing metrics files directly

Usage:
    # Multiple models (parent directory)
    python print_evals.py --base-dir /path/to/evaluations/

    # Single model (direct path)
    python print_evals.py --base-dir /path/to/evaluations/model1/

    # S3 paths work the same way
    python print_evals.py --base-dir s3://bucket/evaluations/ --export-csv results.csv
"""

import argparse
import csv
import glob
import json
import os
import pickle as pkl
import re
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import smart_open
from fsspec import AbstractFileSystem, get_filesystem_class
from prettytable import PrettyTable
from termcolor import colored
from tqdm import tqdm

MMLU_SUBCATEGORIES = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

MMLU_CATEGORIES = {
    "physics": "STEM",
    "chemistry": "STEM",
    "biology": "STEM",
    "computer science": "STEM",
    "math": "STEM",
    "engineering": "STEM",
    "history": "humanities",
    "philosophy": "humanities",
    "law": "humanities",
    "politics": "social sciences",
    "culture": "social sciences",
    "economics": "social sciences",
    "geography": "social sciences",
    "psychology": "social sciences",
    "other": "other (business, health, misc.)",
    "business": "other (business, health, misc.)",
    "health": "other (business, health, misc.)",
}


def _get_s3_kwargs() -> Dict[str, Any]:
    """Read WEKA S3 credentials from AWS profile if available."""
    try:
        import configparser

        config = configparser.ConfigParser()
        config.read(os.path.expanduser("~/.aws/config"))
        creds = configparser.ConfigParser()
        creds.read(os.path.expanduser("~/.aws/credentials"))
        if "WEKA" in creds and "profile WEKA" in config:
            return {
                "key": creds["WEKA"].get("aws_access_key_id", ""),
                "secret": creds["WEKA"].get("aws_secret_access_key", ""),
                "client_kwargs": {"endpoint_url": config["profile WEKA"].get("endpoint_url", "")},
            }
    except Exception:
        pass
    return {}


FS_KWARGS: Dict[str, Dict[str, Any]] = {
    "": {"auto_mkdir": True},
    "s3": _get_s3_kwargs(),
}

RE_GLOB_STAR_ESCAPE = re.compile(r"(?<!\\)\*")
RE_GLOB_ONE_ESCAPE = re.compile(r"(?<!\\)\?")
RE_GLOB_OPEN_ESCAPE = re.compile(r"(?<!\\)\[")
RE_GLOB_CLOSE_ESCAPE = re.compile(r"(?<!\\)\]")
ESCAPE_SYMBOLS_MAP = {"*": "\u2581", "?": "\u2582", "[": "\u2583", "]": "\u2584"}
REVERSE_ESCAPE_SYMBOLS_MAP = {v: k for k, v in ESCAPE_SYMBOLS_MAP.items()}
PATCHED_GLOB = False


def _get_fs(path: Union[Path, str]) -> AbstractFileSystem:
    """Get the filesystem class for a given path."""
    path = str(path)
    protocol = urlparse(path).scheme
    fs = get_filesystem_class(protocol)(**FS_KWARGS.get(protocol, {}))

    global PATCHED_GLOB
    # patch glob method to support recursive globbing
    if protocol == "" and not PATCHED_GLOB:
        fs.glob = partial(glob.glob, recursive=True)
        PATCHED_GLOB = True

    return fs


def _escape_glob(s: Union[str, Path]) -> str:
    """Escape glob characters in a string."""
    s = str(s)
    s = RE_GLOB_STAR_ESCAPE.sub(ESCAPE_SYMBOLS_MAP["*"], s)
    s = RE_GLOB_ONE_ESCAPE.sub(ESCAPE_SYMBOLS_MAP["?"], s)
    s = RE_GLOB_OPEN_ESCAPE.sub(ESCAPE_SYMBOLS_MAP["["], s)
    s = RE_GLOB_CLOSE_ESCAPE.sub(ESCAPE_SYMBOLS_MAP["]"], s)
    return s


def _unescape_glob(s: Union[str, Path]) -> str:
    """Unescape glob characters in a string."""
    s = str(s)
    for k, v in REVERSE_ESCAPE_SYMBOLS_MAP.items():
        s = s.replace(k, v)
    return s


def _pathify(path: Union[Path, str]) -> Tuple[str, Path]:
    """Return the protocol and path of a given path."""
    path = _escape_glob(str(path))
    parsed = urlparse(path)
    path = Path(f"{parsed.netloc}/{parsed.path}") if parsed.netloc else Path(parsed.path)
    return parsed.scheme, path


def join_path(protocol: Union[str, None], *parts: Union[str]) -> str:
    """Join a path from its protocol and path components."""
    from itertools import chain

    all_prots, all_parts = zip(
        *(_pathify(p) for p in chain.from_iterable([p] if isinstance(p, str) else p for p in parts))
    )
    path = str(Path(*all_parts)).rstrip("/")
    protocol = protocol or str(all_prots[0])

    if protocol:
        path = f"{protocol}://{path.lstrip('/')}"
    return _unescape_glob(path)


def glob_path(
    path: Union[Path, str],
    hidden_files: bool = False,
    autoglob_dirs: bool = True,
    recursive_dirs: bool = False,
    yield_dirs: bool = True,
) -> Iterator[str]:
    """Expand a glob path into a list of paths."""
    protocol, parsed_path = _pathify(path)
    fs = _get_fs(path)

    if fs.isdir(path) and autoglob_dirs:
        path = join_path(protocol, _unescape_glob(parsed_path), "*")

    for gl in fs.glob(path):
        gl = str(gl)

        if not hidden_files and Path(gl).name.startswith("."):
            continue

        if fs.isdir(gl):
            if recursive_dirs:
                yield from glob_path(
                    gl,
                    hidden_files=hidden_files,
                    autoglob_dirs=autoglob_dirs,
                    recursive_dirs=recursive_dirs,
                    yield_dirs=yield_dirs,
                )
            if yield_dirs:
                yield join_path(protocol, gl)
        else:
            yield join_path(protocol, gl)


def map_mmlu_cat(subtask):
    subtask = subtask.split("mmlu_")[-1].split(":")[0]
    return MMLU_CATEGORIES[MMLU_SUBCATEGORIES[subtask][0]]


def map_mmlu_subcat(subtask):
    subtask = subtask.split("mmlu_")[-1].split(":")[0]
    return MMLU_SUBCATEGORIES[subtask][0]


def avg_tasks(results, label, task_names):
    """Average scores across multiple tasks into a single metric."""
    for model_name in results:
        # Collect valid (non-None, non-N/A) scores for averaging
        valid_scores = [
            results[model_name][task_name]
            for task_name in task_names
            if task_name in results[model_name]
            and results[model_name][task_name] is not None
            and results[model_name][task_name] != "N/A"
        ]
        # Only compute average if we have at least one valid score
        if valid_scores:
            results[model_name][label] = np.mean(valid_scores)
        else:
            results[model_name][label] = None  # No valid scores to average
        # Remove individual task scores after averaging
        for task_name in task_names:
            if task_name in results[model_name]:
                del results[model_name][task_name]
    return results


def get_task_names(results):
    """Get all unique task names across all models."""
    task_names = set()
    for dic in results.values():
        for task_name in dic:
            task_names.add(task_name)
    return task_names


def main(args):
    # Cache handling
    if not os.path.exists("cached_results.pkl") or args.reset_cache:
        assert not args.load_cache
        results: dict = defaultdict(dict)
        visited_paths = set()
    else:
        with open("cached_results.pkl", "rb") as f:
            results, visited_paths = pkl.load(f)

    if not args.load_cache:
        updated = False
        # Track number of instances per task per model for comparability checking
        num_instances_tracker: dict = defaultdict(lambda: defaultdict(int))

        # Build smart_open transport params for WEKA S3
        _s3_tp: dict = {}
        s3_kw = _get_s3_kwargs()
        if s3_kw and args.base_dir.startswith("s3://"):
            import boto3

            _s3_tp = {
                "client": boto3.client(
                    "s3",
                    aws_access_key_id=s3_kw.get("key"),
                    aws_secret_access_key=s3_kw.get("secret"),
                    endpoint_url=s3_kw.get("client_kwargs", {}).get("endpoint_url"),
                )
            }

        def _add_to_results(model_name, task_name, score, num_instances=None):
            if model_name not in results:
                results[model_name] = {}
            if task_name in results[model_name]:
                if results[model_name][task_name] != score:
                    print(
                        f"Warning: Duplicated results not identical for {task_name} in {model_name}: "
                        f"{results[model_name][task_name]} vs {score}"
                    )
                    exit()
            else:
                results[model_name][task_name] = score

            # Track num_instances if provided
            if num_instances is not None:
                num_instances_tracker[task_name][model_name] = num_instances

        # Model directory discovery
        # Check if base_dir itself contains metrics files
        # if this is not empty, the user is likely looking at results for a single model in this directory
        direct_metrics = list(glob_path(os.path.join(args.base_dir, "task-*metrics.json")))

        if direct_metrics:
            # base_dir is a model directory itself
            model_dirs = [args.base_dir]
        else:
            # if we reach here, we are looking at results for multiple models in this directory
            # base_dir contains model directories
            model_dirs = list(glob_path(os.path.join(args.base_dir, "*")))

        if args.show_models:
            show_models = args.show_models.split(",")
            model_dirs = [dir for dir in model_dirs if any(t in dir for t in show_models)]

        if args.hide_models:
            hide_models = args.hide_models.split(",")
            model_dirs = [dir for dir in model_dirs if not any(t in dir for t in hide_models)]

        for model_dir in tqdm(model_dirs, desc="Processing models"):
            model_name = model_dir.split("/")[-1]

            # Find all metric files
            metric_paths = list(glob_path(os.path.join(model_dir, "task-*metrics.json")))

            if args.show_tasks:
                show_tasks = args.show_tasks.split(",")
                metric_paths = [path for path in metric_paths if any(t in path for t in show_tasks)]

            if args.hide_tasks:
                hide_tasks = args.hide_tasks.split(",")
                metric_paths = [
                    path for path in metric_paths if not any(t in path for t in hide_tasks)
                ]

            for metric_path in metric_paths:
                if metric_path in visited_paths:
                    continue

                with smart_open.open(metric_path, "r", transport_params=_s3_tp) as f:
                    metric = json.load(f)

                task_name = metric["task_name"]
                metrics = metric["metrics"]
                score = metrics["primary_score"]
                num_instances = metric.get("num_instances", None)
                _add_to_results(model_name, task_name, score, num_instances)

                visited_paths.add(metric_path)
                updated = True

        if updated:
            with open("cached_results.pkl", "wb") as f_out:
                pkl.dump([results, visited_paths], f_out)

        # Check for tasks with inconsistent number of examples across models
        print("\n=== Checking task comparability ===")
        has_warnings = False
        for task_name, model_counts in num_instances_tracker.items():
            if len(model_counts) > 1:  # Only check if multiple models have this task
                counts = list(model_counts.values())
                if len(set(counts)) > 1:  # Different counts exist
                    has_warnings = True
                    print(f"\n⚠️  WARNING: Task '{task_name}' has different numbers of examples:")
                    for model_name, count in sorted(model_counts.items()):
                        print(f"   - {model_name}: {count} examples")

        if not has_warnings:
            print("✓ All tasks have consistent numbers of examples across models")
        print()

    # Common task groupings
    core9_tasks = [
        "arc_easy:mc",
        "arc_challenge:mc",
        "boolq:mc",
        "csqa:mc",
        "hellaswag:mc",
        "openbookqa:mc",
        "piqa:mc",
        "socialiqa:mc",
        "winogrande:mc",
    ]

    core9_task_rc = [
        "arc_easy",
        "arc_challenge",
        "boolq",
        "csqa",
        "hellaswag",
        "openbookqa",
        "piqa",
        "socialiqa",
        "winogrande",
    ]

    gen5_tasks = ["coqa", "squad", "naturalqs_open", "triviaqa", "drop"]

    task_names = get_task_names(results)

    # Task averaging options
    if args.avg_core:
        core_tasks_mc_update = [core_task + "_test" for core_task in core9_tasks]
        core_tasks_rc_update = [core_task + ":rc_test" for core_task in core9_task_rc]
        results = avg_tasks(results, "core9:mc", core_tasks_mc_update)
        results = avg_tasks(results, "core9:rc", core_tasks_rc_update)

    if args.avg_mmlu_pro:
        # TODO: default mc ends with nothing, which is ambiguous with rc tasks
        mmlu_pro_tasks = [
            task_name for task_name in task_names if task_name.startswith("mmlu_pro_")
        ]
        if mmlu_pro_tasks:
            results = avg_tasks(results, "mmlu_pro:mc", mmlu_pro_tasks)

    if args.avg_mmlu:
        mmlu_tasks_mc = [
            task_name
            for task_name in task_names
            if task_name.startswith("mmlu_")
            and not task_name.startswith("mmlu_pro_")
            and (
                task_name.endswith(":mc_test")
                or task_name.endswith("_mc")
                or task_name.endswith(":mc")
            )
        ]
        mmlu_tasks_rc = [
            task_name
            for task_name in task_names
            if task_name.startswith("mmlu_")
            and not task_name.startswith("mmlu_pro_")
            and (task_name.endswith(":rc_test") or task_name.endswith(":rc"))
        ]

        if mmlu_tasks_mc:
            results = avg_tasks(results, "mmlu:mc", mmlu_tasks_mc)
        if mmlu_tasks_rc:
            results = avg_tasks(results, "mmlu:rc", mmlu_tasks_rc)

    if args.avg_mmlu_cat or args.avg_mmlu_subcat:
        mmlu_tasks = [
            task_name
            for task_name in task_names
            if task_name.startswith("mmlu_") and not task_name.startswith("mmlu_pro_")
        ]

        mmlu_cats = defaultdict(list)
        for task in mmlu_tasks:
            if args.avg_mmlu_cat:
                mmlu_cats[map_mmlu_cat(task)].append(task)
            else:
                mmlu_cats[map_mmlu_subcat(task)].append(task)

        for mmlu_cat, tasks in mmlu_cats.items():
            results = avg_tasks(results, f"mmlu_{mmlu_cat.split(' ')[0]}:mc", tasks)

    if args.avg_bbh:
        bbh_tasks = [task_name for task_name in task_names if task_name.startswith("bbh_")]
        if bbh_tasks:
            results = avg_tasks(results, "bbh", bbh_tasks)

    if args.avg_gen:
        results = avg_tasks(results, "gen5", gen5_tasks)

    if args.avg_agi_eval:
        # TODO: need to add rc
        agi_eval_tasks = [
            task_name for task_name in task_names if task_name.startswith("agi_eval_")
        ]
        if agi_eval_tasks:
            results = avg_tasks(results, "agi_eval:mc", agi_eval_tasks)

    if args.avg_mm:
        mm_tasks = [task_name for task_name in task_names if task_name.startswith("minerva_math_")]
        if mm_tasks:
            results = avg_tasks(results, "minerva_math", mm_tasks)

    if args.avg_ruler:
        ruler_4k_tasks = [
            task_name for task_name in task_names if task_name.startswith("ruler_4k_")
        ]
        if ruler_4k_tasks:
            results = avg_tasks(results, "ruler_4k", ruler_4k_tasks)

    if args.avg_sciriff:
        sciriff_tasks = [task_name for task_name in task_names if task_name.startswith("sciriff_")]
        if sciriff_tasks:
            results = avg_tasks(results, "sciriff5", sciriff_tasks)

    if args.avg_chembench:
        # MC tasks (accuracy-based) - averaged separately
        chembench_mc_tasks = [
            task_name
            for task_name in task_names
            if task_name.startswith("chembench_") and ":mc" in task_name
        ]
        if chembench_mc_tasks:
            results = avg_tasks(results, "chembench:mc", chembench_mc_tasks)
        # Generative tasks (F1/EM-based) - averaged separately
        # Note: We don't combine MC and gen scores because they use different metrics
        # (accuracy vs F1) and are not directly comparable
        chembench_gen_tasks = [
            task_name
            for task_name in task_names
            if task_name.startswith("chembench_") and ":gen" in task_name
        ]
        if chembench_gen_tasks:
            results = avg_tasks(results, "chembench:gen", chembench_gen_tasks)
        # RC tasks (length-normalized accuracy) - averaged separately
        chembench_rc_tasks = [
            task_name
            for task_name in task_names
            if task_name.startswith("chembench_") and ":rc" in task_name
        ]
        if chembench_rc_tasks:
            results = avg_tasks(results, "chembench:rc", chembench_rc_tasks)

    if args.avg_legalbench:
        # All LegalBench tasks use RC (ranked classification) evaluation
        # Aggregate all ~110 classification tasks into a single score
        legalbench_tasks = [
            task_name
            for task_name in task_names
            if task_name.startswith("legalbench_") and ":rc" in task_name
        ]
        if legalbench_tasks:
            results = avg_tasks(results, "legalbench:rc", legalbench_tasks)

    if args.avg_frenchbench:
        # All FrenchBench tasks use RC (ranked classification) evaluation
        frenchbench_tasks = [
            task_name
            for task_name in task_names
            if task_name.startswith("frenchbench_") and ":rc" in task_name
        ]
        if frenchbench_tasks:
            results = avg_tasks(results, "frenchbench:rc", frenchbench_tasks)

    if args.avg_code:
        code_tasks = [
            task_name
            for task_name in task_names
            if ("mbpp" in task_name or "codex" in task_name) and not task_name.endswith("@10")
        ]
        if code_tasks:
            results = avg_tasks(results, "code_4tasks", code_tasks)

    # Filter results based on task types
    if args.stem_only:
        stem_tasks = [
            "physics",
            "chemistry",
            "biology",
            "computer science",
            "math",
            "engineering",
            "health",
            "arc",
            "drop",
            "bbh",
            "agi_eval",
            "gsm8k",
            "openbook",
            "piqa",
        ]
        results = {
            model_name: {
                k: v for k, v in result.items() if any(keyword in k for keyword in stem_tasks)
            }
            for model_name, result in results.items()
        }

    # Model filtering
    if args.show_models:
        show_models = args.show_models.split(",")
        results = {
            key: value for key, value in results.items() if any(m in key for m in show_models)
        }

    if args.hide_models:
        hide_models = args.hide_models.split(",")
        results = {
            key: value for key, value in results.items() if not any(m in key for m in hide_models)
        }

    # Prepare task list for display
    pt = PrettyTable()
    task_names = sorted(get_task_names(results))

    def _is_code(task_name):
        return (
            "mbpp" in task_name
            or "bigcodebench" in task_name
            or "codex" in task_name
            or "code_4tasks" in task_name
        )

    # Task filtering
    if args.code_only:
        task_names = [task_name for task_name in task_names if _is_code(task_name)]
    elif args.hide_code:
        task_names = [task_name for task_name in task_names if not _is_code(task_name)]
    else:
        # Sort so that code tasks go last
        task_names = sorted(task_names, key=_is_code)

    if args.show_tasks:
        show_tasks = args.show_tasks.split(",")
        task_names = [
            task_name for task_name in task_names if any(t in task_name for t in show_tasks)
        ]

    if args.hide_tasks:
        hide_tasks = args.hide_tasks.split(",")
        task_names = [
            task_name for task_name in task_names if not any(t in task_name for t in hide_tasks)
        ]

    if args.core_and_gen_only:
        task_names = [
            task_name for task_name in task_names if task_name in core9_tasks + gen5_tasks
        ]

    # Compute best scores for highlighting
    best_results = {}
    for task_name in task_names:
        # Filter out None and N/A values when computing best score
        valid_scores = [
            _results.get(task_name, -1)
            for _results in results.values()
            if _results.get(task_name) is not None and _results.get(task_name) != "N/A"
        ]
        if valid_scores:
            best_results[task_name] = (
                np.min(valid_scores) if args.lower_is_better else np.max(valid_scores)
            )
        else:
            best_results[task_name] = -1

    def format_model_name(model):
        """Format model name for display. Applies nicknames if configured."""
        if args.nicknames:
            for mapping in args.nicknames.split(","):
                if ":" in mapping:
                    pattern, nickname = mapping.split(":", 1)
                    if pattern in model:
                        return nickname
        return model

    def format_number(v, task_name):
        """Format score with best result highlighting."""
        # Handle None, "N/A", or missing scores
        if v is None or v == "N/A":
            return "N/A"
        is_best = np.abs(v - best_results[task_name]) < 0.001
        v = f"{v:.3f}"
        return colored(v, "red") if is_best else v

    def format_task(name):
        """Format task name for display."""
        name = name.replace("codex_humaneval", "humaneval")
        name = name.replace("bigcodebench", "bcb")
        name = name.replace("plus", "+")
        # Keep :mc suffix for chembench and medmcqa tasks
        if "medmcqa" not in name and "chembench" not in name:
            name = name.replace(":mc", "")
        return name

    # Build and display table
    if args.transpose:
        formatted_task_names = [format_task(name) for name in task_names]
        pt.field_names = ["Model"] + formatted_task_names
        pt.align["Model"] = "l"
        for task_name in formatted_task_names:
            pt.align[task_name] = "r"
        for model_name in sorted(results.keys()):
            pt.add_row(
                [format_model_name(model_name)]
                + [
                    format_number(results[model_name].get(task_name, None), task_name)
                    for task_name in task_names
                ]
            )
    else:
        model_names = sorted(list(results.keys()))
        formatted_model_names = [format_model_name(model_name) for model_name in model_names]

        model_names = [name for i, name in enumerate(model_names) if formatted_model_names[i]]
        formatted_model_names = [name for i, name in enumerate(formatted_model_names) if name]

        pt.align["Model"] = "l"

        pt.field_names = ["Model"] + formatted_model_names
        pt.align["Model"] = "l"
        for model_name in formatted_model_names:
            pt.align[model_name] = "r"
        for task_name in task_names:
            pt.add_row(
                [format_task(task_name)]
                + [
                    format_number(results[model_name].get(task_name, None), task_name)
                    for model_name in model_names
                ]
            )

    print(pt)

    # Export to CSV if requested
    if args.export_csv:

        def export_csv(results, task_names, model_names, args):
            with open(args.export_csv, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                if args.transpose:
                    header = ["Model"] + [format_task(name) for name in task_names]
                    writer.writerow(header)
                    for model in model_names:
                        row = [format_model_name(model)]
                        for task in task_names:
                            value = results[model].get(task, None)
                            if value is None or value == "N/A":
                                row.append("N/A")
                            else:
                                row.append(f"{value:.3f}")
                        writer.writerow(row)
                else:
                    header = ["Task"] + [format_model_name(m) for m in model_names]
                    writer.writerow(header)
                    for task in task_names:
                        row = [format_task(task)]
                        for m in model_names:
                            value = results[m].get(task, None)
                            if value is None or value == "N/A":
                                row.append("N/A")
                            else:
                                row.append(f"{value:.3f}")
                        writer.writerow(row)

            print(f"Exported results to {args.export_csv}")

        model_names = sorted(results.keys())
        export_csv(results, task_names, model_names, args)

    # Exclude specific tasks from cache (useful for re-running certain evaluations)
    # Example: exclude_tasks = ["mmlu", "arc"] will remove all MMLU and ARC results from cache
    exclude_tasks: list = []
    if len(exclude_tasks) > 0:
        results = {
            k: {t: r for t, r in v.items() if np.any([et not in t for et in exclude_tasks])}
            for k, v in results.items()
        }
        visited_paths = {
            path
            for path in visited_paths
            if np.any([et not in path.split("/")[-1] for et in exclude_tasks])
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate and display evaluation results from multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display results from all models in a parent directory
  python print_evals.py --base-dir ./evaluations/

  # Display results from a single model directory
  python print_evals.py --base-dir ./evaluations/model1/

  # Display results from S3 (works for both parent and model directories)
  python print_evals.py --base-dir s3://bucket/evaluations/

  # Export to CSV and transpose the table
  python print_evals.py --base-dir ./evaluations/ --export-csv results.csv -t

  # Show only specific models and tasks
  python print_evals.py --show-models model1,model2 --show-tasks arc,boolq
        """,
    )

    parser.add_argument(
        "-b",
        "--base-dir",
        type=str,
        required=True,
        help="Base directory path. Can be either: 1) A parent directory containing model folders, or 2) A single model directory with task-*metrics.json files",
    )

    parser.add_argument(
        "-t",
        "--transpose",
        action="store_true",
        help="Transpose the table (models as rows, tasks as columns)",
    )

    parser.add_argument(
        "-c",
        "--load-cache",
        action="store_true",
        help="Load results from cache without re-scanning directories",
    )

    parser.add_argument(
        "-r",
        "--reset-cache",
        action="store_true",
        help="Clear cache and re-scan all directories",
    )

    parser.add_argument(
        "--export-csv",
        type=str,
        help="Export results to CSV file",
    )

    # Task averaging options
    parser.add_argument("--avg-core", action="store_true", help="Average core 9 tasks")
    parser.add_argument("--avg-mmlu", action="store_true", help="Average MMLU tasks")
    parser.add_argument("--avg-mmlu-cat", action="store_true", help="Average MMLU by category")
    parser.add_argument(
        "--avg-mmlu-subcat", action="store_true", help="Average MMLU by subcategory"
    )
    parser.add_argument("--avg-gen", action="store_true", help="Average generation tasks")
    parser.add_argument("--avg-mmlu-pro", action="store_true", help="Average MMLU Pro tasks")
    parser.add_argument("--avg-agi-eval", action="store_true", help="Average AGI Eval tasks")
    parser.add_argument("--avg-bbh", action="store_true", help="Average BBH tasks")
    parser.add_argument("--avg-mm", action="store_true", help="Average Minerva Math tasks")
    parser.add_argument("--avg-ruler", action="store_true", help="Average RULER tasks")
    parser.add_argument("--avg-sciriff", action="store_true", help="Average SciRIFF tasks")
    parser.add_argument("--avg-chembench", action="store_true", help="Average ChemBench tasks")
    parser.add_argument("--avg-legalbench", action="store_true", help="Average LegalBench tasks")
    parser.add_argument("--avg-frenchbench", action="store_true", help="Average FrenchBench tasks")
    parser.add_argument("--avg-code", action="store_true", help="Average coding tasks")
    parser.add_argument(
        "--avg-all", action="store_true", help="Average all tasks into a single score"
    )

    # Task filtering
    parser.add_argument("--stem-only", action="store_true", help="Show only STEM tasks")
    parser.add_argument("--code-only", action="store_true", help="Show only coding tasks")
    parser.add_argument("--hide-code", action="store_true", help="Hide coding tasks")
    parser.add_argument(
        "--core-and-gen-only", action="store_true", help="Show only core and generation tasks"
    )

    # Model/task filtering
    parser.add_argument("--hide-models", help="Comma-separated list of model patterns to hide")
    parser.add_argument("--show-models", help="Comma-separated list of model patterns to show")
    parser.add_argument("--hide-tasks", help="Comma-separated list of task patterns to hide")
    parser.add_argument("--show-tasks", help="Comma-separated list of task patterns to show")
    parser.add_argument(
        "--nicknames",
        help="Comma-separated pattern:nickname mappings for model names (e.g., 'dolma:Dolma,olmoe:OLMoE')",
    )

    parser.add_argument(
        "--lower-is-better",
        action="store_true",
        help="Highlight the lowest score per task instead of the highest (e.g. for bpb/perplexity metrics)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.avg_mmlu and args.avg_mmlu_cat:
        parser.error("Cannot use both --avg-mmlu and --avg-mmlu-cat")

    # --avg-all enables all averaging flags
    if args.avg_all:
        args.avg_core = True
        args.avg_mmlu = True
        args.avg_mmlu_pro = True
        args.avg_agi_eval = True
        args.avg_bbh = True
        args.avg_gen = True
        args.avg_mm = True
        args.avg_ruler = True
        args.avg_sciriff = True
        args.avg_chembench = True
        args.avg_legalbench = True
        args.avg_frenchbench = True
        args.avg_code = True

    main(args)
