#!/usr/bin/env python3
"""Generate diagnostic train/validation-loss figure options for Section 3.2."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import make_section32_figures as base


WINDOW_TOKENS = 100_000_000
ENDPOINT_EPOCHS = [1, 2, 4, 8, 12]
EXTENDED_EPOCHS = {16: [1, 2, 4, 8, 12, 16, 20], 64: [1, 2, 4, 8, 12, 16, 20], 512: ENDPOINT_EPOCHS}
BATCHES = [16, 64, 512]
Y_LIMITS = (2.60, 4.30)
OUTPUT_DIR = base.OUTPUT_DIR
OPTIONS_DATA = base.DATA_DIR / "section3_train_loss_options.json"


def selected_runs() -> dict:
    selected, _ = base.selected_runs(base.load_js_data("1b"))
    return selected


def refresh_full_e12_histories(selected: dict) -> dict:
    """Fetch complete E12 logs so a continuous E12-run trajectory can be shown."""

    cache = base.load_training_history_cache()
    histories = cache["histories"]
    targets = {}
    for batch in BATCHES:
        run = selected[batch, 12.0]
        targets[run["job"]] = run["wandb"]

    def fetch(job: str, wandb: str) -> tuple[str, list[dict[str, float]]]:
        result = subprocess.run(
            ["beaker", "job", "logs", job, "--no-timestamps"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        if result.returncode:
            raise RuntimeError(f"beaker job logs failed for {job}: {result.stdout[-500:]}")
        parsed = base.parse_training_histories(result.stdout)
        if not parsed.get(wandb):
            parsed = base.parse_training_histories(result.stdout, initial_run=wandb)
        if not parsed.get(wandb):
            raise RuntimeError(f"Could not locate W&B run {wandb} in job {job}")
        return wandb, parsed[wandb]

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fetch, job, wandb) for job, wandb in targets.items()]
        for future in as_completed(futures):
            wandb, history = future.result()
            histories[wandb] = history

    base.TRAIN_HISTORY_CACHE.write_text(json.dumps(cache, indent=2) + "\n")
    return cache


def trailing_mean_series(
    history: list[dict[str, float]],
    window_tokens: float = WINDOW_TOKENS,
    min_tokens: float = 1_000_000_000,
    max_tokens: float = 12_000_000_000,
    output_min_tokens: float | None = None,
    max_points: int = 450,
) -> list[dict[str, float]]:
    # Multi-node jobs can print one CE value per worker group at the same global
    # token count. Collapse those values first so every optimizer step receives
    # equal weight and the moving curve does not zig-zag vertically at one x.
    grouped: dict[float, list[float]] = {}
    for point in history:
        if min_tokens <= point["tokens"] <= max_tokens:
            grouped.setdefault(point["tokens"], []).append(point["ce"])
    points = [
        {"tokens": tokens, "ce": float(np.mean(values))}
        for tokens, values in sorted(grouped.items())
    ]
    if not points:
        return []
    tokens = np.asarray([point["tokens"] for point in points], dtype=float)
    losses = np.asarray([point["ce"] for point in points], dtype=float)
    cumulative = np.concatenate(([0.0], np.cumsum(losses)))
    starts = np.searchsorted(tokens, tokens - window_tokens, side="right")
    counts = np.arange(1, len(points) + 1) - starts
    means = (cumulative[1:] - cumulative[starts]) / counts
    stride = max(1, math.ceil(len(points) / max_points))
    indices = list(range(0, len(points), stride))
    if indices[-1] != len(points) - 1:
        indices.append(len(points) - 1)
    output_min = min_tokens if output_min_tokens is None else output_min_tokens
    return [
        {"epoch": round(tokens[i] / 1e9, 4), "ce": round(float(means[i]), 5)}
        for i in indices
        if tokens[i] >= output_min
    ]


def endpoint_values(selected: dict, histories: dict, batch: int, epochs: list[int]) -> list[dict]:
    values = []
    for epoch in epochs:
        run = selected.get((batch, float(epoch)))
        if run is None or run.get("wandb") not in histories:
            continue
        endpoint = epoch * 1e9
        grouped: dict[float, list[float]] = {}
        for point in histories[run["wandb"]]:
            if endpoint - WINDOW_TOKENS < point["tokens"] <= endpoint:
                grouped.setdefault(point["tokens"], []).append(point["ce"])
        step_means = [float(np.mean(values)) for values in grouped.values()]
        train_ce = float(np.mean(step_means)) if step_means else math.nan
        count = len(step_means)
        values.append(
            {
                "epoch": epoch,
                "train": round(train_ce, 5),
                "validation": round(base.validation(run), 5),
                "count": count,
            }
        )
    return values


def build_options_data(selected: dict, cache: dict) -> dict:
    histories = cache["histories"]
    data = {
        "window_tokens": WINDOW_TOKENS,
        "y_limits": list(Y_LIMITS),
        "endpoint": {str(batch): endpoint_values(selected, histories, batch, ENDPOINT_EPOCHS) for batch in BATCHES},
        "extended": {str(batch): endpoint_values(selected, histories, batch, EXTENDED_EPOCHS[batch]) for batch in BATCHES},
        "continuous": {},
    }
    for batch in BATCHES:
        segments = []
        for epoch in ENDPOINT_EPOCHS:
            run = selected[batch, float(epoch)]
            terminal_start = 0.9 * epoch * 1e9
            curve = trailing_mean_series(
                histories[run["wandb"]],
                min_tokens=max(0.0, terminal_start - WINDOW_TOKENS),
                max_tokens=epoch * 1e9,
                output_min_tokens=terminal_start,
                max_points=160,
            )
            segments.append({"target_epoch": epoch, "curve": curve})
        data["continuous"][str(batch)] = segments
    OPTIONS_DATA.write_text(json.dumps(data, indent=2) + "\n")
    return data


def style(ax: plt.Axes, *, xmax: float) -> None:
    ax.set_ylim(*Y_LIMITS)
    ax.set_xlim(0.6, xmax)
    ax.grid(axis="y", color="#D8DEE9", linewidth=0.65, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7.5, length=2.5)
    ax.set_xlabel("Epoch", fontsize=8)


def save(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUTPUT_DIR / f"{stem}.{suffix}",
            dpi=320,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def plot_endpoints(data: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.42), sharey=True)
    for panel, (ax, batch) in enumerate(zip(axes, BATCHES)):
        values = data["endpoint"][str(batch)]
        epochs = [point["epoch"] for point in values]
        ax.plot(
            epochs,
            [point["train"] for point in values],
            color="#2457A6",
            linestyle="--",
            marker="s",
            markersize=2.8,
            linewidth=1.35,
            label="100M-token mean train CE",
        )
        ax.plot(
            epochs,
            [point["validation"] for point in values],
            color="#C65D3A",
            marker="s",
            markersize=2.8,
            linewidth=1.35,
            label="Validation CE",
        )
        ax.set_xticks(epochs)
        ax.set_title(rf"({chr(97 + panel)}) $B={batch}$", fontsize=9, fontweight="bold")
        style(ax, xmax=12.4)
    axes[0].set_ylabel("CE loss", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2, frameon=False, fontsize=7.2)
    fig.subplots_adjust(top=0.78, wspace=0.10)
    save(fig, "section3_train_validation_tight_100m")


def plot_continuous(data: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.42), sharey=True)
    for panel, (ax, batch) in enumerate(zip(axes, BATCHES)):
        endpoint = data["endpoint"][str(batch)]
        for segment_index, segment in enumerate(data["continuous"][str(batch)]):
            curve = segment["curve"]
            ax.plot(
                [point["epoch"] for point in curve],
                [point["ce"] for point in curve],
                color="#2457A6",
                linewidth=1.1,
                label="Terminal train CE (100M smooth)" if segment_index == 0 else None,
            )
        ax.plot(
            [point["epoch"] for point in endpoint],
            [point["validation"] for point in endpoint],
            color="#C65D3A",
            marker="s",
            markersize=2.8,
            linewidth=1.35,
            label="Tuned endpoint validation CE",
        )
        ax.set_xticks(ENDPOINT_EPOCHS)
        ax.set_title(rf"({chr(97 + panel)}) $B={batch}$", fontsize=9, fontweight="bold")
        style(ax, xmax=12.4)
    axes[0].set_ylabel("CE loss", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2, frameon=False, fontsize=7.2)
    fig.subplots_adjust(top=0.78, wspace=0.10)
    save(fig, "section3_train_validation_continuous_100m")


def plot_extended(data: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.42), sharey=True)
    for panel, (ax, batch) in enumerate(zip(axes, BATCHES)):
        values = data["extended"][str(batch)]
        epochs = [point["epoch"] for point in values]
        ax.plot(
            epochs,
            [point["train"] for point in values],
            color="#2457A6",
            linestyle="--",
            marker="s",
            markersize=2.8,
            linewidth=1.35,
            label="100M-token mean train CE",
        )
        ax.plot(
            epochs,
            [point["validation"] for point in values],
            color="#C65D3A",
            marker="s",
            markersize=2.8,
            linewidth=1.35,
            label="Validation CE",
        )
        ax.set_xticks([1, 4, 8, 12, 16, 20] if batch != 512 else ENDPOINT_EPOCHS)
        ax.set_title(rf"({chr(97 + panel)}) $B={batch}$", fontsize=9, fontweight="bold")
        style(ax, xmax=20.4)
    axes[0].set_ylabel("CE loss", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2, frameon=False, fontsize=7.2)
    fig.subplots_adjust(top=0.78, wspace=0.10)
    save(fig, "section3_train_validation_extended_100m")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-full-e12", action="store_true")
    args = parser.parse_args()
    selected = selected_runs()
    cache = refresh_full_e12_histories(selected) if args.refresh_full_e12 else base.load_training_history_cache()
    data = build_options_data(selected, cache)
    plot_endpoints(data)
    plot_continuous(data)
    plot_extended(data)
    for batch in BATCHES:
        print(f"B={batch}", data["extended"][str(batch)])


if __name__ == "__main__":
    main()
