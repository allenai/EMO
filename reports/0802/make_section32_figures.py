#!/usr/bin/env python3
"""Generate the paper figures used in Section 3.2 from the 0802 reports."""

from __future__ import annotations

import json
import math
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPORT_DIR = Path(__file__).resolve().parent
DATA_DIR = REPORT_DIR / "data"
OUTPUT_DIR = REPORT_DIR / "paper_figures"
BATCHES = [16, 32, 64, 128, 256, 512]
TRAIN_VALIDATION_BATCHES = [16, 64, 512]
TRAIN_VALIDATION_EPOCHS = [1, 2, 4, 8, 12]
TRAIN_AVERAGE_TOKENS = 200_000_000
TRAIN_AVERAGE_LOGGED_STEPS = 1
TRAIN_HISTORY_CACHE = DATA_DIR / "section3_train_ce_histories.json"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
WANDB_RUN_RE = re.compile(r"https://wandb\.ai/ai2-llm/sewonm-icsl/runs/([a-z0-9]+)")
TRAIN_CE_RE = re.compile(r"train/CE loss=([0-9.Ee+\-]+)")
TRAIN_TOKENS_RE = re.compile(
    r"throughput/total tokens=([0-9][0-9,]*(?:\.[0-9]+)?)([KMBT]?)"
)


def load_js_data(name: str) -> dict:
    text = (DATA_DIR / f"wsd_batch_size_{name}.js").read_text()
    payload = text[text.index("{") :].strip()
    if payload.endswith(";"):
        payload = payload[:-1]
    return json.loads(payload)


def validation(run: dict) -> float | None:
    value = run.get("validation", run.get("c4"))
    return float(value) if isinstance(value, (int, float)) else None


def selected_runs(report: dict, batches: list[int] = BATCHES) -> tuple[dict, list[dict]]:
    """Replicate the admissibility and coordinate-selection policy in batch_size_charts.js."""

    policy = report.get("selectionPolicy") or {}
    unhealthy = report.get("healthAudit", {}).get("unhealthy", {})

    def admissible(run: dict) -> bool:
        try:
            lr, wd = float(run["lr"]), float(run["wd"])
        except (KeyError, TypeError, ValueError):
            return False
        max_lr = policy.get("maxLearningRate")
        if max_lr is not None and lr > float(max_lr):
            return False
        if policy.get("allowAllCompletedCoordinates"):
            return True
        return lr < 2e-3 or (
            run["batchSequences"] == 256 and lr == 2e-3 and wd == 0.333
        )

    runs: list[dict] = []
    for sweep in report.get("batchSweeps", []):
        for epoch, result in (sweep.get("results") or {}).items():
            run = {**sweep, **result, "epoch": float(epoch)}
            wandb = run.get("wandb") or run.get("activeWandb")
            value = validation(run)
            if (
                run.get("status") == "complete"
                and value is not None
                and math.isfinite(value)
                and admissible(run)
                and wandb not in unhealthy
            ):
                runs.append(run)

    selected: dict[tuple[int, float], dict] = {}
    for batch in batches:
        wd_floor = -math.inf
        epochs = sorted({r["epoch"] for r in runs if r["batchSequences"] == batch})
        for epoch in epochs:
            candidates = [
                r
                for r in runs
                if r["batchSequences"] == batch
                and r["epoch"] == epoch
                and (
                    not policy.get("nondecreasingWd")
                    or float(r["wd"]) >= wd_floor
                )
            ]
            if not candidates:
                continue
            best = min(candidates, key=lambda r: (validation(r), float(r["wd"])))
            selected[batch, epoch] = best
            if policy.get("nondecreasingWd"):
                wd_floor = float(best["wd"])
    return selected, runs


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D8DEE9", linewidth=0.65, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7.5, length=2.5)


def save(fig: plt.Figure, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / filename, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_token_count(value: str, suffix: str) -> float:
    scale = {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
    return float(value.replace(",", "")) * scale[suffix]


def parse_training_histories(
    text: str, initial_run: str | None = None
) -> dict[str, list[dict[str, float]]]:
    """Parse the rank-0 CE series that the trainer prints to Beaker logs."""

    current_run = initial_run
    current_tokens: float | None = None
    histories: dict[str, list[dict[str, float]]] = (
        {initial_run: []} if initial_run is not None else {}
    )
    for line in ANSI_RE.sub("", text).splitlines():
        if match := WANDB_RUN_RE.search(line):
            current_run = match.group(1)
            current_tokens = None
            histories.setdefault(current_run, [])
            continue
        if current_run is None:
            continue
        if match := TRAIN_TOKENS_RE.search(line):
            current_tokens = parse_token_count(match.group(1), match.group(2))
            continue
        if (match := TRAIN_CE_RE.search(line)) and current_tokens is not None:
            histories[current_run].append(
                {"tokens": current_tokens, "ce": float(match.group(1))}
            )
    return histories


def refresh_training_history_cache(selected: dict) -> dict:
    """Fetch only the completed jobs needed by the three-panel paper figure."""

    targets: dict[str, dict] = {}
    for batch in TRAIN_VALIDATION_BATCHES:
        for epoch in TRAIN_VALIDATION_EPOCHS:
            run = selected.get((batch, float(epoch)))
            if run is None or not run.get("wandb") or not run.get("job"):
                continue
            targets[run["wandb"]] = {
                "batch": batch,
                "epoch": epoch,
                "job": run["job"],
            }

    histories: dict[str, list[dict[str, float]]] = {}
    if TRAIN_HISTORY_CACHE.exists():
        histories.update(json.loads(TRAIN_HISTORY_CACHE.read_text()).get("histories", {}))

    jobs: dict[str, set[str]] = {}
    for wandb, target in targets.items():
        endpoint = target["epoch"] * 1_000_000_000
        recent_count = sum(
            endpoint - TRAIN_AVERAGE_TOKENS < point["tokens"] <= endpoint
            for point in histories.get(wandb, [])
        )
        if recent_count < 5:
            jobs.setdefault(target["job"], set()).add(wandb)

    def run_beaker(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )

    def fetch(job: str) -> tuple[str, dict[str, list[dict[str, float]]]]:
        desired_runs = jobs[job]
        command = ["beaker", "job", "logs", job, "--no-timestamps"]

        # The smallest-batch jobs have very long logs. For single-run jobs, request only
        # the final few hours and seed the parser with the known W&B ID because its URL may
        # precede the requested interval. Multi-run jobs remain unsliced so run boundaries
        # are preserved.
        if len(desired_runs) == 1:
            metadata = run_beaker(["beaker", "job", "get", job, "--format", "json"])
            if metadata.returncode == 0:
                payload = json.loads(metadata.stdout)
                record = payload[0] if isinstance(payload, list) else payload
                finalized = record.get("status", {}).get("finalized")
                if finalized:
                    # Python 3.9 only accepts 3 or 6 fractional-second digits, while
                    # Beaker occasionally emits other precisions (for example 5).
                    normalized = re.sub(
                        r"\.(\d+)(?=Z|[+-]\d\d:\d\d$)",
                        lambda match: "." + match.group(1)[:6].ljust(6, "0"),
                        finalized,
                    )
                    end = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
                    since = (end - timedelta(hours=3)).astimezone(timezone.utc)
                    command.extend(["--since", since.isoformat().replace("+00:00", "Z")])

        result = run_beaker(command)
        if result.returncode:
            raise RuntimeError(f"beaker job logs failed for {job}: {result.stdout[-500:]}")
        initial_run = next(iter(desired_runs)) if len(desired_runs) == 1 else None
        parsed = parse_training_histories(result.stdout, initial_run=initial_run)

        if initial_run is not None:
            endpoint = targets[initial_run]["epoch"] * 1_000_000_000
            recent_count = sum(
                endpoint - TRAIN_AVERAGE_TOKENS < point["tokens"] <= endpoint
                for point in parsed.get(initial_run, [])
            )
            if recent_count < 5:
                full = run_beaker(["beaker", "job", "logs", job, "--no-timestamps"])
                if full.returncode:
                    raise RuntimeError(
                        f"beaker job logs fallback failed for {job}: {full.stdout[-500:]}"
                    )
                parsed = parse_training_histories(full.stdout, initial_run=initial_run)

        return job, parsed

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, job): job for job in jobs}
        for future in as_completed(futures):
            job, parsed = future.result()
            for wandb in jobs[job]:
                if parsed.get(wandb):
                    histories[wandb] = parsed[wandb]

    missing = sorted(set(targets) - set(histories))
    if missing:
        raise RuntimeError(f"No training CE history found for W&B runs: {', '.join(missing)}")

    payload = {
        "description": (
            "Rank-0 train/CE loss printed in completed Beaker logs, paired with the "
            "most recent throughput/total tokens value."
        ),
        "targets": targets,
        "histories": histories,
    }
    TRAIN_HISTORY_CACHE.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def load_training_history_cache() -> dict:
    if not TRAIN_HISTORY_CACHE.exists():
        raise FileNotFoundError(
            f"Missing {TRAIN_HISTORY_CACHE}; rerun with --refresh-train-history"
        )
    return json.loads(TRAIN_HISTORY_CACHE.read_text())


def trailing_training_ce(
    history: list[dict[str, float]], endpoint_tokens: float, window_tokens: float
) -> tuple[float, int]:
    values = [
        point["ce"]
        for point in history
        if endpoint_tokens - window_tokens < point["tokens"] <= endpoint_tokens
    ]
    if not values:
        return math.nan, 0
    return float(np.mean(values)), len(values)


def recent_logged_training_ce(
    history: list[dict[str, float]], endpoint_tokens: float, observations: int
) -> tuple[float, int]:
    """Average the final logged optimizer-step losses, collapsing worker duplicates."""

    grouped: dict[float, list[float]] = {}
    for point in history:
        if point["tokens"] <= endpoint_tokens:
            grouped.setdefault(point["tokens"], []).append(point["ce"])
    step_means = [
        float(np.mean(values))
        for _, values in sorted(grouped.items())[-observations:]
    ]
    if not step_means:
        return math.nan, 0
    return float(np.mean(step_means)), len(step_means)


def plot_batch_optima(all_selected: dict[str, dict]) -> None:
    models = [("1b", "1.5B"), ("474m", "474M"), ("153m", "153M")]
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.25))
    color = "#2457A6"
    for ax, (key, title) in zip(axes, models):
        selected = all_selected[key]
        values = []
        epochs = []
        for batch in BATCHES:
            candidates = [r for (b, _), r in selected.items() if b == batch]
            if candidates:
                best = min(candidates, key=validation)
                values.append(validation(best))
                epochs.append(best["epoch"])
            else:
                values.append(np.nan)
                epochs.append(np.nan)
        xs = np.asarray(BATCHES, dtype=float)
        ys = np.asarray(values, dtype=float)
        ax.plot(xs, ys, color=color, marker="o", markersize=4.2, linewidth=1.6)
        if np.isfinite(ys).any():
            best_index = int(np.nanargmin(ys))
            ax.scatter(
                [xs[best_index]],
                [ys[best_index]],
                marker="*",
                s=68,
                color="#D97706",
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )
            ax.annotate(
                f"{ys[best_index]:.3f}\n(E{epochs[best_index]:g})",
                (xs[best_index], ys[best_index]),
                xytext=(0, -4),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=6.8,
                color="#7C4700",
            )
            observed = ys[np.isfinite(ys)]
            pad = max(0.028, 0.40 * (observed.max() - observed.min()))
            ax.set_ylim(observed.min() - pad, observed.max() + pad)
        ax.set_xscale("log", base=2)
        ax.set_xticks(BATCHES, [str(x) for x in BATCHES])
        ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
        ax.set_xlabel("Global batch (sequences)", fontsize=7.5)
        style_axes(ax)
    axes[0].set_ylabel("Best validation CE", fontsize=8)
    fig.subplots_adjust(wspace=0.34)
    save(fig, "section3_batch_optima.png")


def fixed_weight_decay_series(
    selected: dict, runs: list[dict], batch: int
) -> tuple[list[dict], list[dict]]:
    tuned = [r for (b, _), r in selected.items() if b == batch and r["epoch"] >= 1]
    tuned.sort(key=lambda r: r["epoch"])
    epoch_one = selected.get((batch, 1.0))
    if not epoch_one:
        return tuned, []
    wd = float(epoch_one["wd"])
    fixed = []
    for epoch in [r["epoch"] for r in tuned]:
        candidates = [
            r
            for r in runs
            if r["batchSequences"] == batch
            and r["epoch"] == epoch
            and math.isclose(float(r["wd"]), wd)
        ]
        if candidates:
            fixed.append(min(candidates, key=validation))
    return tuned, fixed


def plot_weight_decay(all_selected: dict[str, dict], all_runs: dict[str, list[dict]]) -> None:
    models = [("1b", "1.5B"), ("474m", "474M"), ("153m", "153M")]
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.45))
    styles = {
        (64, "tuned"): ("#2457A6", "-", "o"),
        (64, "fixed"): ("#69A7E0", "--", "o"),
        (512, "tuned"): ("#B33A3A", "-", "s"),
        (512, "fixed"): ("#E79A74", "--", "s"),
    }
    labels = {
        (64, "tuned"): r"$B=64$ · WD tuned",
        (64, "fixed"): r"$B=64$ · epoch-1 WD",
        (512, "tuned"): r"$B=512$ · WD tuned",
        (512, "fixed"): r"$B=512$ · epoch-1 WD",
    }
    for ax, (key, title) in zip(axes, models):
        for batch in (64, 512):
            tuned, fixed = fixed_weight_decay_series(
                all_selected[key], all_runs[key], batch
            )
            for name, series in (("tuned", tuned), ("fixed", fixed)):
                if not series:
                    continue
                color, line, marker = styles[batch, name]
                ax.plot(
                    [r["epoch"] for r in series],
                    [validation(r) for r in series],
                    color=color,
                    linestyle=line,
                    marker=marker,
                    markerfacecolor="white" if name == "fixed" else color,
                    markeredgewidth=0.8,
                    markersize=3.4,
                    linewidth=1.25,
                    label=labels[batch, name],
                    zorder=3 if name == "fixed" else 2,
                )
        ax.set_xscale("log", base=2)
        max_epoch = max(
            [
                r["epoch"]
                for (batch, _), r in all_selected[key].items()
                if batch in (64, 512) and r["epoch"] >= 1
            ],
            default=32,
        )
        ticks = [tick for tick in (1, 2, 4, 8, 16, 32, 64) if tick <= max_epoch]
        ax.set_xticks(ticks, [str(tick) for tick in ticks])
        ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
        ax.set_xlabel("Epoch", fontsize=7.5)
        style_axes(ax)
    axes[0].set_ylabel("Validation CE", fontsize=8)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=4,
        frameon=False,
        fontsize=7,
        handlelength=2.1,
        columnspacing=1.0,
    )
    fig.subplots_adjust(top=0.79, wspace=0.34)
    save(fig, "section3_weight_decay_transfer.png")


def plot_train_validation_by_batch(selected: dict, history_payload: dict) -> None:
    """Plot endpoint training and validation CE with a genuinely shared y-axis."""

    histories = history_payload["histories"]
    fig, axes = plt.subplots(1, 3, figsize=(3.35, 1.38), sharey=True)
    for panel, (ax, batch) in enumerate(zip(axes, TRAIN_VALIDATION_BATCHES)):
        train_values, validation_values = [], []
        counts = []
        for epoch in TRAIN_VALIDATION_EPOCHS:
            run = selected.get((batch, float(epoch)))
            if run is not None and run.get("wandb") in histories:
                train_ce, count = recent_logged_training_ce(
                    histories[run["wandb"]],
                    endpoint_tokens=epoch * 1_000_000_000,
                    observations=TRAIN_AVERAGE_LOGGED_STEPS,
                )
            else:
                train_ce, count = math.nan, 0
            train_values.append(train_ce)
            counts.append(count)
            validation_values.append(validation(run) if run is not None else np.nan)

        train_array = np.asarray(train_values, dtype=float)
        validation_array = np.asarray(validation_values, dtype=float)
        ax.plot(
            TRAIN_VALIDATION_EPOCHS,
            train_array,
            color="#2457A6",
            linestyle="--",
            marker="s",
            markersize=1.45,
            linewidth=0.9,
            label="Training CE",
        )
        ax.plot(
            TRAIN_VALIDATION_EPOCHS,
            validation_array,
            color="#C65D3A",
            linestyle="-",
            marker="s",
            markersize=1.45,
            linewidth=0.9,
            label="Validation CE",
        )
        ax.set_xticks([1, 4, 8, 12])
        ax.set_xlim(0.6, 12.4)
        ax.set_ylim(2.6, 4.3)
        ax.set_yticks([2.75, 3.25, 3.75, 4.25])
        ax.set_title(
            rf"({chr(ord('a') + panel)}) $B={batch}$",
            fontsize=5.7,
            fontweight="bold",
            pad=1.8,
        )
        ax.set_xlabel("Epoch", fontsize=5.1)
        style_axes(ax)
        ax.tick_params(labelsize=4.7, length=1.8)
        print(
            f"B={batch} final-{TRAIN_AVERAGE_LOGGED_STEPS}-logged-step train CE: "
            + ", ".join(
                f"E{epoch}={ce:.4f} (n={count})"
                for epoch, ce, count in zip(
                    TRAIN_VALIDATION_EPOCHS, train_values, counts
                )
            )
        )

    axes[0].set_ylabel("CE loss", fontsize=5.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
        fontsize=4.8,
        handlelength=1.5,
        columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.10, right=0.998, bottom=0.235, top=0.735, wspace=0.06)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUTPUT_DIR / f"section3_train_validation_endpoint.{suffix}",
            dpi=320,
            bbox_inches="tight",
            pad_inches=0.015,
            facecolor="white",
        )
    plt.close(fig)


def plot_unique_equivalence(repeated_best: float, repeated_untuned: float = 3.138) -> None:
    tuned = json.loads(
        (DATA_DIR / "wsd_unique_vs_repeated_batch_tuned_1b.json").read_text()
    )
    baseline = json.loads((DATA_DIR / "wsd_unique_vs_repeated_1b.json").read_text())
    unhealthy = {
        **baseline.get("healthAudit", {}).get("unhealthy", {}),
        **tuned.get("healthAudit", {}).get("unhealthy", {}),
    }

    def eligible(run: dict) -> bool:
        return (
            run.get("status") == "complete"
            and run.get("lr") in {"5e-4", "1e-3"}
            and validation(run) is not None
            and run.get("wandb") not in unhealthy
        )

    runs = [
        {**run, "batchSequences": 1024}
        for run in baseline.get("uniqueRuns", [])
        if eligible(run)
    ]
    runs.extend(run for run in tuned.get("uniqueRuns", []) if eligible(run))

    fig, ax = plt.subplots(figsize=(3.38, 2.72))
    colors = {64: "#2457A6", 256: "#7656A5", 1024: "#C65D3A"}
    for batch in (64, 256, 1024):
        epochs = sorted({float(r["epoch"]) for r in runs if r["batchSequences"] == batch})
        xs, ys = [], []
        for epoch in epochs:
            candidates = [
                r
                for r in runs
                if r["batchSequences"] == batch and float(r["epoch"]) == epoch
            ]
            best = min(candidates, key=validation)
            xs.append(epoch)
            ys.append(validation(best))
        ax.plot(
            xs,
            ys,
            color=colors[batch],
            marker="o",
            markersize=3.3,
            linewidth=1.35,
            label=(
                rf"Unique · $B={batch}$ (best for data efficiency)"
                if batch == 64
                else rf"Unique · $B={batch}$ (best for throughput)"
                if batch == 1024
                else rf"Unique · $B={batch}$"
            ),
        )
    ax.axhline(
        repeated_best,
        color="#222222",
        linestyle="-",
        linewidth=1.25,
        label="Repeated 1B · best w/ tuning $B$",
    )
    ax.axhline(
        repeated_untuned,
        color="#666666",
        linestyle="--",
        linewidth=1.15,
        label="Repeated 1B · best w/o tuning $B$",
    )
    ax.annotate(
        r"$\sim$8B at $B=64$",
        xy=(8, repeated_best),
        xytext=(4.7, repeated_best + 0.36),
        fontsize=6.8,
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "#555555"},
    )
    ax.annotate(
        r"$\sim$16B at $B=1024$",
        xy=(16, repeated_best),
        xytext=(9.2, repeated_best + 0.55),
        fontsize=6.8,
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "#555555"},
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    ax.set_xlim(0.85, 18.5)
    ax.set_ylim(2.80, 5.12)
    ax.set_xlabel("Unique training tokens (billions)", fontsize=8)
    ax.set_ylabel("Validation CE", fontsize=8)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=5.7, loc="upper right", handlelength=1.8)
    save(fig, "section3_unique_equivalence.png")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-train-history",
        action="store_true",
        help="Refresh the read-only Beaker-log cache used for late-training CE.",
    )
    args = parser.parse_args()

    reports = {name: load_js_data(name) for name in ("1b", "474m", "153m")}
    selected, runs = {}, {}
    for name, report in reports.items():
        selected[name], runs[name] = selected_runs(report)

    if args.refresh_train_history:
        history_payload = refresh_training_history_cache(selected["1b"])
    else:
        history_payload = load_training_history_cache()

    plot_batch_optima(selected)
    plot_weight_decay(selected, runs)
    plot_train_validation_by_batch(selected["1b"], history_payload)
    repeated_best = min(
        validation(run)
        for (batch, _), run in selected["1b"].items()
        if batch == 64
    )
    plot_unique_equivalence(repeated_best)

    for name in ("1b", "474m", "153m"):
        print(name)
        for batch in BATCHES:
            candidates = [r for (b, _), r in selected[name].items() if b == batch]
            if not candidates:
                print(f"  BS{batch}: missing")
                continue
            best = min(candidates, key=validation)
            print(
                f"  BS{batch}: CE={validation(best):.3f}, epoch={best['epoch']:g}, "
                f"train={best.get('train')}, LR={best['lr']}, WD={best['wd']}"
            )


if __name__ == "__main__":
    main()
