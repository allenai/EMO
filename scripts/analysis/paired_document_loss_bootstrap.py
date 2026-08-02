#!/usr/bin/env python3
"""Paired cluster bootstrap for document-level language-model loss statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_stats(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def paired_bootstrap(
    a: dict[str, np.ndarray],
    b: dict[str, np.ndarray],
    *,
    samples: int,
    seed: int,
    chunk_size: int = 25,
) -> dict[str, object]:
    for field in ("document_starts", "document_ends", "token_counts"):
        if not np.array_equal(a[field], b[field]):
            raise ValueError(f"paired inputs disagree on '{field}'")

    counts = a["token_counts"].astype(np.int64, copy=False)
    included = counts > 0
    if not np.any(included):
        raise ValueError("paired inputs contain no scored documents")
    counts = counts[included]
    a_sums = a["loss_sums"][included].astype(np.float64, copy=False)
    b_sums = b["loss_sums"][included].astype(np.float64, copy=False)
    delta_sums = a_sums - b_sums

    point = float(delta_sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    probabilities = np.full(len(counts), 1.0 / len(counts), dtype=np.float64)
    draws = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        weights = rng.multinomial(len(counts), probabilities, size=stop - start)
        draws[start:stop] = (weights @ delta_sums) / (weights @ counts)

    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "difference_a_minus_b": point,
        "confidence_level": 0.95,
        "confidence_interval": [float(low), float(high)],
        "statistically_significant": bool(low > 0 or high < 0),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "documents_in_metadata": int(len(included)),
        "documents_included": int(included.sum()),
        "scored_tokens": int(counts.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--a-label", required=True)
    parser.add_argument("--b-label", required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = {
        "a": {"label": args.a_label, "path": str(args.a)},
        "b": {"label": args.b_label, "path": str(args.b)},
        "method": "paired document-cluster nonparametric bootstrap; documents resampled with replacement and token losses re-aggregated within each draw",
        **paired_bootstrap(
            load_stats(args.a),
            load_stats(args.b),
            samples=args.samples,
            seed=args.seed,
        ),
    }
    payload = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)


if __name__ == "__main__":
    main()
