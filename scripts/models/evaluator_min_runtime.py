#!/usr/bin/env python3
"""Estimate a buffered Beaker ``minRuntime`` for decay-and-eval jobs.

The throughput assumptions are conservative round numbers derived from the
one-node/eight-GPU DR+WT+EmbedWD producers and evaluators launched on
2026-08-30.  The estimate covers every requested 10% WSD decay, one 100M-token
heldout pass per source checkpoint, repeated process startup/checkpoint loading,
and a 20% contingency buffer.  Reservations are rounded up to 30-minute blocks.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from typing import Any

SEQUENCE_LENGTH = 4096
DECAY_FRACTION = 0.1
HELDOUT_TOKENS = 100_003_840
STARTUP_SECONDS_PER_EPOCH = 180
BUFFER_FRACTION = 0.20
ROUND_SECONDS = 30 * 60

# Conservative sustained token rates from the live 2026-08-30 jobs.  These are
# intentionally below the typical measured rates before the explicit buffer is
# applied (roughly 449k, 1.1M, and 2.8M tokens/s, respectively).
MODEL_TOKENS_PER_SECOND = {
    "1b": 440_000,
    "474m": 1_000_000,
    "153m": 2_600_000,
}


def format_duration(seconds: int) -> str:
    if seconds <= 0 or seconds % 60:
        raise ValueError("duration must be a positive whole number of minutes")
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    return f"{seconds // 60}m"


def estimate_min_runtime(
    *,
    model: str,
    pool_tokens: int,
    batch_sequences: int,
    epochs: Sequence[int],
) -> dict[str, Any]:
    normalized_model = model.lower()
    if normalized_model not in MODEL_TOKENS_PER_SECOND:
        raise ValueError(f"unsupported evaluator model {model}")
    if pool_tokens <= 0 or batch_sequences <= 0:
        raise ValueError("pool_tokens and batch_sequences must be positive")
    resolved_epochs = [int(epoch) for epoch in epochs]
    if not resolved_epochs or any(epoch <= 0 for epoch in resolved_epochs):
        raise ValueError("epochs must contain positive integers")
    if len(set(resolved_epochs)) != len(resolved_epochs):
        raise ValueError("epochs must be unique")

    decay_steps = 0
    for epoch in resolved_epochs:
        endpoint = math.ceil(
            epoch * pool_tokens / (batch_sequences * SEQUENCE_LENGTH)
        )
        stable = endpoint - round(DECAY_FRACTION * endpoint) - 1
        decay_steps += endpoint - stable

    decay_tokens = decay_steps * batch_sequences * SEQUENCE_LENGTH
    evaluation_tokens = HELDOUT_TOKENS * len(resolved_epochs)
    tokens_per_second = MODEL_TOKENS_PER_SECOND[normalized_model]
    estimated_seconds = math.ceil(
        (decay_tokens + evaluation_tokens) / tokens_per_second
        + STARTUP_SECONDS_PER_EPOCH * len(resolved_epochs)
    )
    buffered_seconds = math.ceil(estimated_seconds * (1 + BUFFER_FRACTION))
    reserved_seconds = math.ceil(buffered_seconds / ROUND_SECONDS) * ROUND_SECONDS

    return {
        "minRuntime": format_duration(reserved_seconds),
        "estimatedSeconds": estimated_seconds,
        "bufferedSeconds": buffered_seconds,
        "reservedSeconds": reserved_seconds,
        "decaySteps": decay_steps,
        "decayTokens": decay_tokens,
        "evaluationTokens": evaluation_tokens,
        "tokensPerSecond": tokens_per_second,
        "bufferFraction": BUFFER_FRACTION,
        "roundSeconds": ROUND_SECONDS,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_TOKENS_PER_SECOND))
    parser.add_argument("--pool-tokens", type=int, required=True)
    parser.add_argument("--batch-sequences", type=int, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            estimate_min_runtime(
                model=args.model,
                pool_tokens=args.pool_tokens,
                batch_sequences=args.batch_sequences,
                epochs=args.epochs,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
