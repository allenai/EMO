"""Benchmark extraction batch sizes to find optimal throughput on multi-GPU."""

import logging
import time

import numpy as np
import torch

from src.scripts.clustering.extract import extract_logits
from src.scripts.clustering.utils import (
    get_moe_config,
    load_model_and_tokenizer,
)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "models/twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301/step30995-hf"

# Generate synthetic documents (250 tokens each, like the real extraction)
DOC_LEN = 250
NUM_WARMUP_BATCHES = 2
NUM_BENCH_BATCHES = 5
BATCH_SIZES = [256, 512, 1024, 2048]


def make_fake_docs(n, doc_len=DOC_LEN):
    """Generate n fake documents of fixed length."""
    rng = np.random.RandomState(42)
    return [rng.randint(100, 50000, size=doc_len, dtype=np.int32) for _ in range(n)]


def benchmark_batch_size(model, device, num_layers, num_standard_experts, batch_size):
    docs = make_fake_docs(batch_size)

    # Warmup
    for _ in range(NUM_WARMUP_BATCHES):
        extract_logits(model, docs, device, num_layers, num_standard_experts)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    total_tokens = 0
    for _ in range(NUM_BENCH_BATCHES):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits, token_info = extract_logits(model, docs, device, num_layers, num_standard_experts)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        total_tokens += len(token_info)

    avg_time = np.mean(times)
    std_time = np.std(times)
    tokens_per_sec = (batch_size * DOC_LEN) / avg_time

    return avg_time, std_time, tokens_per_sec


def main():
    logger.info(f"Loading model from {MODEL_PATH}...")
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)

    if hasattr(model, "hf_device_map"):
        first_device = next(iter(model.hf_device_map.values()))
        device = f"cuda:{first_device}" if isinstance(first_device, int) else str(first_device)
        logger.info(f"Device map: {model.hf_device_map}")
    else:
        device = str(next(model.parameters()).device)
    logger.info(f"Input device: {device}")

    moe_cfg = get_moe_config(model)
    num_layers = moe_cfg["num_layers"]
    num_standard_experts = moe_cfg["num_standard_experts"]

    # Check GPU memory after model load
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.memory_allocated(i) / 1e9
        logger.info(f"  GPU {i}: {mem:.1f} GB allocated")

    logger.info(
        f"\nBenchmarking with {DOC_LEN}-token docs, "
        f"{NUM_WARMUP_BATCHES} warmup + {NUM_BENCH_BATCHES} timed batches each\n"
    )

    results = []
    for bs in BATCH_SIZES:
        logger.info(f"--- batch_size={bs} ---")
        try:
            avg, std, tok_s = benchmark_batch_size(
                model, device, num_layers, num_standard_experts, bs
            )
            # Peak memory across all GPUs
            peak_mem = max(
                torch.cuda.max_memory_allocated(i) / 1e9 for i in range(torch.cuda.device_count())
            )
            results.append((bs, avg, std, tok_s, peak_mem))
            logger.info(
                f"  {avg:.3f}s ± {std:.3f}s per batch | "
                f"{tok_s:,.0f} tok/s | peak GPU mem: {peak_mem:.1f} GB"
            )
        except torch.cuda.OutOfMemoryError:
            logger.info(f"  OOM at batch_size={bs}")
            torch.cuda.empty_cache()
            break

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info(
        f"{'BS':>5}  {'Time/batch':>12}  {'Tok/s':>12}  {'Peak GPU MB':>12}  {'ETA 20M tok':>12}"
    )
    logger.info("-" * 80)
    for bs, avg, std, tok_s, peak in results:
        eta_min = 20_000_000 / tok_s / 60
        logger.info(
            f"{bs:>5}  {avg:>9.3f}s  {tok_s:>12,.0f}  {peak*1024:>10,.0f}  {eta_min:>9.1f} min"
        )
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
