"""
One-off script to analyze the OLMoE-mix-0824 data composition.

For each source in the mix:
  1. Queries S3 for .npy file sizes to estimate total token counts
  2. Downloads the first ~2MB of one file, decodes a few documents,
     and prints them for human sanity-checking

Outputs a composition JSON showing each source's token fraction,
which can be passed to extract_router_embeddings.py for proportional sampling.

Usage:
    conda run -n flexmoe python -m src.scripts.analysis.analyze_data_mix \
        --mix-file src/olmo_core/data/mixes/OLMoE-mix-0824.txt \
        --output-dir claude_outputs/analysis/router_clustering \
        --num-preview-docs 2
"""

import argparse
import json
import logging
import os
import subprocess
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

EOS_TOKEN_ID = 100257
S3_BASE = "s3://ai2-llm/"
TOKENIZER_ID = "allenai/dolma2-tokenizer"
# Approx bytes per token for uint16-stored npy arrays (2 bytes/token + tiny header overhead)
BYTES_PER_TOKEN = 2


# ---------------------------------------------------------------------------
# Mix parsing
# ---------------------------------------------------------------------------


def parse_mix_file(mix_path: str) -> List[Tuple[str, str]]:
    """Returns list of (source_label, s3_path) for all lines in the mix file."""
    pairs = []
    with open(mix_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            label, path = line.split(",", 1)
            path = path.replace("{TOKENIZER}", TOKENIZER_ID)
            pairs.append((label.strip(), S3_BASE + path.strip()))
    return pairs


# ---------------------------------------------------------------------------
# S3 file size queries
# ---------------------------------------------------------------------------


def get_s3_file_sizes(s3_paths: List[str]) -> Dict[str, int]:
    """
    Query S3 for file sizes. Groups paths by their directory prefix and runs
    one `aws s3 ls` per directory for efficiency.
    Returns {s3_path: size_bytes}.
    """
    # Group by directory prefix
    prefix_to_files: Dict[str, List[str]] = defaultdict(list)
    for path in s3_paths:
        prefix = path.rsplit("/", 1)[0] + "/"
        prefix_to_files[prefix].append(path)

    sizes: Dict[str, int] = {}
    for prefix, files in prefix_to_files.items():
        logger.info(f"  Listing {prefix} ...")
        result = subprocess.run(["aws", "s3", "ls", prefix], capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"  Could not list {prefix}: {result.stderr.strip()}")
            continue
        # Parse `aws s3 ls` output: "YYYY-MM-DD HH:MM:SS   <size>   <filename>"
        file_sizes: Dict[str, int] = {}
        for line in result.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) == 4:
                fname, size_str = parts[3], parts[2]
                try:
                    file_sizes[fname] = int(size_str)
                except ValueError:
                    pass
        for s3_path in files:
            fname = s3_path.rsplit("/", 1)[-1]
            if fname in file_sizes:
                sizes[s3_path] = file_sizes[fname]
            else:
                logger.warning(f"  File not found in listing: {s3_path}")
    return sizes


# ---------------------------------------------------------------------------
# Streaming a small chunk from S3 for text decoding
# ---------------------------------------------------------------------------


def stream_first_bytes(s3_path: str, num_bytes: int) -> bytes:
    """Download just the first `num_bytes` of an S3 file using byte-range GET."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".npy") as tmp:
        result = subprocess.run(
            [
                "aws",
                "s3api",
                "get-object",
                "--bucket",
                "ai2-llm",
                "--key",
                s3_path.replace("s3://ai2-llm/", ""),
                "--range",
                f"bytes=0-{num_bytes - 1}",
                tmp.name,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to stream {s3_path}: {result.stderr[:200]}")
        with open(tmp.name, "rb") as f:
            return f.read()


def load_tokens_from_bytes(raw: bytes) -> np.ndarray:
    """
    Load tokens from a raw binary chunk.
    These files are headerless raw uint32 memmap arrays (NOT standard .npy format).
    Each token is 4 bytes, little-endian uint32.
    """
    item_size = 4  # uint32
    n_items = len(raw) // item_size
    tokens = np.frombuffer(raw[: n_items * item_size], dtype=np.uint32)
    return tokens.astype(np.int32)


def iter_documents_from_tokens(tokens: np.ndarray, min_len: int = 32, max_len: int = 4096):
    eos_positions = np.where(tokens == EOS_TOKEN_ID)[0]
    start = 0
    for pos in eos_positions:
        doc = tokens[start : pos + 1]
        if min_len <= len(doc) <= max_len:
            yield doc
        start = pos + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mix-file", default="src/olmo_core/data/mixes/OLMoE-mix-0824.txt")
    parser.add_argument("--output-dir", default="claude_outputs/analysis/router_clustering")
    parser.add_argument(
        "--num-preview-docs",
        type=int,
        default=2,
        help="Number of documents to decode per source for sanity checking",
    )
    parser.add_argument(
        "--stream-bytes",
        type=int,
        default=2_000_000,
        help="Bytes to stream from the first file of each source for decoding",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer for decoding
    from transformers import AutoTokenizer

    logger.info("Loading tokenizer (allenai/dolma2-tokenizer) ...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")

    # Parse mix
    all_pairs = parse_mix_file(args.mix_file)
    logger.info(f"Mix file has {len(all_pairs)} total entries")

    # Group by source label
    source_to_paths: Dict[str, List[str]] = defaultdict(list)
    for label, path in all_pairs:
        source_to_paths[label].append(path)

    num_sources = len(source_to_paths)
    logger.info(f"Found {num_sources} unique sources")

    # -----------------------------------------------------------------------
    # Step 1: Get file sizes for all files
    # -----------------------------------------------------------------------
    logger.info("\n--- Step 1: Querying S3 file sizes ---")
    all_paths = [p for paths in source_to_paths.values() for p in paths]
    sizes = get_s3_file_sizes(all_paths)

    # Compute per-source total bytes and estimated tokens
    source_stats: Dict[str, Dict] = {}
    for label, paths in source_to_paths.items():
        total_bytes = sum(sizes.get(p, 0) for p in paths)
        num_files = len(paths)
        files_with_sizes = sum(1 for p in paths if p in sizes)
        est_tokens = total_bytes // BYTES_PER_TOKEN
        source_stats[label] = {
            "num_files": num_files,
            "files_with_sizes": files_with_sizes,
            "total_bytes": total_bytes,
            "est_tokens": est_tokens,
        }

    grand_total_tokens = sum(s["est_tokens"] for s in source_stats.values())
    for label, stats in source_stats.items():
        stats["fraction"] = (
            stats["est_tokens"] / grand_total_tokens if grand_total_tokens > 0 else 0.0
        )
        stats["target_tokens_500M"] = int(stats["fraction"] * 500_000_000)

    # Print composition table
    print("\n" + "=" * 90)
    print(f"{'SOURCE':<45} {'FILES':>6} {'EST_TOKENS':>14} {'FRACTION':>9} {'500M_ALLOC':>12}")
    print("=" * 90)
    sorted_sources = sorted(source_stats.items(), key=lambda x: -x[1]["est_tokens"])
    for label, stats in sorted_sources:
        print(
            f"{label:<45} {stats['num_files']:>6} {stats['est_tokens']:>14,} "
            f"{stats['fraction']:>8.2%} {stats['target_tokens_500M']:>12,}"
        )
    print("=" * 90)
    print(
        f"{'TOTAL':<45} {len(all_paths):>6} {grand_total_tokens:>14,} {'100.00%':>9} {'500,000,000':>12}"
    )

    # -----------------------------------------------------------------------
    # Step 2: Decode a text sample from the first file of each source
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("--- Step 2: Text samples from each source (sanity check) ---")
    print("=" * 90)

    text_samples = {}
    for label, paths in source_to_paths.items():
        first_path = paths[0]
        print(f"\n[SOURCE: {label}]  ({first_path})")
        try:
            raw = stream_first_bytes(first_path, args.stream_bytes)
            tokens = load_tokens_from_bytes(raw)
            print(f"  Streamed {len(raw):,} bytes → {len(tokens):,} tokens")

            docs = list(iter_documents_from_tokens(tokens))
            print(f"  Found {len(docs)} complete documents in this chunk")

            if not docs:
                print("  WARNING: no complete documents found in streamed chunk")
                text_samples[label] = []
                continue

            source_texts = []
            for i, doc in enumerate(docs[: args.num_preview_docs]):
                decoded = tokenizer.decode(doc[:500].tolist(), skip_special_tokens=True)
                print(f"\n  --- Document {i+1} (len={len(doc)} tokens) ---")
                print(f"  {decoded[:600]}")
                source_texts.append(decoded[:600])
            text_samples[label] = source_texts

        except Exception as e:
            logger.warning(f"  Failed to sample {label}: {e}")
            text_samples[label] = []

    # -----------------------------------------------------------------------
    # Save composition JSON
    # -----------------------------------------------------------------------
    composition = {
        "grand_total_est_tokens": grand_total_tokens,
        "sources": {
            label: {
                "num_files": stats["num_files"],
                "est_tokens": stats["est_tokens"],
                "fraction": round(stats["fraction"], 6),
                "target_tokens_500M": stats["target_tokens_500M"],
                "first_file": source_to_paths[label][0],
                "all_files": source_to_paths[label],
            }
            for label, stats in source_stats.items()
        },
    }
    out_path = os.path.join(args.output_dir, "mix_composition.json")
    with open(out_path, "w") as f:
        json.dump(composition, f, indent=2)
    logger.info(f"\nSaved composition to {out_path}")

    # Also save a short text report
    report_path = os.path.join(args.output_dir, "mix_composition_report.txt")
    with open(report_path, "w") as f:
        f.write(f"OLMoE-mix-0824 composition  (grand total est. {grand_total_tokens:,} tokens)\n\n")
        f.write(
            f"{'SOURCE':<45} {'FILES':>6} {'EST_TOKENS':>14} {'FRACTION':>9} {'500M_ALLOC':>12}\n"
        )
        f.write("-" * 90 + "\n")
        for label, stats in sorted_sources:
            f.write(
                f"{label:<45} {stats['num_files']:>6} {stats['est_tokens']:>14,} "
                f"{stats['fraction']:>8.2%} {stats['target_tokens_500M']:>12,}\n"
            )
        f.write("\n\n--- TEXT SAMPLES ---\n")
        for label, texts in text_samples.items():
            f.write(f"\n[{label}]\n")
            for t in texts:
                f.write(f"  {t[:300]}\n")
    logger.info(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
