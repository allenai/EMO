"""
Generate pretraining data composition file.

Parses the mix file (source label + S3 path per line), queries S3 for file
sizes to estimate per-source token fractions, and optionally decodes sample
documents for sanity checking.

Output: JSON with per-source fractions and S3 file lists, used by extract.py
for proportional sampling.

Usage:
    python -m src.scripts.clustering.generate_mix \\
        --mix-file src/olmo_core/data/mixes/OLMoE-mix-0824.txt \\
        --output claude_outputs/clustering/pretraining_mix.json
"""

import argparse
import json
import logging
import os
import subprocess
import tempfile
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

EOS_TOKEN_ID = 100257
S3_BASE = "s3://ai2-llm/"
TOKENIZER_ID = "allenai/dolma2-tokenizer"
BYTES_PER_TOKEN = 2  # uint16-stored npy arrays


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


def get_s3_file_sizes(s3_paths: List[str]) -> Dict[str, int]:
    """Query S3 for file sizes, batching by directory prefix."""
    prefix_to_files: Dict[str, List[str]] = defaultdict(list)
    for path in s3_paths:
        prefix = path.rsplit("/", 1)[0] + "/"
        prefix_to_files[prefix].append(path)

    sizes: Dict[str, int] = {}
    for prefix, files in prefix_to_files.items():
        logger.info(f"  Listing {prefix} ...")
        result = subprocess.run(
            ["aws", "s3", "ls", prefix],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.warning(f"  Could not list {prefix}: {result.stderr.strip()}")
            continue
        file_sizes: Dict[str, int] = {}
        for line in result.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) == 4:
                try:
                    file_sizes[parts[3]] = int(parts[2])
                except ValueError:
                    pass
        for s3_path in files:
            fname = s3_path.rsplit("/", 1)[-1]
            if fname in file_sizes:
                sizes[s3_path] = file_sizes[fname]
    return sizes


def stream_first_bytes(s3_path: str, num_bytes: int) -> bytes:
    """Download the first `num_bytes` of an S3 file using byte-range GET."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            [
                "aws", "s3api", "get-object",
                "--bucket", "ai2-llm",
                "--key", s3_path.replace("s3://ai2-llm/", ""),
                "--range", f"bytes=0-{num_bytes - 1}",
                tmp_path,
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to stream {s3_path}: {result.stderr[:200]}")
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate pretraining data composition file"
    )
    parser.add_argument("--mix-file", required=True,
                        help="Path to mix file (e.g. OLMoE-mix-0824.txt)")
    parser.add_argument("--output", required=True,
                        help="Output JSON path (e.g. pretraining_mix.json)")
    parser.add_argument("--num-preview-docs", type=int, default=2,
                        help="Documents to decode per source for sanity checking")
    parser.add_argument("--stream-bytes", type=int, default=2_000_000,
                        help="Bytes to stream from each source for decoding")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Parse mix
    all_pairs = parse_mix_file(args.mix_file)
    logger.info(f"Mix file has {len(all_pairs)} entries")

    source_to_paths: Dict[str, List[str]] = defaultdict(list)
    for label, path in all_pairs:
        source_to_paths[label].append(path)
    logger.info(f"Found {len(source_to_paths)} unique sources")

    # Get file sizes
    logger.info("\nQuerying S3 file sizes...")
    all_paths = [p for paths in source_to_paths.values() for p in paths]
    sizes = get_s3_file_sizes(all_paths)

    # Compute per-source stats
    source_stats = {}
    for label, paths in source_to_paths.items():
        total_bytes = sum(sizes.get(p, 0) for p in paths)
        source_stats[label] = {
            "num_files": len(paths),
            "total_bytes": total_bytes,
            "est_tokens": total_bytes // BYTES_PER_TOKEN,
        }

    grand_total = sum(s["est_tokens"] for s in source_stats.values())
    for stats in source_stats.values():
        stats["fraction"] = stats["est_tokens"] / grand_total if grand_total > 0 else 0.0

    # Print composition
    logger.info(f"\n{'SOURCE':<45} {'FILES':>6} {'EST_TOKENS':>14} {'FRACTION':>9}")
    logger.info("=" * 80)
    for label, stats in sorted(source_stats.items(), key=lambda x: -x[1]["est_tokens"]):
        logger.info(f"{label:<45} {stats['num_files']:>6} "
                     f"{stats['est_tokens']:>14,} {stats['fraction']:>8.2%}")
    logger.info(f"{'TOTAL':<45} {len(all_paths):>6} {grand_total:>14,}")

    # Preview docs
    if args.num_preview_docs > 0:
        from transformers import AutoTokenizer
        logger.info("\nLoading tokenizer for previews...")
        tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")

        for label, paths in source_to_paths.items():
            try:
                raw = stream_first_bytes(paths[0], args.stream_bytes)
                tokens = np.frombuffer(
                    raw[:len(raw) // 4 * 4], dtype=np.uint32
                ).astype(np.int32)
                eos_pos = np.where(tokens == EOS_TOKEN_ID)[0]
                start = 0
                count = 0
                for pos in eos_pos:
                    doc = tokens[start:pos + 1]
                    if 32 <= len(doc) <= 4096:
                        decoded = tokenizer.decode(doc[:500].tolist(), skip_special_tokens=True)
                        logger.info(f"  [{label}] doc {count+1}: {decoded[:200]!r}")
                        count += 1
                        if count >= args.num_preview_docs:
                            break
                    start = pos + 1
            except Exception as e:
                logger.warning(f"  Failed to preview {label}: {e}")

    # Save
    composition = {
        "grand_total_est_tokens": grand_total,
        "sources": {
            label: {
                "num_files": stats["num_files"],
                "est_tokens": stats["est_tokens"],
                "fraction": round(stats["fraction"], 6),
                "all_files": source_to_paths[label],
            }
            for label, stats in source_stats.items()
        },
    }
    with open(args.output, "w") as f:
        json.dump(composition, f, indent=2)
    logger.info(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
