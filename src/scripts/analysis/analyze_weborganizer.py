"""
Analyze the cc_all_dressed topic dataset to produce a mix_composition.json
suitable for extract_router_embeddings.py with uniform sampling across topics.

Structure on S3:
  s3://ai2-llm/preprocessed/cc_all_dressed/all_dressed_v3/dclm_plus2_vigilantes/
    allenai/dolma2-tokenizer/
      <topic>/
        vigintile_0001/ ... vigintile_0020/
          *.npy  (headerless raw uint32 binary, BYTES_PER_TOKEN=4)

We use the highest-numbered vigintile per topic (vigintile_0020).
Uniform mixing: each topic gets fraction = 1/num_topics regardless of its
actual data volume (unlike the pretraining analysis which used proportional
mixing).

Usage:
    python -m src.scripts.analysis.analyze_weborganizer \
        --output-dir claude_outputs/analysis/router_clustering_weborganizer \
        --num-preview-docs 2
"""

import argparse
import json
import logging
import os
import subprocess
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

S3_BASE = "s3://ai2-llm"
ALL_DRESSED_PREFIX = "preprocessed/cc_all_dressed/all_dressed_v3/dclm_plus2_vigilantes/allenai/dolma2-tokenizer"
BYTES_PER_TOKEN = 4  # headerless raw uint32 binary
EOS_TOKEN_ID = 100257


def s3_ls(prefix: str) -> List[str]:
    """List immediate children of an S3 prefix (directories and files)."""
    result = subprocess.run(
        ["aws", "s3", "ls", f"s3://ai2-llm/{prefix}/"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"aws s3 ls failed for {prefix}: {result.stderr[:200]}")
    entries = []
    for line in result.stdout.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        # Directories end with "/"
        if line.strip().startswith("PRE"):
            entries.append(parts[1].rstrip("/"))
        elif len(parts) == 4:
            entries.append(parts[3])
    return entries


def list_topics() -> List[str]:
    """List all topic directories under the all-dressed base prefix."""
    entries = s3_ls(ALL_DRESSED_PREFIX)
    return sorted(entries)


def list_vigintiles(topic: str) -> List[str]:
    """List vigintile subdirectories for a topic, sorted."""
    prefix = f"{ALL_DRESSED_PREFIX}/{topic}"
    entries = s3_ls(prefix)
    # Keep only vigintile_* directories
    vigs = sorted(e for e in entries if e.startswith("vigintile_"))
    return vigs


def list_npy_files(topic: str, vigintile: str) -> List[Tuple[str, int]]:
    """
    List .npy files in a topic/vigintile directory.
    Returns list of (s3_path, size_bytes).
    """
    prefix = f"{ALL_DRESSED_PREFIX}/{topic}/{vigintile}"
    result = subprocess.run(
        ["aws", "s3", "ls", f"s3://ai2-llm/{prefix}/"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.warning(f"Could not list {prefix}: {result.stderr[:100]}")
        return []

    files = []
    for line in result.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) == 4:
            fname, size_str = parts[3], parts[2]
            if fname.endswith(".npy"):
                try:
                    s3_path = f"s3://ai2-llm/{prefix}/{fname}"
                    files.append((s3_path, int(size_str)))
                except ValueError:
                    pass
    return sorted(files, key=lambda x: x[0])


def stream_bytes_from_s3(s3_path: str, num_bytes: int) -> bytes:
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            ["aws", "s3api", "get-object",
             "--bucket", "ai2-llm",
             "--key", s3_path.replace("s3://ai2-llm/", ""),
             "--range", f"bytes=0-{num_bytes - 1}",
             tmp_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"S3 range-GET failed: {result.stderr[:200]}")
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def decode_sample_docs(s3_path: str, tokenizer, num_docs: int = 2,
                       stream_bytes: int = 2_000_000) -> List[str]:
    """Stream a small chunk from S3 and decode a few documents."""
    try:
        raw = stream_bytes_from_s3(s3_path, stream_bytes)
    except Exception as e:
        logger.warning(f"  Could not stream {s3_path}: {e}")
        return []

    n = len(raw) // BYTES_PER_TOKEN
    tokens = np.frombuffer(raw[:n * BYTES_PER_TOKEN], dtype=np.uint32).astype(np.int32)

    eos_pos = np.where(tokens == EOS_TOKEN_ID)[0]
    docs = []
    start = 0
    for pos in eos_pos:
        doc = tokens[start:pos + 1]
        if 32 <= len(doc) <= 4096:
            docs.append(doc)
        start = pos + 1
        if len(docs) >= num_docs:
            break

    texts = []
    for doc in docs:
        text = tokenizer.decode(doc[:500].tolist(), skip_special_tokens=True)
        texts.append(text[:600])
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",
                        default="claude_outputs/analysis/router_clustering_weborganizer")
    parser.add_argument("--num-preview-docs", type=int, default=2,
                        help="Documents to decode per topic for sanity checking (0 to skip)")
    parser.add_argument("--stream-bytes", type=int, default=2_000_000,
                        help="Bytes to stream per topic for text previews")
    parser.add_argument("--model-path", default=None,
                        help="HF model path for tokenizer (only needed for --num-preview-docs > 0). "
                             "Defaults to the twolevelbatchlbreducedp model.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Enumerate topics ──────────────────────────────────────────────
    logger.info("Listing topics under all-dressed prefix ...")
    topics = list_topics()
    logger.info(f"Found {len(topics)} topics: {topics}")

    # ── Step 2: For each topic, find highest vigintile and list .npy files ────
    logger.info("\nScanning vigintile_0020 for each topic ...")
    topic_files: Dict[str, List[Tuple[str, int]]] = {}
    topic_max_vigintile: Dict[str, str] = {}

    for topic in topics:
        logger.info(f"  [{topic}] listing vigintiles ...")
        vigs = list_vigintiles(topic)
        if not vigs:
            logger.warning(f"  [{topic}] no vigintiles found, skipping")
            continue
        max_vig = vigs[-1]
        topic_max_vigintile[topic] = max_vig
        logger.info(f"  [{topic}] using {max_vig}")

        files = list_npy_files(topic, max_vig)
        if not files:
            logger.warning(f"  [{topic}] no .npy files in {max_vig}, skipping")
            continue
        topic_files[topic] = files
        total_bytes = sum(sz for _, sz in files)
        logger.info(f"  [{topic}] {len(files)} files, {total_bytes / 1e9:.1f} GB, "
                    f"~{total_bytes // BYTES_PER_TOKEN / 1e9:.1f}B tokens")

    # ── Step 3: Compute stats and uniform fractions ───────────────────────────
    valid_topics = sorted(topic_files.keys())
    num_topics = len(valid_topics)
    uniform_fraction = 1.0 / num_topics

    topic_stats = {}
    for topic in valid_topics:
        files = topic_files[topic]
        total_bytes = sum(sz for _, sz in files)
        est_tokens = total_bytes // BYTES_PER_TOKEN
        topic_stats[topic] = {
            "num_files": len(files),
            "total_bytes": total_bytes,
            "est_tokens": est_tokens,
            "vigintile": topic_max_vigintile[topic],
        }

    grand_total_est_tokens = sum(s["est_tokens"] for s in topic_stats.values())
    proportional_fraction = {t: topic_stats[t]["est_tokens"] / grand_total_est_tokens
                              for t in valid_topics}

    # ── Step 4: Print distribution table ─────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"{'TOPIC':<40} {'VIGINTILE':<15} {'FILES':>6} {'EST_TOKENS':>14} "
          f"{'PROP_FRAC':>10} {'UNIFORM_FRAC':>13}")
    print("=" * 100)
    for topic in sorted(valid_topics, key=lambda t: -topic_stats[t]["est_tokens"]):
        s = topic_stats[topic]
        print(f"{topic:<40} {s['vigintile']:<15} {s['num_files']:>6} "
              f"{s['est_tokens']:>14,} {proportional_fraction[topic]:>9.2%} "
              f"{uniform_fraction:>12.2%}")
    print("=" * 100)
    print(f"{'TOTAL':<40} {'':<15} "
          f"{sum(s['num_files'] for s in topic_stats.values()):>6} "
          f"{grand_total_est_tokens:>14,} {'100.00%':>10} {'100.00%':>13}")
    print(f"\nNote: {num_topics} topics → uniform fraction = {uniform_fraction:.4f} ({uniform_fraction:.2%}) each")
    print(f"Grand total estimated tokens: {grand_total_est_tokens:,}")

    # ── Step 5: Optional text previews ───────────────────────────────────────
    text_samples = {}
    if args.num_preview_docs > 0:
        model_path = args.model_path or (
            "models/twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-1_0211/step30995-hf"
        )
        logger.info(f"\nLoading tokenizer from {model_path} ...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        print("\n" + "=" * 100)
        print("--- Text samples per topic (sanity check) ---")
        print("=" * 100)
        for topic in valid_topics:
            first_file_path = topic_files[topic][0][0]
            print(f"\n[TOPIC: {topic}]  ({first_file_path.split('/')[-1]})")
            texts = decode_sample_docs(first_file_path, tokenizer,
                                       args.num_preview_docs, args.stream_bytes)
            text_samples[topic] = texts
            for i, text in enumerate(texts):
                print(f"  --- Doc {i+1} ---")
                print(f"  {text[:400]}")
            if not texts:
                print("  (no complete documents in streamed chunk)")

    # ── Step 6: Save mix_composition.json ────────────────────────────────────
    composition = {
        "description": "cc_all_dressed vigintile_0020, uniform mixing across topics (weborganizer)",
        "num_topics": num_topics,
        "grand_total_est_tokens": grand_total_est_tokens,
        "sources": {
            topic: {
                "vigintile": topic_max_vigintile[topic],
                "num_files": topic_stats[topic]["num_files"],
                "est_tokens": topic_stats[topic]["est_tokens"],
                "total_bytes": topic_stats[topic]["total_bytes"],
                "proportional_fraction": round(proportional_fraction[topic], 6),
                "fraction": round(uniform_fraction, 6),  # uniform!
                "all_files": [path for path, _ in topic_files[topic]],
            }
            for topic in valid_topics
        },
    }

    comp_path = os.path.join(args.output_dir, "mix_composition.json")
    with open(comp_path, "w") as f:
        json.dump(composition, f, indent=2)
    logger.info(f"\nSaved mix_composition.json → {comp_path}")

    # Save text report
    report_path = os.path.join(args.output_dir, "mix_composition_report.txt")
    with open(report_path, "w") as f:
        f.write(f"cc_all_dressed (weborganizer) distribution (grand total est. {grand_total_est_tokens:,} tokens)\n")
        f.write(f"Uniform mixing: {num_topics} topics × {uniform_fraction:.4f} each\n\n")
        f.write(f"{'TOPIC':<40} {'VIGINTILE':<15} {'FILES':>6} {'EST_TOKENS':>14} "
                f"{'PROP_FRAC':>10} {'UNIFORM_FRAC':>13}\n")
        f.write("-" * 100 + "\n")
        for topic in sorted(valid_topics, key=lambda t: -topic_stats[t]["est_tokens"]):
            s = topic_stats[topic]
            f.write(f"{topic:<40} {s['vigintile']:<15} {s['num_files']:>6} "
                    f"{s['est_tokens']:>14,} {proportional_fraction[topic]:>9.2%} "
                    f"{uniform_fraction:>12.2%}\n")
        if text_samples:
            f.write("\n\n--- TEXT SAMPLES ---\n")
            for topic, texts in text_samples.items():
                f.write(f"\n[{topic}]\n")
                for text in texts:
                    f.write(f"  {text[:400]}\n")
    logger.info(f"Saved report → {report_path}")


if __name__ == "__main__":
    main()
