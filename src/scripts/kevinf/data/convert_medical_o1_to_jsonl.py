"""
Convert FreedomIntelligence/medical-o1-reasoning-SFT to Dolma JSONL format.

Dataset has columns: Question, Complex_CoT, Response
Configs: en, zh, en_mix, zh_mix

Output subdirs are named <config>_cot or <config>_nocot so both variants
can coexist under the same base dir and be referenced independently in mixes.

Usage:
    # With CoT (default):
    python src/scripts/kevinf/data/convert_medical_o1_to_jsonl.py \
        --output-dir /data/output/medical-o1/jsonl \
        --configs en zh

    # Without CoT (ablation):
    python src/scripts/kevinf/data/convert_medical_o1_to_jsonl.py \
        --output-dir /data/output/medical-o1/jsonl \
        --configs en --no-cot

Output structure:
    <output-dir>/en_cot/chunk000000.jsonl.gz
    <output-dir>/en_nocot/chunk000000.jsonl.gz
    <output-dir>/zh_cot/chunk000000.jsonl.gz

Next step (tokenize):
    bash scripts/kevinf/data/tokenize_datasets.sh \
        /data/output/medical-o1/jsonl \
        /data/output/medical-o1/tokenized \
        64 en_cot en_nocot zh_cot
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

DATASET_NAME = "FreedomIntelligence/medical-o1-reasoning-SFT"
EOS_TOKEN = "<|endoftext|>"
ALL_CONFIGS = ["en", "zh", "en_mix", "zh_mix"]
DOCS_PER_CHUNK = 5000


def make_id(config: str, idx: int, question: str) -> str:
    key = f"medical_o1:{config}:{idx}:{question[:80]}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def format_document(question: str, cot: str, response: str, include_cot: bool = True) -> str:
    parts = [question]
    if include_cot and cot:
        parts.append(cot)
    parts.append(response)
    return "\n\n".join(parts) + EOS_TOKEN


def convert_config(subdir: str, output_dir: Path, include_cot: bool = True) -> int:
    """Download and convert one HF config to chunked JSONL."""
    # subdir is e.g. "en_cot"; HF config name is the part before the last "_"
    hf_config = subdir.rsplit("_", 1)[0]
    config_dir = output_dir / subdir
    config_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Loading {DATASET_NAME} [{hf_config}] ...")
    ds = load_dataset(DATASET_NAME, hf_config, split="train")
    total = len(ds)
    print(f"  {total:,} examples")

    chunk_idx = 0
    docs_written = 0
    current_chunk = []

    for i, example in enumerate(tqdm(ds, desc=subdir, total=total)):
        question = (example.get("Question") or "").strip()
        cot = (example.get("Complex_CoT") or "").strip()
        response = (example.get("Response") or "").strip()

        if not question and not response:
            continue

        text = format_document(question, cot, response, include_cot=include_cot)

        doc = {
            "id": make_id(hf_config, i, question),
            "text": text,
            "source": "medical_o1_reasoning_sft",
            "config": subdir,
        }

        current_chunk.append(json.dumps(doc).encode())
        docs_written += 1

        if len(current_chunk) >= DOCS_PER_CHUNK:
            chunk_file = config_dir / f"chunk{chunk_idx:06d}.jsonl.gz"
            with gzip.open(chunk_file, "wb", compresslevel=1) as f:
                f.write(b"\n".join(current_chunk) + b"\n")
            chunk_idx += 1
            current_chunk = []

    # Write remaining
    if current_chunk:
        chunk_file = config_dir / f"chunk{chunk_idx:06d}.jsonl.gz"
        with gzip.open(chunk_file, "wb", compresslevel=1) as f:
            f.write(b"\n".join(current_chunk) + b"\n")

    print(f"  Wrote {docs_written:,} docs in {chunk_idx + 1} chunks -> {config_dir}")
    return docs_written


def main():
    parser = argparse.ArgumentParser(description="Convert medical-o1-reasoning-SFT to Dolma JSONL")
    parser.add_argument("--output-dir", type=Path, default=Path("/data/output/medical-o1/jsonl"))
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["en"],
        choices=ALL_CONFIGS,
        help="HF configs to download (default: en)",
    )
    parser.add_argument(
        "--no-cot", action="store_true", help="Exclude Complex_CoT field from output text"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("medical-o1-reasoning-SFT → Dolma JSONL")
    print("=" * 60)
    print(f"Output: {args.output_dir}")
    print(f"Configs: {args.configs}")
    print("=" * 60)

    total_docs = 0
    include_cot = not args.no_cot
    suffix = "cot" if include_cot else "nocot"
    print(f"Include CoT: {include_cot}  (suffix: {suffix})")
    for config in args.configs:
        subdir_name = f"{config}_{suffix}"
        print(f"\n[{subdir_name}]")
        total_docs += convert_config(subdir_name, args.output_dir, include_cot=include_cot)


if __name__ == "__main__":
    main()
