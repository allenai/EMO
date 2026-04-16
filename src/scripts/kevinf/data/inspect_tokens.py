"""
Inspect tokenized .npy files by decoding back to text.

Usage:
    python inspect_tokens.py /path/to/tokens.npy
"""

import sys

import numpy as np

from transformers import AutoTokenizer


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_tokens.py <path_to_npy>")
        sys.exit(1)

    npy_path = sys.argv[1]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")

    # Load tokens - Dolma outputs raw binary uint32, not numpy format
    tokens = np.fromfile(npy_path, dtype=np.uint32)
    print(f"Loaded {len(tokens):,} tokens from {npy_path}")
    print(f"dtype: {tokens.dtype}, shape: {tokens.shape}")
    print(f"Token range: {tokens.min()} - {tokens.max()}")

    # Decode first N tokens
    n = 500
    sample_tokens = tokens[:n].tolist()
    decoded = tokenizer.decode(sample_tokens)

    print(f"\n=== First {n} tokens decoded ===")
    print(decoded)

    # Show EOS token locations (document boundaries)
    eos_id = 100257
    eos_positions = np.where(tokens == eos_id)[0]
    print(f"\n=== Found {len(eos_positions)} documents (EOS tokens) ===")
    if len(eos_positions) > 0:
        print(f"First 10 doc boundaries at positions: {eos_positions[:10].tolist()}")


if __name__ == "__main__":
    main()
