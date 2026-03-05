import os

import click
import numpy as np
from transformers import AutoTokenizer


@click.command()
@click.argument("tokenized_file")
@click.option("--tokenizer-name-or-path", default="allenai/dolma2-tokenizer")
@click.option("--dtype", default="uint32")
@click.option("--chunk-size", default=1024**2, type=int)
@click.option("--num-docs", default=3, type=int, help="Number of documents to print")
def inspect_tokenized(
    tokenized_file: str, tokenizer_name_or_path: str, dtype: str, chunk_size: int, num_docs: int
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    path = tokenized_file
    size = os.path.getsize(path)
    data = np.memmap(path, dtype=dtype, mode="r", shape=(size // 4,))

    collection = []
    doc_count = 0
    i = 0
    while i < len(data) and doc_count < num_docs:
        chunk = data[i : i + chunk_size]
        i += chunk_size

        while (chunk == tokenizer.eos_token_id).any() and doc_count < num_docs:
            eos_idx = np.where(chunk == tokenizer.eos_token_id)[0][0] + 1
            collection.extend(chunk[:eos_idx].tolist())
            print(f"\n{'='*80}")
            print(f"DOCUMENT {doc_count + 1}  (tokens: {len(collection)})")
            print("=" * 80)
            print(tokenizer.decode(collection))
            collection = []
            chunk = chunk[eos_idx:]
            doc_count += 1

        if doc_count < num_docs:
            collection.extend(chunk.tolist())


if __name__ == "__main__":
    inspect_tokenized()
