#!/bin/bash
# Convert ChemPile datasets to chunked JSONL format

python src/scripts/kevinf/data/convert_chempile_to_jsonl.py \
    --output-dir /data/input/kf/FlexMoE/data/chempile_chunked \
    --datasets all \
    --max-workers 220 \
    --docs-per-chunk 50000
