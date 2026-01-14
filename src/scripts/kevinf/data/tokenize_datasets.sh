#!/bin/bash
# Tokenize ChemPile datasets with Dolma

dolma tokens \
    --documents "/data/input/kf/FlexMoE/data/chempile_chunked/chempile_paper/*.jsonl.gz" \
    --destination /data/input/kf/FlexMoE/data/chempile/tokenized/chempile_paper \
    --tokenizer.name_or_path allenai/dolma2-tokenizer \
    --tokenizer.eos_token_id 100257 \
    --tokenizer.pad_token_id 100277 \
    --dtype uint32 \
    --processes 200
