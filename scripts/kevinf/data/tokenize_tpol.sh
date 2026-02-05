#!/bin/bash
# tokenize_pile_of_law.sh

# Tokenize Pile of Law
dolma tokens \
    --documents "/data/input/ai2-llm/pretraining-data/sources/the-pile-of-law/train.*.jsonl.xz" \
    --destination "/data/input/ai2-llm/preprocessed/the-pile-of-law/allenai/dolma2-tokenizer/" \
    --tokenizer.name_or_path allenai/dolma2-tokenizer \
    --tokenizer.eos_token_id 100257 \
    --tokenizer.pad_token_id 100277 \
    --dtype uint32 \
    --processes 200

echo "Pile of Law tokenization complete!"