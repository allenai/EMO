dolma tokens \
    --documents "${INPUT_DIR}/chempile_paper_*.jsonl.gz" \
    --destination "${OUTPUT_BASE}/chempile_paper" \
    --tokenizer.name_or_path allenai/dolma2-tokenizer \
    --tokenizer.eos_token_id 100257 \
    --tokenizer.pad_token_id 100277 \
    --dtype uint32 \
    --processes 220