#!/bin/bash
# =============================================================================
# Tokenize JSONL datasets with Dolma
# =============================================================================
#
# Tokenizes JSONL files into .npy format for training. Run configs sequentially
# with max workers (not parallel with fewer workers) for best performance.
#
# Usage:
#   ./tokenize_datasets.sh <jsonl-base> <output-base> <processes> [config1 config2 ...]
#
# Examples:
#   # Tokenize specific configs:
#   ./tokenize_datasets.sh /data/croissant/jsonl /data/croissant/tokenized 200 \
#       french_303b_1 french_303b_2 code_140b
#
#   # Tokenize all configs in jsonl dir:
#   ./tokenize_datasets.sh /data/croissant/jsonl /data/croissant/tokenized 200
#
# Performance notes:
#   - Run configs SEQUENTIALLY with max workers (e.g., 200)
#   - NOT parallel with fewer workers per config
#   - This ensures full worker utilization when small configs finish
#   - Monitor with: iostat -x 1 (check if CPU or I/O bound)
#
# Output:
#   <output-base>/<config>/part-XX-00000.npy  (tokenized data)
#   <output-base>/<config>/part-XX-00000.csv.gz  (document boundary metadata)
# =============================================================================

set -e

JSONL_BASE="${1:?Usage: $0 <jsonl-base> <output-base> <processes> [config1 config2 ...]}"
OUTPUT_BASE="${2:?Usage: $0 <jsonl-base> <output-base> <processes> [config1 config2 ...]}"
PROCESSES="${3:-200}"
shift 3
CONFIGS="$@"

# If no configs specified, find all subdirs
if [ -z "$CONFIGS" ]; then
    CONFIGS=$(ls -d ${JSONL_BASE}/*/ 2>/dev/null | xargs -I{} basename {})
fi

echo "=========================================="
echo "Dolma Tokenization"
echo "=========================================="
echo "Input:    $JSONL_BASE"
echo "Output:   $OUTPUT_BASE"
echo "Workers:  $PROCESSES"
echo "Configs:  $CONFIGS"
echo "=========================================="

for config in $CONFIGS; do
    echo ""
    echo ">>> Tokenizing: $config"
    dolma tokens \
        --documents "${JSONL_BASE}/${config}/**/*.jsonl.gz" \
        --destination "${OUTPUT_BASE}/${config}" \
        --tokenizer.name_or_path allenai/dolma2-tokenizer \
        --tokenizer.eos_token_id 100257 \
        --tokenizer.pad_token_id 100277 \
        --dtype uint32 \
        --processes "$PROCESSES"
    echo "<<< Completed: $config"
done

echo ""
echo "=========================================="
echo "ALL DONE!"
echo "=========================================="
echo "Token count: ./count_tokens.sh $OUTPUT_BASE"