#!/bin/bash
# =============================================================================
# Unified HuggingFace Download & Convert Pipeline
# =============================================================================
#
# Downloads HF datasets locally (fast) then converts Arrow → JSONL for Dolma.
# This is ~123x faster than streaming directly from HuggingFace.
#
# Usage:
#   ./download_and_convert.sh <hf-repo> <output-base> [dir1 dir2 ...]
#
# Examples:
#   # Download and convert specific configs:
#   ./download_and_convert.sh croissantllm/croissant_dataset /data/croissant \
#       french_303b_1 french_303b_2 french_303b_3 code_140b aligned_36b
#
#   # Download entire dataset (all configs):
#   ./download_and_convert.sh allenai/dolma /data/dolma
#
# Output structure:
#   <output-base>/raw/      - Downloaded Arrow files
#   <output-base>/jsonl/    - Converted JSONL files (ready for Dolma)
#
# Next step (tokenization):
#   dolma tokens --documents '<output-base>/jsonl/**/*.jsonl.gz' \
#       --destination <output-base>/tokenized \
#       --tokenizer.name_or_path allenai/dolma2-tokenizer \
#       --dtype uint32 --processes 200
# =============================================================================

set -e

REPO="$1"
OUTPUT_BASE="$2"
shift 2
DIRS="$@"

if [ -z "$REPO" ] || [ -z "$OUTPUT_BASE" ]; then
    echo "Usage: $0 <hf-repo> <output-base> [dir1 dir2 ...]"
    exit 1
fi

DOWNLOAD_DIR="${OUTPUT_BASE}/raw"
JSONL_DIR="${OUTPUT_BASE}/jsonl"
DATASET_NAME=$(echo "$REPO" | tr '/' '_')

echo "=========================================="
echo "Step 1: Download from HuggingFace"
echo "=========================================="

if [ -n "$DIRS" ]; then
    HF_HUB_ENABLE_HF_TRANSFER=1 python src/scripts/kevinf/data/download_hf_dataset.py \
        --repo "$REPO" \
        --output-dir "$DOWNLOAD_DIR" \
        --dirs $DIRS \
        --parallel
else
    HF_HUB_ENABLE_HF_TRANSFER=1 python src/scripts/kevinf/data/download_hf_dataset.py \
        --repo "$REPO" \
        --output-dir "$DOWNLOAD_DIR"
fi

echo ""
echo "=========================================="
echo "Step 2: Convert Arrow to JSONL"
echo "=========================================="

# If no dirs specified, find all subdirs with Arrow files
if [ -z "$DIRS" ]; then
    DIRS=$(find "$DOWNLOAD_DIR" -name "*.arrow" -exec dirname {} \; | xargs -I{} dirname {} | sort -u | xargs -I{} basename {})
fi

python src/scripts/kevinf/data/convert_arrow_to_jsonl.py \
    --input-dir "$DOWNLOAD_DIR" \
    --output-dir "$JSONL_DIR" \
    --data-dirs $DIRS \
    --name "$DATASET_NAME" \
    --max-workers 100

echo ""
echo "=========================================="
echo "Done! Output: $JSONL_DIR/"
echo "=========================================="
echo ""
echo "Next: Tokenize with Dolma:"
echo "dolma tokens \\"
echo "    --documents '${JSONL_DIR}/**/*.jsonl.gz' \\"
echo "    --destination ${OUTPUT_BASE}/tokenized \\"
echo "    --tokenizer.name_or_path allenai/dolma2-tokenizer \\"
echo "    --dtype uint32 --processes 64"
