#!/bin/bash
set -euo pipefail

# Downloads all .npy files listed in mose_code_mix.txt from S3.
#
# Usage:
#   bash scripts/download_mose_code_mix.sh <tokenizer_id> [base_s3_dir] [local_output_dir]
#
# Example:
#   bash scripts/download_mose_code_mix.sh dolma2-tokenizer s3://ai2-llm ./data

MIX_FILE="src/olmo_core/data/mixes/mose_code_mix.txt"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <tokenizer_id> [base_s3_dir] [local_output_dir]"
    echo "  tokenizer_id   : e.g. dolma2-tokenizer, gpt-neox-olmo-dolma-v1_5"
    echo "  base_s3_dir    : S3 prefix (default: s3://ai2-llm)"
    echo "  local_output_dir: local destination (default: ./mose_code_mix_data)"
    exit 1
fi

TOKENIZER="$1"
BASE_DIR="${2:-s3://ai2-llm}"
OUTPUT_DIR="${3:-./mose_code_mix_data}"

# Strip trailing slash from base dir
BASE_DIR="${BASE_DIR%/}"

echo "Tokenizer:  $TOKENIZER"
echo "S3 base:    $BASE_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Mix file:   $MIX_FILE"
echo ""

count=0
failed=0

while IFS= read -r line; do
    # Skip empty lines and comments
    [[ -z "$line" || "$line" == \#* ]] && continue

    label="${line%%,*}"
    rel_path="${line#*,}"

    # Replace {TOKENIZER} placeholder
    rel_path="${rel_path//\{TOKENIZER\}/$TOKENIZER}"

    s3_path="${BASE_DIR}/${rel_path}"
    local_path="${OUTPUT_DIR}/${rel_path}"

    mkdir -p "$(dirname "$local_path")"

    echo "[$((count + 1))] Downloading: $s3_path"
    if aws s3 cp "$s3_path" "$local_path" --quiet; then
        count=$((count + 1))
    else
        echo "  FAILED: $s3_path"
        failed=$((failed + 1))
    fi
done < "$MIX_FILE"

echo ""
echo "Done. Downloaded: $count, Failed: $failed"
