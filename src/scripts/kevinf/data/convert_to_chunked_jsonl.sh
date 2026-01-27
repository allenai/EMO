#!/bin/bash
# Convert HuggingFace datasets to chunked JSONL format for Dolma tokenization.
#
# Usage:
#   ./convert_to_chunked_jsonl.sh <hf-path> [output-dir] [name]
#
# Examples:
#   # Convert ChemPile education dataset:
#   ./convert_to_chunked_jsonl.sh jablonkagroup/chempile-education
#
#   # Convert with custom output directory:
#   ./convert_to_chunked_jsonl.sh jablonkagroup/chempile-education ./my_data
#
#   # Convert with custom name:
#   ./convert_to_chunked_jsonl.sh jablonkagroup/chempile-education ./my_data chempile_edu

set -e

HF_PATH="${1:?Error: HuggingFace dataset path required (e.g., jablonkagroup/chempile-education)}"
OUTPUT_DIR="${2:-/data/input/kf/FlexMoE/data/chunked}"
NAME="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CMD="python ${SCRIPT_DIR}/convert_hf_to_jsonl.py \
    --hf-path ${HF_PATH} \
    --output-dir ${OUTPUT_DIR} \
    --max-workers 220 \
    --docs-per-chunk 50000"

if [ -n "${NAME}" ]; then
    CMD="${CMD} --name ${NAME}"
fi

echo "Running: ${CMD}"
eval "${CMD}"
