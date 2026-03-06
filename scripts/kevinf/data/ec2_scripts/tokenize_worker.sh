#!/bin/bash
# =============================================================================
# Generic Tokenization — Worker Pipeline
# =============================================================================
#
# Runs the full tokenization pipeline: download → convert → tokenize → upload.
# Reads all configuration from a YAML config file via parse_config.py.
#
# Works both locally and on EC2 (launched by tokenize_remote.sh).
#
# Usage:
#   ./tokenize_worker.sh configs/mimic-iv-note.yaml
#   ./tokenize_worker.sh configs/the-pile-of-law.yaml
#
# Supported source formats:
#   csv   — convert via convert_csv_to_jsonl.py (needs text_field, id_field, etc.)
#   arrow — convert via convert_arrow_to_jsonl.py (needs data_dirs)
#   jsonl — skip conversion, tokenize directly from source
# =============================================================================

set -e

# Source profile to pick up PATH for uv, s5cmd, dolma, etc.
source ~/.bashrc 2>/dev/null || true

CONFIG="${1:?Usage: $0 <config.yaml>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARSE_SCRIPT="$SCRIPT_DIR/parse_config.py"

# If parse_config.py is not next to us (local dev), look in the repo
if [ ! -f "$PARSE_SCRIPT" ]; then
    REPO_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
    PARSE_SCRIPT="$REPO_DIR/src/scripts/kevinf/data/parse_config.py"
fi

# Load config into env vars
eval "$(python3 "$PARSE_SCRIPT" "$CONFIG")"

# =============================================================================
# Resolve working directories
# =============================================================================
# Remote default: /mnt/raid0/<name>/...
# Local: use LOCAL_SOURCE / LOCAL_DOLMA_DOCS / LOCAL_TOKENIZED from config
if [ -n "$LOCAL_SOURCE" ]; then
    BASE_DIR="$LOCAL_SOURCE"
else
    BASE_DIR="/mnt/raid0/$DATASET_NAME"
fi

RAW_DIR="${BASE_DIR}/raw"
DOLMA_DIR="${LOCAL_DOLMA_DOCS:-${BASE_DIR}/dolma_docs}"
TOKENIZED_DIR="${LOCAL_TOKENIZED:-${BASE_DIR}/tokenized}"

# Conversion scripts (co-located on remote, or in repo locally)
CONVERT_CSV="$SCRIPT_DIR/convert_csv_to_jsonl.py"
CONVERT_ARROW="$SCRIPT_DIR/convert_arrow_to_jsonl.py"
if [ ! -f "$CONVERT_CSV" ]; then
    REPO_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
    CONVERT_CSV="$REPO_DIR/src/scripts/kevinf/data/convert_csv_to_jsonl.py"
    CONVERT_ARROW="$REPO_DIR/src/scripts/kevinf/data/convert_arrow_to_jsonl.py"
fi

# Dolma command (uv run dolma vs dolma)
if [ "$USE_UV" = "true" ]; then
    DOLMA_CMD="uv run dolma"
else
    DOLMA_CMD="dolma"
fi

echo "=========================================="
echo "Tokenization Worker: $DATASET_NAME"
echo "Started: $(date)"
echo "=========================================="
echo "Format:     $SOURCE_FORMAT"
echo "Raw:        $RAW_DIR"
echo "Dolma docs: $DOLMA_DIR"
echo "Tokenized:  $TOKENIZED_DIR"
echo "Processes:  $NUM_PROCESSES"
echo "=========================================="

# =============================================================================
# STEP 1: Download from S3 (skipped if no S3 source)
# =============================================================================
if [ -n "$S3_SOURCE" ]; then
    echo ""
    echo ">>> Step 1: Downloading from S3..."
    mkdir -p "$RAW_DIR"
    s5cmd --numworkers 64 cp "${S3_SOURCE}/*" "${RAW_DIR}/"
    echo "Downloaded files:"
    ls -lh "$RAW_DIR"
else
    echo ""
    echo ">>> Step 1: Skipped (no S3 source, using local data)"
fi

# =============================================================================
# STEP 2: Convert to Dolma JSONL (format-dependent)
# =============================================================================
echo ""
echo ">>> Step 2: Converting to Dolma JSONL (format: $SOURCE_FORMAT)..."

case "$SOURCE_FORMAT" in
    csv)
        mkdir -p "$DOLMA_DIR"
        CSV_ARGS=(
            --input-dir "$RAW_DIR"
            --output-dir "$DOLMA_DIR"
            --source "$DATASET_NAME"
            --text-field "$TEXT_FIELD"
            --id-field "$ID_FIELD"
            --max-workers "$CONVERT_WORKERS"
            --docs-per-shard "$DOCS_PER_SHARD"
        )
        if [ -n "$METADATA_FIELDS" ]; then
            CSV_ARGS+=(--metadata-fields $METADATA_FIELDS)
        fi
        if [ -n "$FILES" ]; then
            CSV_ARGS+=(--files $FILES)
        fi
        python3 "$CONVERT_CSV" "${CSV_ARGS[@]}"
        echo "Dolma docs output:"
        du -sh "$DOLMA_DIR"/*
        # Set the documents pattern for tokenization
        DOC_PATTERN="${DOLMA_DIR}/**/*.jsonl.gz"
        ;;

    arrow)
        mkdir -p "$DOLMA_DIR"
        ARROW_ARGS=(
            --input-dir "$RAW_DIR"
            --output-dir "$DOLMA_DIR"
            --name "$DATASET_NAME"
            --max-workers "$CONVERT_WORKERS"
            --docs-per-chunk "$DOCS_PER_SHARD"
        )
        if [ -n "$DATA_DIRS" ]; then
            ARROW_ARGS+=(--data-dirs $DATA_DIRS)
        fi
        python3 "$CONVERT_ARROW" "${ARROW_ARGS[@]}"
        echo "Dolma docs output:"
        du -sh "$DOLMA_DIR"/*
        DOC_PATTERN="${DOLMA_DIR}/**/*.jsonl.gz"
        ;;

    jsonl)
        echo "Skipping conversion (source is already JSONL)."
        # Use source path directly with custom pattern or default
        SRC_PATH="${LOCAL_SOURCE:-$RAW_DIR}"
        if [ -n "$DOCUMENTS_PATTERN" ]; then
            DOC_PATTERN="${SRC_PATH}/${DOCUMENTS_PATTERN}"
        else
            DOC_PATTERN="${SRC_PATH}/**/*.jsonl.gz"
        fi
        ;;

    *)
        echo "ERROR: Unknown source format: $SOURCE_FORMAT"
        echo "Supported: csv, arrow, jsonl"
        exit 1
        ;;
esac

echo "Documents pattern: $DOC_PATTERN"

# =============================================================================
# STEP 3: Tokenize with Dolma
# =============================================================================
echo ""
echo ">>> Step 3: Tokenizing with Dolma..."
mkdir -p "$TOKENIZED_DIR"

# Build dolma command
DOLMA_ARGS=(
    tokens
    --documents "$DOC_PATTERN"
    --destination "$TOKENIZED_DIR"
    --tokenizer.name_or_path "$TOKENIZER_NAME"
    --tokenizer.eos_token_id "$EOS_TOKEN_ID"
    --tokenizer.pad_token_id "$PAD_TOKEN_ID"
    --dtype "$DTYPE"
    --processes "$NUM_PROCESSES"
)

# Append any extra args from config
if [ -n "$DOLMA_EXTRA_ARGS" ]; then
    DOLMA_ARGS+=($DOLMA_EXTRA_ARGS)
fi

$DOLMA_CMD "${DOLMA_ARGS[@]}"

echo ""
echo "Tokenized output:"
ls "$TOKENIZED_DIR"/*.npy 2>/dev/null | wc -l
echo "npy files"
du -sh "$TOKENIZED_DIR"

# =============================================================================
# STEP 4: Count tokens
# =============================================================================
echo ""
echo ">>> Step 4: Counting tokens..."
TOTAL_BYTES=$(find "$TOKENIZED_DIR" -name "*.npy" -print0 | xargs -0 du -cb 2>/dev/null | tail -1 | cut -f1)
TOTAL_TOKENS=$((TOTAL_BYTES / 4))
echo "Total tokens: $(printf "%'d" $TOTAL_TOKENS)"

# =============================================================================
# STEP 5: Upload to S3 (skipped if no S3 destination)
# =============================================================================
if [ -n "$S3_TOKENIZED_DEST" ] || [ -n "$S3_DOLMA_DEST" ]; then
    echo ""
    echo ">>> Step 5: Uploading to S3..."

    if [ -n "$S3_TOKENIZED_DEST" ]; then
        echo "Uploading tokenized .npy files..."
        s5cmd cp "${TOKENIZED_DIR}/*" "${S3_TOKENIZED_DEST}/"
    fi

    if [ -n "$S3_DOLMA_DEST" ]; then
        echo "Uploading dolma docs..."
        for subdir in "$DOLMA_DIR"/*/; do
            dirname=$(basename "$subdir")
            s5cmd cp "${subdir}*.jsonl.gz" "${S3_DOLMA_DEST}/${dirname}/"
        done
    fi

    # Verify
    echo ""
    echo ">>> Step 6: Verifying upload..."
    if [ -n "$S3_TOKENIZED_DEST" ]; then
        echo "Tokenized files on S3:"
        aws s3 ls "${S3_TOKENIZED_DEST}/" | wc -l
        echo "files"
        aws s3 ls "${S3_TOKENIZED_DEST}/" | head -5
    fi
    if [ -n "$S3_DOLMA_DEST" ]; then
        echo ""
        echo "Dolma docs on S3:"
        aws s3 ls --recursive "${S3_DOLMA_DEST}/" | wc -l
        echo "files"
    fi
else
    echo ""
    echo ">>> Step 5: Skipped (no S3 destination configured)"
fi

# =============================================================================
# Done
# =============================================================================
echo ""
echo "=========================================="
echo "ALL DONE!"
echo "Finished: $(date)"
echo "Total tokens: $(printf "%'d" $TOTAL_TOKENS)"
echo ""
echo "Tokenized data: $TOKENIZED_DIR"
if [ -n "$S3_TOKENIZED_DEST" ]; then
    echo "S3 tokenized:   $S3_TOKENIZED_DEST"
fi
if [ -n "$S3_DOLMA_DEST" ]; then
    echo "S3 dolma docs:  $S3_DOLMA_DEST"
fi
if [ "$COMPUTE_MODE" = "remote" ]; then
    echo ""
    echo "Safe to terminate the cluster."
fi
echo "=========================================="
