#!/bin/bash
# =============================================================================
# MIMIC-IV-Note Tokenization — Worker (runs on EC2 instance)
# =============================================================================
#
# This script runs on the c6a.48xlarge instance. It is launched by the local
# orchestrator (tokenize_mimic_iv_note.sh).
#
# Steps: download CSVs → convert to Dolma JSONL → tokenize (per subset) → upload to S3
# =============================================================================

set -e

# Source profile to pick up PATH for uv, s5cmd, dolma, etc.
source ~/.bashrc 2>/dev/null || true

DATA_DIR="/mnt/raid0/mimic-iv-note"
RAW_DIR="${DATA_DIR}/raw"
DOLMA_DIR="${DATA_DIR}/dolma_docs"
TOK_DISCHARGE="${DATA_DIR}/tokenized_discharge"
TOK_RADIOLOGY="${DATA_DIR}/tokenized_radiology"

S3_SOURCE="s3://ai2-llm/pretraining-data/sources/mimic-iv-note/note"
S3_DOLMA_DEST="s3://ai2-llm/pretraining-data/sources/mimic-iv-note/dolma_docs"
S3_TOKENIZED_BASE="s3://ai2-llm/preprocessed/mimic-iv-note"

CONVERT_SCRIPT="/home/ec2-user/scripts/convert_csv_to_jsonl.py"

echo "=========================================="
echo "MIMIC-IV-Note Worker Pipeline"
echo "Started: $(date)"
echo "=========================================="

# =============================================================================
# STEP 1: Download CSV files from S3
# =============================================================================
echo ""
echo ">>> Step 1: Downloading MIMIC-IV-Note CSVs from S3..."
mkdir -p "$RAW_DIR"

s5cmd --numworkers 64 cp "${S3_SOURCE}/*" "${RAW_DIR}/"

echo "Downloaded files:"
ls -lh "$RAW_DIR"

# =============================================================================
# STEP 2: Convert CSV → Dolma JSONL
# =============================================================================
echo ""
echo ">>> Step 2: Converting CSV to Dolma JSONL..."
mkdir -p "$DOLMA_DIR"

python3 "$CONVERT_SCRIPT" \
    --input-dir "$RAW_DIR" \
    --output-dir "$DOLMA_DIR" \
    --source mimic-iv-note \
    --text-field text \
    --id-field note_id \
    --metadata-fields subject_id hadm_id note_type note_seq charttime \
    --files discharge.csv.gz radiology.csv.gz \
    --max-workers 64 \
    --docs-per-shard 50000

echo ""
echo "Dolma docs output:"
du -sh "$DOLMA_DIR"/*

# =============================================================================
# STEP 3: Tokenize with Dolma
# =============================================================================
echo ""
echo ">>> Step 3: Tokenizing with Dolma..."
mkdir -p "$TOKENIZED_DIR"

uv run dolma tokens \
    --documents "${DOLMA_DIR}/**/*.jsonl.gz" \
    --destination "$TOKENIZED_DIR" \
    --tokenizer.name_or_path allenai/dolma2-tokenizer \
    --tokenizer.eos_token_id 100257 \
    --tokenizer.pad_token_id 100277 \
    --no-tokenizer.segment_before_tokenization \
    --tokenizer.encode_special_tokens \
    --processes 192 \
    --max_size 1_000_000_000 \
    --sample_ring_prop \
    --dtype uint32

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
TOTAL_BYTES=$(find "$TOKENIZED_DIR" -name "*.npy" -exec stat --format="%s" {} + | paste -sd+ | bc)
TOTAL_TOKENS=$((TOTAL_BYTES / 4))
echo "Total tokens: $(printf "%'d" $TOTAL_TOKENS)"

# =============================================================================
# STEP 5: Upload to S3
# =============================================================================
echo ""
echo ">>> Step 5: Uploading to S3..."

echo "Uploading tokenized .npy files..."
s5cmd cp "${TOKENIZED_DIR}/*" "${S3_TOKENIZED_DEST}/"

echo "Uploading dolma docs..."
for subdir in "$DOLMA_DIR"/*/; do
    dirname=$(basename "$subdir")
    s5cmd cp "${subdir}*.jsonl.gz" "${S3_DOLMA_DEST}/${dirname}/"
done

# =============================================================================
# STEP 6: Verify upload
# =============================================================================
echo ""
echo ">>> Step 6: Verifying upload..."

echo "Tokenized files on S3:"
aws s3 ls "${S3_TOKENIZED_DEST}/" | wc -l
echo "files"
aws s3 ls "${S3_TOKENIZED_DEST}/" | head -5

echo ""
echo "Dolma docs on S3:"
aws s3 ls --recursive "${S3_DOLMA_DEST}/" | wc -l
echo "files"

echo ""
echo "=========================================="
echo "ALL DONE!"
echo "Finished: $(date)"
echo "Total tokens: $(printf "%'d" $TOTAL_TOKENS)"
echo ""
echo "Tokenized data at: ${S3_TOKENIZED_DEST}/"
echo "Dolma docs at:     ${S3_DOLMA_DEST}/"
echo ""
echo "Safe to terminate the cluster."
echo "=========================================="
