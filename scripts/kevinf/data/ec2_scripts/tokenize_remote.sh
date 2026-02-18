#!/bin/bash
# =============================================================================
# Generic Tokenization — Local Orchestrator
# =============================================================================
#
# Reads a YAML config and either runs the pipeline locally or provisions an
# EC2 instance via poormanray and runs it there.
#
# Usage:
#   ./tokenize_remote.sh configs/mimic-iv-note.yaml
#   ./tokenize_remote.sh configs/the-pile-of-law.yaml
#
# For remote mode:
#   - Provisions EC2, installs tools, copies scripts + config, launches worker
#   - Prints monitoring/SSH/terminate instructions when done
#
# For local mode:
#   - Runs tokenize_worker.sh directly on this machine
# =============================================================================

set -e

CONFIG="${1:?Usage: $0 <config.yaml>}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PARSE_SCRIPT="$REPO_DIR/src/scripts/kevinf/data/parse_config.py"

# Load config into env vars
eval "$(python3 "$PARSE_SCRIPT" "$CONFIG")"

echo "=========================================="
echo "Tokenization Pipeline: $DATASET_NAME"
echo "=========================================="
echo "Mode:     $COMPUTE_MODE"
echo "Format:   $SOURCE_FORMAT"
echo "Workers:  $NUM_PROCESSES"
if [ "$COMPUTE_MODE" = "remote" ]; then
    echo "Instance: $INSTANCE_TYPE"
    echo "Storage:  ${STORAGE_SIZE}GB EBS"
fi
echo "=========================================="

# =============================================================================
# LOCAL MODE: run worker directly
# =============================================================================
if [ "$COMPUTE_MODE" = "local" ]; then
    echo ""
    echo ">>> Running pipeline locally..."
    bash "$SCRIPT_DIR/tokenize_worker.sh" "$CONFIG"
    exit 0
fi

# =============================================================================
# REMOTE MODE: provision EC2 and run on instance
# =============================================================================

# Step 1: Provision cluster
echo ""
echo ">>> Step 1: Creating cluster..."
poormanray create -n "$DATASET_NAME" -t "$INSTANCE_TYPE" --number 1 --storage-size "$STORAGE_SIZE"

# Step 2: Setup tools
echo ""
echo ">>> Step 2: Setting up tools (AWS creds, d2tk, dolma)..."
poormanray setup -n "$DATASET_NAME"
poormanray setup-d2tk -n "$DATASET_NAME"
poormanray setup-dolma-python -n "$DATASET_NAME"

# Verify
echo ""
echo "Verifying installations..."
poormanray run -n "$DATASET_NAME" -c 'source ~/.bashrc && uv run dolma --help | head -5'
poormanray run -n "$DATASET_NAME" -c 'source ~/.bashrc && which s5cmd'

# Step 3: Copy scripts + config to instance
echo ""
echo ">>> Step 3: Copying scripts to instance..."

REMOTE_SCRIPTS="/home/ec2-user/scripts"
poormanray run -n "$DATASET_NAME" -c "mkdir -p $REMOTE_SCRIPTS"

# Scripts to copy (base64 for safe transfer)
declare -A SCRIPT_FILES=(
    ["parse_config.py"]="$REPO_DIR/src/scripts/kevinf/data/parse_config.py"
    ["convert_csv_to_jsonl.py"]="$REPO_DIR/src/scripts/kevinf/data/convert_csv_to_jsonl.py"
    ["convert_arrow_to_jsonl.py"]="$REPO_DIR/src/scripts/kevinf/data/convert_arrow_to_jsonl.py"
    ["tokenize_worker.sh"]="$SCRIPT_DIR/tokenize_worker.sh"
    ["config.yaml"]="$CONFIG"
)

for name in "${!SCRIPT_FILES[@]}"; do
    local_path="${SCRIPT_FILES[$name]}"
    if [ -f "$local_path" ]; then
        B64=$(base64 < "$local_path")
        poormanray run -n "$DATASET_NAME" -c "echo '$B64' | base64 -d > $REMOTE_SCRIPTS/$name"
        echo "  Copied $name"
    fi
done

poormanray run -n "$DATASET_NAME" -c "chmod +x $REMOTE_SCRIPTS/tokenize_worker.sh"

# Verify
poormanray run -n "$DATASET_NAME" -c "ls -la $REMOTE_SCRIPTS/"
echo "Scripts copied successfully."

# Step 4: Launch pipeline on instance (detached)
echo ""
echo ">>> Step 4: Launching pipeline on instance (detached)..."
poormanray run -n "$DATASET_NAME" -c "source ~/.bashrc 2>/dev/null; screen -dmS worker bash -c 'bash $REMOTE_SCRIPTS/tokenize_worker.sh $REMOTE_SCRIPTS/config.yaml > ~/worker.log 2>&1'"

echo ""
echo "=========================================="
echo "Pipeline launched in detached mode!"
echo ""
echo "Monitor progress:"
echo "  poormanray run -n $DATASET_NAME -c 'tail -50 ~/worker.log'"
echo ""
echo "SSH in to check:"
echo "  poormanray ssh -n $DATASET_NAME"
echo "  tail -f ~/worker.log"
echo ""
echo "When done, terminate:"
echo "  poormanray terminate -n $DATASET_NAME"
echo "=========================================="
