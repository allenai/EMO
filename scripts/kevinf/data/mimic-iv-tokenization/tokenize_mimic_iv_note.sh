#!/bin/bash
# =============================================================================
# MIMIC-IV-Note Tokenization — Local Orchestrator
# =============================================================================
#
# Run this locally. It provisions a c6a.48xlarge, installs tools, copies the
# conversion script, and kicks off the full pipeline on the instance.
#
# Data: ~331K discharge summaries + ~2.3M radiology reports
# Source: s3://ai2-llm/pretraining-data/sources/mimic-iv-note/note/
# Output: s3://ai2-llm/preprocessed/mimic-iv-note/dolma2-tokenizer/
#
# Usage:
#   ./tokenize_mimic_iv_note.sh
# =============================================================================

set -e

CLUSTER_NAME="mimic-iv-note"
INSTANCE_TYPE="c6a.48xlarge"
STORAGE_SIZE=200
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "=========================================="
echo "MIMIC-IV-Note Tokenization Pipeline"
echo "=========================================="
echo "Cluster:  $CLUSTER_NAME"
echo "Instance: $INSTANCE_TYPE"
echo "Storage:  ${STORAGE_SIZE}GB EBS"
echo "=========================================="

# =============================================================================
# STEP 1: Provision cluster
# =============================================================================
echo ""
echo ">>> Step 1: Creating cluster..."
poormanray create -n $CLUSTER_NAME -t $INSTANCE_TYPE --number 1 --storage-size $STORAGE_SIZE

# =============================================================================
# STEP 2: Setup tools
# =============================================================================
echo ""
echo ">>> Step 2: Setting up tools (AWS creds, d2tk, dolma)..."
poormanray setup -n $CLUSTER_NAME
# Run setup WITHOUT -d so they block until fully installed
poormanray setup-d2tk -n $CLUSTER_NAME
poormanray setup-dolma-python -n $CLUSTER_NAME

# Verify (source profile to pick up PATH additions from setup)
echo ""
echo "Verifying installations..."
poormanray run -n $CLUSTER_NAME -c 'source ~/.bashrc && uv run dolma --help | head -5'
poormanray run -n $CLUSTER_NAME -c 'source ~/.bashrc && which s5cmd'

# =============================================================================
# STEP 3: Copy scripts to instance
# =============================================================================
echo ""
echo ">>> Step 3: Copying scripts to instance..."
CONVERT_SCRIPT="$REPO_DIR/src/scripts/kevinf/data/convert_csv_to_jsonl.py"
WORKER_SCRIPT="$SCRIPT_DIR/tokenize_mimic_iv_note_worker.sh"

poormanray run -n $CLUSTER_NAME -c 'mkdir -p /home/ec2-user/scripts'

# Use base64 to safely transfer scripts (avoids heredoc escaping issues)
B64_CONVERT=$(base64 < "$CONVERT_SCRIPT")
poormanray run -n $CLUSTER_NAME -c "echo '$B64_CONVERT' | base64 -d > /home/ec2-user/scripts/convert_csv_to_jsonl.py"

B64_WORKER=$(base64 < "$WORKER_SCRIPT")
poormanray run -n $CLUSTER_NAME -c "echo '$B64_WORKER' | base64 -d > /home/ec2-user/scripts/tokenize_mimic_iv_note_worker.sh && chmod +x /home/ec2-user/scripts/tokenize_mimic_iv_note_worker.sh"

# Verify
poormanray run -n $CLUSTER_NAME -c 'head -3 /home/ec2-user/scripts/convert_csv_to_jsonl.py'
poormanray run -n $CLUSTER_NAME -c 'head -3 /home/ec2-user/scripts/tokenize_mimic_iv_note_worker.sh'
echo "Scripts copied successfully."

# =============================================================================
# STEP 4: Run the full pipeline on the instance (detached)
# =============================================================================
echo ""
echo ">>> Step 4: Launching pipeline on instance (detached)..."
poormanray run -n $CLUSTER_NAME -c 'source ~/.bashrc 2>/dev/null; screen -dmS worker bash -c "bash ~/scripts/tokenize_mimic_iv_note_worker.sh > ~/worker.log 2>&1"'

echo ""
echo "=========================================="
echo "Pipeline launched in detached mode!"
echo ""
echo "Monitor progress:"
echo "  poormanray run -n $CLUSTER_NAME -c 'tail -50 ~/worker.log'"
echo ""
echo "SSH in to check:"
echo "  poormanray ssh -n $CLUSTER_NAME"
echo "  tail -f ~/worker.log"
echo ""
echo "When done, terminate:"
echo "  poormanray terminate -n $CLUSTER_NAME"
echo "=========================================="
