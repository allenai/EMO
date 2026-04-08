#!/bin/bash
# Test script to verify ChemBench gen evaluation produces correct metrics locally

set -e  # Exit on error

echo "============================================"
echo "Testing ChemBench Gen Evaluation Locally"
echo "============================================"
echo

# Configuration
MODEL_PATH="/data/input/kevinf/checkpoints/new-kevinf-olmo3-1b-130b-dolma3-0625-150Bsample/step30995-hf"
OUTPUT_DIR="/tmp/test-chembench-eval-$(date +%s)"
TASK="chembench_organic_chemistry:gen"
LIMIT=5  # Only test on 5 examples for speed

echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Task: $TASK"
echo "Limit: $LIMIT examples"
echo

# Run evaluation
echo "Running evaluation..."
PYTHONPATH=. python -u -m offline_evals.run_eval \
    --model "$MODEL_PATH" \
    --model-type hf \
    --task "$TASK" \
    --limit "$LIMIT" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 2 \
    --fewshot-seed 1234 \
    --random-subsample-seed 1234

echo
echo "============================================"
echo "Checking Results"
echo "============================================"
echo

# Find the metrics file
METRICS_FILE=$(find "$OUTPUT_DIR" -name "*gen-metrics.json" | head -1)

if [ -z "$METRICS_FILE" ]; then
    echo "❌ ERROR: No metrics file found!"
    exit 1
fi

echo "Found metrics file: $METRICS_FILE"
echo

# Check the metrics
python3 << EOF
import json
import sys

with open("$METRICS_FILE") as f:
    data = json.load(f)

metrics = data.get("metrics", {})
task_name = data.get("task_name", "unknown")
num_instances = data.get("num_instances", 0)

print(f"Task: {task_name}")
print(f"Instances: {num_instances}")
print()

print("Metrics keys:", list(metrics.keys()))
print()

# Check required metrics
required = ["exact_match", "f1", "recall", "all_correct", "primary_score"]
missing = [m for m in required if m not in metrics]

if missing:
    print(f"❌ FAILED: Missing metrics: {missing}")
    sys.exit(1)

print("✅ All required metrics present!")
print()

# Show metric values
print("Metric Values:")
for key in ["exact_match", "f1", "recall", "all_correct", "primary_score"]:
    print(f"  {key}: {metrics[key]}")
print()

# Verify primary_score equals all_correct
if abs(metrics["primary_score"] - metrics["all_correct"]) > 1e-6:
    print(f"❌ FAILED: primary_score ({metrics['primary_score']}) != all_correct ({metrics['all_correct']})")
    sys.exit(1)

print("✅ primary_score correctly set from all_correct")
print()

# Check task config
task_config = data.get("task_config", {})
primary_metric = task_config.get("primary_metric", "unknown")
print(f"Task config primary_metric: {primary_metric}")

if primary_metric != "all_correct":
    print(f"❌ WARNING: Expected primary_metric='all_correct', got '{primary_metric}'")
else:
    print("✅ Task config correctly set")

print()
print("=" * 44)
print("✅ ALL TESTS PASSED!")
print("=" * 44)
EOF

TEST_RESULT=$?

echo
echo "Output directory: $OUTPUT_DIR"
echo "(You can delete this with: rm -rf $OUTPUT_DIR)"

exit $TEST_RESULT
