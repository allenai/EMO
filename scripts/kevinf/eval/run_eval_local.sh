#!/bin/bash
# Simple script to run evaluations locally (no Beaker/Gantry)
# Usage: bash src/scripts/kevinf/eval/run_eval_local.sh [OPTIONS]

set -e

# Default configuration
MODEL_PATH="/data/input/kevinf/checkpoints/new-kevinf-olmo3-1b-130b-dolma3-0625-150Bsample/step30995-hf"
OUTPUT_DIR="/tmp/eval-local-$(date +%Y%m%d-%H%M%S)"
TASKS=(
    "chembench:gen"
)
LIMIT=1000
BATCH_SIZE=4
MODEL_TYPE="hf"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --tasks)
            IFS=',' read -ra TASKS <<< "$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model PATH          Model path (default: OLMo3-1B dolma3 checkpoint)"
            echo "  --output-dir DIR      Output directory (default: /tmp/eval-local-TIMESTAMP)"
            echo "  --tasks TASKS         Comma-separated task list (default: chembench:gen)"
            echo "  --limit N             Number of examples (default: 10)"
            echo "  --batch-size N        Batch size (default: 4)"
            echo "  --model-type TYPE     Model type: hf or vllm (default: hf)"
            echo "  --help                Show this help"
            echo ""
            echo "Examples:"
            echo "  # Run chembench gen (default):"
            echo "  bash $0"
            echo ""
            echo "  # Run multiple tasks:"
            echo "  bash $0 --tasks chembench:gen,chembench:mc,gsm8k"
            echo ""
            echo "  # Run on specific model with more examples:"
            echo "  bash $0 --model /path/to/model --limit 100"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo "============================================"
echo "Running Local Evaluation"
echo "============================================"
echo ""
echo "Model:      $MODEL_PATH"
echo "Output:     $OUTPUT_DIR"
echo "Tasks:      ${TASKS[@]}"
echo "Limit:      $LIMIT examples per task"
echo "Batch size: $BATCH_SIZE"
echo "Model type: $MODEL_TYPE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation for each task
for TASK in "${TASKS[@]}"; do
    echo "============================================"
    echo "Running task: $TASK"
    echo "============================================"
    echo ""

    PYTHONPATH=. python -u src/scripts/eval/launch_eval.py \
        --model "$MODEL_PATH" \
        --model-type "$MODEL_TYPE" \
        --task "$TASK" \
        --limit "$LIMIT" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size "$BATCH_SIZE" \
        --fewshot-seed 1234 \
        --random-subsample-seed 1234

    echo ""
done

# Show results
echo "============================================"
echo "Results Summary"
echo "============================================"
echo ""

# Find and display metrics
METRICS_FILES=$(find "$OUTPUT_DIR" -name "*-metrics.json")

if [ -z "$METRICS_FILES" ]; then
    echo "No metrics files found!"
else
    python3 << EOF
import json
import glob

metrics_files = sorted(glob.glob("$OUTPUT_DIR/*-metrics.json"))

print("Task Results:")
print("-" * 80)

for f in metrics_files:
    with open(f) as file:
        data = json.load(file)
        task = data.get("task_name", "unknown")
        num_inst = data.get("num_instances", 0)
        metrics = data.get("metrics", {})
        primary = metrics.get("primary_score", "N/A")

        print(f"\n{task} ({num_inst} examples):")

        # Show key metrics
        for key in ["primary_score", "all_correct", "exact_match", "f1", "acc_raw", "acc_per_char"]:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, float):
                    print(f"  {key:20s}: {value:.4f}")
                else:
                    print(f"  {key:20s}: {value}")

print("\n" + "-" * 80)
print(f"\nFull results saved to: $OUTPUT_DIR")
EOF
fi

echo ""
echo "============================================"
echo "Done!"
echo "============================================"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "To view predictions:"
echo "  cat $OUTPUT_DIR/*-predictions.jsonl | jq"
echo ""
echo "To clean up:"
echo "  rm -rf $OUTPUT_DIR"
echo ""
