#!/bin/bash

# Script to launch beaker evaluations for all available tasks
# Usage: bash src/scripts/eval/launch_beaker_eval.sh

# Configuration
MODELS=(
    # # need: general model, pretrained model (hf versions)
    # "/data/input/kevinf/checkpoints/olmo3-1b-10B-chempile-papers_education_lift/step2385-hf"
    # "/data/input/kevinf/checkpoints/olmo3-1b-10B-chempile-papers_education_lift-ckpt_1B_dolma3/step2385-hf"
    # "/data/input/kevinf/checkpoints/olmo3-1b-10B-chempile-papers_education_lift-ckpt_1B_dolma3/step2385-hf"
    # "/data/input/kevinf/checkpoints/olmo3-1b-130b-OLMo-mix-0625-150Bsample-dclm/step30995-hf"
    # "/data/input/kevinf/checkpoints/olmo3-1b-10B-chempile-papers_education_lift-continued_pt_30B_from_130B_dolma3/step7153-hf"
    # "/data/input/kevinf/checkpoints/new-kevinf-olmo3-1b-130b-dolma3-0625-150Bsample/step30995-hf"
    # "/data/input/kevinf/checkpoints/new-kevinf-olmo3-1b-130b-olmoemix-0824/step30995-hf"
    # "/data/input/kevinf/checkpoints/olmo3-1b-the-pile-of-law-10B-lr5e-5-warmup0.1-ctd/step2385-hf"
    # "/data/input/kevinf/checkpoints/new-kevinf-olmo3-1b-130b-dolma3-0625-150Bsample/step30995-hf"
    # "/data/input/kevinf/checkpoints/new-kevinf-olmo3-1b-130b-olmoemix-0824/step30995-hf"
    # "/data/input/kevinf/checkpoints/olmo3-1b-chempile-10B-lr2e-4-warmup715-ctd/step2385-hf"
    # "/data/input/kevinf/checkpoints/olmo3-1b-chempile-10B-lr5e-5-warmup715-ctd/step2385-hf"
    # "/data/input/kevinf/checkpoints/olmo3-1b-croissant-10B-lr5e-5-warmup0.1-ctd/step2385-hf/"
    # "/data/input/kevinf/checkpoints/olmo3-1b-croissant-10B-lr5e-5-warmup0.1-ctd/step2385-hf"
    # "/data/input/kevinf/checkpoints/new-kevinf-olmo3-1b-130b-dolma3-0625-150Bsample/step30995-hf"
    # "/data/input/kevinf/checkpoints/new-kevinf-olmo3-1b-130b-olmoemix-0824/step30995-hf"
    # "/data/input/kevinf/checkpoints/olmo3-1b-chempile-10B-lr5e-5-warmup0.1-ctd/step2385-hf"
    # "/data/input/kevinf/checkpoints/olmo3-1b-the-pile-of-law-10B-lr5e-5-warmup0.1-ctd"/step2385-hf
    # "/data/input/kevinf/checkpoints/olmo3-1b-croissant-10B-lr5e-5-warmup0.1-ctd/step2385-hf"
    # "/data/input/kevinf/checkpoints/olmo3-1b-pmc-30B-lr5e-5-warmup0.1-ctd/step7153-hf"
    # "/data/input/kevinf/checkpoints/olmo3-1b-mimic-iv-note-2B-lr5e-5-warmup0.1-ctd/step477-hf"
    # "/data/input/kevinf/checkpoints/olmo3-1b-medical-o1-en-cot-0.1B-lr5e-5-warmup0.1-ctd/step24-hf"
    # "/data/input/kevinf/checkpoints/olmo3-1b-medical-o1-en-nocot-0.1B-lr5e-5-warmup0.1-ctd/step48-hf/"
    # # dolma2 code mixes
    # "/data/input/kevinf/checkpoints/train-olmo3-1b-dolma2-code-java-10B-lr5e-5-eval-on-java-only-new-ctd/step2385-hf"
    # "/data/input/kevinf/checkpoints/train-olmo3-1b-dolma2-code-python-10B-lr5e-5-eval-on-python-only-new-ctd/step2385-hf"
    # # stack-v2 code mixes (quality p75+)
    # "/data/input/kevinf/checkpoints/train-olmo3-1b-stack-v2-python-p75-10B-lr5e-5-ctd/step2385-hf"
    # "/data/input/kevinf/checkpoints/train-olmo3-1b-stack-v2-cpp-p75-10B-lr5e-5-ctd/step2385-hf"
    # "/data/input/kevinf/checkpoints/train-olmo3-1b-stack-v2-java-p75-10B-lr5e-5-ctd/step2385-hf"
    # # other code models (for reference)
    # "/data/input/kevinf/checkpoints/train-olmo3-1b-dolma2-code-java-10B-lr5e-5-eval-on-java-only-new-ctd/step2385-hf"
    # "/data/input/kevinf/checkpoints/train-olmo3-1b-dolma2-code-python-10B-lr5e-5-eval-on-python-only-new-ctd/step2385-hf"
    # "/data/input/kevinf/checkpoints/olmo3-1b-dolma2-code-python-10B-lr5e-5-warmup0.1-ctd/step2385-hf"
    # "/data/input/kevinf/checkpoints/olmo3-1b-dolma2-code-java-10B-lr5e-5-warmup0.1-ctd/step2385-hf"
    # "/data/input/kevinf/checkpoints/olmo3-1b-dolma2-code-cpp-10B-lr5e-5-warmup0.1-ctd/step2385-hf"
    # "/data/input/kevinf/checkpoints/train-olmo3-1b-stack-v2-cpp-p75-10B-lr5e-5-ctd/step2385-hf"
    # "/data/input/kevinf/checkpoints/train-olmo3-1b-stack-v2-java-p75-10B-lr5e-5-ctd/step2385-hf"
    # "/data/input/kevinf/checkpoints/olmo3-1b-code_fim_python-2B-lr5e-5-warmup0.1-pplx-raw-ctd/step477-hf"
    # "/data/input/kevinf/checkpoints/olmo3-1b-code_fim_cpp-2B-lr5e-5-warmup0.1-pplx-raw-ctd/step477-hf"
    # "/data/input/kevinf/checkpoints/olmo3-1b-code_fim_java-2B-lr5e-5-warmup0.1-pplx-raw-ctd/step477-hf"
    # "/data/input/kevinf/checkpoints/train-olmo3-1b-sponge-code-prose-p75-10B-lr5e-5-ctd/step2385-hf"
    
    # code_fresh_rolling:bpb active models (verified)
    "/data/input/kevinf/checkpoints/new-kevinf-olmo3-1b-130b-dolma3-0625-150Bsample/step30995-hf"
    "/data/input/kevinf/checkpoints/olmo3-1b-dolma2-code-python-10B-lr5e-5-warmup0.1-ctd/step2385-hf"
    "/data/input/kevinf/checkpoints/train-olmo3-1b-stack-v2-python-p75-10B-lr5e-5-ctd/step2385-hf"
    "/data/input/kevinf/checkpoints/train-olmo3-1b-sponge-code-prose-p75-10B-lr5e-5-ctd/step2385-hf"
    "/data/input/kevinf/checkpoints/olmo3-1b-code_fim_python-2B-lr5e-5-warmup0.1-pplx-raw-ctd/step477-hf"
)

BASE_OUTPUT_DIR="/data/input/kevinf/eval_results/flexmoe"
BATCH_SIZE=4
CLUSTER="ai2/saturn"
LIMIT=1000
model_type=hf

# Define all available tasks from run_eval.sh (ALL tasks from all groups)
TASKS=(
    # code_fresh rolling BPB (all 42 languages)
    code_fresh_rolling:bpb

    # # # MC9 tasks
    # arc_easy:mc::olmes
    # arc_challenge:mc::olmes
    # boolq:mc::olmes
    # csqa:mc::olmes
    # hellaswag:mc::olmes
    # openbookqa:mc::olmes
    # piqa:mc::olmes
    # socialiqa:mc::olmes
    # winogrande:mc::olmes

    # # # Gen5 tasks
    # coqa::olmes
    # squad::olmes
    # naturalqs::olmes
    # triviaqa::olmes
    # drop::olmes

    # # # MMLU tasks
    # mmlu:mc::olmes
    # mmlu_pro:mc::none

    # # # AGI eval
    # agi_eval_english:1shot::olmes

    # # # BBH
    # # bbh:cot-v1::olmes

    # # # Math2 tasks
    # gsm8k::olmes
    # minerva_math_algebra::olmes
    # minerva_math_counting_and_probability::olmes
    # minerva_math_geometry::olmes
    # minerva_math_intermediate_algebra::olmes
    # minerva_math_number_theory::olmes
    # minerva_math_prealgebra::olmes
    # minerva_math_precalculus::olmes

    # # Code4 tasks
    # codex_humaneval:temp0.8
    # codex_humanevalplus:temp0.8
    # mbpp::none
    # mbppplus::none

    # # ChemBench MC and generative tasks
    # chembench:mc
    # chembench:gen
    # chembench:rc
    # frenchbench:rc
    # legalbench:rc

    # medqa
    # medmcqa:mc

    # mt_mbpp
)

# Function to get checkpoint name - extracts run name and step from path
function get_checkpoint_name {
    local path=$1
    # Get last two path components (run name and step)
    local step_dir=$(basename "$path")
    local run_name=$(basename "$(dirname "$path")")
    echo "${run_name}_${step_dir}"
}

echo "Launching beaker evaluations for ${#MODELS[@]} models and ${#TASKS[@]} tasks..."
echo "Models: ${MODELS[@]}"
echo "Base output directory: $BASE_OUTPUT_DIR"
echo "Cluster: $CLUSTER"
echo ""

# Launch evaluation for each model and task combination
for MODEL_PATH in "${MODELS[@]}"; do
    echo "Processing model: $MODEL_PATH"
    
    # For setting the output_dir (matching original script logic)
    if [[ $MODEL_PATH == "/"* ]]; then
        # internal model
        model=$(get_checkpoint_name $MODEL_PATH)
    else
        # HF model
        model=$(echo $MODEL_PATH | cut -d'/' -f2)
    fi
    
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/$model"
    
    for TASK in "${TASKS[@]}"; do
        echo "Launching evaluation for model: $model, task: $TASK"
    
    gpus=1
    
    # Batch size adjustment (matching original script)
    if [[ $TASK == *"cot"* || $TASK == "minerva_math_"* || $TASK == "mbpp"* || $TASK == "bigcodebench"* || $TASK == "ruler"* || $TASK == "sciriff"* || $TASK == *"code_fresh"* ]]; then
        batch_size=1
    else
        batch_size=4
    fi
    
    # Create job name - remove invalid characters only
    safe_model_name=$(echo $model | sed 's/[^a-zA-Z0-9_-]//g')
    safe_task_name=$(echo $TASK | sed 's/[^a-zA-Z0-9_-]//g')
    job_name="eval-${safe_model_name}-${safe_task_name}"
    
    echo "  Model name: $model"
    echo "  Output dir: $OUTPUT_DIR"
    echo "  GPUs: $gpus"
    echo "  Batch size: $batch_size"
    echo "  Job name: $job_name"
    
    uv run gantry run \
        --name $job_name \
        --weka oe-training-default:/data/input \
        --install "pip install setuptools uv && UV_CACHE_DIR=/tmp/uv-cache uv pip install -e '.[eval]'" \
        --budget ai2/oceo \
        --workspace ai2/flex2 \
        --cluster $CLUSTER \
        --priority urgent \
        --gpus $gpus \
        --env-secret HF_TOKEN=KEVINF_HF_TOKEN \
        --env-secret AWS_ACCESS_KEY_ID=KEVINF_AWS_ACCESS_KEY_ID \
        --env-secret AWS_SECRET_ACCESS_KEY=KEVINF_AWS_SECRET_ACCESS_KEY \
        --allow-dirty \
        -- \
        bash -c "PYTHONPATH=. python -u src/scripts/eval/launch_eval.py \
            --model $MODEL_PATH \
            --model-type hf \
            --model-args logits_cache=false \
            --task $TASK \
            --limit $LIMIT \
            --output-dir $OUTPUT_DIR \
            --batch-size $batch_size \
            --gpus $gpus \
            --fewshot-seed 4321 \
            --random-subsample-seed 4321 \
            "
    
        echo "Launched evaluation for model: $model, task: $TASK"
        echo "----------------------------------------"
    done
    
    echo "Completed all tasks for model: $model"
    echo "========================================"
done

echo "All beaker evaluations have been launched!"
echo "Total jobs: $((${#MODELS[@]} * ${#TASKS[@]}))"
echo "Check the beaker dashboard for job status."

 # test without && pip install 'fsspec>=2024.2.0,<=2025.3.0' 's3fs>=2024.2.0,<=2025.3.0'
