---
name: launch-beaker
description: Launch eval or training jobs on Beaker. Use when the user asks to run evals, training, or any job on Beaker/Gantry.
---

# Launch on Beaker

Launch evaluation or training jobs on Beaker: $ARGUMENTS

Parse the user's request to determine the job type (eval or training) and parameters.

## 1. Eval Jobs (via Gantry)

Reference script: `scripts/kevinf/eval/launch.sh`

Required info (ask user if not provided):
- **Model path(s)**: local checkpoint path (e.g., `/data/input/kevinf/checkpoints/...`) or HF model name
- **Task(s)**: task specs like `chembench:mc`, `arc_easy:mc::olmes`, `legalbench:rc`, or task suites
- **Limit**: number of eval instances (default: 1000, use 5 for quick tests)

Defaults (override if user specifies):
- `CLUSTER=ai2/saturn`
- `BASE_OUTPUT_DIR=/data/input/kevinf/flexmoe/eval/results`
- `batch_size=4` (use 1 for: gen tasks, cot, minerva_math, mbpp, bigcodebench, ruler, sciriff)
- `gpus=1`

### Deriving model name for output dir
```bash
# For local paths: combine run_name + step_dir
step_dir=$(basename "$MODEL_PATH")
run_name=$(basename "$(dirname "$MODEL_PATH")")
model="${run_name}_${step_dir}"

# For HF models: use the model name after /
model=$(echo $MODEL_PATH | cut -d'/' -f2)
```

### Launch command template
```bash
source .venv/bin/activate

uv run gantry run \
    --name eval-${safe_model}-${safe_task} \
    --weka oe-training-default:/data/input \
    --install "pip install uv && UV_CACHE_DIR=/tmp/uv-cache uv pip install -e '.[eval]'" \
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
        --task $TASK \
        --limit $LIMIT \
        --output-dir $OUTPUT_DIR \
        --batch-size $batch_size \
        --gpus $gpus \
        --fewshot-seed 1234 \
        --random-subsample-seed 1234 \
        "
```

### Important: commit and push first!
Beaker pulls code from git. Before launching, ensure all relevant Python source changes are committed and pushed. Check with `git status` and prompt the user if there are uncommitted changes to `src/` files.

## 2. Training Jobs (via olmo_core.launch.beaker)

Reference script: `scripts/kevinf/train/launch_croissant.sh`

Required info (ask user if not provided):
- **Run name**: descriptive name for the training run
- **Training script**: path to training script (e.g., `src/scripts/kevinf/train/OLMo3-1B.py`)
- **Dataset mix**: dataset name (e.g., `croissant`, `chempile`, `the-pile-of-law`)
- **Training tokens**: in billions (e.g., 10)
- **Learning rate**: (e.g., `5e-5`)

Optional:
- **Load path**: checkpoint to continue from (e.g., `/weka/oe-training-default/kevinf/checkpoints-new/.../step30995`)
- **Warmup fraction**: default 0.1
- **Cluster**: default `ai2/jupiter` for training (8 GPUs)

### Launch command template
```bash
python -m olmo_core.launch.beaker \
    --name $runname \
    --gpus 8 \
    --nodes 1 \
    --weka=oe-training-default \
    --is_private_repo \
    --priority urgent \
    --shared-filesystem \
    --workspace ai2/flex2 \
    --cluster ai2/jupiter \
    --preemptible \
    --allow-dirty \
    --env-secret "GITHUB_TOKEN=KEVINF_GITHUB_TOKEN" "WANDB_API_KEY=KEVINF_WANDB_API_KEY" "BEAKER_TOKEN=KEVINF_BEAKER_TOKEN" \
    -- $TRAIN_SCRIPT \
    $runname \
    --save-folder="/weka/oe-training-default/kevinf/checkpoints/${runname}/" \
    --work-dir="/weka/oe-training-default/kevinf/dataset-cache" \
    --trainer.max_duration="{value: ${train_tokens_raw}, unit: tokens}" \
    --trainer.hard_stop="{value: ${train_tokens_raw}, unit: tokens}" \
    --dataset.mix=$dataset \
    --train_module.optim.lr=$lr \
    ${load_path:+--load_path=$load_path}
```

## 3. After Launching

- Report the Beaker experiment URL(s) to the user
- If multiple jobs, list them in a table with task/model/experiment URL
- Suggest using `/monitor-experiment <experiment-id>` to track progress

## Common Task Suites for Evals

| Suite | Description |
|-------|-------------|
| `chembench:mc`, `chembench:rc`, `chembench:gen` | Chemistry benchmark |
| `legalbench:rc` | Legal reasoning |
| `frenchbench:rc` | French language |
| `sciriff5` | Scientific reasoning |
| Standard OLMES: `arc_easy:mc::olmes`, `hellaswag:mc::olmes`, `mmlu:mc::olmes`, etc. | General benchmarks |

## Key Differences: Eval vs Training

| | Eval | Training |
|---|------|----------|
| Launcher | `uv run gantry run` | `python -m olmo_core.launch.beaker` |
| Cluster | `ai2/saturn` (1 GPU) | `ai2/jupiter` (8 GPUs) |
| Weka mount | `--weka oe-training-default:/data/input` | `--weka=oe-training-default` |
| Secrets | HF_TOKEN, AWS keys | GITHUB_TOKEN, WANDB, BEAKER_TOKEN |
| Preemptible | No | Yes (with `--preemptible`) |
