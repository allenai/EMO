# Run templates for scripts/

General templates for each type of run in this directory. The shared preamble (paths, launcher) lives in [`launch_common.sh`](launch_common.sh); every run script in `models_0116/` and `extensions/` sources it.

## Preamble — `launch_common.sh`

Source it from any script:

```bash
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"
```

`launch_common.sh` exports:

- **Paths** (override via env vars before sourcing):
  - `PREFIX` (default `/weka/oe-training-default/ryanwang/phdbrainstorm/Emo`) — root for outputs.
  - `MODELS_DIR` — derived from `PREFIX` (`${PREFIX}/models`).
  - `DATASET_CACHE` (default `/weka/oe-training-default/ryanwang/dataset-cache`) — tokenizer-mapped dataset cache.
- **`launch()` function** — wraps either `torchrun --nproc-per-node=${NPROC}` (default, `MODE=local`) or `python -m olmo_core.launch.beaker` with the `tylerr/olmo-core-tch280cu128-2025-11-25` image (when `MODE=beaker`). Call as: `launch <script.py> <run_name> [args...]`.

Switch a script to a beaker submission with:
```bash
MODE=beaker bash scripts/models_0116/dense_1b_lr-4e-3_0213.sh
```

Override cluster sizing per script if needed: `BEAKER_GPUS=8 BEAKER_NODES=4 ...` or set `BEAKER_GPUS=${gpus} BEAKER_NODES=${nodes}` at the top of the script (some scripts already define `gpus=` / `nodes=` for this purpose).

The hand-written templates below (templates 5 and 6) — the eval and pruning loops — are not yet adapted to source `launch_common.sh`; they're standalone references for `extensions/launch_eval.sh` and `pruning_hf/launch_pruning_hf.sh`.

---

## 1. Dense 1B pretraining

Pretrain a dense 1B model on `OLMoE-mix-0824`. Entry point: `src/scripts/train/olmo2-1B.py`.

```bash
lr=4e-3
runname="dense_1b_lr-${lr}_0213"

launch src/scripts/train/olmo2-1B.py $runname \
    --save-folder="${MODELS_DIR}/${runname}" \
    --dataset.mix=OLMoE-mix-0824 \
    --work-dir="${DATASET_CACHE}" \
    --trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
    --trainer.callbacks.wandb.enabled=true \
    --trainer.callbacks.wandb.entity=ryanyxw \
    --trainer.callbacks.wandb.project=olmoe-modular \
    --trainer.callbacks.wandb.name="${runname}" \
    --trainer.callbacks.wandb.tags='[pretraining]' \
    --dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
    --model.block.name="default" \
    --model.block.sequence_mixer.qk_norm=null \
    --lr=${lr}
```

## 2. MoE 1B/14B (single-level) pretraining

Pretrain a 128-expert single-level MoE. Entry point: `src/scripts/train/olmoe-1B-7B_fsl.py`.

```bash
lr=4e-4
lb=1e-1
runname="moe_1b14b_lb-1e-1_0118"

launch src/scripts/train/olmoe-1B-7B_fsl.py $runname \
    --save-folder="${MODELS_DIR}/${runname}" \
    --dataset.mix=OLMoE-mix-0824 \
    --work-dir="${DATASET_CACHE}" \
    --trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
    --trainer.callbacks.wandb.enabled=true \
    --trainer.callbacks.wandb.entity=ryanyxw \
    --trainer.callbacks.wandb.project=olmoe-modular \
    --trainer.callbacks.wandb.name="${runname}" \
    --trainer.callbacks.wandb.tags='[pretraining]' \
    --model-type="moe" \
    --model.block.feed_forward_moe.num_experts=128 \
    --dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
    --model.block.name="moe" \
    --model.block.sequence_mixer.qk_norm=null \
    --lr=${lr} \
    --model.block.feed_forward_moe.lb_loss_weight=${lb}
```

## 3. MoE 1B/14B two-level with shared-expert pool pretraining

Pretrain the `twolevelbatchlbreducedp_sharedexp...randpool` family. Adds `--document-expert-pool`, `--num_shared_experts*`, and a custom `--model-type`.

```bash
lr=4e-3
lb=1e-1
document_expert_pool=32
num_shared_experts_pool=4
num_shared_experts=2  # must be ≥2 so softmax gradients backprop

runname="twolevelbatchlbreducedp_sharedexp${num_shared_experts_pool}c${num_shared_experts}-${document_expert_pool}_1b14b_lr-${lr}_lb-${lb}_0214"

launch src/scripts/train/olmoe-1B-7B_fsl.py $runname \
    --save-folder="${MODELS_DIR}/${runname}" \
    --dataset.mix=OLMoE-mix-0824 \
    --work-dir="${DATASET_CACHE}" \
    --trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
    --trainer.callbacks.wandb.enabled=true \
    --trainer.callbacks.wandb.entity=ryanyxw \
    --trainer.callbacks.wandb.project=olmoe-modular \
    --trainer.callbacks.wandb.name="${runname}" \
    --trainer.callbacks.wandb.tags='[pretraining]' \
    --model.block.feed_forward_moe.num_experts=128 \
    --dataset.generate_doc_lengths=true \
    --model.block.sequence_mixer.backend=flash_2 \
    --model-type="two-level_lb-batch_reduce-dp_sharedexppool" \
    --document-expert-pool=${document_expert_pool} \
    --num_shared_experts=${num_shared_experts} \
    --num_shared_experts_pool=${num_shared_experts_pool} \
    --dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
    --model.block.name="moe" \
    --model.block.sequence_mixer.qk_norm=null \
    --lr=${lr} \
    --model.block.feed_forward_moe.lb_loss_weight=${lb}
```

Model-type variants (pick one `--model-type`):
- `moe` — single-level
- `two-level_lb-batch_reduce-dp` — two-level, no shared experts
- `two-level_lb-batch_reduce-dp_sharedexppool` — two-level with fixed shared-expert pool
- `two-level_lb-batch_reduce-dp_sharedexp_randpool` — two-level with random shared-expert pool sampling (used in the anneal runs)

## 4. Continual pretraining / extension

Load an existing checkpoint and continue training on a domain mix (e.g. `mj_finemath4plus`). Entry point: `src/scripts/train/olmoe-1B-7B_fsl_extension.py`.

```bash
lr=4e-4
lb=1e-1
min_document_expert_pool=8
max_document_expert_pool=8
eval_document_expert_pool=32
num_shared_experts=1

num_billion_tokens=10
num_tokens=$((num_billion_tokens * 1000000000))

base_model_path="${MODELS_DIR}/twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419/step250339"
runname="twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419_ct-math_8"

launch src/scripts/train/olmoe-1B-7B_fsl_extension.py $runname \
    --save-folder="${MODELS_DIR}/${runname}" \
    --dataset.mix=mj_finemath4plus \
    --work-dir="${DATASET_CACHE}" \
    --trainer.callbacks.wandb.enabled=true \
    --trainer.callbacks.wandb.entity=ryanyxw \
    --trainer.callbacks.wandb.project=olmoe-modular \
    --trainer.callbacks.wandb.name="${runname}" \
    --trainer.callbacks.wandb.tags='[extension, contpretrain, finemath]' \
    --num-tokens=${num_tokens} \
    --lr=${lr} \
    --load-path=${base_model_path}/model_and_optim \
    --model.block.feed_forward_moe.num_experts=128 \
    --model-type="two-level_lb-batch_reduce-dp_sharedexp_randpool" \
    --min_document_expert_pool=${min_document_expert_pool} \
    --max_document_expert_pool=${max_document_expert_pool} \
    --eval_document_expert_pool=${eval_document_expert_pool} \
    --num_shared_experts=${num_shared_experts} \
    --dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
    --model.block.name="moe" \
    --model.block.sequence_mixer.qk_norm=null \
    --model.block.feed_forward_moe.lb_loss_weight=${lb}
```

Notes:
- `--load-path` points at `.../step<N>/model_and_optim` (not `step<N>-hf`).
- `load_trainer_state=False` (default) resets the step counter; pair with a fresh LR / schedule.

## 5. Eval (no training)

For each (model, task) pair, run OLMES eval on an HF checkpoint and upload results to S3. Loop replicates `extensions/launch_eval.sh` with torchrun instead of `python -m olmo_core.launch.beaker`.

```bash
S3_BASE="s3://ai2-sewonm/ryanwang/extension_evals_0414"

MODELS=(
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419/step250339-hf"
    # add more here
)

TASKS=(
    "arc_easy:rc_test::olmes"
    "arc_challenge:rc_test::olmes"
    "hellaswag:rc_test::olmes"
    "squad::olmes"
    "gsm8k::olmes"
    "minerva_math_500::olmes"
    "mbpp:3shot:bpb::none"
    "codex_humaneval:3shot:bpb::none"
)

for MODEL in "${MODELS[@]}"; do
    stringified_model=$(echo ${MODEL} | sed 's/[^a-zA-Z0-9_-]//g')
    for TASK in "${TASKS[@]}"; do
        # Heuristic GPU / batch-size overrides
        batch_size=16
        gpus=${NPROC}
        if [[ ${TASK} == *mmlu* || ${TASK} == *agi_eval* || ${TASK} == *bbh* || ${TASK} == *gsm8k* || ${TASK} == *minerva_math_* || ${TASK} == *codex* || ${TASK} == *mbpp* ]]; then
            gpus=$(( NPROC < 4 ? NPROC : 4 ))
        fi
        if [[ ${TASK} == *"cot"* || ${TASK} == *"minerva_math_"* || ${TASK} == *"mbpp"* || ${TASK} == *"bigcodebench"* || ${TASK} == *"boolq"* ]]; then
            batch_size=$((batch_size / 4))
        fi

        safe_task=$(echo "${TASK}" | sed 's/[^a-zA-Z0-9_-]//g')
        relative_dir="${stringified_model}/${safe_task}"
        s3_output_dir="${S3_BASE}/${relative_dir}"

        aws s3 rm --recursive --quiet "${s3_output_dir}/" || true

        torchrun --nproc-per-node=${gpus} -m src.scripts.eval.launch_eval \
            --model-path="${MODELS_DIR}/${MODEL}" \
            --task="${TASK}" \
            --batch-size=${batch_size} \
            --output-dir="${EVAL_OUTPUT_DIR}/${relative_dir}" \
            --s3-output="${s3_output_dir}"
    done
done
```

## 6. Pruning + task-specific finetuning

For each (model, keep-k, task), compute router activations → prune to top-k experts → finetune. Worker: `scripts/pruning_hf/hf_finetune_with_pruning.sh`.

```bash
PRUNING_MODE="layerwise"   # global | layerwise | layerwise_variable | easy_ep
PRUNE_KEEP_K_VALUES=(8 16 32 64)
num_epochs=1
batch_size=32
lr=5e-5

MODELS=(
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419/step250339-hf"
    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_1T_0322_anneal_from_step238419/step250339-hf"
)

TASKS=(
    "gsm8k_generation_0shot"
    "mmlu_math"
    "codex_humaneval"
)

for MODEL in "${MODELS[@]}"; do
    # Infer num_shared_experts from the model-family substring
    if [[ ${MODEL} == *"twolevelbatchlbreducedp512sharedexp1"* || ${MODEL} == *"moereducedp512sharedexp1"* ]]; then
        num_shared_experts=1
    elif [[ ${MODEL} == *"twolevelbatchlbreducedp512sharedexp2"* || ${MODEL} == *"twolevelbatchlbreducedp512sharedexp4c2"* ]]; then
        num_shared_experts=2
    else
        num_shared_experts=0
    fi

    stringified_model=$(echo ${MODEL} | sed 's/[^a-zA-Z0-9_-]//g')

    for prune_keep_k in "${PRUNE_KEEP_K_VALUES[@]}"; do
        for TASK in "${TASKS[@]}"; do
            micro_batch_size=8
            gpus=${NPROC}

            relative_dir="${stringified_model}/${TASK}_keepk_${prune_keep_k}_bs-${batch_size}_lr-${lr}_epoch-${num_epochs}_prunemode-${PRUNING_MODE}"

            bash scripts/pruning_hf/hf_finetune_with_pruning.sh \
                --model "${MODEL}" \
                --task "${TASK}" \
                --prune-keep-k ${prune_keep_k} \
                --num-shared-experts ${num_shared_experts} \
                --relative-dir "${relative_dir}" \
                --base-dir "${PREFIX}" \
                --num-gpus ${gpus} \
                --num-epochs ${num_epochs} \
                --batch-size ${batch_size} \
                --micro-batch-size ${micro_batch_size} \
                --learning-rate ${lr}
        done
    done
done
```

Pruning-mode variants (pick one `PRUNING_MODE`):
- `global` — single-pass activation collection + top-k prune across the whole model
- `layerwise` — greedy layer-by-layer pruning (each layer conditioned on earlier pruned layers)
- `layerwise_variable` — greedy layerwise with a per-layer keep-k schedule (`KEEP_K_PER_LAYER=128,128,32,...`)
- `easy_ep` — EASY-EP (arXiv 2504.06792): domain-specific one-shot prune on calibration data

Optional calibration-size control:
- `NUM_PRUNE_EXAMPLES=""` — use the full validation pool
- `NUM_PRUNE_EXAMPLES=50` — subsample N prompts (deterministic, seed=0)
- `NUM_PRUNE_EXAMPLES="random"` — skip calibration entirely and pick experts at random (output dir gets `_prunemode-random`)

---

## Model family → naming cheatsheet

Pieces that appear in `runname`:
- `dense_1b` / `moe_1b14b` / `moe_1b35b` / `moe_1b4b` — size / expert family
- `twolevelbatchlb-<N>` — two-level router with batch-level LB, document-expert-pool `N`
- `reducedp<K>` — reduced data-parallel degree to `K` (e.g. `reducedp512`)
- `sharedexp<P>c<S>` — shared-expert pool of `P` experts, `S` of them active per token (`c` = "chosen")
- `sharedexp<S>randpool-<min>-<max>[eval<E>]` — single shared expert, randomized pool sampled in `[min, max]` during train, fixed to `E` at eval
- `lr-<X>`, `lb-<Y>` — learning rate, load-balance loss weight
- `_0213`, `_0308`, etc. — date tag (MMDD)
- `_1T` — 1-trillion-token schedule
- `_anneal` / `_anneal_from_step<N>` — annealing phase, optionally from an intermediate step
- `_ct-<data>_<K>` — continual-training on `<data>` with extension config key `<K>` (e.g. `m8`, `math_8`)

Use these to build a new `runname` that matches the convention. Keep the parent script's name in a `# PARENT:` header comment so provenance stays traceable.
