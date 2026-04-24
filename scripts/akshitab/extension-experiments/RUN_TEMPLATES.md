# Extension experiment run templates

General templates for each type of extension-experiment run in this directory. All paths are parametrized by a single `${PREFIX}` variable; set it once per environment (e.g. `/weka/oe-training-default/akshitab/FlexMoE` on Ai2 weka, or `/path/to/local/FlexMoE` locally). All templates use `torchrun` for local / single-node launches — swap in `python -m olmo_core.launch.beaker ... --` to submit to Beaker instead.

## Preamble (put at the top of every run script)

```bash
# Root for model checkpoints and dataset cache. Override via env var.
PREFIX=${PREFIX:-/path/to/FlexMoE}

BASE_MODELS="${PREFIX}/base_models"        # pretrained 128-expert bases (from upstream)
EXTENSIONS="${PREFIX}/models/extensions"   # pre-training extensions (new experts added, not yet trained)
MODELS="${PREFIX}/models"                  # trained run outputs
DATASET_CACHE="${PREFIX}/dataset-cache"    # tokenizer-mapped dataset cache

# Launcher: torchrun on a single node with ${NPROC} GPUs.
# Swap with `python -m olmo_core.launch.beaker --name ${RUN_NAME} <beaker-flags> --`
# for cluster runs.
NPROC=${NPROC:-8}
LAUNCH="torchrun --nproc-per-node=${NPROC}"
```

## Regular MoE base model

All regular-MoE extensions start from:

```bash
REGULAR_BASE="${BASE_MODELS}/moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_0308/step30995"
```

## Twolevel MoE base model

```bash
TWOLEVEL_BASE="${BASE_MODELS}/twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301/step30995"
```

---

## 1. Initialization + train new experts (regular MoE)

Add `${NUM_NEW_EXPERTS}` new experts to the 128-expert base, then freeze everything except the new experts and train on a domain dataset.

**Part 1 — add experts (run once, locally):**

```bash
NUM_NEW_EXPERTS=4
TOTAL_EXPERTS=$((128 + NUM_NEW_EXPERTS))

NEW_BASE_MODEL_PATH="${EXTENSIONS}/moereducedp512sharedexp1_1b14b_${TOTAL_EXPERTS}experts_0308_step30995_init_top2_math_average_noise"
EVAL_DIR="s3://ai2-sewonm/akshitab/mose/evals/extensions/moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_0308_step30995-hf"

python src/scripts/akshitab/add_finegrained_expert/add_new_expert.py \
    -c ${REGULAR_BASE} \
    -o ${NEW_BASE_MODEL_PATH} \
    --num_new_experts ${NUM_NEW_EXPERTS} \
    --init_method similar \
    --activation_file ${EVAL_DIR}/task-gsm8k_generation_test_0shot-router.jsonl \
    -k 2 \
    --noise_std_fraction 0.1
```

Alternative `--init_method` values: `average`, `random_expert`, `similar` (with `-k 2`), `similar` with `--noise_std_fraction 0.1`.

**Part 2 — train new experts:**

```bash
NUM_BILLION_TOKENS=10
NUM_TOKENS=$((NUM_BILLION_TOKENS * 1000000000))
LR=4e-4
DATA_MIX=mj_finemath4plus   # math: mj_finemath4plus | code: code_mix / starcoder_mix | french: croissant

RUN_NAME="freeze-fix-moe1b14b_${TOTAL_EXPERTS}experts_${NUM_NEW_EXPERTS}trained_math_init_top2_average_noise_${NUM_BILLION_TOKENS}B_lr_${LR}"

${LAUNCH} src/scripts/akshitab/add_finegrained_expert/train_new_expert.py \
    ${RUN_NAME} \
    --trainer.load_path="${NEW_BASE_MODEL_PATH}/model_and_optim" \
    --save-folder="${MODELS}/${RUN_NAME}" \
    --dataset.mix=${DATA_MIX} \
    --work-dir="${DATASET_CACHE}" \
    --trainer.max_duration="{value: ${NUM_TOKENS}, unit: tokens}" \
    --trainer.callbacks.wandb.enabled=true \
    --trainer.callbacks.wandb.entity=akshitab \
    --trainer.callbacks.wandb.project=olmoe-modular \
    --trainer.callbacks.wandb.name="${RUN_NAME}" \
    --trainer.callbacks.wandb.tags='[extension]' \
    --dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
    --model.block.name="moe" \
    --model.block.sequence_mixer.qk_norm=null \
    --model.block.feed_forward_moe.lb_loss_weight=1e-2 \
    --train_module.scheduler.warmup_fraction=0.1 \
    --lr=${LR} \
    --num-experts-to-train=${NUM_NEW_EXPERTS}
```

Variants:
- `--init_method` drives the `_init_*` substring in `RUN_NAME`.
- To freeze the router too, swap `train_new_expert.py` → `train_new_expert_no_router.py` and add `_no_router` to `RUN_NAME`.

## 2. Num experts / tokens grid

Same as template 1 with `init_top2_average` / `init_top2_average_noise`, sweeping `NUM_NEW_EXPERTS ∈ {1, 2, 4, 8}` and `NUM_BILLION_TOKENS ∈ {5, 10, 20}`.

## 3. Selective training (train specific expert indices)

Train specific existing experts on a domain dataset without adding any new experts.

```bash
EXPERTS_TO_TRAIN=69,30,3,6   # e.g. top-4 math experts from router analysis
TOTAL_EXPERTS=128

NUM_BILLION_TOKENS=10
NUM_TOKENS=$((NUM_BILLION_TOKENS * 1000000000))
LR=4e-4

RUN_NAME="moereducedp512sharedexp1_1b14b_${TOTAL_EXPERTS}experts_${EXPERTS_TO_TRAIN//,/_}_trained_math_${NUM_BILLION_TOKENS}B_lr_${LR}"

${LAUNCH} src/scripts/akshitab/add_finegrained_expert/train_selected_experts.py \
    ${RUN_NAME} \
    --trainer.load_path="${REGULAR_BASE}/model_and_optim" \
    --save-folder="${MODELS}/${RUN_NAME}" \
    --dataset.mix=mj_finemath4plus \
    --work-dir="${DATASET_CACHE}" \
    --trainer.max_duration="{value: ${NUM_TOKENS}, unit: tokens}" \
    --trainer.callbacks.wandb.enabled=true \
    --trainer.callbacks.wandb.entity=akshitab \
    --trainer.callbacks.wandb.project=olmoe-modular \
    --trainer.callbacks.wandb.name="${RUN_NAME}" \
    --trainer.callbacks.wandb.tags='[extension]' \
    --dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
    --model.block.feed_forward_moe.lb_loss_weight=1e-2 \
    --train_module.scheduler.warmup_fraction=0.1 \
    --lr=${LR} \
    --base-model-config="${REGULAR_BASE}" \
    --experts-to-train=${EXPERTS_TO_TRAIN}
```

Pick `EXPERTS_TO_TRAIN` via `python src/scripts/eval/router_analysis.py --router-files <task-router.jsonl>`.

## 4. Baselines (full-finetune / all-experts-trained / freeze-embeddings-only)

Same as template 3, but set `EXPERTS_TO_TRAIN` to all experts:

```bash
TOTAL_EXPERTS=128
EXPERTS_TO_TRAIN=$(seq -s, 0 $((TOTAL_EXPERTS - 1)))   # all experts

RUN_NAME="moereducedp512sharedexp1_1b14b_${TOTAL_EXPERTS}experts_all_trained_math_${NUM_BILLION_TOKENS}B_lr_${LR}"
# or: ..._full_finetune_math_...   (also add --model.freeze_params='[]')
# or: ..._freeze_emb_only_math_... (freeze only embeddings)
```

Full-finetune additionally sets `--model.freeze_params='[]'`.

## 5. Shared / always-active expert

Same as template 1 (part 2), but launch with `train_new_expert_always_active.py` and set `--always-active-experts`.

```bash
NUM_NEW_EXPERTS=4
SHARED_EXPERTS=56  # expert index to keep always active, e.g. top global expert
TOTAL_EXPERTS=$((128 + NUM_NEW_EXPERTS))

RUN_NAME="ff-moe1b14b_${TOTAL_EXPERTS}experts_${NUM_NEW_EXPERTS}trained_sharedexp${SHARED_EXPERTS}math_init_top2_average_${NUM_BILLION_TOKENS}B_lr_${LR}"

${LAUNCH} src/scripts/akshitab/add_finegrained_expert/train_new_expert_always_active.py \
    ${RUN_NAME} \
    ... \
    --num-experts-to-train=${NUM_NEW_EXPERTS} \
    --always-active-experts=${SHARED_EXPERTS}
```

## 6. Merging extensions from multiple domains

Merge new experts from two domain-trained extension runs (e.g. 4 math + 4 code experts) into a single 136-expert model. Runs locally — no training.

```bash
NUM_NEW_EXPERTS=8   # 4 math + 4 code
TOTAL_EXPERTS=$((128 + NUM_NEW_EXPERTS))

MATH_MODEL="${MODELS}/moereducedp512sharedexp1_132experts_4trained_math_init_top2_average_10B_lr_4e-4/step2385"
CODE_MODEL="${MODELS}/moereducedp512sharedexp1_132experts_4trained_code_mix_init_top2_average_noise_10B_lr_4e-4/step2385"

SAVE_PATH="${MODELS}/merged_moereducedp512sharedexp1_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise"

python src/scripts/akshitab/add_finegrained_expert/merge_experts.py \
    -b ${REGULAR_BASE} \
    -m ${MATH_MODEL} ${CODE_MODEL} \
    -e 128 129 130 131 -e 128 129 130 131 \
    -o ${SAVE_PATH}
```

For weight-space merging (no router reassignment) use `weight_merge_moe_models.py`; run name gets a `weight_merge_` prefix instead of `merged_`.

## 7. Router training on merged model

After merging, do a short (~1B-token) router-only finetune on a mixed dataset.

```bash
NUM_NEW_EXPERTS=8
TOTAL_EXPERTS=$((128 + NUM_NEW_EXPERTS))

MERGED_MODEL_PATH="${MODELS}/merged_moe_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise"

NUM_BILLION_TOKENS=1
NUM_TOKENS=$((NUM_BILLION_TOKENS * 1000000000))
LR=4e-4

RUN_NAME="rt-realdata-merged_moe_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise_${NUM_BILLION_TOKENS}B_lr_${LR}"

${LAUNCH} src/scripts/akshitab/add_finegrained_expert/train_router.py \
    ${RUN_NAME} \
    --trainer.load_path="${MERGED_MODEL_PATH}/model_and_optim" \
    --save-folder="${MODELS}/${RUN_NAME}" \
    --dataset.mix=base_math_code \
    --work-dir="${DATASET_CACHE}" \
    --trainer.max_duration="{value: ${NUM_TOKENS}, unit: tokens}" \
    --trainer.callbacks.wandb.enabled=true \
    --trainer.callbacks.wandb.entity=akshitab \
    --trainer.callbacks.wandb.project=olmoe-modular \
    --trainer.callbacks.wandb.name="${RUN_NAME}" \
    --trainer.callbacks.wandb.tags='[extension]' \
    --dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
    --model.block.name="moe" \
    --model.block.sequence_mixer.qk_norm=null \
    --model.block.feed_forward_moe.lb_loss_weight=1e-2 \
    --train_module.scheduler.warmup_fraction=0.1 \
    --lr=${LR} \
    --num-new-experts=${NUM_NEW_EXPERTS}
```

Name prefix encodes the merge source: `rt-merged_` for stock merge, `rt-realdata-merged_` when trained on real data (vs synthetic), `rt-merged_moereducedp512sharedexp1_` for the reducedp base, `rt-merged_twolevel_` for the twolevel base.

---

## Twolevel MoE variants

Twolevel runs use `${TWOLEVEL_BASE}` and add two extra knobs to the `train_selected_experts.py` invocation:

```bash
NUM_SHARED_EXPERTS=1                                       # shared experts at the end of the expert pool
INSERT_POS=$((128 - NUM_SHARED_EXPERTS))                   # where new experts get inserted
EXPERTS_TO_TRAIN=$(seq -s, $INSERT_POS $((INSERT_POS + NUM_NEW_EXPERTS - 1)))
```

And the trainer flags include:

```bash
    --base-model-config="${NEW_BASE_MODEL_PATH}" \
    --experts-to-train=${EXPERTS_TO_TRAIN} \
    --model.block.feed_forward_moe.router.num_forced_experts=${NUM_NEW_EXPERTS}
```

Run-name prefix: `twolevel_` for direct extension, `freeze-fix-twolevel_` for the frozen-everything-else variant, `merged_twolevel_` / `rt-merged_twolevel_` for merge and router-training. The `_forced_` segment in the name signals `num_forced_experts > 0`.

Initialization (`add_new_expert.py`) takes `--num_shared_experts 1 --exclude_experts 127` to respect the shared-expert slot.
