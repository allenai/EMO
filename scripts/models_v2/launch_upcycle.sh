#!/bin/bash
# Reusable expert-UPCYCLING launcher for models_v2.
#
# Goal: take a trained 64-expert stdMoE checkpoint at 25B tokens, expand it to 128 experts
# (slot-aware surgery, see scripts/models_v2/expand_moe_experts.py), and continue WSD training so
# the upcycled 128-expert model can be compared head-to-head with the from-scratch
# stdmoe_128exp_50b_wsd_lr2e-3 run. Three init modes x two optimizer axes = 8 ablation leaves.
#
# Two steps per launch:
#   1. SEED (local, CPU): run expand_moe_experts.py to write an expanded step5960 checkpoint into the
#      run's save folder. Idempotent -- skipped if it already exists. Runs in THIS session against
#      the LOCAL weka mount ($HOME), mirroring how launch_merged_eval.sh converts HF locally.
#   2. LAUNCH (Beaker): run the EXACT stdmoe_128exp_50b_wsd_lr2e-3.sh config; the trainer auto-resumes
#      from the seeded step5960 (model + optim + global_step + data cursor) and trains to MAX_B.
#
# Because the seeded checkpoint preserves global_step=5960 and the from-scratch config uses flat WSD
# (warmup 2000, decay_steps=1), step5960 is past warmup => LR is flat 2e-3, exactly matching how the
# from-scratch trunk behaves over its 25B->50B span. Clean trunk-vs-trunk comparison.
#
# Env (set by the per-leaf wrappers):
#   INIT_MODE   random | upcycle | upcycle_jitter
#   KEPT_OPTIM  carry | reset     (kept experts + non-MoE params keep moments, or fresh Adam)
#   NEW_OPTIM   copy | zero       (new experts inherit source moments; ignored for random/reset)
#   LEAF        grid leaf, e.g. copy_keptcarry_newcopy (run name + save subdir)
#   MAX_B       total token budget in B (default 30 = 25B branch + 5B convergence check; 50 to extend)
#   FROM_RUN/FROM_STEP   source trunk + step (default stdmoe_64exp_50b_wsd_lr2e-3 / 5960)
#
# NOTE: commit AND push before launching (gantry clones source from origin on each worker).
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

EXPERIMENT_NAME="models_v2"
WEKA_MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/${EXPERIMENT_NAME}"
# In this GPU/launch session the literal /weka/... path is not mounted; ryanwang -> $HOME. Seeding
# runs locally and must use the local mount; the Beaker workers see the weka path.
LOCAL_MODELS_DIR="${WEKA_MODELS_DIR/\/weka\/oe-training-default\/ryanwang/${HOME}}"
MODELS_DIR="${WEKA_MODELS_DIR}"
DATA_ROOT="s3://ai2-llm"

BEAKER_NODES=8   # match the trunk: at lb!=0 the reduce-dp batch-LB stats depend on node count
BEAKER_GPUS=8

# stdMoE config (match stdmoe_128exp_50b_wsd_lr2e-3.sh exactly).
lr=2e-3
lb=1e-1
num_shared_experts=1
num_experts=128
warmup_steps=2000
decay_steps=1

# Upcycling spec.
INIT_MODE="${INIT_MODE:?set INIT_MODE}"
KEPT_OPTIM="${KEPT_OPTIM:?set KEPT_OPTIM}"
NEW_OPTIM="${NEW_OPTIM:-zero}"
LEAF="${LEAF:?set LEAF}"
FROM_RUN="${FROM_RUN:-stdmoe_64exp_50b_wsd_lr2e-3}"
FROM_STEP="${FROM_STEP:-5960}"
FROM_EXPERTS="${FROM_EXPERTS:-64}"
JITTER_STD="${JITTER_STD:-0.02}"
SEED="${SEED:-0}"
MAX_B="${MAX_B:-30}"

runname="stdmoe_128exp_up${FROM_EXPERTS}to128_lr2e-3_${LEAF}"
save_folder="${MODELS_DIR}/upcycle_${FROM_EXPERTS}to128/${LEAF}"               # weka path (workers)
local_save_folder="${LOCAL_MODELS_DIR}/upcycle_${FROM_EXPERTS}to128/${LEAF}"   # local mount (seeding)
local_src="${LOCAL_MODELS_DIR}/${FROM_RUN}/step${FROM_STEP}"
seed_step_dir="${local_save_folder}/step${FROM_STEP}"
max_duration="$(awk "BEGIN{printf \"%.0f\", ${MAX_B}*1000000000}")"

echo "Expert upcycling:"
echo "  leaf:          ${LEAF}  (init=${INIT_MODE} kept-optim=${KEPT_OPTIM} new-optim=${NEW_OPTIM})"
echo "  source:        ${local_src}"
echo "  seed ckpt:     ${seed_step_dir}"
echo "  run name:      ${runname}"
echo "  save folder:   ${save_folder}"
echo "  max tokens:    ${max_duration} (${MAX_B}B)"

# --- Step 1: seed the expanded checkpoint locally (idempotent) ---
if [ -f "${seed_step_dir}/model_and_optim/.metadata" ]; then
    echo "=== seed checkpoint already present, skipping surgery ==="
else
    if [ ! -d "${local_src}/model_and_optim" ]; then
        echo "ERROR: source checkpoint ${local_src} not found." >&2
        exit 1
    fi
    echo "=== seeding expanded checkpoint (CPU surgery) ==="
    python "$(dirname "${BASH_SOURCE[0]}")/expand_moe_experts.py" \
        --input-checkpoint "${local_src}" \
        --output-checkpoint "${seed_step_dir}" \
        --from-experts "${FROM_EXPERTS}" \
        --to-experts "${num_experts}" \
        --num-shared "${num_shared_experts}" \
        --init-mode "${INIT_MODE}" \
        --kept-optim "${KEPT_OPTIM}" \
        --new-optim "${NEW_OPTIM}" \
        --jitter-std "${JITTER_STD}" \
        --seed "${SEED}"
fi

# --- Step 2: launch training ---
# The trainer's auto-resume (maybe_load_checkpoint) does NOT recognize our hand-built step5960
# (it lacks the checkpointer's discovery marker), so we load it EXPLICITLY via --load_path with
# --load_trainer_state (continues global_step=5960 + data cursor) and --load_optim_state. load_path
# is only used when the save folder has no trainer-written checkpoint, so on the MAX_B=50 extend
# pass the run auto-resumes from its OWN later checkpoint and ignores load_path.
launch src/scripts/train/olmoe-1B-7B_fsl.py "$runname" \
		--save-folder="${save_folder}" \
		--load_path="${save_folder}/step${FROM_STEP}" \
		--load_trainer_state=true \
		--load_optim_state=true \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration="{value: ${max_duration}, unit: tokens}" \
		--scheduler=wsd \
		--warmup_steps=${warmup_steps} \
		--decay_steps=${decay_steps} \
		--trainer.callbacks.checkpointer.save_interval=1000000 \
		--trainer.callbacks.checkpointer.fixed_steps="[1192, 2384, 3576, 4768, 5960, 7153, 8345, 9537, 10729]" \
		--trainer.callbacks.checkpointer.ephemeral_save_interval=500 \
		--trainer.callbacks.checkpointer.keep_ephemeral=2 \
		--trainer.callbacks.checkpointer.pre_train_checkpoint=false \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=emo-extension \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags="[pretraining, ${EXPERIMENT_NAME}, upcycle]" \
		--model-type="moe_lbreducedp_sharedexp" \
		--num_shared_experts=$num_shared_experts \
		--model.block.feed_forward_moe.num_experts=${num_experts} \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name="moe" \
		--model.block.sequence_mixer.qk_norm=null \
		--lr=${lr} \
		--model.block.feed_forward_moe.lb_loss_weight=${lb}
