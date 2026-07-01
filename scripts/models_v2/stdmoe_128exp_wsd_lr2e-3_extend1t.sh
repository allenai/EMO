# PARENT: "scripts/models_v2/stdmoe_128exp_50b_wsd_lr2e-3.sh" (the 50B WSD-2e-3 stable trunk /
#         extension-methods upperbound).
# DESCRIPTION:
#     - CONTINUAL TRAINING of the 128e WSD-2e-3 stable trunk out to 1T tokens, to get a stronger
#       (longer-trained) 128e upperbound while opportunistically using idle GPUs. Because the trunk
#       is pure WSD with decay_steps=1 (LR flat at the peak the whole run, only the single final
#       step touches 0), we can extend the stable phase indefinitely just by resuming with a larger
#       max_duration -- no re-warmup, no schedule discontinuity:
#         * The WSD scheduler's t_max = trainer.max_steps is recomputed from max_duration on resume
#           (max_steps = global_step + ceil((max_tokens - tokens_seen)/tokens_per_batch)); it is NOT
#           persisted in the checkpoint. current = trainer.global_step (absolute). At the resume step
#           (11921, past warmup=2000, far below t_max-1) get_lr returns the peak LR -> flat.
#         * Data order is deterministic in (seed, epoch) and cached in the work-dir, INDEPENDENT of
#           max_duration. Resume restores epoch / batches_processed / tokens_processed, so training
#           continues from the exact 50B data position; when epoch 1 is exhausted the loader
#           reshuffles into epoch 2, etc. (standard multi-epoch). OLMoE-mix-0824 is far larger than
#           50B, so no repetition until well into the run, and any repetition is a clean reshuffle.
#     - This is a SEPARATE run (its own W&B id + save folder) that LOADS the trunk's 50B checkpoint
#       (step11921) with full trainer state -- it does NOT mutate the published 50B trunk, so the
#       matched-50B-compute upperbound stays intact as a reference. step11921 is an exact stable
#       checkpoint: the trunk's final step ran at LR=0 (zero-effect update), so its weights ARE the
#       50B stable-trunk weights and its data position is exactly 50B. Same load_path + full-state
#       pattern the WSD decay branches use.
#     - Model flags MUST match the trunk exactly (the model is rebuilt from CLI then the checkpoint
#       state is loaded into it -- a mismatch would fail the load): model-type moe_lbreducedp_sharedexp,
#       128 experts / 1 shared, lb 1e-1, OLMoE-mix-0824, WSD warmup=2000 / decay_steps=1, peak LR 2e-3.
#     - Opportunistic run: normal priority + preemptible (fills idle capacity, yields to others),
#       fire-and-forget (--no-follow). Permanent checkpoints every ~100B tokens so any 100B point is
#       usable even if the run is stopped/preempted early; 2-deep rolling ephemerals for restart
#       safety. On a preemption restart, the (now non-empty) save folder auto-resumes with full
#       trainer state, so load_path only matters for the very first launch.
#
#   git add . && git commit && git push origin <branch>      # gantry clones from origin
#   MODE=beaker bash scripts/models_v2/stdmoe_128exp_wsd_lr2e-3_extend1t.sh
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

EXPERIMENT_NAME="models_v2"
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/${EXPERIMENT_NAME}"
DATA_ROOT="s3://ai2-llm"

BEAKER_NODES=8              # match the trunk: at lb!=0 the reduce-dp batch-LB stats depend on node count
BEAKER_GPUS=8
BEAKER_PRIORITY=normal      # opportunistic idle-fill (not urgent): yield to higher-priority jobs
BEAKER_NO_FOLLOW=1          # fire-and-forget: submit and return, monitor via W&B

# --- match the trunk's model/objective exactly ---
lr=2e-3
lb=1e-1
num_shared_experts=1
num_experts=128
warmup_steps=2000
decay_steps=1              # pure stable trunk: flat at peak LR; only the final step touches 0.

# --- continuation spec ---
TRUNK_RUN="stdmoe_128exp_50b_wsd_lr2e-3"
LOAD_STEP=11921            # the trunk's 50B checkpoint (exact stable weights, data at 50B)
MAX_TOKENS=1000000000000   # 1T-token cap for the extended stable trunk

runname="stdmoe_128exp_50b_wsd_lr2e-3_extend1t"
load_path="${MODELS_DIR}/${TRUNK_RUN}/step${LOAD_STEP}"
save_folder="${MODELS_DIR}/${runname}"

# Permanent checkpoint cadence: ~every 100B tokens. 100B / 4,194,304 tok/step = 23,842 steps.
save_interval=23842

echo "Continual-training the 128e WSD-2e-3 trunk:"
echo "  load_path (50B ckpt): ${load_path}"
echo "  save_folder:          ${save_folder}"
echo "  max tokens:           ${MAX_TOKENS} (1T)"
echo "  permanent every:      ${save_interval} steps (~100B tokens)"

launch src/scripts/train/olmoe-1B-7B_fsl.py $runname \
		--save-folder="${save_folder}" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.load_path="${load_path}" \
		--trainer.load_trainer_state=true \
		--trainer.load_optim_state=true \
		--trainer.max_duration="{value: ${MAX_TOKENS}, unit: tokens}" \
		--scheduler=wsd \
		--warmup_steps=${warmup_steps} \
		--decay_steps=${decay_steps} \
		--trainer.callbacks.checkpointer.save_interval=${save_interval} \
		--trainer.callbacks.checkpointer.ephemeral_save_interval=500 \
		--trainer.callbacks.checkpointer.keep_ephemeral=2 \
		--trainer.callbacks.checkpointer.pre_train_checkpoint=false \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=emo-extension \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags="[pretraining, ${EXPERIMENT_NAME}]" \
		--model-type="moe_lbreducedp_sharedexp" \
		--num_shared_experts=$num_shared_experts \
		--model.block.feed_forward_moe.num_experts=${num_experts} \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name="moe" \
		--model.block.sequence_mixer.qk_norm=null \
		--lr=${lr} \
		--model.block.feed_forward_moe.lb_loss_weight=${lb}
