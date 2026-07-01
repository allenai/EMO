# PARENT: "scripts/models_v2/emo_64exp_50b_wsd_lr2e-3.sh" (the EMO 64e 50B WSD-2e-3 stable trunk) +
#         "scripts/models_v2/stdmoe_128exp_wsd_lr2e-3_extend1t.sh" (the in-place extension pattern).
# DESCRIPTION:
#     - CONTINUAL TRAINING of the EMO 64e WSD-2e-3 stable trunk out to 1T tokens, to get a stronger
#       (longer-trained) EMO model while opportunistically using idle GPUs. The EMO counterpart of
#       stdmoe_128exp_wsd_lr2e-3_extend1t.sh -- identical mechanism, EMO model flags.
#     - Because the trunk is pure WSD with decay_steps=1 (LR flat at the peak the whole run, only the
#       single final step touches 0), we extend the STABLE phase indefinitely just by resuming with a
#       larger max_duration -- no re-warmup, no schedule discontinuity:
#         * The WSD scheduler's t_max = trainer.max_steps is recomputed from max_duration on resume
#           (max_steps = global_step + ceil((max_tokens - tokens_seen)/tokens_per_batch)); it is NOT
#           persisted. current = trainer.global_step (absolute). At the resume step (11921, past
#           warmup=2000, far below t_max-1) get_lr returns the peak LR -> flat stable phase.
#         * Data order is deterministic in (seed, epoch) and cached in the work-dir, INDEPENDENT of
#           max_duration. Resume restores epoch / batches_processed / tokens_processed (verified:
#           step11921 = epoch 1, 50.00B, seed 0), so training continues from the exact 50B data
#           position; epoch 1 reshuffles into epoch 2 when exhausted (standard multi-epoch).
#     - THIS IS THE SAME RUN, CONTINUED IN PLACE (not a separate run/line):
#         * save-folder = the trunk's OWN folder -> the trainer auto-resumes from its latest
#           checkpoint (step11921) with full trainer + optimizer state. No load_path needed.
#         * W&B: resume the trunk's existing run (id s9024txw) via WANDB_RUN_ID + WANDB_RESUME=allow
#           (WandBCallback doesn't expose id/resume, but wandb.init honors these env vars), so the
#           curve continues past 50B on the SAME W&B run -- the report's `emo64wsd2e3` key already
#           points at s9024txw, so its LR-tab / eval line extends on the same line too.
#         * On a preemption restart the (non-empty) save folder auto-resumes; WANDB_RESUME=allow keeps
#           re-attaching to the same run.
#     - Model flags MUST match the trunk exactly (model rebuilt from CLI then checkpoint state loaded
#       in -- a mismatch fails the load): EMO model-type two-level_lb-batch_reduce-dp_sharedexp_randpool,
#       64 experts / 1 shared, lb 1e-1, generate_doc_lengths, flash_2, document-expert pool
#       min=8/max=64/eval=64, OLMoE-mix-0824, WSD warmup=2000 / decay_steps=1, peak LR 2e-3, 8 nodes
#       (reduce-dp batch-LB stats depend on node count at lb!=0; keeping 8 also matches world_size ->
#       RNG restore).
#     - Same treatment as the trunk: urgent priority + preemptible, fire-and-forget (--no-follow).
#       Permanent checkpoints every ~100B tokens so any 100B point is usable even if stopped/preempted
#       early; 2-deep rolling ephemerals for restart safety. Nothing changes vs the trunk except the
#       token budget (50B -> 1T) and the periodic-permanent cadence (the trunk's every-5B fixed_steps
#       are all in the past, so a periodic save_interval is needed to keep saving as it extends).
#
#   git add . && git commit && git push origin <branch>      # gantry clones from origin
#   MODE=beaker bash scripts/models_v2/emo_64exp_wsd_lr2e-3_extend1t.sh
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

EXPERIMENT_NAME="models_v2"
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/${EXPERIMENT_NAME}"
DATA_ROOT="s3://ai2-llm"

BEAKER_NODES=8              # match the trunk (reduce-dp batch-LB + world_size/RNG continuity)
BEAKER_GPUS=8
BEAKER_PRIORITY=urgent      # same treatment as the trunk (launch_common default) -- just trained longer
BEAKER_NO_FOLLOW=1          # fire-and-forget: submit and return, monitor via W&B

# Resume the trunk's EXISTING W&B run so the curve stays one continuous line (id verified: s9024txw,
# name emo_64exp_50b_wsd_lr2e-3). wandb.init picks these env vars up on every (re)start.
TRUNK_RUN="emo_64exp_50b_wsd_lr2e-3"
WANDB_ID="s9024txw"
BEAKER_ENV_VARS=("WANDB_RUN_ID=${WANDB_ID}" "WANDB_RESUME=allow")

# --- match the trunk's model/objective exactly ---
lr=2e-3
lb=1e-1
num_shared_experts=1
num_experts=64
min_document_expert_pool=8
max_document_expert_pool=64
eval_document_expert_pool=64
warmup_steps=2000
decay_steps=1              # pure stable trunk: flat at peak LR; only the final step touches 0.

MAX_TOKENS=1000000000000   # 1T-token cap for the extended stable trunk
save_folder="${MODELS_DIR}/${TRUNK_RUN}"        # the trunk's OWN folder -> auto-resume from step11921
save_interval=23842        # permanent ckpt ~every 100B tokens (100e9 / 4,194,304 tok/step)

# Beaker job label (findable as the extend job); the W&B run + save folder stay the trunk's.
runname="${TRUNK_RUN}_extend1t"

echo "Continual-training the EMO 64e WSD-2e-3 trunk IN PLACE (same run/line):"
echo "  save_folder (auto-resume): ${save_folder}"
echo "  W&B run (resumed):         ${WANDB_ID} (${TRUNK_RUN})"
echo "  max tokens:                ${MAX_TOKENS} (1T)"
echo "  permanent every:           ${save_interval} steps (~100B tokens)"

launch src/scripts/train/olmoe-1B-7B_fsl.py $runname \
		--save-folder="${save_folder}" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
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
		--trainer.callbacks.wandb.name="${TRUNK_RUN}" \
		--trainer.callbacks.wandb.tags="[pretraining, ${EXPERIMENT_NAME}]" \
		--model.block.feed_forward_moe.num_experts=${num_experts} \
		--dataset.generate_doc_lengths=true \
		--model.block.sequence_mixer.backend=flash_2 \
		--model-type="two-level_lb-batch_reduce-dp_sharedexp_randpool" \
		--min_document_expert_pool=${min_document_expert_pool} \
		--max_document_expert_pool=${max_document_expert_pool} \
		--eval_document_expert_pool=${eval_document_expert_pool} \
		--num_shared_experts=$num_shared_experts \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name="moe" \
		--model.block.sequence_mixer.qk_norm=null \
		--lr=${lr} \
		--model.block.feed_forward_moe.lb_loss_weight=${lb}
