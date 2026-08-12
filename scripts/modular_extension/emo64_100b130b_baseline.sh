# PARENT: "scripts/models_v2/emo_64exp_wsd_lr2e-3_extend1t.sh" (the in-place 1T extension whose
#         100B-130B window this reproduces).
# DESCRIPTION:
#     - STAGE-3 BASELINE for modular_extension: re-train the EMO 64e model over EXACTLY the
#       100B->130B token window that was oracle-partitioned (k=32/k=64 doc clusters), normally
#       (no partitioning), and save the 130B checkpoint. The original extend1t run consumed this
#       window but only keeps permanent checkpoints every 23842 steps (~100B), so no ~130B
#       checkpoint exists; this recreates it as a separate run so the in-flight extend1t
#       (same save folder, now ~835B) is untouched.
#     - Exact-same-tokens guarantee: --load_path the trunk's step23842 (100B) checkpoint WITH
#       trainer state (data cursor, RNG) + optimizer state. Data order is deterministic in
#       (seed, epoch) and cached in the work-dir (global_indices chunk2, world-size independent),
#       so steps 23843..30996 revisit exactly the tokens the extraction pipeline pulled for
#       100e9-130e9 (verified against the run's own global-indices cache).
#     - Exact-same-schedule guarantee: max_duration stays 1T (same scheduler t_max as the real
#       extend1t at this phase -> LR flat at peak 2e-3 through the whole window; decay_steps=1
#       only touches the final step of a full 1T run) and we stop with
#       --trainer.hard_stop=130e9 tokens instead. The checkpointer's post_train hook saves the
#       final permanent checkpoint at the hard stop (step 30996, 130.0B tokens).
#     - 8 nodes to match the trunk (reduce-dp batch-LB stats depend on node count at lb!=0, and
#       world_size must match for the checkpoint's RNG/data-cursor restore).
#     - Preemption-safe: on restart the (non-empty) save folder auto-resumes from its own
#       ephemeral checkpoint and load_path is ignored.
#
#   git add . && git commit && git push origin <branch>      # gantry clones from origin
#   MODE=beaker bash scripts/modular_extension/emo64_100b130b_baseline.sh
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

EXPERIMENT_NAME="modular_extension"
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/${EXPERIMENT_NAME}"
DATA_ROOT="s3://ai2-llm"

BEAKER_NODES=8              # match the trunk (reduce-dp batch-LB + world_size/RNG continuity)
BEAKER_GPUS=8
BEAKER_PRIORITY=urgent
BEAKER_NO_FOLLOW=1          # fire-and-forget: submit and return, monitor via W&B

# --- match the trunk's model/objective exactly (see extend1t) ---
lr=2e-3
lb=1e-1
num_shared_experts=1
num_experts=64
min_document_expert_pool=8
max_document_expert_pool=64
eval_document_expert_pool=64
warmup_steps=2000
decay_steps=1

MAX_TOKENS=1000000000000    # SAME 1T cap as extend1t -> identical (flat) LR through the window
HARD_STOP=130000000000      # ...but stop at 130B; post_train saves the final checkpoint there

runname="emo64_100b130b_baseline"
save_folder="${MODELS_DIR}/${runname}"
INIT_CHECKPOINT="/weka/oe-training-default/ryanwang/EMO/models_v2/emo_64exp_50b_wsd_lr2e-3/step23842"

echo "Stage-3 BASELINE: re-training the 100B->130B window normally from step23842:"
echo "  load_path (full state):  ${INIT_CHECKPOINT}"
echo "  save_folder:             ${save_folder}"
echo "  hard stop:               ${HARD_STOP} tokens (~step 30996); LR stays flat 2e-3"

launch src/scripts/train/olmoe-1B-7B_fsl.py $runname \
		--save-folder="${save_folder}" \
		--load_path="${INIT_CHECKPOINT}" \
		--load_trainer_state=true \
		--load_optim_state=true \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration="{value: ${MAX_TOKENS}, unit: tokens}" \
		--trainer.hard_stop="{value: ${HARD_STOP}, unit: tokens}" \
		--scheduler=wsd \
		--warmup_steps=${warmup_steps} \
		--decay_steps=${decay_steps} \
		--trainer.callbacks.checkpointer.save_interval=23842 \
		--trainer.callbacks.checkpointer.ephemeral_save_interval=500 \
		--trainer.callbacks.checkpointer.keep_ephemeral=2 \
		--trainer.callbacks.checkpointer.pre_train_checkpoint=false \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=emo-extension \
		--trainer.callbacks.wandb.name="${runname}" \
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
