# PARENT: "scripts/models_fullextend/emo_1b14b_130b.sh"
# DESCRIPTION:
#     - models_fullextend ghost-expert SWEEP run, config #1.
#       Ghost knobs: coeff_mode=usage, route=always, detach_coeff=false, num=1, random_k=8.
#       (num and random_k are held fixed across the whole sweep; the runname encodes the
#       three varied knobs: coeff_mode / route / detach_coeff.)
#     - 16 nodes. Trains on the full 130B-token schedule (max_duration=130B so the LR
#       cosine decay targets 130B) but HARD-STOPS at 50B tokens for the sweep.
#     - Checkpointing: keep the 2 most-recent rolling (ephemeral) checkpoints for
#       crash-resume + the final model; no permanent intermediates accumulate.
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

EXPERIMENT_NAME="models_fullextend"
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/${EXPERIMENT_NAME}"
DATA_ROOT="s3://ai2-llm"

BEAKER_NODES=16
BEAKER_GPUS=8

min_document_expert_pool=8
max_document_expert_pool=128
eval_document_expert_pool=32
lr=4e-3
lb=1e-1

num_shared_experts=1 # 1 out of 8 will be shared experts

# --- ghost-expert knobs (this run) ---
ghost_extend_num=1               # fixed across the sweep
ghost_extend_random_k=8          # fixed across the sweep
ghost_extend_coeff_mode="usage"  # swept: usage | uniform | random
ghost_extend_route="always"      # swept: always (topk not implemented)
ghost_extend_detach_coeff=false  # swept: false | true

runname="emo_1b14b_130b_ghost_usage_always_detachF"

launch src/scripts/train/olmoe-1B-7B_fsl.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
		--trainer.hard_stop='{value: 50_000_000_000, unit: tokens}' \
		--trainer.callbacks.checkpointer.save_interval=1000000 \
		--trainer.callbacks.checkpointer.ephemeral_save_interval=500 \
		--trainer.callbacks.checkpointer.keep_ephemeral=2 \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=emo-extension \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags="[pretraining, ${EXPERIMENT_NAME}]" \
		--model.block.feed_forward_moe.num_experts=128 \
		--dataset.generate_doc_lengths=true \
		--model.block.sequence_mixer.backend=flash_2 \
		--model-type="two-level_lb-batch_reduce-dp_sharedexp_randpool_ghost" \
		--min_document_expert_pool=${min_document_expert_pool} \
		--max_document_expert_pool=${max_document_expert_pool} \
		--eval_document_expert_pool=${eval_document_expert_pool} \
		--num_shared_experts=$num_shared_experts \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name="moe" \
		--model.block.sequence_mixer.qk_norm=null \
		--lr=${lr} \
		--model.block.feed_forward_moe.lb_loss_weight=${lb} \
		--model.block.feed_forward_moe.router.ghost_extend_mode=true \
		--model.block.feed_forward_moe.router.ghost_extend_num=${ghost_extend_num} \
		--model.block.feed_forward_moe.router.ghost_extend_coeff_mode=${ghost_extend_coeff_mode} \
		--model.block.feed_forward_moe.router.ghost_extend_random_k=${ghost_extend_random_k} \
		--model.block.feed_forward_moe.router.ghost_extend_route=${ghost_extend_route} \
		--model.block.feed_forward_moe.router.ghost_extend_detach_coeff=${ghost_extend_detach_coeff}
