# PARENT: "scripts/models_fullextend/emo_1b14b_130b_ghost_uniform_always_detachF.sh"
# DESCRIPTION:
#     - models_fullextend ghost-expert SWEEP run, config #3.
#       Ghost knobs: coeff_mode=random, route=always, detach_coeff=false, num=1, random_k=8.
#       Completes the coeff_mode sweep (usage / uniform / random). Each ghost is the
#       uniform average over a random sample of random_k=8 document-pool experts
#       (re-sampled per forward). detach_coeff is a no-op for random (alpha is constant).
#     - 16 nodes, max_duration=130B, hard_stop=50B. 2-deep rolling checkpoints + final;
#       pre_train_checkpoint disabled (skip the useless random-init step0).
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
ghost_extend_coeff_mode="random" # swept: usage | uniform | random
ghost_extend_route="always"      # swept: always (topk not implemented)
ghost_extend_detach_coeff=false  # swept: false | true (no-op for random)

runname="emo_1b14b_130b_ghost_random_always_detachF"

launch src/scripts/train/olmoe-1B-7B_fsl.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
		--trainer.hard_stop='{value: 50_000_000_000, unit: tokens}' \
		--trainer.callbacks.checkpointer.save_interval=1000000 \
		--trainer.callbacks.checkpointer.ephemeral_save_interval=500 \
		--trainer.callbacks.checkpointer.keep_ephemeral=2 \
		--trainer.callbacks.checkpointer.pre_train_checkpoint=false \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=emo-extension \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags="[pretraining, ${EXPERIMENT_NAME}]" \
		--model.block.feed_forward_moe.num_experts=128 \
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
		--model.block.feed_forward_moe.lb_loss_weight=${lb} \
		--model.block.feed_forward_moe.router.ghost_extend_mode=true \
		--model.block.feed_forward_moe.router.ghost_extend_num=${ghost_extend_num} \
		--model.block.feed_forward_moe.router.ghost_extend_coeff_mode=${ghost_extend_coeff_mode} \
		--model.block.feed_forward_moe.router.ghost_extend_random_k=${ghost_extend_random_k} \
		--model.block.feed_forward_moe.router.ghost_extend_route=${ghost_extend_route} \
		--model.block.feed_forward_moe.router.ghost_extend_detach_coeff=${ghost_extend_detach_coeff}
