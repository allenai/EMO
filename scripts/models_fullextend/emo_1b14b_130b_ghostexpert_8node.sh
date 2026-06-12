# PARENT: "scripts/models_fullextend/emo_1b14b_130b_ghostexpert.sh"
# DESCRIPTION:
#     - 8-node Beaker smoke test of the ghost-expert run: confirm it launches and
#       trains, and measure throughput vs the baseline. Identical recipe/knobs to
#       emo_1b14b_130b_ghostexpert.sh, but hardcodes BEAKER_NODES=8 (64 H100s) and
#       uses a distinct runname so it does not collide with the real run's save path.
#     - Not the production launch — intended to be stopped once throughput is read.
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

EXPERIMENT_NAME="models_fullextend"
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/${EXPERIMENT_NAME}"
DATA_ROOT="s3://ai2-llm"

# 8-node smoke test (default is 16).
BEAKER_NODES=8
BEAKER_GPUS=8

min_document_expert_pool=8
max_document_expert_pool=128
eval_document_expert_pool=32
lr=4e-3
lb=1e-1

num_shared_experts=1 # 1 out of 8 will be shared experts

# --- ghost-expert knobs ---
ghost_extend_num=1
ghost_extend_coeff_mode="usage"
ghost_extend_random_k=8
ghost_extend_route="always"
ghost_extend_detach_coeff=false

runname="emo_1b14b_130b_ghostexpert_8node"

launch src/scripts/train/olmoe-1B-7B_fsl.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=emo-extension \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags="[pretraining, ${EXPERIMENT_NAME}, smoketest]" \
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
