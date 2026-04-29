# PARENT: "twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121.sh"
# DESCRIPTION:
#     - the same but added on some new elements
# STATUS: USED
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

document_expert_pool=32
lr=4e-4
lb=1e-1

#runname="twolevelbatchlb-${document_expert_pool}_1b14b_lr-${lr}_poolsched-true_ablations_0116"
runname="twolevelbatchlb-${document_expert_pool}_1b14b_lr-${lr}_lb-${lb}_poolsched_0119"


#torchrun --nproc-per-node=1 src/scripts/train/olmoe-1B-7B_fsl.py \
#  $runname \
#  --save-folder="/root/ryanwang/phdbrainstorm/FlexMoE/models/$runname" \
#  --dataset.mix=arc-easy-train \
#  --work-dir="/root/ryanwang/dataset-cache" \
#  --trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
#  --trainer.callbacks.wandb.enabled=true \
#  --trainer.callbacks.wandb.entity=ryanyxw \
#  --trainer.callbacks.wandb.project=olmoe-modular \
#  --trainer.callbacks.wandb.name="${runname}" \
#  --model.block.feed_forward_moe.num_experts=16 \
#  --model-type="two-level" \
#  --document-expert-pool=${document_expert_pool} \
#  --train_module.compile_model=false \
#  --dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
#  --model.block.feed_forward_moe.z_loss_weight=0.004 \


launch src/scripts/train/olmoe-1B-7B_fsl.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
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
		--model-type="two-level_lb-batch" \
		--document-expert-pool=${document_expert_pool} \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name="moe" \
		--model.block.sequence_mixer.qk_norm=null \
		--lr=${lr} \
		--model.block.feed_forward_moe.lb_loss_weight=${lb} \
		--poolsched="{min_pool: ${document_expert_pool}, decay_steps: 2000}"












