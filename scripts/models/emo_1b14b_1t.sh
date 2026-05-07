# PARENT: "twolevelbatchlbreducedpsharedexp-32_1b14b_lr-4e-3_lb-1e-1_0211.sh"
# DESCRIPTION:
#     - Same as parent but with random document_expert_pool sampled uniformly from [8, 128] per document
# STATUS: NEW
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

min_document_expert_pool=8
max_document_expert_pool=128
eval_document_expert_pool=32
lr=4e-3
lb=1e-1

nodes=16
gpus=8
# calculate by taking nodes multiply by gpus multiply by 4 (since we have 4 as micro batch size)
lb_global_batch_size=$((nodes * gpus * 4))

num_shared_experts=1 # 1 out of 8 will be shared experts

runname="emo_1b14b_1t"


#torchrun --nproc-per-node=1 src/scripts/train/olmoe-1B-7B_fsl.py \
#  $runname \
#  --save-folder="/root/ryanwang/phdbrainstorm/Emo/models/$runname" \
#  --dataset.mix=arc-easy-train \
#  --work-dir="/root/ryanwang/dataset-cache" \
#  --trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
#  --trainer.callbacks.wandb.enabled=false \
#  --trainer.callbacks.wandb.entity=ryanyxw \
#  --trainer.callbacks.wandb.project=olmoe-modular \
#  --trainer.callbacks.wandb.name="${runname}" \
#  --global_batch_size=2 \
#  --model.block.feed_forward_moe.num_experts=16 \
#  --model-type="two-level_lb-batch_reduce-dp_sharedexp_randpool" \
#  --min_document_expert_pool=${min_document_expert_pool} \
#  --max_document_expert_pool=${max_document_expert_pool} \
#  --eval_document_expert_pool=${eval_document_expert_pool} \
#  --num_shared_experts=2 \
#  --train_module.compile_model=false \
#  --dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
#  --model.block.name="moe" \
#  --model.block.sequence_mixer.qk_norm=null \
#  --lr=${lr} \
#  --model.block.feed_forward_moe.lb_loss_weight=${lb}


launch src/scripts/train/olmoe-1B-7B_fsl.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration='{value: 1_000_000_000_000, unit: tokens}' \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=olmoe-modular \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags='[pretraining]' \
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
		--trainer.callbacks.checkpointer.save_interval=20000 \
		--trainer.callbacks.downstream_evaluator.eval_interval=2500
