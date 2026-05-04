# PARENT: "twolevelbatchlbreducedpsharedexp1randpool-8-128_1b14b_lr-4e-3_lb-1e-1_1T_0313.sh"
# DESCRIPTION:
#     - Annealing run: resumes from 1T checkpoint and linearly decays LR to 0 over anneal_tokens
# STATUS: NEW
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

min_document_expert_pool=8
max_document_expert_pool=128
eval_document_expert_pool=32
lb=1e-1
# NOTE: --lr is no longer needed; the anneal script auto-extracts it from the checkpoint

anneal_tokens=50000000000  # 50B tokens
anneal_checkpoint="${MODELS_DIR}/twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313/step238419"

nodes=16
gpus=8
# calculate by taking nodes multiply by gpus multiply by 4 (since we have 4 as micro batch size)
lb_global_batch_size=$((nodes * gpus * 4))

num_shared_experts=1 # 1 out of 8 will be shared experts

runname="twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419"


#torchrun --nproc-per-node=1 src/scripts/train/olmoe-1B-7B_fsl_anneal.py \
#  $runname \
#  --save-folder="./claude_outputs/models/$runname" \
#  --dataset.mix=arc-easy-train \
#  --work-dir="./claude_outputs/dataset-cache" \
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
#  --model.block.feed_forward_moe.lb_loss_weight=${lb} \
#  --anneal-tokens=${anneal_tokens} \
#  --anneal-checkpoint=${anneal_checkpoint}


launch src/scripts/train/olmoe-1B-7B_fsl_anneal.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=olmoe-modular \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags='[annealing]' \
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
		--model.block.feed_forward_moe.lb_loss_weight=${lb} \
		--trainer.callbacks.checkpointer.save_interval=20000 \
		--trainer.callbacks.downstream_evaluator.eval_interval=2500 \
		--anneal-tokens=${anneal_tokens} \
		--anneal-checkpoint=${anneal_checkpoint}
