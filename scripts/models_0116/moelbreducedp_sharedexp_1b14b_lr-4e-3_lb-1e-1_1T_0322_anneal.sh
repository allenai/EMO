# PARENT: "moelbreducedp_sharedexp_1b14b_lr-4e-3_lb-1e-1_1T_0322.sh"
# DESCRIPTION:
#     - Annealing run: resumes from 1T checkpoint and linearly decays LR to 0 over anneal_tokens
# STATUS: NEW
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

lb=1e-1
# NOTE: --lr is no longer needed; the anneal script auto-extracts it from the checkpoint

anneal_tokens=50000000000  # 50B tokens
anneal_checkpoint="${MODELS_DIR}/moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_1T_0322/step238419"

nodes=16
gpus=8
# calculate by taking nodes multiply by gpus multiply by 4 (since we have 4 as micro batch size)
lb_global_batch_size=$((nodes * gpus * 4))

num_shared_experts=1

runname="moereducedp${lb_global_batch_size}sharedexp${num_shared_experts}_1b14b_lr-4e-3_lb-${lb}_1T_0322_anneal_from_step238419"


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
#  --model.block.feed_forward_moe.num_experts=128 \
#  --model-type="moe_lbreducedp_sharedexp" \
#  --num_shared_experts=${num_shared_experts} \
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
		--model-type="moe_lbreducedp_sharedexp" \
		--num_shared_experts=$num_shared_experts \
		--model.block.feed_forward_moe.num_experts=128 \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name="moe" \
		--model.block.sequence_mixer.qk_norm=null \
		--model.block.feed_forward_moe.lb_loss_weight=${lb} \
		--trainer.callbacks.checkpointer.save_interval=20000 \
		--trainer.callbacks.downstream_evaluator.eval_interval=2500 \
		--anneal-tokens=${anneal_tokens} \
		--anneal-checkpoint=${anneal_checkpoint}
