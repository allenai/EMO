# PARENT: "extensions/twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419_ct-m8_lb0.sh"
# DESCRIPTION:
#     - Same as _ct-m8_lb0 (lb=0) continual-pretrain, but with weight_decay=0.1 restored
#       (matching the anneal script's WD; ct-math_8 and ct-m8_lb0 use WD=0.0).
#     - Note: runname drops trailing "419" from step238419 to fit within the 128-char
#       beaker name limit. Actual base checkpoint step is 238419 (see base_model_path).
# STATUS: NEW
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

min_document_expert_pool=8
max_document_expert_pool=8
eval_document_expert_pool=32
lb=0
lr=4e-4
wd=0.1

num_billion_tokens=10
num_tokens=$((num_billion_tokens * 1000000000))

base_model_path="${MODELS_DIR}/twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419/step250339"

nodes=16
gpus=8

num_shared_experts=1

runname="twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238_ct-m8_lb0_wd"


#torchrun --nproc-per-node=1 src/scripts/train/olmoe-1B-7B_fsl_extension.py \
#  $runname \
#  --save-folder="./claude_outputs/models/$runname" \
#  --dataset.mix=arc-easy-train \
#  --work-dir="./claude_outputs/dataset-cache" \
#  --trainer.callbacks.wandb.enabled=false \
#  --trainer.callbacks.wandb.entity=ryanyxw \
#  --trainer.callbacks.wandb.project=olmoe-modular \
#  --trainer.callbacks.wandb.name="${runname}" \
#  --global_batch_size=2 \
#  --num-tokens=100000 \
#  --lr=${lr} \
#  --load-path=${base_model_path}/model_and_optim \
#  --model.block.feed_forward_moe.num_experts=128 \
#  --model-type="two-level_lb-batch_reduce-dp_sharedexp_randpool" \
#  --min_document_expert_pool=${min_document_expert_pool} \
#  --max_document_expert_pool=${max_document_expert_pool} \
#  --eval_document_expert_pool=${eval_document_expert_pool} \
#  --num_shared_experts=${num_shared_experts} \
#  --train_module.compile_model=false \
#  --dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
#  --model.block.name="moe" \
#  --model.block.sequence_mixer.qk_norm=null \
#  --model.block.sequence_mixer.backend=torch \
#  --model.block.feed_forward_moe.lb_loss_weight=${lb} \
#  --train_module.optim.weight_decay=${wd}


launch src/scripts/train/olmoe-1B-7B_fsl_extension.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
		--dataset.mix=mj_finemath4plus \
		--work-dir="${DATASET_CACHE}" \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=olmoe-modular \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags='[extension, contpretrain, finemath, lb0, wd]' \
		--num-tokens=${num_tokens} \
		--lr=${lr} \
		--load-path=${base_model_path}/model_and_optim \
		--model.block.feed_forward_moe.num_experts=128 \
		--dataset.generate_doc_lengths=true \
		--model.block.sequence_mixer.backend=flash_2 \
		--model-type="two-level_lb-batch_reduce-dp_sharedexp_randpool" \
		--min_document_expert_pool=${min_document_expert_pool} \
		--max_document_expert_pool=${max_document_expert_pool} \
		--eval_document_expert_pool=${eval_document_expert_pool} \
		--num_shared_experts=${num_shared_experts} \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name="moe" \
		--model.block.sequence_mixer.qk_norm=null \
		--model.block.feed_forward_moe.lb_loss_weight=${lb} \
		--train_module.optim.weight_decay=${wd} \
		--trainer.callbacks.checkpointer.save_interval=600 \
		--trainer.callbacks.downstream_evaluator.eval_interval=250
