# PARENT: "models_0116/twolevelbatchlbreducedpsharedexp1randpool-8-128_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal.sh"
# DESCRIPTION:
#     - Continual pretraining of the annealed two-level 1T checkpoint on mj_finemath4plus (10B tokens).
#     - Fresh LR (4e-4) with CosWithWarmup(0.1), WD=0.0. load_trainer_state=False (step counter resets).
#     - Router config + all weights trainable (no gradient masking).
# STATUS: NEW
##############################################################
min_document_expert_pool=8
max_document_expert_pool=8
eval_document_expert_pool=32
lb=1e-1
lr=4e-4

num_billion_tokens=10
num_tokens=$((num_billion_tokens * 1000000000))

base_model_path="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models/twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_twolevel_1T_annealed_from_step238419/step250339"

nodes=16
gpus=8
# global batch size in instances (nodes * gpus * 4 microbatch) — multiplied by SEQUENCE_LENGTH inside the script
global_batch_size=$((nodes * gpus * 4))

num_shared_experts=1

runname="twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_twolevel_1T_annealed_from_step238419_contpretrain-finemath4plus_pool8fixed"


#torchrun --nproc-per-node=1 src/scripts/train/olmoe-1B-7B_fsl_extension.py \
#  $runname \
#  --save-folder="./claude_outputs/models/$runname" \
#  --dataset.mix=arc-easy-train \
#  --work-dir="./claude_outputs/dataset-cache" \
#  --trainer.callbacks.wandb="{enabled: false, entity: ryanyxw, project: olmoe-modular, name: ${runname}}" \
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
#  --model.block.attention.qk_norm=null \
#  --model.block.attention.backend=torch \
#  --model.block.feed_forward_moe.lb_loss_weight=${lb}


python -m olmo_core.launch.beaker \
  --name $runname \
	--gpus $gpus \
  --nodes $nodes \
	--weka=oe-training-default \
  --shared-filesystem \
	--workspace ai2/flex2 \
	--cluster ai2/jupiter \
	--preemptible \
	--allow-dirty \
	--priority urgent \
	--env-secret "GITHUB_TOKEN=RYAN_GITHUB_TOKEN" "WANDB_API_KEY=RYAN_WANDB_API_KEY" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY" "HF_TOKEN=RYAN_HF_TOKEN" \
	-- src/scripts/train/olmoe-1B-7B_fsl_extension.py \
    $runname \
		--save-folder="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models/$runname" \
		--dataset.mix=mj_finemath4plus \
		--work-dir="/weka/oe-training-default/ryanwang/dataset-cache" \
		--trainer.callbacks.wandb="{enabled: true, entity: ryanyxw, project: olmoe-modular, name: ${runname}, tags: [extension, contpretrain, finemath]}" \
		--global_batch_size=${global_batch_size} \
		--num-tokens=${num_tokens} \
		--lr=${lr} \
		--load-path=${base_model_path}/model_and_optim \
		--model.block.feed_forward_moe.num_experts=128 \
		--dataset.generate_doc_lengths=true \
		--model.block.attention.backend=flash_2 \
		--model-type="two-level_lb-batch_reduce-dp_sharedexp_randpool" \
		--min_document_expert_pool=${min_document_expert_pool} \
		--max_document_expert_pool=${max_document_expert_pool} \
		--eval_document_expert_pool=${eval_document_expert_pool} \
		--num_shared_experts=${num_shared_experts} \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name="moe" \
		--model.block.attention.qk_norm=null \
		--model.block.feed_forward_moe.lb_loss_weight=${lb} \
		--trainer.callbacks.checkpointer.save_interval=600 \
		--trainer.callbacks.downstream_evaluator.eval_interval=250
