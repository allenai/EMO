# PARENT: "extensions/twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step23_ct-math_8-128.sh"
# DESCRIPTION:
#     - Same as _ct-m8-128_ef8 but ALSO detaches the router for non-top-8 experts.
#     - Per-doc top-8 expert MLPs train AND those experts' router rows train (via CE +
#       LB + z paths). For non-top-8 expert MLPs and shared expert: no grad. For
#       non-top-8 router rows: CE/LB/z paths cut (only softmax cross-coupling leaks).
#     - Note: runname drops trailing "38419" from step238419 to fit the 128-char
#       beaker name limit. Actual base checkpoint step is 238419 (see base_model_path).
# STATUS: NEW
##############################################################
min_document_expert_pool=8
max_document_expert_pool=128
eval_document_expert_pool=32
lb=1e-1
lr=4e-4

# extension_finetune_* flags: top_e=8 expert MLPs train; router CE/LB/z cut for non-top-e
ef_mode=true
ef_top_e=8
ef_detach_router=true

num_billion_tokens=10
num_tokens=$((num_billion_tokens * 1000000000))

base_model_path="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models/twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419/step250339"

nodes=16
gpus=8

num_shared_experts=1

runname="twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step2_ct-m8-128_ef8r"


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
#  --model.block.feed_forward_moe.lb_loss_weight=${lb} \
#  --model.block.feed_forward_moe.router.extension_finetune_mode=${ef_mode} \
#  --model.block.feed_forward_moe.router.extension_finetune_top_e=${ef_top_e} \
#  --model.block.feed_forward_moe.router.extension_finetune_detach_router=${ef_detach_router}


python -m olmo_core.launch.beaker \
  --name $runname \
	--gpus $gpus \
  --nodes $nodes \
	--weka=oe-training-default \
  --shared-filesystem \
	--workspace ai2/flex2 \
	--cluster ai2/jupiter \
	--beaker-image tylerr/olmo-core-tch280cu128-2025-11-25 \
	--preemptible \
	--allow-dirty \
	--priority urgent \
	--env-secret "GITHUB_TOKEN=RYAN_GITHUB_TOKEN" "WANDB_API_KEY=RYAN_WANDB_API_KEY" "BEAKER_TOKEN=RYAN_BEAKER_TOKEN" "AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY" "HF_TOKEN=RYAN_HF_TOKEN" \
	-- src/scripts/train/olmoe-1B-7B_fsl_extension.py \
    $runname \
		--save-folder="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models/$runname" \
		--dataset.mix=mj_finemath4plus \
		--work-dir="/weka/oe-training-default/ryanwang/dataset-cache" \
		--trainer.callbacks.wandb="{enabled: true, entity: ryanyxw, project: olmoe-modular, name: ${runname}, tags: [extension, contpretrain, finemath, top-e8, detach-router]}" \
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
		--model.block.feed_forward_moe.router.extension_finetune_mode=${ef_mode} \
		--model.block.feed_forward_moe.router.extension_finetune_top_e=${ef_top_e} \
		--model.block.feed_forward_moe.router.extension_finetune_detach_router=${ef_detach_router} \
		--trainer.callbacks.checkpointer.save_interval=600 \
		--trainer.callbacks.downstream_evaluator.eval_interval=250
