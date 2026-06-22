# PARENT: "scripts/models_sizescaling/emo_1b14b_130b.sh"
# DESCRIPTION:
#     - No-ghost EMO baseline for the models_fullextend experiment, run at the SAME
#       compute as the ghost coeff-mode sweep's config #3: 8 nodes / 64 GPUs,
#       max_duration=130B but hard_stop=50B (= step 11921), so it is apples-to-apples
#       with the 8-node random ghost run. The unmodified EMO 1B/14B randpool recipe
#       (128 experts, 1 shared, pool 8-128 / eval 32, lr 4e-3, lb 1e-1), no ghost knobs.
#       2-deep rolling ephemeral checkpoints + final; pre_train_checkpoint disabled.
#       CAVEAT: at 8 nodes the reduce-dp batch-level LB statistics are reduced over a
#       smaller sequence population (64 vs 128 GPUs) than the 16-node usage/uniform runs
#       -- same caveat as config #3.
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

EXPERIMENT_NAME="models_fullextend"
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/${EXPERIMENT_NAME}"
DATA_ROOT="s3://ai2-llm"

BEAKER_NODES=8
BEAKER_GPUS=8

min_document_expert_pool=8
max_document_expert_pool=128
eval_document_expert_pool=32
lr=4e-3
lb=1e-1

num_shared_experts=1 # 1 out of 8 will be shared experts

runname="emo_1b14b_50bof130b"

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
		--model.block.feed_forward_moe.lb_loss_weight=${lb}
