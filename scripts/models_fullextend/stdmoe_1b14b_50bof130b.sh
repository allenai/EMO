# PARENT: "scripts/models/stdmoe_1b14b_130b.sh"
# DESCRIPTION:
#     - Standard-MoE (token-level top-k, no EMO randpool / two-level routing) baseline
#       for the models_fullextend experiment. Model type moe_lbreducedp_sharedexp:
#       128 experts, 1 shared, lr 4e-3, lb 1e-1. Copied from scripts/models/ and
#       repointed to this experiment (wandb project emo-extension, tag models_fullextend,
#       weka save root, s3 data root).
#     - Run at the SAME compute as the ghost sweep's config #3 and the EMO baseline:
#       8 nodes / 64 GPUs, max_duration=130B but hard_stop=50B (= step 11921), so it is
#       apples-to-apples. 2-deep rolling ephemeral checkpoints + final; pre_train
#       checkpoint disabled.
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

EXPERIMENT_NAME="models_fullextend"
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/${EXPERIMENT_NAME}"
DATA_ROOT="s3://ai2-llm"

BEAKER_NODES=8
BEAKER_GPUS=8

lr=4e-3
lb=1e-1

num_shared_experts=1

runname="stdmoe_1b14b_50bof130b"

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
		--model-type="moe_lbreducedp_sharedexp" \
		--num_shared_experts=$num_shared_experts \
		--model.block.feed_forward_moe.num_experts=128 \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name="moe" \
		--model.block.sequence_mixer.qk_norm=null \
		--lr=${lr} \
		--model.block.feed_forward_moe.lb_loss_weight=${lb}
