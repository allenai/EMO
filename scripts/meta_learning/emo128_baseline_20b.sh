# PARENT: "scripts/models_v2/emo_64exp_50b_wsd_lr2e-3.sh"
# DESCRIPTION:
#     meta_learning pilot arm (i): vanilla EMO-128e baseline, from scratch. 128 experts / 1
#     shared, pool min=8 / max=128 / eval=128, lb 1e-1, lr 2e-3 flat WSD trunk (decay_steps=1),
#     OLMoE-mix-0824, 8 nodes / 64 GPUs. Runs through the meta entry script with
#     meta_mode=vanilla (bit-identical to the parent trainer) so it gets the same pool-pinned LM
#     evaluators (lm-full vs lm-pool32 -> the selective-vs-full CE gap) as the meta arms.
#
#     Trained to 20B tokens: 10B (step 2384) is the matched-TOKEN comparison point vs the meta
#     arms; 20B is the matched-COMPUTE point vs the ~2x-cost same_tokens arm.
#     Fixed checkpoints every ~2.5B tokens (2.5e9 / 4,194,304 tokens-per-step).
#
#   git add . && git commit && git push origin <branch>   # gantry clones from origin!
#   MODE=beaker bash scripts/meta_learning/emo128_baseline_10b.sh
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

EXPERIMENT_NAME="meta_learning"
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/${EXPERIMENT_NAME}"
DATA_ROOT="s3://ai2-llm"

BEAKER_NODES=8
BEAKER_GPUS=8

lr=2e-3
lb=1e-1

num_shared_experts=1
num_experts=128

min_document_expert_pool=8
max_document_expert_pool=128
eval_document_expert_pool=128

warmup_steps=500   # 10B pilot = ~2384 steps; the models_v2 warmup of 2000 would eat most of it
decay_steps=1      # pure stable trunk: flat at peak LR

runname="meta128_vanilla_20b"

launch src/scripts/train/olmoe-1B-7B_fsl_meta.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration='{value: 20_000_000_000, unit: tokens}' \
		--scheduler=wsd \
		--warmup_steps=${warmup_steps} \
		--decay_steps=${decay_steps} \
		--trainer.callbacks.checkpointer.save_interval=1000000 \
		--trainer.callbacks.checkpointer.fixed_steps="[596, 1192, 1788, 2384, 2980, 3576, 4172, 4768]" \
		--trainer.callbacks.checkpointer.ephemeral_save_interval=500 \
		--trainer.callbacks.checkpointer.keep_ephemeral=2 \
		--trainer.callbacks.checkpointer.pre_train_checkpoint=false \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=emo-extension \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags="[pretraining, ${EXPERIMENT_NAME}, vanilla]" \
		--model.block.feed_forward_moe.num_experts=${num_experts} \
		--dataset.generate_doc_lengths=true \
		--model.block.sequence_mixer.backend=flash_2 \
		--min_document_expert_pool=${min_document_expert_pool} \
		--max_document_expert_pool=${max_document_expert_pool} \
		--eval_document_expert_pool=${eval_document_expert_pool} \
		--num_shared_experts=$num_shared_experts \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name="moe" \
		--model.block.sequence_mixer.qk_norm=null \
		--lr=${lr} \
		--model.block.feed_forward_moe.lb_loss_weight=${lb} \
		--train_module.meta_mode=vanilla
