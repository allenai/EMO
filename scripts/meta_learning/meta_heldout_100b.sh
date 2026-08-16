# PARENT: "scripts/meta_learning/meta_sametok_100b.sh"
# DESCRIPTION:
#     meta_learning FULL RUN arm (iii): FOMAML held-out outer split, 100B tokens. Identical to
#     the same-tokens 100B arm EXCEPT meta_mode=heldout: each rank's micro-batches are split in
#     half — inner pass (selective top-32 + pseudo-step) on the first half, outer full-routing
#     pass on the second half, so the meta-gradient rewards inner updates that GENERALIZE
#     (standard MAML support/query). Per-step compute ~= the vanilla baseline, but the real
#     (outer) update sees half the tokens per step (effective outer batch 512 instances) —
#     documented caveat. Requires even micro-batches/rank: 1024/64 = 16 instances -> 4 mbs. OK.
#
#   git add . && git commit && git push origin <branch>   # gantry clones from origin!
#   MODE=beaker bash scripts/meta_learning/meta_heldout_100b.sh
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

EXPERIMENT_NAME="meta_learning"
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/${EXPERIMENT_NAME}"
DATA_ROOT="s3://ai2-llm"

BEAKER_NODES=8
BEAKER_GPUS=8
BEAKER_NO_FOLLOW=1

lr=2e-3
lb=1e-1

num_shared_experts=1
num_experts=128

min_document_expert_pool=8
max_document_expert_pool=128
eval_document_expert_pool=128

warmup_steps=2000
decay_steps=1

inner_lr=3e-2
inner_pool_size=32

runname="meta128_heldout_100b"

launch src/scripts/train/olmoe-1B-7B_fsl_meta.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration='{value: 100_000_000_000, unit: tokens}' \
		--scheduler=wsd \
		--warmup_steps=${warmup_steps} \
		--decay_steps=${decay_steps} \
		--trainer.callbacks.checkpointer.save_interval=1000000 \
		--trainer.callbacks.checkpointer.fixed_steps="[2384, 4768, 7153, 9537, 11921, 14305, 16689, 19073, 21458]" \
		--trainer.callbacks.checkpointer.ephemeral_save_interval=500 \
		--trainer.callbacks.checkpointer.keep_ephemeral=2 \
		--trainer.callbacks.checkpointer.pre_train_checkpoint=false \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=emo-extension \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags="[pretraining, ${EXPERIMENT_NAME}, heldout, 100b]" \
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
		--train_module.meta_mode=heldout \
		--train_module.inner_lr=${inner_lr} \
		--train_module.inner_pool_size=${inner_pool_size}
