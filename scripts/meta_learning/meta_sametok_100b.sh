# PARENT: "scripts/meta_learning/emo128_vanilla_100b.sh"
# DESCRIPTION:
#     meta_learning FULL RUN arm (ii): FOMAML same-tokens, 100B tokens. Identical recipe to the
#     vanilla-128e 100B baseline EXCEPT meta_mode=same_tokens: every step runs a selective inner
#     pass (per-doc router top-32 of 127 non-shared experts), a temporary SGD pseudo-step on the
#     expert weights (inner_lr from the 2026-08-16 smoke sweep: best CE, cosine ~0.96;
#     delta/weight ~7e-5 at toy scale — WATCH train/meta delta/weight norm in the first steps,
#     raise alpha if it stays <~1e-4 under bf16), then a full-routing outer pass on the SAME
#     tokens whose gradient (at the perturbed weights) is the real update. ~2.1x step cost vs
#     the baseline (compare at matched tokens, or vs baseline half-token points for matched
#     compute).
#
#   git add . && git commit && git push origin <branch>   # gantry clones from origin!
#   MODE=beaker bash scripts/meta_learning/meta_sametok_100b.sh
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

runname="meta128_sametok_100b"

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
		--trainer.callbacks.wandb.tags="[pretraining, ${EXPERIMENT_NAME}, same_tokens, 100b]" \
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
		--train_module.meta_mode=same_tokens \
		--train_module.inner_lr=${inner_lr} \
		--train_module.inner_pool_size=${inner_pool_size}
