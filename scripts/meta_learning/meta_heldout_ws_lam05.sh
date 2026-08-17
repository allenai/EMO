# PARENT: "scripts/meta_learning/meta_heldout_ws.sh" + "scripts/meta_learning/meta_sametok_ws_lam05.sh"
# DESCRIPTION:
#     meta_learning arm (vi): HELDOUT working-set + LAMBDA ANCHOR, 20B tokens. The heldout split
#     (inner selective pass + pseudo-step on the first half of each rank's sequences, outer
#     full-routing working-set-masked pass on the other half) PLUS lambda_inner=0.5 +
#     lb_on_inner=true: the inner half's selective-forward loss (with LB/z aux) contributes
#     directly to the real update (.grad = 0.5*g_inner(half A, at theta) + g_outer(half B, at
#     theta')). Motivations: (a) the lambda term anchors theta and restores restricted-forward
#     training (the selective-inference gap lever); (b) it also fixes heldout's "only half the
#     batch updates" weakness — both halves now carry update signal; (c) heldout's cross-data
#     meta term is immune to the pseudo-step-leaning transient by construction.
# BUDGET CUT 2026-08-17: phase-1 shortened from 100B to 20B tokens (step 4768; fixed ckpts at
# 10B/20B). Names carry no token budget (WSD flat trunk; budgets change). Phase 2 = k=32-CPT-style cluster-wise selective CPT on tokens 20B-40B (window
# extraction: scripts/meta_learning/run_extract_20b_40b.sh).
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

inner_lr=3e-1
inner_grad_clip=10
inner_pool_size=32
lambda_inner=0.5

runname="meta128_heldout_ws_lam05"

launch src/scripts/train/olmoe-1B-7B_fsl_meta.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration='{value: 20_000_000_000, unit: tokens}' \
		--scheduler=wsd \
		--warmup_steps=${warmup_steps} \
		--decay_steps=${decay_steps} \
		--trainer.callbacks.checkpointer.save_interval=1000000 \
		--trainer.callbacks.checkpointer.fixed_steps="[2384, 4768]" \
		--trainer.callbacks.checkpointer.ephemeral_save_interval=500 \
		--trainer.callbacks.checkpointer.keep_ephemeral=2 \
		--trainer.callbacks.checkpointer.pre_train_checkpoint=false \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=emo-extension \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags="[pretraining, ${EXPERIMENT_NAME}, heldout_ws_lam05]" \
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
		--train_module.inner_grad_clip=${inner_grad_clip} \
		--train_module.inner_pool_size=${inner_pool_size} \
		--train_module.outer_expert_update=working_set \
		--train_module.lambda_inner=${lambda_inner} \
		--train_module.lb_on_inner=true
