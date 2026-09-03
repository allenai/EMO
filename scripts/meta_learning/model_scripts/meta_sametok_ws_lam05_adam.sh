# PARENT: "scripts/meta_learning/model_scripts/meta_sametok_ws_lam05.sh"
# DESCRIPTION:
#     meta_learning REDESIGN arm: the sametok_ws_lam05 config with the pseudo-step upgraded from
#     the raw-SGD probe to an Adam-preconditioned, schedule-matched, magnitude-randomized step —
#     the (1)+(2) fixes for why one selective step didn't transfer in phase-2:
#       (1) inner_optim=adam   -- the pseudo-step is the AdamW step the real optimizer WOULD take
#                                 on g_inner at the CURRENT expert moments (read-only; moments are
#                                 consumed unchanged by the outer step). theta' is the true "next
#                                 selective AdamW step", not a raw-SGD probe of a different shape;
#                                 Adam-scale delta (~lr/coord) is immune to bf16 self-extinction.
#       (2) inner_lr_mode=match_lr + inner_lr_scale=[1,32] -- base step size = the LIVE scheduler
#                                 lr (so theta' tracks the real committed step), and each step
#                                 samples a log-uniform displacement scale in [1,32] so the transfer
#                                 property is trained across a band of committed-step magnitudes
#                                 (the update-magnitude analog of vanilla's random pools).
#     Everything else is IDENTICAL to meta128_sametok_ws_lam05 (same_tokens, working-set outer
#     update, lambda_inner=0.5 + lb_on_inner, inner_pool_size=32, 128e/1-shared, lr 2e-3, lb 1e-1,
#     WSD warmup 2000, 20B tokens, fixed ckpts 10B/20B) so this is a clean A/B on the pseudo-step
#     mechanism. inner_grad_clip lowered 10 -> 1.0 to match the trainer's global grad clip (Adam
#     normalizes; the large SGD-probe clip is no longer needed). inner_lr is a fixed-mode fallback,
#     unused under match_lr.
#
#   git add . && git commit && git push origin <branch>   # gantry clones from origin!
#   MODE=beaker PREEMPTIBLE=0 bash scripts/meta_learning/model_scripts/meta_sametok_ws_lam05_adam.sh
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../../launch_common.sh"

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

inner_lr=2e-3            # fixed-mode fallback; ignored under match_lr
inner_grad_clip=1.0      # match the trainer's max_grad_norm (Adam normalizes)
inner_pool_size=32
lambda_inner=0.5
inner_lr_scale_min=1
inner_lr_scale_max=32

runname="meta128_sametok_ws_lam05_adam"

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
		--trainer.callbacks.wandb.tags="[pretraining, ${EXPERIMENT_NAME}, same_tokens_ws_lam05_adam]" \
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
		--train_module.inner_optim=adam \
		--train_module.inner_lr_mode=match_lr \
		--train_module.inner_lr=${inner_lr} \
		--train_module.inner_grad_clip=${inner_grad_clip} \
		--train_module.inner_pool_size=${inner_pool_size} \
		--train_module.outer_expert_update=working_set \
		--train_module.lambda_inner=${lambda_inner} \
		--train_module.lb_on_inner=true \
		--train_module.inner_lr_scale_min=${inner_lr_scale_min} \
		--train_module.inner_lr_scale_max=${inner_lr_scale_max}
