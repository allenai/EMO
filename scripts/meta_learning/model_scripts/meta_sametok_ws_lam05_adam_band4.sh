# PARENT: "scripts/meta_learning/model_scripts/meta_sametok_ws_lam05_adam.sh"
# DESCRIPTION:
#     meta_learning REDESIGN arm, GENTLE-BAND variant. Identical to the adam arm EXCEPT:
#       - inner_lr_scale = [1, 4]   -- a GENTLE magnitude band (vs the original [1,32], which
#                                      regressed full CE by +0.30 because up-to-32x committed-step
#                                      displacements damaged base quality). Keeps some update-
#                                      magnitude randomization (the analog of vanilla's random
#                                      pools) but caps theta' at ~4x the true next selective step.
#       - BEAKER_NODES = 4          -- half the compute (user request). NOTE: at lb=1e-1 the
#                                      reduce-dp LB statistic granularity scales with DP width,
#                                      so this is NOT a bit-clean A/B vs the 8-node baselines;
#                                      it is a diagnostic re-run.
#     Everything else IDENTICAL to meta_sametok_ws_lam05_adam.sh. Run alongside the scale1
#     variant: scale1 isolates the mechanism, band4 tests whether a mild magnitude band helps
#     the selective-vs-full gap without the base-quality cost of the [1,32] band.
#
#   git add . && git commit && git push origin <branch>   # gantry clones from origin!
#   MODE=beaker PREEMPTIBLE=0 bash scripts/meta_learning/model_scripts/meta_sametok_ws_lam05_adam_band4.sh
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../../launch_common.sh"

EXPERIMENT_NAME="meta_learning"
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/${EXPERIMENT_NAME}"
DATA_ROOT="s3://ai2-llm"

BEAKER_NODES=4
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
inner_lr_scale_max=4

runname="meta128_sametok_ws_lam05_adam_band4"

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
		--trainer.callbacks.wandb.tags="[pretraining, ${EXPERIMENT_NAME}, same_tokens_ws_lam05_adam_band4]" \
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
