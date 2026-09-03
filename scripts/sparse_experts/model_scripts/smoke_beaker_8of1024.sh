# PARENT: "scripts/sparse_experts/model_scripts/sparse_8of1024_10b.sh"
# DESCRIPTION:
#     1-GPU Beaker smoke for the >512-expert grouped-GEMM path (DroplessMoEMLP._gmm_batch_sizes
#     moves batch_sizes to host beyond 512 groups; grouped_gemm's CUTLASS backend rejects CUDA
#     batch_sizes with >512 groups: "At most 512 experts are supported when batch_sizes is a
#     CUDA tensor"). The local `emo` env has no grouped_gemm (python-loop fallback), so this can
#     only be exercised on the Beaker image.
#
#     Tiny model: 2 layers, 1024 experts x hidden 256, top_k 8 (7 routed + 1 shared), pool
#     8..1024, compile on (as in production), same 4 x 4096 rank micro-batch as the real run,
#     ${STEPS:-20} steps, no checkpoints / W&B / evals. Pass = training steps print without a
#     RuntimeError from grouped_gemm.
#
#   git add . && git commit && git push origin <branch>   # gantry clones from origin!
#   MODE=beaker BEAKER_NO_FOLLOW=1 bash scripts/sparse_experts/model_scripts/smoke_beaker_8of1024.sh
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../../launch_common.sh"

EXPERIMENT_NAME="sparse_experts"
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/${EXPERIMENT_NAME}"
DATA_ROOT="s3://ai2-llm"

BEAKER_NODES=1
BEAKER_GPUS=1
# small job: leave preemptible (default) so it schedules from the unallocated pool

STEPS="${STEPS:-20}"
num_experts=1024
expert_hidden_size=256

runname="smoke_8of1024_gg"

launch src/scripts/train/olmoe-1B-7B_fsl_meta.py $runname \
		--save-folder="${MODELS_DIR}/smoke/$runname" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.hard_stop="{value: ${STEPS}, unit: steps}" \
		--trainer.max_duration='{value: 10_000_000_000, unit: tokens}' \
		--scheduler=wsd \
		--warmup_steps=10 \
		--decay_steps=1 \
		--global_batch_size=4 \
		--trainer.no_checkpoints=true \
		--trainer.callbacks.checkpointer.save_interval=1000000 \
		--trainer.callbacks.checkpointer.ephemeral_save_interval=null \
		--trainer.callbacks.checkpointer.pre_train_checkpoint=false \
		--trainer.callbacks.wandb.enabled=false \
		--trainer.callbacks.downstream_evaluator.enabled=false \
		--trainer.callbacks.lm_eval_full.enabled=false \
		--trainer.callbacks.lm_eval_pool32.enabled=false \
		--model.n_layers=2 \
		--model.block.feed_forward_moe.num_experts=${num_experts} \
		--model.block.feed_forward_moe.hidden_size=${expert_hidden_size} \
		--dataset.generate_doc_lengths=true \
		--model.block.sequence_mixer.backend=flash_2 \
		--min_document_expert_pool=8 \
		--max_document_expert_pool=${num_experts} \
		--eval_document_expert_pool=${num_experts} \
		--num_shared_experts=1 \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name="moe" \
		--model.block.sequence_mixer.qk_norm=null \
		--lr=2e-3 \
		--model.block.feed_forward_moe.lb_loss_weight=1e-1 \
		--train_module.meta_mode=vanilla \
		"$@"
