# PARENT: "scripts/models_v2/emo_64exp_50b_wsd_lr2e-3.sh" (from ryanyxw/emo_extend)
# DESCRIPTION:
#     - Short throughput benchmark of the on-GPU (seg_id) two-level routing at the EXACT config of
#       the matched EMO-vs-StdMoE 8-node comparison: EMO (two-level randpool), 64 experts / 1 shared,
#       lb 1e-1, WSD, lr 2e-3, 8 nodes x 8 GPUs. Only the routing CODE differs from the CPU-routing
#       run emo_64exp_50b_wsd_lr2e-3 (wandb emo-extension/s9024txw, ~21,592 TPS / 20.3% MFU).
#     - Compare throughput/device/TPS + MFU against StdMoE 64exp (emo-extension/0ucu7x8n, ~26,892 TPS).
#     - ALL checkpointing disabled (pre_train + fixed + ephemeral + save) so this benchmark writes
#       essentially nothing; runs ~480 steps then stops. Output isolated under models_gpurouting/.
# STATUS: NEW (throwaway speed benchmark)
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

# Fresh, non-colliding output root (throwaway benchmark dir; safe to delete wholesale).
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/models_gpurouting"
DATA_ROOT="s3://ai2-llm"

BEAKER_NODES=8
BEAKER_GPUS=8

lr=2e-3
lb=1e-1

num_shared_experts=1
num_experts=64

min_document_expert_pool=8
max_document_expert_pool=64
eval_document_expert_pool=64

# NOTE: this branch predates the emo_extend WSD scheduler args (--scheduler/--warmup_steps/
# --decay_steps), so we use the default cosine schedule. The scheduler only changes the per-step
# LR value, not per-step compute, so training throughput is unaffected — the comparison holds.

runname="emo_64exp_50b_wsd_lr2e-3_gpurouting_speedtest_0709"

# ~480 steps at global_batch 1024 seqs x 4096 = 4.19M tokens/step.
max_tokens=2_000_000_000

launch src/scripts/train/olmoe-1B-7B_fsl.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration="{value: ${max_tokens}, unit: tokens}" \
		--trainer.callbacks.checkpointer.save_interval=2000000 \
		--trainer.callbacks.checkpointer.ephemeral_save_interval=1000000 \
		--trainer.callbacks.checkpointer.fixed_steps="[]" \
		--trainer.callbacks.checkpointer.pre_train_checkpoint=false \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=emo-extension \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags='[speedtest, gpurouting]' \
		--model.block.feed_forward_moe.num_experts=${num_experts} \
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
		--model.block.feed_forward_moe.lb_loss_weight=${lb} \
		--trainer.callbacks.downstream_evaluator.eval_interval=1000000
