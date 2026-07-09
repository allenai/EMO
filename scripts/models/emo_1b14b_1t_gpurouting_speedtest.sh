# PARENT: "emo_1b14b_1t.sh"
# DESCRIPTION:
#     - Short throughput benchmark of the on-GPU (seg_id) two-level routing on branch
#       ryanyxw/emo_gpurouting. IDENTICAL settings to emo_1b14b_1t.sh (128 experts, lr=4e-3,
#       lb=1e-1, 16 nodes x 8 GPUs, seq 4096, global_batch 1024, compile on) — only the code
#       path differs. Runs ~a few hundred steps then stops.
#     - Compare throughput/device/TPS and throughput/device/MFU against the prior cosine run
#       emo_1b14b_1t (wandb ryanyxw/olmoe-modular/d3x89bkp: ~19k TPS, ~18% MFU).
# STATUS: NEW
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

# Fresh, non-colliding output root (no models_gpurouting dir exists on weka/in the repo tree).
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/models_gpurouting"

min_document_expert_pool=8
max_document_expert_pool=128
eval_document_expert_pool=32
lr=4e-3
lb=1e-1

nodes=16
gpus=8
# calculate by taking nodes multiply by gpus multiply by 4 (since we have 4 as micro batch size)
lb_global_batch_size=$((nodes * gpus * 4))

num_shared_experts=1 # 1 out of 8 will be shared experts

runname="emo_1b14b_1t_gpurouting_speedtest_0709"

# ~480 steps at global_batch 1024 seqs x 4096 tokens = ~4.19M tokens/step.
max_tokens=2_000_000_000

launch src/scripts/train/olmoe-1B-7B_fsl.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration="{value: ${max_tokens}, unit: tokens}" \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=olmoe-modular \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags='[pretraining, speedtest, gpurouting]' \
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
		--model.block.feed_forward_moe.lb_loss_weight=${lb} \
		--trainer.callbacks.checkpointer.save_interval=1000000 \
		--trainer.callbacks.checkpointer.ephemeral_save_interval=1000000 \
		--trainer.callbacks.downstream_evaluator.eval_interval=1000000
