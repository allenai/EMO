# PARENT: "emo_1b14b_1t.sh" (config) + the deleted gpurouting speedtest
#         (git show 11e180f9^:scripts/models/emo_1b14b_1t_gpurouting_speedtest.sh)
# DESCRIPTION:
#     - Short throughput benchmark of the MERGED on-GPU (seg_id) two-level routing at the exact
#       emo_1b14b_1t config: EMO randpool, 128 experts / 1 shared, lr 4e-3, lb 1e-1, cosine,
#       16 nodes x 8 GPUs, seq 4096, global_batch 1024, compile on. Twin of
#       stdmoe_1b14b_128exp_speedtest_0727.sh — only the model-type (and EMO pool args) differ.
#     - Reference: old CPU-routing emo_1b14b_1t (wandb ryanyxw/olmoe-modular/d3x89bkp,
#       ~19k TPS/device, ~18% MFU).
#     - ALL checkpointing disabled so this writes essentially nothing; runs ~480 steps
#       (2B tokens at 4.19M tokens/step) then stops. Output isolated under models_speedtest_0727/.
# STATUS: NEW (throwaway speed benchmark — delete after reporting numbers)
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

# Fresh, non-colliding throwaway output root (safe to delete wholesale afterwards).
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/models_speedtest_0727"

# The weka ai2-llm mirror is incomplete; S3 is the source of truth for tokenized data.
DATA_ROOT="s3://ai2-llm"

# Assign unconditionally AFTER sourcing launch_common (it sets BEAKER_NODES at source time).
BEAKER_NODES=16
BEAKER_GPUS=8

lr=4e-3
lb=1e-1

num_shared_experts=1
num_experts=128

min_document_expert_pool=8
max_document_expert_pool=128
eval_document_expert_pool=32

runname="emo_1b14b_128exp_speedtest_0727"

# ~480 steps at global_batch 1024 seqs x 4096 tokens = ~4.19M tokens/step.
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
