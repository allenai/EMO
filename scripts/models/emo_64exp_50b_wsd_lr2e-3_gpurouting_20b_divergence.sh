# PARENT: "scripts/models/emo_64exp_50b_wsd_lr2e-3_gpurouting_speedtest.sh"
# DESCRIPTION:
#     - Longer (~20B token, ~4770 step) run of the on-GPU (seg_id) two-level routing to confirm the
#       loss trajectory does NOT diverge from the old CPU-routing run over a longer horizon.
#     - Matched to the CPU-routing run emo_64exp_50b_wsd_lr2e-3 (wandb emo-extension/s9024txw):
#       64 experts / 1 shared, lb 1e-1, lr 2e-3, 8 nodes. That run used WSD = linear warmup (2000
#       steps) then FLAT at peak 2e-3. This branch lacks the WSD scheduler, so we reproduce the same
#       schedule with the default CosWithWarmup + alpha_f=1.0 (verified flat at peak after warmup).
#     - Overlay train/CE loss vs s9024txw; expect tight tracking, with only the sanctioned RNG drift
#       (per-doc pool size now drawn with GPU randint). ALL checkpointing disabled (throwaway).
# STATUS: NEW (throwaway divergence check)
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

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

runname="emo_64exp_50b_wsd_lr2e-3_gpurouting_20b_0710"

# ~4770 steps at global_batch 1024 seqs x 4096 = 4.19M tokens/step.
max_tokens=20_000_000_000

launch src/scripts/train/olmoe-1B-7B_fsl.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration="{value: ${max_tokens}, unit: tokens}" \
		--train_module.scheduler.alpha_f=1.0 \
		--trainer.callbacks.checkpointer.save_interval=20000000 \
		--trainer.callbacks.checkpointer.ephemeral_save_interval=10000000 \
		--trainer.callbacks.checkpointer.fixed_steps="[]" \
		--trainer.callbacks.checkpointer.pre_train_checkpoint=false \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=emo-extension \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags='[speedtest, gpurouting, divergence]' \
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
