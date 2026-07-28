# PARENT: stdmoe_1b14b_128exp_speedtest_0727.sh (git show 6ad03e63:scripts/models/...)
# DESCRIPTION:
#     - flash_2-backend variant of the 0727 StdMoE speedtest, to isolate the attention-backend
#       contribution to the EMO-vs-StdMoE throughput comparison. Identical to the 0727 run
#       (which measured 20,411 TPS/device, 19.21% MFU steady-state on torch SDPA) EXCEPT
#       --model.block.sequence_mixer.backend=flash_2, matching the EMO twin's backend
#       (still no generate_doc_lengths — full causal attention, as StdMoE trains).
#     - ALL checkpointing disabled; runs ~480 steps (2B tokens) then stops. Output isolated
#       under models_speedtest_0728/.
# STATUS: NEW (throwaway speed benchmark — delete after reporting numbers)
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

# Fresh, non-colliding throwaway output root (safe to delete wholesale afterwards).
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/models_speedtest_0728"

# The weka ai2-llm mirror is incomplete; S3 is the source of truth for tokenized data.
DATA_ROOT="s3://ai2-llm"

# Assign unconditionally AFTER sourcing launch_common (it sets BEAKER_NODES at source time).
BEAKER_NODES=16
BEAKER_GPUS=8

lr=4e-3
lb=1e-1

num_shared_experts=1
num_experts=128

runname="stdmoe_1b14b_128exp_flash2_speedtest_0728"

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
		--model.block.sequence_mixer.backend=flash_2 \
		--model-type="moe_lbreducedp_sharedexp" \
		--num_shared_experts=$num_shared_experts \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name="moe" \
		--model.block.sequence_mixer.qk_norm=null \
		--lr=${lr} \
		--model.block.feed_forward_moe.lb_loss_weight=${lb} \
		--trainer.callbacks.downstream_evaluator.eval_interval=1000000
