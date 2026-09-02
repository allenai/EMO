# PARENT: "scripts/meta_learning/model_scripts/emo128_baseline_20b.sh"
# DESCRIPTION:
#     sparse_experts arm: EMO with QUARTER-SIZED experts (expert hidden_size 256 instead of the
#     olmoe_1B_7B preset's 0.5*d_model = 1024), 8 active (top_k=8 = 7 routed + 1 shared) out of
#     256 total experts. Everything else matches the meta_learning vanilla EMO-128e baseline:
#     pool min=8 / max=256 / eval=256, lb 1e-1, lr 2e-3 flat WSD trunk (decay_steps=1),
#     OLMoE-mix-0824, 8 nodes / 64 GPUs, 10B tokens (= 2385 steps; flat WSD trunk, extendable
#     later by resuming with a larger max_duration), meta_mode=vanilla (stock trainer +
#     pool-pinned lm-full / lm-pool32 evaluators).
#
#     Size (TransformerConfig.num_params, dolma2 vocab 100352, d_model 2048, 16 layers):
#         total 7.130B | active 0.889B | expert weights 402.7M / layer
#     Reference: emo128_baseline_20b.sh is 13.569B total / 1.489B active / 805.3M expert / layer.
#
#     Fixed checkpoints every ~2.5B tokens (2.5e9 / 4,194,304 tokens-per-step).
#
#   git add . && git commit && git push origin <branch>   # gantry clones from origin!
#   MODE=beaker bash scripts/sparse_experts/model_scripts/sparse_8of256_10b.sh
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../../launch_common.sh"

EXPERIMENT_NAME="sparse_experts"
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/${EXPERIMENT_NAME}"
DATA_ROOT="s3://ai2-llm"

BEAKER_NODES=8
BEAKER_GPUS=8
PREEMPTIBLE=0   # allocated slot: multi-node preemptible gangs queue for days / die mid-run

lr=2e-3
lb=1e-1

num_shared_experts=1
num_experts=256
expert_hidden_size=256   # quarter of the preset's int(0.5 * d_model) = 1024

min_document_expert_pool=8
max_document_expert_pool=${num_experts}
eval_document_expert_pool=${num_experts}

warmup_steps=500   # matches the meta_learning 20B baseline
decay_steps=1      # pure stable trunk: flat at peak LR

runname="sparse_8of256_10b"

launch src/scripts/train/olmoe-1B-7B_fsl_meta.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration='{value: 10_000_000_000, unit: tokens}' \
		--scheduler=wsd \
		--warmup_steps=${warmup_steps} \
		--decay_steps=${decay_steps} \
		--trainer.callbacks.checkpointer.save_interval=1000000 \
		--trainer.callbacks.checkpointer.fixed_steps="[596, 1192, 1788, 2384]" \
		--trainer.callbacks.checkpointer.ephemeral_save_interval=500 \
		--trainer.callbacks.checkpointer.keep_ephemeral=2 \
		--trainer.callbacks.checkpointer.pre_train_checkpoint=false \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=emo-extension \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags="[pretraining, ${EXPERIMENT_NAME}, sparse_h${expert_hidden_size}]" \
		--model.block.feed_forward_moe.num_experts=${num_experts} \
		--model.block.feed_forward_moe.hidden_size=${expert_hidden_size} \
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
		--train_module.meta_mode=vanilla \
		"$@"
