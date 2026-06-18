# PARENT: "scripts/models_fullextend/emo_1b14b_130b.sh"
# EXPERIMENT: models_routerfixed -- does the router need to be LEARNED during pretraining?
# DESCRIPTION:
#     Take the trained routers from emo_1b14b_130b (step 11921 = 50B tokens), graft them onto a
#     FRESH model init (everything else random-init, byte-exact to the original run's step 0),
#     FREEZE the routers, and retrain from scratch on the identical EMO recipe. If loss still
#     converges, a good routing function found once can be held fixed while the experts organise
#     around it.
#
#     This is the KEEPAUX ablation: the router-shaping auxiliary losses are kept IDENTICAL to the
#     baseline (lb_loss_weight=1e-1, z_loss_weight=1e-3 default). With the router frozen these no
#     longer move it, but still backprop into the activations (pushing earlier layers to balance the
#     fixed router) and the reduce-dp all-reduce still runs. The NOAUX sibling zeroes them out.
#
#     The init checkpoint is built once by scripts/models_routerfixed/build_step0.sh and loaded with
#     a fresh optimizer at step 0 (--load_trainer_state/--load_optim_state false). Freezing uses the
#     existing TransformerConfig.freeze_params glob mechanism (same as extend_finemath_frz_*).
#     Same compute/recipe as the baseline: 8 nodes / 64 GPUs, max_duration=130B, hard_stop=50B.
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"

EXPERIMENT_NAME="models_routerfixed"
MODELS_DIR="/weka/oe-training-default/ryanwang/EMO/${EXPERIMENT_NAME}"
DATA_ROOT="s3://ai2-llm"

BEAKER_NODES=8
BEAKER_GPUS=8

min_document_expert_pool=8
max_document_expert_pool=128
eval_document_expert_pool=32
lr=4e-3
lb=1e-1   # keepaux: auxiliary LB loss kept identical to the baseline

num_shared_experts=1 # 1 out of 8 will be shared experts

runname="emo_1b14b_130b_routerfixed_keepaux"

# Router-fixed init checkpoint (built once by build_step0.sh): fresh weights + trained, grafted routers.
INIT_CHECKPOINT="${MODELS_DIR}/init_routerfixed_step0/model_and_optim"

launch src/scripts/train/olmoe-1B-7B_fsl.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
		--load_path="${INIT_CHECKPOINT}" \
		--load_trainer_state=false \
		--load_optim_state=false \
		--model.freeze_params='[blocks.*.feed_forward_moe.router.*]' \
		--dataset.mix=OLMoE-mix-0824 \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration='{value: 130_000_000_000, unit: tokens}' \
		--trainer.hard_stop='{value: 50_000_000_000, unit: tokens}' \
		--trainer.callbacks.checkpointer.save_interval=1000000 \
		--trainer.callbacks.checkpointer.ephemeral_save_interval=500 \
		--trainer.callbacks.checkpointer.keep_ephemeral=2 \
		--trainer.callbacks.checkpointer.pre_train_checkpoint=false \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ryanyxw \
		--trainer.callbacks.wandb.project=emo-extension \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags="[pretraining, ${EXPERIMENT_NAME}, routerfixed, keepaux]" \
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
		--model.block.feed_forward_moe.lb_loss_weight=${lb}
