# PARENT: "scripts/models_routerfixed/emo_1b14b_130b_routerfixed_noaux.sh"
# EXPERIMENT: models_routerfixed -- does the FROZEN router need to be TRAINED, or does any fixed routing work?
# DESCRIPTION:
#     Control for the noaux run. The noaux run grafted the TRAINED step-11921 routers onto a fresh
#     init and froze them. This run is identical in every way EXCEPT we DON'T graft anything: the
#     model is FULLY RANDOMLY INITIALIZED from scratch and we freeze its own RANDOM router init.
#
#     No --load_path / init checkpoint is needed. The non-router weights come out identical to the
#     noaux run for free: EMO's weight init is topology-independent and seed-deterministic (driven by
#     the model-level init_seed), and noaux's non-router weights are themselves just that same fresh
#     init (init_routerfixed_step0 grafted ONLY the routers onto the fresh init). So same seed/config
#     => same non-router weights; the ONLY difference between this run and noaux is random-vs-trained
#     FROZEN routers.
#
#     If this converges as well as noaux, freezing the trained router was not special -- any fixed
#     routing function suffices. If it does markedly worse, the trained routing function genuinely
#     matters. Still a NOAUX run: lb_loss_weight=0, z_loss_weight=0 (router is frozen, so the
#     router-shaping aux losses are off; this also disables the randpool reduce-dp all-reduce).
#
#     Same compute/recipe as noaux: 8 nodes / 64 GPUs, max_duration=130B, hard_stop=50B. Routers
#     frozen via the existing TransformerConfig.freeze_params glob (requires_grad=False before FSDP,
#     so the random routers never move and are excluded from the optimizer).
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
lb=0   # noaux: router-shaping LB loss switched off (router is frozen)

num_shared_experts=1 # 1 out of 8 will be shared experts

runname="emo_1b14b_130b_routerrandom_noaux"

# No init checkpoint: train fully from scratch and freeze the model's own random router init.
launch src/scripts/train/olmoe-1B-7B_fsl.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
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
		--trainer.callbacks.wandb.tags="[pretraining, ${EXPERIMENT_NAME}, routerrandom, noaux]" \
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
		--model.block.feed_forward_moe.z_loss_weight=0
