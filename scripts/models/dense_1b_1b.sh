# PARENT: "dense_1b_olmoe-mix_1019.sh"
# DESCRIPTION:
#     - Different from parent by removing qknorm and using prenorm
# STATUS: USED
##############################################################
DATA_ROOT="${DATA_ROOT:-/weka/oe-training-default/sewonm}"
source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"


lr=4e-3

# 104 hours for 130B tokens
# 1 hour for 1B token

runname="dense_1b_1b"
launch src/scripts/train/olmo2-1B.py $runname \
		--save-folder="${MODELS_DIR}/$runname" \
		--dataset.mix=codelion-dclm-baseline-1B \
		--work-dir="${DATASET_CACHE}" \
		--trainer.max_duration='{value: 1_000_000_000, unit: tokens}' \
		--trainer.callbacks.wandb.enabled=true \
		--trainer.callbacks.wandb.entity=ai2-llm \
		--trainer.callbacks.wandb.project=sewonm-icsl \
		--trainer.callbacks.wandb.name="${runname}" \
		--trainer.callbacks.wandb.tags='[pretraining]' \
		--trainer.callbacks.downstream_evaluator.tasks='[hellaswag]' \
		--trainer.callbacks.downstream_evaluator.eval_interval=null \
		--trainer.callbacks.downstream_evaluator.eval_on_finish=true \
		--trainer.callbacks.heldout_evaluator='{_CLASS_: olmo_core.train.callbacks.evaluator_callback.LMEvaluatorCallbackConfig, eval_dataset: {_CLASS_: olmo_core.data.numpy_dataset.NumpyPaddedFSLDatasetConfig, tokenizer: {_CLASS_: olmo_core.data.tokenizer.TokenizerConfig, vocab_size: 100278, eos_token_id: 100257, pad_token_id: 100277, identifier: allenai/dolma2-tokenizer}, paths: [/weka/oe-training-default/ai2-llm/eval-data/perplexity/v3_small_dolma2-tokenizer/c4_en/val/part-0-00000.npy], metadata: [{label: c4_en-validation}], sequence_length: 4096, work_dir: /weka/oe-training-default/sewonm/dataset-cache}, eval_interval: null, eval_on_finish: true, name: heldout}' \
		--dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
		--model.block.name="default" \
		--model.block.sequence_mixer.qk_norm=null \
		--train_module.scheduler.warmup=20 \
		--lr=${lr}
