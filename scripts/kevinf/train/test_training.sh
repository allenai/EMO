python src/scripts/kevinf/train/OLMo3-1B.py test-run \
    --dry-run \
    --save-folder="/tmp/test" \
    --work-dir="/tmp/dataset-cache" \
    --trainer.max_duration="{value: 10000000000, unit: tokens}" \
    --trainer.hard_stop="{value: 10000000000, unit: tokens}" \
    --trainer.callbacks.downstream_evaluator.eval_interval=100 \
    --dataset.mix=the-pile-of-law \
    --train_module.optim.lr=5e-5 \
    --train_module.scheduler.warmup_fraction=0.5