python -m offline_evals.run_eval \
    --task '{"task_name": "gsm8k:perplexity", "split": "test", "primary_metric": "bits_per_byte", "num_shots": 8, "fewshot_source": "STD:GSM8k", "metadata": {"regimes": ["OLMES-v0.2"]}}' \
    --batch-size 16 \
    --output-dir /root/ryanwang/phdbrainstorm/FlexMoE/evals/dense_1b_olmoe-mix_300B_1030_step71526-hf \
    --save-raw-requests true \
    --num-workers 1 \
    --gpus 1 \
    --model /root/ryanwang/phdbrainstorm/FlexMoE/models/dense_1b_olmoe-mix_300B_1030/step71526-hf \
    --model-args '{"model_type": "hf"}'