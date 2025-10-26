python -u -m offline_evals.run_eval\
    --task '{"task_name": "hellaswag", "split": "validation", "primary_metric": "acc_per_char", "num_shots": 5, "limit": 1000, "fewshot_source": "OLMES:hellaswag", "metadata": {"regimes": ["OLMES-v0.1"], "alias": "hellaswag:rc::olmes"}}'\
    --batch-size 4 \
    --output-dir /root/phdbrainstorm/evals/testbed_hellaswag \
    --save-raw-requests true \
    --num-workers 1 \
    --gpus 1 \
    --model /root/phdbrainstorm/models/olmoe-pretrain-replicate/step30995-hf \
    --model-args '{"model_type": "hf"}'