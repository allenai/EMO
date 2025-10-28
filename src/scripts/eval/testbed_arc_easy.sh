python -m offline_evals.run_eval \
    --task '{"task_name": "arc_easy", "split": "test", "primary_metric": "acc_per_char", "num_shots": 5, "limit": 1000, "fewshot_source": "OLMES:ARC-Easy", "metadata": {"description": "ARC-Easy (RC) using OLMES-v0.1", "regimes": ["OLMES-v0.1"], "alias": "arc_easy:rc::olmes"}}' \
    --batch-size 4 \
    --output-dir /root/ryanwang/phdbrainstorm/evals/testbed_arc_easy \
    --save-raw-requests true \
    --num-workers 1 \
    --gpus 1 \
    --model /root/ryanwang/phdbrainstorm/models/olmoe-pretrain-mose-natural-1022/step30995/olmoe-finetune-arc_easy-mc/step1686-hf \
    --model-args '{"model_type": "hf"}'