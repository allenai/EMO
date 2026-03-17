python -u -m offline_evals.run_eval\
    --task '{"task_name": "mmlu_biology:rc_test", "split": "test", "num_shots": 5, "primary_metric": "acc_per_char", "category_name": "biology", "metadata": {"regimes": ["OLMES-v0.1"], "alias": "mmlu_biology:rc_test::olmes"}}' \
    --batch-size 4 \
    --output-dir /root/ryanwang/phdbrainstorm/evals/testbed_mmlu \
    --save-raw-requests true \
    --num-workers 1 \
    --gpus 1 \
    --model /root/ryanwang/phdbrainstorm/FlexMoE/models/dense_1b_olmoe-mix_prenorm_noqknorm_1123/step30995-hf \
    --model-args '{"model_type": "hf"}'