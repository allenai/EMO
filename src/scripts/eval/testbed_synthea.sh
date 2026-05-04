
python -m offline_evals.run_eval \
    --task '{"task_name": "synthea:rc_test_0shot", "split": "test", "num_shots": 0, "metadata": {"description": "synthea test using OLMES-v0.1", "regimes": ["OLMES-v0.1"], "alias": "synthea:rc_test_0shot::olmes"}}' \
    --batch-size 1 \
    --output-dir /root/ryanwang/phdbrainstorm/evals/testbed_synthea \
    --save-raw-requests true \
    --num-workers 1 \
    --gpus 1 \
    --model /root/ryanwang/phdbrainstorm/Emo/models/dense_1b_olmoe-mix_prenorm_noqknorm_1123/step30995-hf \
    --model-args '{"model_type": "hf"}'