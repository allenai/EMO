
python -m offline_evals.run_eval \
    --task "synthea:rc_test_0shot::olmes" \
    --batch-size 1 \
    --output-dir /root/ryanwang/phdbrainstorm/evals/testbed_synthea \
    --save-raw-requests true \
    --num-workers 1 \
    --gpus 1 \
    --model /root/ryanwang/phdbrainstorm/models/dense_1b_olmoe-mix_prenorm_noqknorm_1123/step30995-hf \
    --model-args '{"model_type": "hf"}'