python -m offline_evals.run_eval \
    --task '{"task_name": "arc_easy:mc", "split": "test", "num_shots": 5, "limit": 1000, "fewshot_source": "OLMES:ARC-Easy", "metadata": {"description": "ARC-Easy (MC) using OLMES-v0.1", "regimes": ["OLMES-v0.1"], "alias": "arc_easy:mc::olmes"}}' \
    --batch-size 4 \
    --output-dir /weka/oe-training-default/ryanwang/phdbrainstorm/evals/weka_oe-training-default_ryanwang_phdbrainstorm_models_olmoe-pretrain-mose-unbalanced-1012_step30995-hf \
    --save-raw-requests true \
    --num-workers 1 \
    --gpus 1 \
    --model /root/ryanwang/phdbrainstorm/models/olmoe-pretrain-mose-natural-1022/step30995/olmoe-finetune-arc_easy-mc/step1686-hf \
    --model-args '{"model_type": "hf"}'