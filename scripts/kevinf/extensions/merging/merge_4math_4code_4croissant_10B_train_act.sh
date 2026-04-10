#!/bin/bash
# Merge math, code, and croissant trained experts into the base model.
# Base: 128 experts (0-126 regular, 127 shared)
# Each extension: 132 experts (0-126 frozen, 127-130 trained, 131 shared)
# Merged: 140 experts (0-126 base, 127-130 math, 131-134 code, 135-138 croissant, 139 shared)

BASE_MODEL_PATH="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models/moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_0308/step30995"

MERGE_MODEL_PATH1="/weka/oe-training-default/kevinf/FlexMoE/models/moereducedp512sharedexp1_132experts_4trained_math_init_top2_average_train_act_10B_lr_4e-4_20260407a/step2385"
MERGE_MODEL_PATH2="/weka/oe-training-default/kevinf/extension-experiments/code-ta-01/step30995/runs/code-ta-01_lr4e-4_10B_20260407-234403/step2385"
MERGE_MODEL_PATH3="/weka/oe-training-default/kevinf/extension-experiments/croissant-ta-01/step30995/runs/croissant-ta-01_lr4e-4_10B_20260407-234459/step2385"

SAVE_PATH="/weka/oe-training-default/kevinf/extension-experiments/merged-math-code-croissant-train-act"

PYTHONPATH=. python -u src/scripts/akshitab/add_finegrained_expert/merge_experts.py \
    -b ${BASE_MODEL_PATH} \
    -m ${MERGE_MODEL_PATH1} ${MERGE_MODEL_PATH2} ${MERGE_MODEL_PATH3} \
    -e 127 128 129 130 -e 127 128 129 130 -e 127 128 129 130 \
    -o ${SAVE_PATH} \
    --num_shared_experts 1
