BASE_MODEL_PATH="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models/twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301/step30995"

MERGE_MODEL_PATH1="/weka/oe-training-default/akshitab/FlexMoE/models/twolevel_132experts_4trained_forced_code_mix_init_top2_average_noise_10B_lr_4e-4/step2385"
MERGE_MODEL_PATH2="/weka/oe-training-default/akshitab/FlexMoE/models/twolevel_132experts_4trained_forced_math_init_top2_average_noise_10B_lr_4e-4/step2385"

SAVE_PATH="/weka/oe-training-default/akshitab/FlexMoE/models/merged_twolevel_1b14b_128base_4math_10B_4code_mix_10B_forced_init_top2_average_noise"

python src/scripts/akshitab/add_finegrained_expert/merge_experts.py \
    -b ${BASE_MODEL_PATH} \
    -m ${MERGE_MODEL_PATH1} ${MERGE_MODEL_PATH2} \
    -e 128 129 130 131 -e 128 129 130 131 \
    -o ${SAVE_PATH}