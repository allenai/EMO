BASE_MODEL_PATH="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models/moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123/step30995"

MERGE_MODEL_PATH1="/weka/oe-training-default/akshitab/FlexMoE/models/freeze-fix-moe1b14b_132experts_4trained_math_init_top2_average_noise_10B_lr_4e-4/step2385"
MERGE_MODEL_PATH2="/weka/oe-training-default/akshitab/FlexMoE/models/ff-moe1b14b_132experts_4trained_starcoder_init_top2_average_noise_10B_lr_4e-4/step2385"

SAVE_PATH="/weka/oe-training-default/akshitab/FlexMoE/models/extensions/merged_moe_1b14b_128base_4math_10B_4starcoder_10B_init_top2_average_noise"

python src/scripts/akshitab/add_finegrained_expert/merge_experts.py \
    -b ${BASE_MODEL_PATH} \
    -m ${MERGE_MODEL_PATH1} ${MERGE_MODEL_PATH2} \
    -e 128 129 130 131 -e 128 129 130 131 \
    -o ${SAVE_PATH}