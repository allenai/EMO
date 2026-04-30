source "$(dirname "${BASH_SOURCE[0]}")/../../launch_common.sh"

MATH_TRAINED_TOKENS=5B
MATH_TRAINED_STEPS=1193

BASE_MODEL_PATH="${NONSHARED_BASE}"

MERGE_MODEL_PATH1="${MODELS}/freeze-fix-moe1b14b_129experts_1trained_math_init_top2_average_${MATH_TRAINED_TOKENS}_lr_4e-4/step${MATH_TRAINED_STEPS}"
MERGE_MODEL_PATH2="${MODELS}/ff-moe1b14b_132experts_4trained_code_mix_init_top2_average_noise_10B_lr_4e-4/step2385"

SAVE_PATH="${MODELS}/merged_moe_1b14b_128base_1math_${MATH_TRAINED_TOKENS}_4code_mix_10B_init_top2_average_noise"

python src/scripts/akshitab/add_finegrained_expert/merge_experts.py \
    -b ${BASE_MODEL_PATH} \
    -m ${MERGE_MODEL_PATH1} ${MERGE_MODEL_PATH2} \
    -e 128 -e 128 129 130 131 \
    -o ${SAVE_PATH}



MERGE_MODEL_PATH1="${MODELS}/freeze-fix-moe1b14b_130experts_2trained_math_init_top2_average_noise_${MATH_TRAINED_TOKENS}_lr_4e-4/step${MATH_TRAINED_STEPS}"
MERGE_MODEL_PATH2="${MODELS}/ff-moe1b14b_132experts_4trained_code_mix_init_top2_average_noise_10B_lr_4e-4/step2385"

SAVE_PATH="${MODELS}/merged_moe_1b14b_128base_2math_${MATH_TRAINED_TOKENS}_4code_mix_10B_init_top2_average_noise"

python src/scripts/akshitab/add_finegrained_expert/merge_experts.py \
    -b ${BASE_MODEL_PATH} \
    -m ${MERGE_MODEL_PATH1} ${MERGE_MODEL_PATH2} \
    -e 128 129 -e 128 129 130 131 \
    -o ${SAVE_PATH}

MERGE_MODEL_PATH1="${MODELS}/freeze-fix-moe1b14b_132experts_4trained_math_init_top2_average_noise_${MATH_TRAINED_TOKENS}_lr_4e-4/step${MATH_TRAINED_STEPS}"
MERGE_MODEL_PATH2="${MODELS}/ff-moe1b14b_132experts_4trained_code_mix_init_top2_average_noise_10B_lr_4e-4/step2385"

SAVE_PATH="${MODELS}/merged_moe_1b14b_128base_4math_${MATH_TRAINED_TOKENS}_4code_mix_10B_init_top2_average_noise"

python src/scripts/akshitab/add_finegrained_expert/merge_experts.py \
    -b ${BASE_MODEL_PATH} \
    -m ${MERGE_MODEL_PATH1} ${MERGE_MODEL_PATH2} \
    -e 128 129 130 131 -e 128 129 130 131 \
    -o ${SAVE_PATH}