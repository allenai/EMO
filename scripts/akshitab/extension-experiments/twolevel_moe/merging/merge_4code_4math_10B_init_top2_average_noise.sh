source "$(dirname "${BASH_SOURCE[0]}")/../../launch_common.sh"

BASE_MODEL_PATH="${TWOLEVEL_BASE}"

MERGE_MODEL_PATH1="${MODELS}/twolevel_132experts_4trained_forced_code_mix_init_top2_average_noise_10B_lr_4e-4/step2385"
MERGE_MODEL_PATH2="${MODELS}/twolevel_132experts_4trained_forced_math_init_top2_average_noise_10B_lr_4e-4/step2385"

SAVE_PATH="${MODELS}/merged_twolevel_1b14b_128base_4math_10B_4code_mix_10B_forced_init_top2_average_noise"

python src/scripts/akshitab/add_finegrained_expert/merge_experts.py \
    -b ${BASE_MODEL_PATH} \
    -m ${MERGE_MODEL_PATH1} ${MERGE_MODEL_PATH2} \
    -e 128 129 130 131 -e 128 129 130 131 \
    -o ${SAVE_PATH} \
    --num_shared_experts 1