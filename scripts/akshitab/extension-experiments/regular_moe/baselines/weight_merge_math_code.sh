MODEL_DIR="/weka/oe-training-default/akshitab/FlexMoE/models"
MODEL_PATH1="${MODEL_DIR}/moe1b14b_128experts_76_5_122_126_trained_math_10B_lr_4e-4/step2385"
MODEL_PATH2="${MODEL_DIR}/moe1b14b_128experts_76_41_120_3_trained_code_10B_lr_4e-4/step2385"

WEIGHT_MERGED_MODEL_PATH="${MODEL_DIR}/weight_merge_moe_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise"

python src/scripts/akshitab/add_finegrained_expert/weight_merge_moe_models.py \                                                                                                                                                                                                                    
      -c ${MODEL_PATH1} ${MODEL_PATH2} \                                                                                                                                                                                                                                                
      -o ${WEIGHT_MERGED_MODEL_PATH}