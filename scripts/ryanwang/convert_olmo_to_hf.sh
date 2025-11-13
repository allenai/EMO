BASE_FOLDER="/root/ryanwang/phdbrainstorm/FlexMoE/models"
MODELS=(
#  "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995/task-arc_easy_rc_train_0shot_finetune-keepk32"
#  "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995/task-arc_challenge_rc_train_0shot_finetune-keepk32"
#  "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995/task-boolq_rc_train_0shot_finetune-keepk32"
#  "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995/task-csqa_rc_train_0shot_finetune-keepk32"
  "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995/task-hellaswag_rc_train_0shot_finetune-keepk32"
  "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995/task-openbookqa_rc_train_0shot_finetune-keepk32"
  "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995/task-piqa_rc_train_0shot_finetune-keepk32"
  "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995/task-socialiqa_rc_train_0shot_finetune-keepk32"
  "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995/task-winogrande_rc_train_0shot_finetune-keepk32"

#  "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995/task-arc_challenge_rc_train_0shot_finetune-keepk32"
#  "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995/task-boolq_rc_train_0shot_finetune-keepk32"
#  "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995/task-csqa_rc_train_0shot_finetune-keepk32"
#  "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995/task-hellaswag_rc_train_0shot_finetune-keepk32"
#  "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995/task-openbookqa_rc_train_0shot_finetune-keepk32"
#  "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995/task-piqa_rc_train_0shot_finetune-keepk32"
#  "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995/task-socialiqa_rc_train_0shot_finetune-keepk32"
#  "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995/task-winogrande_rc_train_0shot_finetune-keepk32"

)

for MODEL in "${MODELS[@]}"; do
  # list all the files in the model directory, only include directories that start with "step"
  MODEL_DIR="${BASE_FOLDER}/${MODEL}"
  STEPS=$(ls -d ${MODEL_DIR}/step*/ | xargs -n 1 basename)
  for STEP in ${STEPS}; do
    # check that the hf converted folder does not already exist
    if [ -d "${MODEL_DIR}/${STEP}-hf" ]; then
      break 1
    fi
    echo "Converting model ${MODEL} at step ${STEP} to Huggingface format..."

    python src/examples/huggingface/convert_checkpoint_to_hf.py \
      --checkpoint-input-path "${MODEL_DIR}/${STEP}" \
      --max-sequence-length 4096 \
      --huggingface-output-dir "${MODEL_DIR}/${STEP}-hf" \
      --dtype float32

  done
done

#python src/examples/huggingface/convert_checkpoint_to_hf.py \
#      --checkpoint-input-path "/root/ryanwang/phdbrainstorm/FlexMoE/models/twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995" \
#      --max-sequence-length 4096 \
#      --huggingface-output-dir "/root/ryanwang/phdbrainstorm/FlexMoE/models/twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995-hf" \
#      --dtype float32

