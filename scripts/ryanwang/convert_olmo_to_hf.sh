BASE_FOLDER="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models"
#BASE_FOLDER="/root/ryanwang/phdbrainstorm/FlexMoE/models"
MODELS=(
#  "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995/task-arc_easy_rc_train_0shot_finetune-keepk32"
#  "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995/task-arc_challenge_rc_train_0shot_finetune-keepk32"
  "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995/task-boolq_rc_train_0shot_finetune-keepk32"
  "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995/task-csqa_rc_train_0shot_finetune-keepk32"
#  "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995/task-hellaswag_rc_train_0shot_finetune-keepk32"
#  "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995/task-openbookqa_rc_train_0shot_finetune-keepk32"
#  "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995/task-piqa_rc_train_0shot_finetune-keepk32"
#  "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995/task-socialiqa_rc_train_0shot_finetune-keepk32"
#  "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995/task-winogrande_rc_train_0shot_finetune-keepk32"

#  "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995/task-arc_challenge_rc_train_0shot_finetune-keepk32"
#  "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995/task-boolq_rc_train_0shot_finetune-keepk32"
#  "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995/task-csqa_rc_train_0shot_finetune-keepk32"
#  "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995/task-hellaswag_rc_train_0shot_finetune-keepk32"
#  "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995/task-openbookqa_rc_train_0shot_finetune-keepk32"
#  "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995/task-piqa_rc_train_0shot_finetune-keepk32"
#  "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995/task-socialiqa_rc_train_0shot_finetune-keepk32"
#  "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995/task-winogrande_rc_train_0shot_finetune-keepk32"

)



PARENT_MODELS=(
    "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995"
#    "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995"
)

FINETUNE_TASKS=(
#    "task-arc_easy_rc_train_0shot_finetune-keepk32/step0"
#    "task-arc_easy_rc_train_0shot_finetune-keepk32/step84"
#    "task-arc_easy_rc_train_0shot_finetune-keepk32/step168"
#    "task-arc_easy_rc_train_0shot_finetune-keepk32/step252"
#    "task-arc_easy_rc_train_0shot_finetune-keepk32/step336"
#    "task-arc_easy_rc_train_0shot_finetune-keepk32/step420"
#
#    "task-arc_challenge_rc_train_0shot_finetune-keepk32/step0"
#    "task-arc_challenge_rc_train_0shot_finetune-keepk32/step41"
#    "task-arc_challenge_rc_train_0shot_finetune-keepk32/step82"
#    "task-arc_challenge_rc_train_0shot_finetune-keepk32/step123"
#    "task-arc_challenge_rc_train_0shot_finetune-keepk32/step164"
#    "task-arc_challenge_rc_train_0shot_finetune-keepk32/step207"

#    "task-boolq_rc_train_0shot_finetune-keepk32/step0"
#    "task-boolq_rc_train_0shot_finetune-keepk32/step315"
    "task-boolq_rc_train_0shot_finetune-keepk32/step630"
    "task-boolq_rc_train_0shot_finetune-keepk32/step945"
    "task-boolq_rc_train_0shot_finetune-keepk32/step1260"
    "task-boolq_rc_train_0shot_finetune-keepk32/step1578"
#
    "task-csqa_rc_train_0shot_finetune-keepk32/step0"
    "task-csqa_rc_train_0shot_finetune-keepk32/step327"
    "task-csqa_rc_train_0shot_finetune-keepk32/step654"
    "task-csqa_rc_train_0shot_finetune-keepk32/step981"
    "task-csqa_rc_train_0shot_finetune-keepk32/step1308"
    "task-csqa_rc_train_0shot_finetune-keepk32/step1638"
#
#    "task-hellaswag_rc_train_0shot_finetune-keepk32/step0"
#    "task-hellaswag_rc_train_0shot_finetune-keepk32/step1458"
#    "task-hellaswag_rc_train_0shot_finetune-keepk32/step2916"
#    "task-hellaswag_rc_train_0shot_finetune-keepk32/step4370"
#    "task-hellaswag_rc_train_0shot_finetune-keepk32/step5832"
#    "task-hellaswag_rc_train_0shot_finetune-keepk32/step7293"
#
    "task-openbookqa_rc_train_0shot_finetune-keepk32/step0"
    "task-openbookqa_rc_train_0shot_finetune-keepk32/step185"
    "task-openbookqa_rc_train_0shot_finetune-keepk32/step370"
    "task-openbookqa_rc_train_0shot_finetune-keepk32/step555"
    "task-openbookqa_rc_train_0shot_finetune-keepk32/step740"
    "task-openbookqa_rc_train_0shot_finetune-keepk32/step927"

    "task-piqa_rc_train_0shot_finetune-keepk32/step0"
    "task-piqa_rc_train_0shot_finetune-keepk32/step566"
    "task-piqa_rc_train_0shot_finetune-keepk32/step1132"
    "task-piqa_rc_train_0shot_finetune-keepk32/step1698"
    "task-piqa_rc_train_0shot_finetune-keepk32/step2264"
    "task-piqa_rc_train_0shot_finetune-keepk32/step2832"

    "task-socialiqa_rc_train_0shot_finetune-keepk32/step0"
    "task-socialiqa_rc_train_0shot_finetune-keepk32/step1215"
    "task-socialiqa_rc_train_0shot_finetune-keepk32/step2430"
    "task-socialiqa_rc_train_0shot_finetune-keepk32/step3645"
    "task-socialiqa_rc_train_0shot_finetune-keepk32/step4860"
    "task-socialiqa_rc_train_0shot_finetune-keepk32/step6075"

    "task-winogrande_rc_train_0shot_finetune-keepk32/step0"
    "task-winogrande_rc_train_0shot_finetune-keepk32/step1477"
    "task-winogrande_rc_train_0shot_finetune-keepk32/step2954"
    "task-winogrande_rc_train_0shot_finetune-keepk32/step4431"
    "task-winogrande_rc_train_0shot_finetune-keepk32/step5908"
    "task-winogrande_rc_train_0shot_finetune-keepk32/step7386"

)

for BASE in "${PARENT_MODELS[@]}"; do
  for FINETUNE in "${FINETUNE_TASKS[@]}"; do
    # list all the files in the model directory, only include directories that start with "step"
    MODEL_DIR="${BASE_FOLDER}/${BASE}/${FINETUNE}"

    echo "checkpoint-input-path is ${MODEL_DIR}"
    echo "output_dir is ${MODEL_DIR}-hf"

    # Beaker names can only contain letters, digits, periods, dashes, and underscores.
    job_name="convert_${FINETUNE//\//_}"

    gantry run \
    --name $job_name \
    --weka oe-training-default:/weka/oe-training-default \
    --install 'pip install -e .[all]' \
    --budget ai2/oceo \
    --workspace ai2/flex2 \
    --cluster "ai2/jupiter-cirrascale-2" \
    --cpus 16 \
    --gpus 0 \
    --priority urgent \
    --env-secret HF_TOKEN=RYAN_HF_TOKEN \
    --env-secret AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY \
    -- \
    bash -c "python src/examples/huggingface/convert_checkpoint_to_hf.py \
      --checkpoint-input-path "${MODEL_DIR}" \
      --max-sequence-length 4096 \
      --huggingface-output-dir "${MODEL_DIR}-hf" \
      --dtype float32
      "

  done
done

#for MODEL in "${MODELS[@]}"; do
#  # list all the files in the model directory, only include directories that start with "step"
#  MODEL_DIR="${BASE_FOLDER}/${MODEL}"
#  STEPS=$(ls -d ${MODEL_DIR}/step*/ | xargs -n 1 basename)
#  for STEP in ${STEPS}; do
#    # check that the hf converted folder does not already exist
#    if [ -d "${MODEL_DIR}/${STEP}-hf" ]; then
#      break 1
#    fi
#    echo "Converting model ${MODEL} at step ${STEP} to Huggingface format..."
#
#    gantry run \
#            --name $job_name \
#            --weka oe-training-default:/weka/oe-training-default \
#            --install "pip install -e \".[all]\"" \
#            --budget ai2/oceo \
#            --workspace ai2/flex2 \
#            --cluster $CLUSTER \
#            --priority urgent \
#            --env-secret HF_TOKEN=RYAN_HF_TOKEN \
#            --env-secret AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID \
#            --env-secret AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY \
#            -- \
#            bash -c "python src/examples/huggingface/convert_checkpoint_to_hf.py \
#                --checkpoint-input-path "${MODEL_DIR}/${STEP}" \
#                --max-sequence-length 4096 \
#                --huggingface-output-dir "${MODEL_DIR}/${STEP}-hf" \
#                --dtype float32
#                "
#  done
#done

#python src/examples/huggingface/convert_checkpoint_to_hf.py \
#      --checkpoint-input-path "/root/ryanwang/phdbrainstorm/FlexMoE/models/twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995" \
#      --max-sequence-length 4096 \
#      --huggingface-output-dir "/root/ryanwang/phdbrainstorm/FlexMoE/models/twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995-hf" \
#      --dtype float32

