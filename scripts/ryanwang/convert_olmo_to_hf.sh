BASE_FOLDER="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models"
#BASE_FOLDER="/root/ryanwang/phdbrainstorm/FlexMoE/models"


PARENT_MODELS=(
#    "dense_1b_olmoe-mix_1028/step30995"
    "twolevelbatchlb-32_1b14b_stability_filter-true_zlossweight-1e-3_1115/step30995"
#    "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995"
#    "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995"
)
  
#postfix="-keepk8"
postfix="-keepk32"
#postfix=""

taskvariant=""

FINETUNE_TASKS=(
    "task-arc_easy_rc_train${taskvariant}_finetune${postfix}/step0"
#    "task-arc_easy_rc_train${taskvariant}_finetune${postfix}/step84"
#    "task-arc_easy_rc_train${taskvariant}_finetune${postfix}/step168"
#    "task-arc_easy_rc_train${taskvariant}_finetune${postfix}/step252"
#    "task-arc_easy_rc_train${taskvariant}_finetune${postfix}/step336"
#    "task-arc_easy_rc_train${taskvariant}_finetune${postfix}/step420"

#    "task-arc_challenge_rc_train${taskvariant}_finetune${postfix}/step0"
#    "task-arc_challenge_rc_train${taskvariant}_finetune${postfix}/step41"
#    "task-arc_challenge_rc_train${taskvariant}_finetune${postfix}/step82"
#    "task-arc_challenge_rc_train${taskvariant}_finetune${postfix}/step123"
#    "task-arc_challenge_rc_train${taskvariant}_finetune${postfix}/step164"
#    "task-arc_challenge_rc_train${taskvariant}_finetune${postfix}/step207"
#
#    "task-boolq_rc_train${taskvariant}_finetune${postfix}/step0"
#    "task-boolq_rc_train${taskvariant}_finetune${postfix}/step315"
#    "task-boolq_rc_train${taskvariant}_finetune${postfix}/step630"
#    "task-boolq_rc_train${taskvariant}_finetune${postfix}/step945"
#    "task-boolq_rc_train${taskvariant}_finetune${postfix}/step1260"
#    "task-boolq_rc_train${taskvariant}_finetune${postfix}/step1578"
#
#    "task-csqa_rc_train${taskvariant}_finetune${postfix}/step0"
#    "task-csqa_rc_train${taskvariant}_finetune${postfix}/step327"
#    "task-csqa_rc_train${taskvariant}_finetune${postfix}/step654"
#    "task-csqa_rc_train${taskvariant}_finetune${postfix}/step981"
#    "task-csqa_rc_train${taskvariant}_finetune${postfix}/step1308"
#    "task-csqa_rc_train${taskvariant}_finetune${postfix}/step1638"
#
#    "task-hellaswag_rc_train${taskvariant}_finetune${postfix}/step0"
#    "task-hellaswag_rc_train${taskvariant}_finetune${postfix}/step1458"
#    "task-hellaswag_rc_train${taskvariant}_finetune${postfix}/step2916"
#    "task-hellaswag_rc_train${taskvariant}_finetune${postfix}/step4374"
#    "task-hellaswag_rc_train${taskvariant}_finetune${postfix}/step5832"
#    "task-hellaswag_rc_train${taskvariant}_finetune${postfix}/step7293"
#
#    "task-openbookqa_rc_train${taskvariant}_finetune${postfix}/step0"
#    "task-openbookqa_rc_train${taskvariant}_finetune${postfix}/step185"
#    "task-openbookqa_rc_train${taskvariant}_finetune${postfix}/step370"
#    "task-openbookqa_rc_train${taskvariant}_finetune${postfix}/step555"
#    "task-openbookqa_rc_train${taskvariant}_finetune${postfix}/step740"
#    "task-openbookqa_rc_train${taskvariant}_finetune${postfix}/step927"
#
#    "task-piqa_rc_train${taskvariant}_finetune${postfix}/step0"
#    "task-piqa_rc_train${taskvariant}_finetune${postfix}/step566"
#    "task-piqa_rc_train${taskvariant}_finetune${postfix}/step1132"
#    "task-piqa_rc_train${taskvariant}_finetune${postfix}/step1698"
#    "task-piqa_rc_train${taskvariant}_finetune${postfix}/step2264"
#    "task-piqa_rc_train${taskvariant}_finetune${postfix}/step2832"
#
#    "task-socialiqa_rc_train${taskvariant}_finetune${postfix}/step0"
#    "task-socialiqa_rc_train${taskvariant}_finetune${postfix}/step1215"
#    "task-socialiqa_rc_train${taskvariant}_finetune${postfix}/step2430"
#    "task-socialiqa_rc_train${taskvariant}_finetune${postfix}/step3645"
#    "task-socialiqa_rc_train${taskvariant}_finetune${postfix}/step4860"
#    "task-socialiqa_rc_train${taskvariant}_finetune${postfix}/step6075"
#
#    "task-winogrande_rc_train${taskvariant}_finetune${postfix}/step0"
#    "task-winogrande_rc_train${taskvariant}_finetune${postfix}/step1477"
#    "task-winogrande_rc_train${taskvariant}_finetune${postfix}/step2954"
#    "task-winogrande_rc_train${taskvariant}_finetune${postfix}/step4431"
#    "task-winogrande_rc_train${taskvariant}_finetune${postfix}/step5908"
#    "task-winogrande_rc_train${taskvariant}_finetune${postfix}/step7386"

)

for BASE in "${PARENT_MODELS[@]}"; do
  for FINETUNE in "${FINETUNE_TASKS[@]}"; do
    # list all the files in the model directory, only include directories that start with "step"
    MODEL_DIR="${BASE_FOLDER}/${BASE}/${FINETUNE}"

    echo "checkpoint-input-path is ${MODEL_DIR}"
    echo "output_dir is ${MODEL_DIR}-hf"

    # Beaker names can only contain letters, digits, periods, dashes, and underscores.
    job_name="convert_${FINETUNE//\//_}"

    # launch the gantry run and delete the original model

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
    bash -c '
    python src/examples/huggingface/convert_checkpoint_to_hf.py \
      --checkpoint-input-path "'"${MODEL_DIR}"'" \
      --max-sequence-length 4096 \
      --huggingface-output-dir "'"${MODEL_DIR}"'-hf" \
      --dtype float32 \
      --skip-validation \
    && \
      rm -rf "'"${MODEL_DIR}"'"
  '
  done
done

#MODELS=(
##  "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995/task-arc_easy_rc_train_0shot_finetune${postfix}"
##  "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995/task-arc_challenge_rc_train_0shot_finetune${postfix}"
#  "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995/task-boolq_rc_train_0shot_finetune${postfix}"
#  "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995/task-csqa_rc_train_0shot_finetune${postfix}"
##  "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995/task-hellaswag_rc_train_0shot_finetune${postfix}"
##  "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995/task-openbookqa_rc_train_0shot_finetune${postfix}"
##  "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995/task-piqa_rc_train_0shot_finetune${postfix}"
##  "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995/task-socialiqa_rc_train_0shot_finetune${postfix}"
##  "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995/task-winogrande_rc_train_0shot_finetune${postfix}"
#
##  "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995/task-arc_challenge_rc_train_0shot_finetune${postfix}"
##  "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995/task-boolq_rc_train_0shot_finetune${postfix}"
##  "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995/task-csqa_rc_train_0shot_finetune${postfix}"
##  "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995/task-hellaswag_rc_train_0shot_finetune${postfix}"
##  "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995/task-openbookqa_rc_train_0shot_finetune${postfix}"
##  "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995/task-piqa_rc_train_0shot_finetune${postfix}"
##  "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995/task-socialiqa_rc_train_0shot_finetune${postfix}"
##  "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995/task-winogrande_rc_train_0shot_finetune${postfix}"
#
#)

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
#      --checkpoint-input-path "/root/ryanwang/phdbrainstorm/FlexMoE/models/twolevelsamplingnolb-32_1b14b_stability_filter-true_zlossweight-1e-3_1116/step30995" \
#      --max-sequence-length 4096 \
#      --huggingface-output-dir "/root/ryanwang/phdbrainstorm/FlexMoE/models/twolevelsamplingnolb-32_1b14b_stability_filter-true_zlossweight-1e-3_1116/step30995-hf" \
#      --dtype float32
#      --skip-validation

