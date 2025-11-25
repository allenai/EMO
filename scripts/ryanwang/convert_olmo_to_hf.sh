BASE_FOLDER="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models"
#BASE_FOLDER="/root/ryanwang/phdbrainstorm/FlexMoE/models"


PARENT_MODELS=(
#    "moe_1b14b_128experts_olmoe-mix_130B_1117/step30995"

#    "dense_1b_olmoe-mix_1119/step30995"
#    "dense_1b_olmoe-mix_1119/step30995/noloadoptim"

    "twolevelbatchlb-32_1b14b_stability_filter-true_zlossweight-1e-3_1115/step30995"

)

# used to iterate on different finetuning variations
#postfix="_keepk32"
postfix="_keepk32/lr-7e-5_warmup-0.2"

FINETUNE_TASKS=(
#    "task-arc_easy_rc_validation${postfix}/finetune-task-arc_easy_rc_train/step0"
#    "task-arc_easy_rc_validation${postfix}/finetune-task-arc_easy_rc_train/step84"
#    "task-arc_easy_rc_validation${postfix}/finetune-task-arc_easy_rc_train/step168"
#    "task-arc_easy_rc_validation${postfix}/finetune-task-arc_easy_rc_train/step252"
#    "task-arc_easy_rc_validation${postfix}/finetune-task-arc_easy_rc_train/step336"
#    "task-arc_easy_rc_validation${postfix}/finetune-task-arc_easy_rc_train/step420"
##
    "task-arc_challenge_rc_validation${postfix}/finetune-task-arc_challenge_rc_train/step0"
    "task-arc_challenge_rc_validation${postfix}/finetune-task-arc_challenge_rc_train/step41"
    "task-arc_challenge_rc_validation${postfix}/finetune-task-arc_challenge_rc_train/step82"
    "task-arc_challenge_rc_validation${postfix}/finetune-task-arc_challenge_rc_train/step123"
    "task-arc_challenge_rc_validation${postfix}/finetune-task-arc_challenge_rc_train/step164"
    "task-arc_challenge_rc_validation${postfix}/finetune-task-arc_challenge_rc_train/step207"
##
#    "task-boolq_rc_validation${postfix}/finetune-task-boolq_rc_train/step0"
#    "task-boolq_rc_validation${postfix}/finetune-task-boolq_rc_train/step315"
#    "task-boolq_rc_validation${postfix}/finetune-task-boolq_rc_train/step630"
#    "task-boolq_rc_validation${postfix}/finetune-task-boolq_rc_train/step945"
#    "task-boolq_rc_validation${postfix}/finetune-task-boolq_rc_train/step1260"
#    "task-boolq_rc_validation${postfix}/finetune-task-boolq_rc_train/step1578"
##
    "task-csqa_rc_validation${postfix}/finetune-task-csqa_rc_train/step0"
    "task-csqa_rc_validation${postfix}/finetune-task-csqa_rc_train/step327"
    "task-csqa_rc_validation${postfix}/finetune-task-csqa_rc_train/step654"
    "task-csqa_rc_validation${postfix}/finetune-task-csqa_rc_train/step981"
    "task-csqa_rc_validation${postfix}/finetune-task-csqa_rc_train/step1308"
    "task-csqa_rc_validation${postfix}/finetune-task-csqa_rc_train/step1638"
#
#    "task-hellaswag_rc_validation${postfix}/finetune-task-hellaswag_rc_train/step0"
#    "task-hellaswag_rc_validation${postfix}/finetune-task-hellaswag_rc_train/step1458"
#    "task-hellaswag_rc_validation${postfix}/finetune-task-hellaswag_rc_train/step2916"
#    "task-hellaswag_rc_validation${postfix}/finetune-task-hellaswag_rc_train/step4374"
#    "task-hellaswag_rc_validation${postfix}/finetune-task-hellaswag_rc_train/step5832"
#    "task-hellaswag_rc_validation${postfix}/finetune-task-hellaswag_rc_train/step7293"
#
#    "task-openbookqa_rc_validation${postfix}/finetune-task-openbookqa_rc_train/step0"
#    "task-openbookqa_rc_validation${postfix}/finetune-task-openbookqa_rc_train/step185"
#    "task-openbookqa_rc_validation${postfix}/finetune-task-openbookqa_rc_train/step370"
#    "task-openbookqa_rc_validation${postfix}/finetune-task-openbookqa_rc_train/step555"
#    "task-openbookqa_rc_validation${postfix}/finetune-task-openbookqa_rc_train/step740"
#    "task-openbookqa_rc_validation${postfix}/finetune-task-openbookqa_rc_train/step927"
##
    "task-piqa_rc_validation${postfix}/finetune-task-piqa_rc_train/step0"
    "task-piqa_rc_validation${postfix}/finetune-task-piqa_rc_train/step566"
    "task-piqa_rc_validation${postfix}/finetune-task-piqa_rc_train/step1132"
    "task-piqa_rc_validation${postfix}/finetune-task-piqa_rc_train/step1698"
    "task-piqa_rc_validation${postfix}/finetune-task-piqa_rc_train/step2264"
    "task-piqa_rc_validation${postfix}/finetune-task-piqa_rc_train/step2832"
#
#    "task-socialiqa_rc_validation${postfix}/finetune-task-socialiqa_rc_train/step0"
#    "task-socialiqa_rc_validation${postfix}/finetune-task-socialiqa_rc_train/step1215"
#    "task-socialiqa_rc_validation${postfix}/finetune-task-socialiqa_rc_train/step2430"
#    "task-socialiqa_rc_validation${postfix}/finetune-task-socialiqa_rc_train/step3645"
#    "task-socialiqa_rc_validation${postfix}/finetune-task-socialiqa_rc_train/step4860"
#    "task-socialiqa_rc_validation${postfix}/finetune-task-socialiqa_rc_train/step6075"
#
#    "task-winogrande_rc_validation${postfix}/finetune-task-winogrande_rc_train/step0"
#    "task-winogrande_rc_validation${postfix}/finetune-task-winogrande_rc_train/step1477"
#    "task-winogrande_rc_validation${postfix}/finetune-task-winogrande_rc_train/step2954"
#    "task-winogrande_rc_validation${postfix}/finetune-task-winogrande_rc_train/step4431"
#    "task-winogrande_rc_validation${postfix}/finetune-task-winogrande_rc_train/step5908"
#    "task-winogrande_rc_validation${postfix}/finetune-task-winogrande_rc_train/step7386"

)

for BASE in "${PARENT_MODELS[@]}"; do
  for FINETUNE in "${FINETUNE_TASKS[@]}"; do

    # check if "dense" appears in BASE, if so then change dir structure (dense did not go through pruning)
    if [[ "$BASE" == *"dense"* ]]; then
      # remove everything before the first "/" in FINETUNE
      FINETUNE="${FINETUNE#*/}"
      MODEL_DIR="${BASE_FOLDER}/${BASE}/${FINETUNE}"
    else
      MODEL_DIR="${BASE_FOLDER}/${BASE}_${FINETUNE}"
    fi

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

#python src/examples/huggingface/convert_checkpoint_to_hf.py \
#      --checkpoint-input-path "/root/ryanwang/phdbrainstorm/FlexMoE/models/dense_1b_olmoe-mix_1119/step30995" \
#      --max-sequence-length 4096 \
#      --huggingface-output-dir "/root/ryanwang/phdbrainstorm/FlexMoE/models/dense_1b_olmoe-mix_1119/step30995-hf" \
#      --dtype float32 \
#      --skip-validation

