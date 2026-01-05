BASE_FOLDER="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models"
#BASE_FOLDER="/root/ryanwang/phdbrainstorm/FlexMoE/models"


PARENT_MODELS=(
    "twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121/step30995"
#    "twolevelbatchlb-32_1b14b_stability_lr-6e-4_1203/step30995"
#    "twolevelbatchlb-8_1b7b_stability_1207/step30995"

#    "mutualinfo_1b14b_cond-1e-2_uncond-1e-2_1205/step30995"

    "moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123/step30995"
#    "moe_1b4b_32experts_1224/step30995"
#    "twolevelsamplingnolb-32_1b14b_stability_1127/step30995"
    "dense_1b_olmoe-mix_prenorm_noqknorm_1123/step30995"

#    "moe_1b35b_320experts_lb-1e-1_1214/step30995"
#    "twolevelbatchlb-128_1b35b_320experts_lb-1e-1_poolsched-lineardecay2000_1217/step30995"
#    "twolevelbatchlb-32_1b35b_320experts_lb-1e-1_1216/step30995"
#    "twolevelbatchlb-128_1b35b_320experts_lb-1e-1_1219/step30995"

)

# used to iterate on different finetuning variations
#postfix=""
#postfix="_keepk128/newdefault_lr-4e-5"
#postfix="_keepk32/newdefault_lr-4e-5_bs-128"
postfix="_keepk32/newdefault_lr-4e-6_bs-128"
#postfix="_keepk32/newdefault_lr-4e-5"
#postfix="_keepk8/newdefault_lr-4e-5"
#postfix="_keepk32"
#postfix="_keepk32/lr-3e-5_warmup-0.2"

FINETUNE_TASKS=(
#    "task-arc_easy_rc_validation${postfix}/finetune-task-arc_easy_rc_train/step0"
#    "task-arc_easy_rc_validation${postfix}/finetune-task-arc_easy_rc_train/step42"
#    "task-arc_easy_rc_validation${postfix}/finetune-task-arc_easy_rc_train/step84"
#    "task-arc_easy_rc_validation${postfix}/finetune-task-arc_easy_rc_train/step126"
#    "task-arc_easy_rc_validation${postfix}/finetune-task-arc_easy_rc_train/step168"
#    "task-arc_easy_rc_validation${postfix}/finetune-task-arc_easy_rc_train/step210"
#
#    "task-arc_challenge_rc_validation${postfix}/finetune-task-arc_challenge_rc_train/step0"
#    "task-arc_challenge_rc_validation${postfix}/finetune-task-arc_challenge_rc_train/step20"
#    "task-arc_challenge_rc_validation${postfix}/finetune-task-arc_challenge_rc_train/step40"
#    "task-arc_challenge_rc_validation${postfix}/finetune-task-arc_challenge_rc_train/step60"
#    "task-arc_challenge_rc_validation${postfix}/finetune-task-arc_challenge_rc_train/step80"
#    "task-arc_challenge_rc_validation${postfix}/finetune-task-arc_challenge_rc_train/step102"
##
#    "task-boolq_rc_validation${postfix}/finetune-task-boolq_rc_train/step0"
#    "task-boolq_rc_validation${postfix}/finetune-task-boolq_rc_train/step157"
#    "task-boolq_rc_validation${postfix}/finetune-task-boolq_rc_train/step314"
#    "task-boolq_rc_validation${postfix}/finetune-task-boolq_rc_train/step471"
#    "task-boolq_rc_validation${postfix}/finetune-task-boolq_rc_train/step628"
#    "task-boolq_rc_validation${postfix}/finetune-task-boolq_rc_train/step789"
#
#    "task-csqa_rc_validation${postfix}/finetune-task-csqa_rc_train/step0"
#    "task-csqa_rc_validation${postfix}/finetune-task-csqa_rc_train/step163"
#    "task-csqa_rc_validation${postfix}/finetune-task-csqa_rc_train/step326"
#    "task-csqa_rc_validation${postfix}/finetune-task-csqa_rc_train/step489"
#    "task-csqa_rc_validation${postfix}/finetune-task-csqa_rc_train/step652"
#    "task-csqa_rc_validation${postfix}/finetune-task-csqa_rc_train/step819"
#
#    "task-hellaswag_rc_validation${postfix}/finetune-task-hellaswag_rc_train/step0"
#    "task-hellaswag_rc_validation${postfix}/finetune-task-hellaswag_rc_train/step729"
#    "task-hellaswag_rc_validation${postfix}/finetune-task-hellaswag_rc_train/step1458"
#    "task-hellaswag_rc_validation${postfix}/finetune-task-hellaswag_rc_train/step2187"
#    "task-hellaswag_rc_validation${postfix}/finetune-task-hellaswag_rc_train/step2916"
#    "task-hellaswag_rc_validation${postfix}/finetune-task-hellaswag_rc_train/step3645"
#
#    "task-openbookqa_rc_validation${postfix}/finetune-task-openbookqa_rc_train/step0"
#    "task-openbookqa_rc_validation${postfix}/finetune-task-openbookqa_rc_train/step92"
#    "task-openbookqa_rc_validation${postfix}/finetune-task-openbookqa_rc_train/step184"
#    "task-openbookqa_rc_validation${postfix}/finetune-task-openbookqa_rc_train/step276"
#    "task-openbookqa_rc_validation${postfix}/finetune-task-openbookqa_rc_train/step368"
#    "task-openbookqa_rc_validation${postfix}/finetune-task-openbookqa_rc_train/step462"
##
#    "task-piqa_rc_validation${postfix}/finetune-task-piqa_rc_train/step0"
#    "task-piqa_rc_validation${postfix}/finetune-task-piqa_rc_train/step283"
#    "task-piqa_rc_validation${postfix}/finetune-task-piqa_rc_train/step566"
#    "task-piqa_rc_validation${postfix}/finetune-task-piqa_rc_train/step849"
#    "task-piqa_rc_validation${postfix}/finetune-task-piqa_rc_train/step1132"
#    "task-piqa_rc_validation${postfix}/finetune-task-piqa_rc_train/step1416"
#
#    "task-socialiqa_rc_validation${postfix}/finetune-task-socialiqa_rc_train/step0"
#    "task-socialiqa_rc_validation${postfix}/finetune-task-socialiqa_rc_train/step607"
#    "task-socialiqa_rc_validation${postfix}/finetune-task-socialiqa_rc_train/step1214"
#    "task-socialiqa_rc_validation${postfix}/finetune-task-socialiqa_rc_train/step1821"
#    "task-socialiqa_rc_validation${postfix}/finetune-task-socialiqa_rc_train/step2428"
#    "task-socialiqa_rc_validation${postfix}/finetune-task-socialiqa_rc_train/step3036"
#
#    "task-winogrande_rc_validation${postfix}/finetune-task-winogrande_rc_train/step0"
#    "task-winogrande_rc_validation${postfix}/finetune-task-winogrande_rc_train/step738"
#    "task-winogrande_rc_validation${postfix}/finetune-task-winogrande_rc_train/step1476"
#    "task-winogrande_rc_validation${postfix}/finetune-task-winogrande_rc_train/step2214"
#    "task-winogrande_rc_validation${postfix}/finetune-task-winogrande_rc_train/step2952"
#    "task-winogrande_rc_validation${postfix}/finetune-task-winogrande_rc_train/step3693"
#
#    "task-gsm8k_generation_validation_0shot${postfix}/finetune-task-gsm8k_generation_train_0shot/step0"
#    "task-gsm8k_generation_validation_0shot${postfix}/finetune-task-gsm8k_generation_train_0shot/step121"
#    "task-gsm8k_generation_validation_0shot${postfix}/finetune-task-gsm8k_generation_train_0shot/step242"
#    "task-gsm8k_generation_validation_0shot${postfix}/finetune-task-gsm8k_generation_train_0shot/step363"
#    "task-gsm8k_generation_validation_0shot${postfix}/finetune-task-gsm8k_generation_train_0shot/step484"
#    "task-gsm8k_generation_validation_0shot${postfix}/finetune-task-gsm8k_generation_train_0shot/step606"

#    "task-synthea_rc_validation_0shot${postfix}/finetune-task-synthea_rc_train_0shot/step0"
#    "task-synthea_rc_validation_0shot${postfix}/finetune-task-synthea_rc_train_0shot/step161"
#    "task-synthea_rc_validation_0shot${postfix}/finetune-task-synthea_rc_train_0shot/step322"
#    "task-synthea_rc_validation_0shot${postfix}/finetune-task-synthea_rc_train_0shot/step483"
#    "task-synthea_rc_validation_0shot${postfix}/finetune-task-synthea_rc_train_0shot/step644"
#    "task-synthea_rc_validation_0shot${postfix}/finetune-task-synthea_rc_train_0shot/step807"

    "task-coqa_validation_0shot${postfix}/finetune-task-coqa_train_0shot/step0"
    "task-coqa_validation_0shot${postfix}/finetune-task-coqa_train_0shot/step28"
    "task-coqa_validation_0shot${postfix}/finetune-task-coqa_train_0shot/step56"
    "task-coqa_validation_0shot${postfix}/finetune-task-coqa_train_0shot/step84"
    "task-coqa_validation_0shot${postfix}/finetune-task-coqa_train_0shot/step112"
    "task-coqa_validation_0shot${postfix}/finetune-task-coqa_train_0shot/step144"

#    "task-squad_validation_0shot${postfix}/finetune-task-squad_train_0shot/step0"
#    "task-squad_validation_0shot${postfix}/finetune-task-squad_train_0shot/step1623"
#    "task-squad_validation_0shot${postfix}/finetune-task-squad_train_0shot/step3246"
#    "task-squad_validation_0shot${postfix}/finetune-task-squad_train_0shot/step4869"
#    "task-squad_validation_0shot${postfix}/finetune-task-squad_train_0shot/step6492"
#    "task-squad_validation_0shot${postfix}/finetune-task-squad_train_0shot/step8118"


)
#
for BASE in "${PARENT_MODELS[@]}"; do
  for FINETUNE in "${FINETUNE_TASKS[@]}"; do

    # check if "dense" appears in BASE, if so then change dir structure (dense did not go through pruning)
    if [[ "$BASE" == *"dense"* || "$BASE" == *"1b4b"* ]]; then
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
    # limit to 120 char
    job_name=${job_name:0:120}

    # launch the gantry run and delete the original model

    gantry run \
    --name $job_name \
    --weka oe-training-default:/weka/oe-training-default \
    --install 'pip install -e .[all]' \
    --budget ai2/oceo \
    --workspace ai2/flex2 \
    --allow-dirty \
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

MODELS=(
#    "dense_1b_olmoe-mix_prenorm_noqknorm_1123/step30995"
#    "moe_1b14b_128experts_olmoe-mix_130B_prenorm_1121/step30995"
#    "twolevelbatchlb-32_1b14b_stability_prenorm_1120/step30995"
#    "twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121/step30995"
#    "twolevelbatchlb-32_1b14b_stability_lr-6e-4_1203/step30995"
#    "twolevelbatchlb-8_1b7b_stability_1207/step30995"


#      "moe_1b35b_320experts_lb-1e-1_1214/step30995"
#      "moe_1b4b_32experts_1224/step30995"

    "twoleveltoppbatchlb_1b14b_topp-0.35_max-64_min-1_lb-1e-1_1222/step30995"


#      "twolevelbatchlb-128_1b35b_320experts_lb-1e-1_poolsched-lineardecay2000_1217/step30995"

#      "twolevelbatchlb-32_1b35b_320experts_lb-1e-1_1216/step30995"
#      "twolevelbatchlb-128_1b35b_320experts_lb-1e-1_1219/step30995"

#    "twolevelsamplingnolb-32_1b10b_stability_1127/step30995"
#    "twolevelsamplingnolb-32_1b14b_stability_1127/step30995"
#    "mutualinfo_1b14b_cond-1e-2_uncond-1e-2_1205/step30995"

)

#for MODEL in "${MODELS[@]}"; do
##
#    python src/examples/huggingface/convert_checkpoint_to_hf.py \
#          --checkpoint-input-path "/root/ryanwang/phdbrainstorm/FlexMoE/models/${MODEL}" \
#          --max-sequence-length 4096 \
#          --huggingface-output-dir "/root/ryanwang/phdbrainstorm/FlexMoE/models/${MODEL}-hf" \
#          --dtype float32 \
#          --skip-validation
###  gantry run \
###    --name convert-${MODEL//\//_} \
###    --weka oe-training-default:/weka/oe-training-default \
###    --beaker-image "ai2/cuda12.8-dev-ubuntu22.04-notorch" \
###    --install 'pip install -e .[all] && pip install --no-build-isolation flash-attn==2.8.2' \
###    --budget ai2/oceo \
###    --workspace ai2/flex2 \
###    --allow-dirty \
###    --cluster "ai2/jupiter-cirrascale-2" \
###    --cpus 16 \
###    --gpus 0 \
###    --priority urgent \
###    --env-secret HF_TOKEN=RYAN_HF_TOKEN \
###    --env-secret AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID \
###    --env-secret AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY \
###    -- \
###    bash -c '
###    python src/examples/huggingface/convert_checkpoint_to_hf.py \
###      --checkpoint-input-path "'"${BASE_FOLDER}/${MODEL}"'" \
###      --max-sequence-length 4096 \
###      --huggingface-output-dir "'"${BASE_FOLDER}/${MODEL}"'-hf" \
###      --dtype float32 \
###      --skip-validation \
###  '
###
#done
