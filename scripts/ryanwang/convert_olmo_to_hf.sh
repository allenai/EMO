#BASE_FOLDER="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models"
#BASE_FOLDER="/root/ryanwang/phdbrainstorm/FlexMoE/models"



MODELS=(
#    "dense_1b_olmoe-mix_prenorm_noqknorm_1123/step30995"
#    "moe_1b14b_128experts_olmoe-mix_130B_prenorm_1121/step30995"
#    "twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121/step30995"

#      "moe_1b35b_320experts_lb-1e-1_1214/step30995"
#      "moe_1b4b_32experts_1224/step30995"

#    "twoleveltoppbatchlb_1b14b_topp-0.35_max-64_min-1_lb-1e-1_1222/step30995"
#    "twolevelbatchlb-32_1b14b_lr-4e-3_lb-1e-1_0119/step30995"
#    "twolevelbatchlb-32_1b14b_lr-4e-3_lb-1e-2_0118/step30995"
#    "twolevelbatchlb-32_1b14b_lr-4e-4_lb-1e-1_0118/step30995"
#    "twolevelbatchlb-32_1b14b_lr-4e-4_lb-1e-1_poolsched_0119/step30995"
#    "twolevelbatchlbreducedp512-32_1b14b_lr-4e-3_lb-1e-1_0119/step30995"
#    "twolevelbatchlbreducedp512-32_1b14b_lr-4e-3_lb-1e-2_0207/step30995"

#    "twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-1_0211/step30995"
#    "twolevelbatchlbreducedp512sharedexp1densefirst-32_1b14b_lr-4e-3_lb-1e-1_0227/step30995"
#    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301/step30995"
#    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313/step238419"
#    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419/step250339"
#    "twolevelbatchlbreducedp512sharedexp2randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0305/step30995"
#    "twolevelbatchlbreducedp512sharedexp4c2-32_1b14b_lr-4e-3_lb-1e-2_sharelb-1e-2_0214/step30995"
#    "twolevelbatchlbreducedp512sharedexp4c2-32_1b14b_lr-4e-3_lb-1e-1_sharelb-1e-1_0214/step30995"
#    "twolevelbatchlbreducedp512sharedexp1-32_1b14b_lr-4e-3_lb-1e-2_0213/step30995"
#    "dense_1b_lr-4e-3_0213/step30995"
#    "moereducedp256_1b4b_lr-4e-3_lb-1e-1_0212/step30995"
#    "moereducedp512sharedexp1_1b4b_lr-4e-3_lb-1e-1_0308/step30995"
#    "moereducedp512_1b14b_lr-4e-3_lb-1e-1_0211/step30995"
#    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_0308/step30995"
#    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_1T_0322/step238419"
#    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_1T_0322_anneal_from_step238419/step250339"

#    "moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_1T_0322_anneal_twolevel_randpool-8-128_from_step238419/step250339"

    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419_ct-m8_lb0/step2385"
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419_ct-math_8/step2385"
    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238_ct-m8_lb0_wd/step2385"
#    "twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step23_ct-m8_lb0_frz/step2385"


#    "moe_1b14b_128experts_lb-1e-1_1217/step30995"

)

for MODEL in "${MODELS[@]}"; do
#
    python src/examples/huggingface/convert_checkpoint_to_hf.py \
          --checkpoint-input-path "/root/ryanwang/phdbrainstorm/FlexMoE/models/${MODEL}" \
          --max-sequence-length 4096 \
          --huggingface-output-dir "/root/ryanwang/phdbrainstorm/FlexMoE/models/${MODEL}-hf" \
          --dtype float32 \
          --skip-validation
##  gantry run \
##    --name convert-${MODEL//\//_} \
##    --weka oe-training-default:/weka/oe-training-default \
##    --beaker-image "ai2/cuda12.8-dev-ubuntu22.04-notorch" \
##    --install 'pip install -e .[all] && pip install --no-build-isolation flash-attn==2.8.2' \
##    --budget ai2/oceo \
##    --workspace ai2/flex2 \
##    --allow-dirty \
##    --cluster "ai2/jupiter-cirrascale-2" \
##    --cpus 16 \
##    --gpus 0 \
##    --priority urgent \
##    --env-secret HF_TOKEN=RYAN_HF_TOKEN \
##    --env-secret AWS_ACCESS_KEY_ID=RYAN_AWS_ACCESS_KEY_ID \
##    --env-secret AWS_SECRET_ACCESS_KEY=RYAN_AWS_SECRET_ACCESS_KEY \
##    -- \
##    bash -c '
##    python src/examples/huggingface/convert_checkpoint_to_hf.py \
##      --checkpoint-input-path "'"${BASE_FOLDER}/${MODEL}"'" \
##      --max-sequence-length 4096 \
##      --huggingface-output-dir "'"${BASE_FOLDER}/${MODEL}"'-hf" \
##      --dtype float32 \
##      --skip-validation \
##  '
##
done
