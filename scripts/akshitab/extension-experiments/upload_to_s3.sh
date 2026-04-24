#!/bin/bash

BASE_S3_PATH="s3://ai2-llm/flexolmo/mose/akshitab/models"
LOG_FILE="${LOG_FILE:-upload_to_s3.log}"

touch "$LOG_FILE"
TOTAL=$(grep -c '^s3sync ' "${BASH_SOURCE[0]}")
IDX=0

s3sync() {
    IDX=$((IDX + 1))
    local src="$1" dst="$2" key="${1} -> ${2}"
    if grep -qxF "$key" "$LOG_FILE"; then
        echo "[$IDX/$TOTAL] SKIP (done): $src"
        return 0
    fi
    echo "[$IDX/$TOTAL] SYNC: $src -> $dst"
    if aws s3 sync --exclude "wandb/*" --exclude "*/wandb/*" "$src" "$dst"; then
        echo "$key" >> "$LOG_FILE"
        echo "[$IDX/$TOTAL] OK: $src"
    else
        echo "[$IDX/$TOTAL] FAIL: $src" >&2
        return 1
    fi
}

# Base model extensions
s3sync extensions/moereducedp512sharedexp1_1b14b_132experts_0308_step30995_init_top2_average_noise ${BASE_S3_PATH}/regular_moe/moereducedp512sharedexp1_1b14b_132experts_0308_step30995_init_top2_math_average_noise
s3sync extensions/moereducedp512sharedexp1_1b14b_132experts_0308_step30995_init_top2_code_average_noise ${BASE_S3_PATH}/regular_moe/moereducedp512sharedexp1_1b14b_132experts_0308_step30995_init_top2_code_average_noise

s3sync extensions/twolevelbatchlbreducedp512sharedexp1randpool_1b14b_132experts_0301_step30995_init_top2_average_noise ${BASE_S3_PATH}/twolevel_moe/twolevelbatchlbreducedp512sharedexp1randpool_1b14b_132experts_0301_step30995_init_top2_math_average_noise
s3sync extensions/twolevelbatchlbreducedp512sharedexp1randpool_1b14b_132experts_0301_step30995_init_top2_code_average_noise ${BASE_S3_PATH}/twolevel_moe/twolevelbatchlbreducedp512sharedexp1randpool_1b14b_132experts_0301_step30995_init_top2_code_average_noise

## Regular MoE base extensions (128-expert olmoe-mix_130B base + new init-only experts)
s3sync extensions/moe_1b14b_129experts_olmoe-mix_130B_1103_step30995_init_average ${BASE_S3_PATH}/regular_moe/moe_1b14b_129experts_olmoe-mix_130B_1103_step30995_init_average
s3sync extensions/moe_1b14b_129experts_olmoe-mix_130B_1103_step30995_init_random_expert ${BASE_S3_PATH}/regular_moe/moe_1b14b_129experts_olmoe-mix_130B_1103_step30995_init_random_expert
s3sync extensions/moe_1b14b_129experts_olmoe-mix_130B_1103_step30995_init_random_expert-hf ${BASE_S3_PATH}/regular_moe/moe_1b14b_129experts_olmoe-mix_130B_1103_step30995_init_random_expert-hf
s3sync extensions/moe_1b14b_129experts_olmoe-mix_130B_1103_step30995_init_top2_average ${BASE_S3_PATH}/regular_moe/moe_1b14b_129experts_olmoe-mix_130B_1103_step30995_init_top2_average
s3sync extensions/moe_1b14b_130experts_olmoe-mix_130B_1103_step30995_init_average ${BASE_S3_PATH}/regular_moe/moe_1b14b_130experts_olmoe-mix_130B_1103_step30995_init_average
s3sync extensions/moe_1b14b_130experts_olmoe-mix_130B_1103_step30995_init_average_noise ${BASE_S3_PATH}/regular_moe/moe_1b14b_130experts_olmoe-mix_130B_1103_step30995_init_average_noise
s3sync extensions/moe_1b14b_130experts_olmoe-mix_130B_1103_step30995_init_average_noise_10perc ${BASE_S3_PATH}/regular_moe/moe_1b14b_130experts_olmoe-mix_130B_1103_step30995_init_average_noise_10perc
s3sync extensions/moe_1b14b_130experts_olmoe-mix_130B_1103_step30995_init_top2 ${BASE_S3_PATH}/regular_moe/moe_1b14b_130experts_olmoe-mix_130B_1103_step30995_init_top2
s3sync extensions/moe_1b14b_130experts_olmoe-mix_130B_1103_step30995_init_top2_average_noise_10perc ${BASE_S3_PATH}/regular_moe/moe_1b14b_130experts_olmoe-mix_130B_1103_step30995_init_top2_average_noise_10perc
s3sync extensions/moe_1b14b_132experts_olmoe-mix_130B_1103_step30995_init_average ${BASE_S3_PATH}/regular_moe/moe_1b14b_132experts_olmoe-mix_130B_1103_step30995_init_average
s3sync extensions/moe_1b14b_132experts_olmoe-mix_130B_1103_step30995_init_average_noise_10perc ${BASE_S3_PATH}/regular_moe/moe_1b14b_132experts_olmoe-mix_130B_1103_step30995_init_average_noise_10perc
s3sync extensions/moe_1b14b_132experts_olmoe-mix_130B_1103_step30995_init_top2_average_noise_10perc ${BASE_S3_PATH}/regular_moe/moe_1b14b_132experts_olmoe-mix_130B_1103_step30995_init_top2_average_noise_10perc
s3sync extensions/moe_1b14b_132experts_olmoe-mix_130B_1103_step30995_init_top2_average_noise_10perc-hf ${BASE_S3_PATH}/regular_moe/moe_1b14b_132experts_olmoe-mix_130B_1103_step30995_init_top2_average_noise_10perc-hf
s3sync extensions/moe_1b14b_132experts_olmoe-mix_130B_1103_step30995_init_top2_code_average_noise ${BASE_S3_PATH}/regular_moe/moe_1b14b_132experts_olmoe-mix_130B_1103_step30995_init_top2_code_average_noise
s3sync extensions/moe_1b14b_132experts_olmoe-mix_130B_1103_step30995_init_top2_code_average_noise-hf ${BASE_S3_PATH}/regular_moe/moe_1b14b_132experts_olmoe-mix_130B_1103_step30995_init_top2_code_average_noise-hf
s3sync extensions/moe_1b14b_136experts_olmoe-mix_130B_1103_step30995_init_average_noise_10perc ${BASE_S3_PATH}/regular_moe/moe_1b14b_136experts_olmoe-mix_130B_1103_step30995_init_average_noise_10perc
s3sync extensions/moe_1b14b_136experts_olmoe-mix_130B_1103_step30995_init_top2_average_noise_10perc ${BASE_S3_PATH}/regular_moe/moe_1b14b_136experts_olmoe-mix_130B_1103_step30995_init_top2_average_noise_10perc

## Regular MoE base extensions (reducedp512sharedexp1 base, 0308)
s3sync extensions/moereducedp512sharedexp1_1b14b_132experts_0308_step30995_init_top2_average_noise_split ${BASE_S3_PATH}/regular_moe/moereducedp512sharedexp1_1b14b_132experts_0308_step30995_init_top2_average_noise_split

## Regular MoE sequential base (math-trained model + starcoder code experts added, pre-training)
s3sync extensions/moe_1b14b_128base_4math_10B_4code_init_top2_starcoder_average_noise ${BASE_S3_PATH}/regular_moe/moe_1b14b_128base_4math_10B_4code_init_top2_starcoder_average_noise

## Twolevel MoE base extensions (stability prenorm noqknorm base, 1121)
s3sync extensions/twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121_step30995_init_average ${BASE_S3_PATH}/twolevel_moe/twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121_step30995_init_average
s3sync extensions/twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121_step30995_init_top2 ${BASE_S3_PATH}/twolevel_moe/twolevelbatchlb-32_1b14b_stability_prenorm_noqknorm_1121_step30995_init_top2
s3sync extensions/twolevelbatchlb-32_1b14b_129experts_stability_prenorm_noqknorm_1121_step30995_init_average ${BASE_S3_PATH}/twolevel_moe/twolevelbatchlb-32_1b14b_129experts_stability_prenorm_noqknorm_1121_step30995_init_average
s3sync extensions/twolevelbatchlb-32_1b14b_129experts_stability_prenorm_noqknorm_1121_step30995_init_top2_average ${BASE_S3_PATH}/twolevel_moe/twolevelbatchlb-32_1b14b_129experts_stability_prenorm_noqknorm_1121_step30995_init_top2_average
s3sync extensions/twolevelbatchlb-32_1b14b_130experts_stability_prenorm_noqknorm_1121_step30995_init_average_noise_10perc ${BASE_S3_PATH}/twolevel_moe/twolevelbatchlb-32_1b14b_130experts_stability_prenorm_noqknorm_1121_step30995_init_average_noise_10perc
s3sync extensions/twolevelbatchlb-32_1b14b_130experts_stability_prenorm_noqknorm_1121_step30995_init_top2_average_noise ${BASE_S3_PATH}/twolevel_moe/twolevelbatchlb-32_1b14b_130experts_stability_prenorm_noqknorm_1121_step30995_init_top2_average_noise


# =============================================================================
# Regular MoE math
# =============================================================================

## LR sweep (129experts_1trained, 5B, init_top2_average)
s3sync freeze-fix-moe1b14b_129experts_1trained_math_init_top2_average_5B_lr_4e-3 ${BASE_S3_PATH}/regular_moe/freeze-fix-moe1b14b_129experts_1trained_math_init_top2_average_5B_lr_4e-3
s3sync freeze-fix-moe1b14b_129experts_1trained_math_init_top2_average_5B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/freeze-fix-moe1b14b_129experts_1trained_math_init_top2_average_5B_lr_4e-4
s3sync freeze-fix-moe1b14b_129experts_1trained_math_init_top2_average_5B_lr_4e-5 ${BASE_S3_PATH}/regular_moe/freeze-fix-moe1b14b_129experts_1trained_math_init_top2_average_5B_lr_4e-5
s3sync freeze-fix-moe1b14b_129experts_1trained_math_init_top2_average_10B_lr_4e-3 ${BASE_S3_PATH}/regular_moe/freeze-fix-moe1b14b_129experts_1trained_math_init_top2_average_10B_lr_4e-3

## Initialization (different init strategies, math data)
s3sync freeze-fix-moe1b14b_129experts_1trained_math_init_average_5B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/freeze-fix-moe1b14b_129experts_1trained_math_init_average_5B_lr_4e-4
s3sync freeze-fix-moe1b14b_129experts_1trained_math_init_average_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/freeze-fix-moe1b14b_129experts_1trained_math_init_average_10B_lr_4e-4
s3sync freeze-fix-moe1b14b_129experts_1trained_math_init_random_expert_5B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/freeze-fix-moe1b14b_129experts_1trained_math_init_random_expert_5B_lr_4e-4
s3sync freeze-fix-moe1b14b_experts_2trained_math_init_top2_5B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/freeze-fix-moe1b14b_experts_2trained_math_init_top2_5B_lr_4e-4
s3sync freeze-fix-moe1b14b_130experts_2trained_math_init_average_noise_10pc_5B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/freeze-fix-moe1b14b_130experts_2trained_math_init_average_noise_10pc_5B_lr_4e-4
s3sync freeze-fix-moe1b14b_132experts_4trained_math_init_average_noise_10pc_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/freeze-fix-moe1b14b_132experts_4trained_math_init_average_noise_10pc_10B_lr_4e-4
s3sync ff-moe1b14b_132experts_4trained_math_init_top2_average_no_router_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/ff-moe1b14b_132experts_4trained_math_init_top2_average_no_router_10B_lr_4e-4

## Num experts / tokens grid (init_top2_average[_noise], varying experts & tokens)
s3sync freeze-fix-moe1b14b_129experts_1trained_math_init_top2_average_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/freeze-fix-moe1b14b_129experts_1trained_math_init_top2_average_10B_lr_4e-4
s3sync freeze-fix-moe1b14b_129experts_1trained_math_init_top2_average_20B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/freeze-fix-moe1b14b_129experts_1trained_math_init_top2_average_20B_lr_4e-4
s3sync freeze-fix-moe1b14b_130experts_2trained_math_init_top2_average_noise_5B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/freeze-fix-moe1b14b_130experts_2trained_math_init_top2_average_noise_5B_lr_4e-4
s3sync freeze-fix-moe1b14b_130experts_2trained_math_init_top2_average_noise_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/freeze-fix-moe1b14b_130experts_2trained_math_init_top2_average_noise_10B_lr_4e-4
s3sync freeze-fix-moe1b14b_130experts_2trained_math_init_top2_average_noise_20B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/freeze-fix-moe1b14b_130experts_2trained_math_init_top2_average_noise_20B_lr_4e-4
s3sync freeze-fix-moe1b14b_132experts_4trained_math_init_top2_average_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/freeze-fix-moe1b14b_132experts_4trained_math_init_top2_average_10B_lr_4e-4
s3sync freeze-fix-moe1b14b_132experts_4trained_math_init_top2_average_noise_5B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/freeze-fix-moe1b14b_132experts_4trained_math_init_top2_average_noise_5B_lr_4e-4
s3sync freeze-fix-moe1b14b_132experts_4trained_math_init_top2_average_noise_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/freeze-fix-moe1b14b_132experts_4trained_math_init_top2_average_noise_10B_lr_4e-4
s3sync freeze-fix-moe1b14b_132experts_4trained_math_init_top2_average_noise_20B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/freeze-fix-moe1b14b_132experts_4trained_math_init_top2_average_noise_20B_lr_4e-4
s3sync freeze-fix-moe1b14b_136experts_8trained_math_init_top2_average_noise_5B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/freeze-fix-moe1b14b_136experts_8trained_math_init_top2_average_noise_5B_lr_4e-4
s3sync freeze-fix-moe1b14b_136experts_8trained_math_init_top2_average_noise_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/freeze-fix-moe1b14b_136experts_8trained_math_init_top2_average_noise_10B_lr_4e-4
s3sync freeze-fix-moe1b14b_136experts_8trained_math_init_top2_average_noise_20B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/freeze-fix-moe1b14b_136experts_8trained_math_init_top2_average_noise_20B_lr_4e-4

## Num experts / tokens grid (reducedp512sharedexp1 base)
s3sync moereducedp512sharedexp1_132experts_4trained_math_init_top2_average_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/moereducedp512sharedexp1_132experts_4trained_math_init_top2_average_10B_lr_4e-4
s3sync moereducedp512sharedexp1_132experts_4trained_math_init_top2_average_split_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/moereducedp512sharedexp1_132experts_4trained_math_init_top2_average_split_10B_lr_4e-4

## Shared expert (always-active shared experts init)
s3sync ff-moe1b14b_132experts_4trained_sharedexp56math_init_top2_average_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/ff-moe1b14b_132experts_4trained_sharedexp56math_init_top2_average_10B_lr_4e-4


# =============================================================================
# Regular MoE code
# =============================================================================

## Initialization (code / code-mix / starcoder data)
s3sync ff-moe1b14b_132experts_4trained_code_mix_init_top2_average_noise_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/ff-moe1b14b_132experts_4trained_code_mix_init_top2_average_noise_10B_lr_4e-4
s3sync ff-moe1b14b_132experts_4trained_code_mix_init_top2_average_noise_no_router_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/ff-moe1b14b_132experts_4trained_code_mix_init_top2_average_noise_no_router_10B_lr_4e-4
s3sync ff-moe1b14b_132experts_4trained_starcoder_init_top2_average_noise_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/ff-moe1b14b_132experts_4trained_starcoder_init_top2_average_noise_10B_lr_4e-4
s3sync freeze-fix-moe1b14b_132experts_4trained_code_init_average_noise_10pc_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/freeze-fix-moe1b14b_132experts_4trained_code_init_average_noise_10pc_10B_lr_4e-4
s3sync freeze-fix-moe1b14b_132experts_4trained_code_mix_init_average_noise_10pc_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/freeze-fix-moe1b14b_132experts_4trained_code_mix_init_average_noise_10pc_10B_lr_4e-4
s3sync freeze-fix-moe1b14b_132experts_4trained_starcoder_init_average_noise_10pc_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/freeze-fix-moe1b14b_132experts_4trained_starcoder_init_average_noise_10pc_10B_lr_4e-4

## Initialization (reducedp512sharedexp1 base)
s3sync moereducedp512sharedexp1_132experts_4trained_code_mix_init_top2_average_noise_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/moereducedp512sharedexp1_132experts_4trained_code_mix_init_top2_average_noise_10B_lr_4e-4


# =============================================================================
# Regular MoE french (croissant)
# =============================================================================
s3sync ff-moe1b14b_132experts_4trained_croissant_init_average_noise_10pc_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/ff-moe1b14b_132experts_4trained_croissant_init_average_noise_10pc_10B_lr_4e-4


# =============================================================================
# Regular MoE merge and router train
# =============================================================================

## Merged models (token-weighted averaging of math + code experts)
s3sync merged_moe_1b14b_128base_1math_5B_4code_mix_10B_init_top2_average_noise ${BASE_S3_PATH}/regular_moe/merged_moe_1b14b_128base_1math_5B_4code_mix_10B_init_top2_average_noise
s3sync merged_moe_1b14b_128base_1math_5B_4code_mix_10B_init_top2_average_noise-hf ${BASE_S3_PATH}/regular_moe/merged_moe_1b14b_128base_1math_5B_4code_mix_10B_init_top2_average_noise-hf
s3sync merged_moe_1b14b_128base_1math_10B_4code_mix_10B_init_top2_average_noise ${BASE_S3_PATH}/regular_moe/merged_moe_1b14b_128base_1math_10B_4code_mix_10B_init_top2_average_noise
s3sync merged_moe_1b14b_128base_1math_10B_4code_mix_10B_init_top2_average_noise-hf ${BASE_S3_PATH}/regular_moe/merged_moe_1b14b_128base_1math_10B_4code_mix_10B_init_top2_average_noise-hf
s3sync merged_moe_1b14b_128base_2math_5B_4code_mix_10B_init_top2_average_noise ${BASE_S3_PATH}/regular_moe/merged_moe_1b14b_128base_2math_5B_4code_mix_10B_init_top2_average_noise
s3sync merged_moe_1b14b_128base_2math_5B_4code_mix_10B_init_top2_average_noise-hf ${BASE_S3_PATH}/regular_moe/merged_moe_1b14b_128base_2math_5B_4code_mix_10B_init_top2_average_noise-hf
s3sync merged_moe_1b14b_128base_2math_10B_4code_mix_10B_init_top2_average_noise ${BASE_S3_PATH}/regular_moe/merged_moe_1b14b_128base_2math_10B_4code_mix_10B_init_top2_average_noise
s3sync merged_moe_1b14b_128base_2math_10B_4code_mix_10B_init_top2_average_noise-hf ${BASE_S3_PATH}/regular_moe/merged_moe_1b14b_128base_2math_10B_4code_mix_10B_init_top2_average_noise-hf
s3sync merged_moe_1b14b_128base_4math_5B_4code_mix_10B_init_top2_average_noise ${BASE_S3_PATH}/regular_moe/merged_moe_1b14b_128base_4math_5B_4code_mix_10B_init_top2_average_noise
s3sync merged_moe_1b14b_128base_4math_5B_4code_mix_10B_init_top2_average_noise-hf ${BASE_S3_PATH}/regular_moe/merged_moe_1b14b_128base_4math_5B_4code_mix_10B_init_top2_average_noise-hf
s3sync merged_moe_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise ${BASE_S3_PATH}/regular_moe/merged_moe_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise
s3sync merged_moe_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise-hf ${BASE_S3_PATH}/regular_moe/merged_moe_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise-hf
s3sync merged_moereducedp512sharedexp1_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise ${BASE_S3_PATH}/regular_moe/merged_moereducedp512sharedexp1_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise
s3sync merged_moereducedp512sharedexp1_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise-hf ${BASE_S3_PATH}/regular_moe/merged_moereducedp512sharedexp1_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise-hf

## Weight merge (raw parameter averaging)
s3sync weight_merge_moe_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise ${BASE_S3_PATH}/regular_moe/weight_merge_moe_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise
s3sync weight_merge_moe_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise-hf ${BASE_S3_PATH}/regular_moe/weight_merge_moe_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise-hf

## Sequential add (math experts trained, then code experts added on top)
s3sync ff-moe_1b14b_128base_4math_10B_4code_init_top2_code_mix_average_noise_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/ff-moe_1b14b_128base_4math_10B_4code_init_top2_code_mix_average_noise_10B_lr_4e-4
s3sync ff-moe_1b14b_128base_4math_10B_4code_init_top2_starcoder_average_noise_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/ff-moe_1b14b_128base_4math_10B_4code_init_top2_starcoder_average_noise_10B_lr_4e-4

## Router training on merged models (rt- prefix)
s3sync rt-merged_moe_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise_1B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/rt-merged_moe_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise_1B_lr_4e-4
s3sync rt-merged_moereducedp512sharedexp1_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise_1B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/rt-merged_moereducedp512sharedexp1_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise_1B_lr_4e-4
s3sync rt-realdata-merged_moe_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise_1B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/rt-realdata-merged_moe_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise_1B_lr_4e-4


# =============================================================================
# Regular MoE selective training
# =============================================================================

## Baselines (full finetune / all experts trained)
s3sync moereducedp512sharedexp1_1b14b_128experts_all_trained_math_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/moereducedp512sharedexp1_1b14b_128experts_all_trained_math_10B_lr_4e-4
s3sync moereducedp512sharedexp1_1b14b_128experts_full_finetune_math_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/moereducedp512sharedexp1_1b14b_128experts_full_finetune_math_10B_lr_4e-4

## Selective - math (train specific expert indices)
s3sync moe1b14b_128experts_76_5_122_126_trained_math_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/moe1b14b_128experts_76_5_122_126_trained_math_10B_lr_4e-4
s3sync moe1b14b_128experts_76_5_122_126_trained_math_no_router_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/moe1b14b_128experts_76_5_122_126_trained_math_no_router_10B_lr_4e-4
s3sync moereducedp512sharedexp1_1b14b_128experts_127_69_30_3_trained_math_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/moereducedp512sharedexp1_1b14b_128experts_127_69_30_3_trained_math_10B_lr_4e-4
s3sync moereducedp512sharedexp1_1b14b_128experts_69_30_3_6_trained_math_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/moereducedp512sharedexp1_1b14b_128experts_69_30_3_6_trained_math_10B_lr_4e-4

## Selective - code (train specific expert indices)
s3sync moe1b14b_128experts_76_41_120_3_trained_code_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/moe1b14b_128experts_76_41_120_3_trained_code_10B_lr_4e-4
s3sync moe1b14b_128experts_76_41_120_3_trained_code_no_router_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/moe1b14b_128experts_76_41_120_3_trained_code_no_router_10B_lr_4e-4
s3sync moereducedp512sharedexp1_1b14b_128experts_123_1_76_6_trained_code_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/moereducedp512sharedexp1_1b14b_128experts_123_1_76_6_trained_code_10B_lr_4e-4
s3sync moereducedp512sharedexp1_1b14b_128experts_127_123_1_76_trained_code_10B_lr_4e-4 ${BASE_S3_PATH}/regular_moe/moereducedp512sharedexp1_1b14b_128experts_127_123_1_76_trained_code_10B_lr_4e-4


# =============================================================================
# Twolevel MoE math
# =============================================================================

## Initialization
s3sync freeze-fix-twolevel_129experts_1trained_math_init_average_5B_lr_4e-4 ${BASE_S3_PATH}/twolevel_moe/freeze-fix-twolevel_129experts_1trained_math_init_average_5B_lr_4e-4
s3sync freeze-fix-twolevel_129experts_1trained_math_init_top2_average_5B_lr_4e-4 ${BASE_S3_PATH}/twolevel_moe/freeze-fix-twolevel_129experts_1trained_math_init_top2_average_5B_lr_4e-4
s3sync freeze-fix-twolevel_130experts_2trained_math_init_average_noise_10pc_5B_lr_4e-4 ${BASE_S3_PATH}/twolevel_moe/freeze-fix-twolevel_130experts_2trained_math_init_average_noise_10pc_5B_lr_4e-4
s3sync freeze-fix-twolevel_130experts_2trained_math_init_top2_average_noise_5B_lr_4e-4 ${BASE_S3_PATH}/twolevel_moe/freeze-fix-twolevel_130experts_2trained_math_init_top2_average_noise_5B_lr_4e-4
s3sync twolevel_132experts_4trained_forced_math_init_top2_average_noise_10B_lr_4e-4 ${BASE_S3_PATH}/twolevel_moe/twolevel_132experts_4trained_forced_math_init_top2_average_noise_10B_lr_4e-4


# =============================================================================
# Twolevel MoE code
# =============================================================================

## Initialization
s3sync twolevel_132experts_4trained_forced_code_mix_init_top2_average_noise_10B_lr_4e-4 ${BASE_S3_PATH}/twolevel_moe/twolevel_132experts_4trained_forced_code_mix_init_top2_average_noise_10B_lr_4e-4


# =============================================================================
# Twolevel MoE merge and router train
# =============================================================================

## Merged models
s3sync merged_twolevel_1b14b_128base_4math_10B_4code_mix_10B_forced_init_top2_average_noise ${BASE_S3_PATH}/twolevel_moe/merged_twolevel_1b14b_128base_4math_10B_4code_mix_10B_forced_init_top2_average_noise
s3sync merged_twolevel_1b14b_128base_4math_10B_4code_mix_10B_forced_init_top2_average_noise-hf ${BASE_S3_PATH}/twolevel_moe/merged_twolevel_1b14b_128base_4math_10B_4code_mix_10B_forced_init_top2_average_noise-hf

## Router training on merged models
s3sync rt-merged_twolevel_1b14b_128base_4math_10B_4code_mix_10B_forced_init_top2_average_noise_1B_lr_4e-4 ${BASE_S3_PATH}/twolevel_moe/rt-merged_twolevel_1b14b_128base_4math_10B_4code_mix_10B_forced_init_top2_average_noise_1B_lr_4e-4


# =============================================================================
# Twolevel MoE selective training
# =============================================================================
s3sync twolevel_1b14b_128experts_99_0_42_88_trained_math_10B_lr_4e-4 ${BASE_S3_PATH}/twolevel_moe/twolevel_1b14b_128experts_99_0_42_88_trained_math_10B_lr_4e-4
s3sync twolevel_1b14b_128experts_63_26_6_19_trained_code_10B_lr_4e-4 ${BASE_S3_PATH}/twolevel_moe/twolevel_1b14b_128experts_63_26_6_19_trained_code_10B_lr_4e-4
