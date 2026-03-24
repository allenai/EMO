#!/bin/bash

S3_BASE="s3://ai2-sewonm/akshitab/mose/evals/extensions"
OUTPUT_DIR="router_evals"

MODELS=(
  # # "freeze-fix-moe1b14b_132experts_4trained_math_init_top2_average_noise_10B_lr_4e-4_step2385-hf"
  # # "ff-moe1b14b_132experts_4trained_code_mix_init_top2_average_noise_10B_lr_4e-4_step2385-hf"
  # # "ff-moe_1b14b_128base_4math_10B_4code_init_top2_code_mix_average_noise_10B_lr_4e-4_step2385-hf"
  # # "merged_moe_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise-hf"

  # # base model:
  # moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123_step30995-hf
  # # math extension before training:
  # extensions_moe_1b14b_132experts_olmoe-mix_130B_1103_step30995_init_top2_average_noise_10perc-hf
  # # code extension before training:
  # extensions_moe_1b14b_132experts_olmoe-mix_130B_1103_step30995_init_top2_code_average_noise-hf

  # # math extension after training:
  # freeze-fix-moe1b14b_132experts_4trained_math_init_top2_average_noise_10B_lr_4e-4_step2385-hf
  # # code extension after training:
  # ff-moe1b14b_132experts_4trained_code_mix_init_top2_average_noise_10B_lr_4e-4_step2385-hf

  # # merged model:
  # merged_moe_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise-hf

  # moe1b14b_128experts_76_5_122_126_trained_math_10B_lr_4e-4_step2385-hf

  # rt-realdata-merged_moe_1b14b_128base_4math_10B_4code_mix_10B_init_top2_average_noise_1B_lr_4e-4_step239-hf

  # moe1b14b_128experts_76_5_122_126_trained_math_10B_lr_4e-4_step2385-hf
  # moe1b14b_128experts_76_41_120_3_trained_code_10B_lr_4e-4_step2385-hf

  # moe1b14b_128experts_76_5_122_126_trained_math_no_router_10B_lr_4e-4_step2385-hf
  # moe1b14b_128experts_76_41_120_3_trained_code_no_router_10B_lr_4e-4_step2385-hf

  # moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_0308_step30995-hf
  twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301_step30995-hf
)

mkdir -p "$OUTPUT_DIR"

for model in "${MODELS[@]}"; do
  # Use the full model path as the local directory name (replace / with _)
  local_dir="$OUTPUT_DIR/${model//\//_}"
  mkdir -p "$local_dir"

  echo "==> Downloading *-router.jsonl from $model"
  aws s3 cp "${S3_BASE}/${model}/" "$local_dir/" \
    --recursive \
    --exclude "*" \
    --include "*-router.jsonl"

  echo ""
done

echo "Done. Files saved under $OUTPUT_DIR/"
