prune_keep_k=32

MODEL_DIR="/root/ryanwang/phdbrainstorm/FlexMoE/models/twolevelbatchlb-32_1b14b_stability_filter-true_zlossweight-1e-3_1115/step30995"

SAVE_PATH="/root/ryanwang/phdbrainstorm/FlexMoE/models/twolevelbatchlb-32_1b14b_stability_filter-true_zlossweight-1e-3_1115/step30995_keepk32"

python src/scripts/eval/prune_moe_checkpoint.py \
  --checkpoint_path "$MODEL_DIR" \
  --save_path "$SAVE_PATH" \
  --prune_keep_k "$prune_keep_k"

