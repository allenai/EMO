#!/usr/bin/env bash
# Re-creates the EMO 512e baseline checkpoint at step 20,000 (10.5B tokens), which the 20B baseline run no longer has (pruned), so
# that the Q5 EMO baseline line has its first matched point: same recipe as olmoe3_275m_emo_20b_baseline.sh (WSD trunk, constant LR,
# same stream from the 10B checkpoint), stopped at step 20,000. 1 jupiter node x 8 H100, allocated. Then run the held-out pass:
#   extract_routing.py --checkpoint .../olmoe3_275m_emo_20b_step20000/step20000 --out-dir .../runs_heldout300b_emo/baseline_step20000/none/rank0
export OLMOE3_NUM_EXPERTS=512
export OLMOE3_EMO=1
export OLMOE3_TOKENS=$((20000 * 524288))
export OLMOE3_NUM_NODES=1
export OLMOE3_FIXED_STEPS=20000
export OLMOE3_RUNNAME=olmoe3_275m_emo_20b_step20000
export OLMOE3_WANDB_TAGS=20b,olmoe3_squares_emorand,step20000
export OLMOE3_FOLLOW="${OLMOE3_FOLLOW:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "${1:-launch}" \
    --trainer.load_path=/weka/oe-training-default/ryanwang/EMO/sparse_experts/olmoe3_275m_emo_10b/step19074 \
    --trainer.load_strategy=always
