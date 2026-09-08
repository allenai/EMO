#!/usr/bin/env bash
# Launch the full prefix-restriction routing sweep: 2 models x 13 conditions, one 1-GPU unallocated
# filler Beaker job each (~8 min per job: 8000 instances = 65.5M tokens at ~230k tok/s + model load).
#   bash scripts/sparse_experts/olmoe3_routing/launch_sweep.sh [max_instances] [out_subdir]
# Conditions: none; layers 1-3 and 1-6 at pools 32/64/128/256 (later layers free); all 9 layers at
# each pool (reference). Skips conditions whose DONE marker exists. Push before launching.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
MAX=${1:-0}; SUB=${2:-runs}
W=/weka/oe-training-default/ryanwang/EMO/sparse_experts
declare -A CK=([std]=olmoe3_275m_10b [emo]=olmoe3_275m_emo_10b)
CONDS=(none)
for P in 32 64 128 256; do CONDS+=("1-3:$P" "1-6:$P" "1-9:$P"); done
export PATH=/root/.conda/envs/emo/bin:$PATH
for m in std emo; do for cond in "${CONDS[@]}"; do
  if [[ -f sparse_experts/olmoe3_routing/$SUB/$m/$cond/DONE || -f sparse_experts/olmoe3_routing/$SUB/$m/$cond/rank0/DONE ]]; then echo "skip $m $cond (done)"; continue; fi
  tag=$(echo "$cond" | tr ':' 'P' | tr -d '-')
  PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "routing-$SUB-$m-$tag" --gpus 1 -- \
    python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$W/${CK[$m]}/step19074" \
      --instances "$W/olmoe3_routing/sample_8k.npz" --out-dir "$W/olmoe3_routing/$SUB/$m/$cond/rank0" \
      --restrict "$cond" --max-instances "$MAX" --batch-size 8 --log-every 25 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -oE 'beaker.org/ex/[A-Z0-9]+' | head -1 | sed "s|^|$m $cond: |"
done; done
