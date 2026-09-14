#!/usr/bin/env bash
# Launch the learned-d hyper-parameter sweep (2000 steps each, 1 allocated node per config, fire-and-forget).
# One-at-a-time variations around the default (T=2, lambda=0.01, warmup=500, d_head LR x10) + the uniform-pool EMO control.
#   bash scripts/sparse_experts/learnedd/launch_sweep.sh [config ...]     (default: all)
# Experiment ids are appended to sparse_experts/learnedd_sweep/experiments.tsv
set -u
cd "$(git rev-parse --show-toplevel)"
OUT=sparse_experts/learnedd_sweep; mkdir -p "$OUT"
declare -A CFG=(
  [base]="OLMOE3_LD_TEMP=2.0 OLMOE3_LD_LAMBDA=0.01 OLMOE3_LD_WARMUP=500"
  [T1]="OLMOE3_LD_TEMP=1.0 OLMOE3_LD_LAMBDA=0.01 OLMOE3_LD_WARMUP=500"
  [T4]="OLMOE3_LD_TEMP=4.0 OLMOE3_LD_LAMBDA=0.01 OLMOE3_LD_WARMUP=500"
  [l0.003]="OLMOE3_LD_TEMP=2.0 OLMOE3_LD_LAMBDA=0.003 OLMOE3_LD_WARMUP=500"
  [l0.03]="OLMOE3_LD_TEMP=2.0 OLMOE3_LD_LAMBDA=0.03 OLMOE3_LD_WARMUP=500"
  [w0]="OLMOE3_LD_TEMP=2.0 OLMOE3_LD_LAMBDA=0.01 OLMOE3_LD_WARMUP=0"
  [w1000]="OLMOE3_LD_TEMP=2.0 OLMOE3_LD_LAMBDA=0.01 OLMOE3_LD_WARMUP=1000"
  [m1]="OLMOE3_LD_TEMP=2.0 OLMOE3_LD_LAMBDA=0.01 OLMOE3_LD_WARMUP=500 OLMOE3_LD_LR_MULT=1"
  [m30]="OLMOE3_LD_TEMP=2.0 OLMOE3_LD_LAMBDA=0.01 OLMOE3_LD_WARMUP=500 OLMOE3_LD_LR_MULT=30"
  [control]="OLMOE3_LD_CONTROL=1"
)
ORDER=(base T1 T4 l0.003 l0.03 w0 w1000 m1 m30 control)
sel=("$@"); [ ${#sel[@]} -eq 0 ] && sel=("${ORDER[@]}")
for name in "${sel[@]}"; do
  spec="${CFG[$name]:?unknown config $name}"
  for attempt in 1 2 3 4; do
    log="$OUT/launch_${name}.log"
    env $spec OLMOE3_FOLLOW=0 bash scripts/sparse_experts/model_scripts/olmoe3_275m_emo_learnedd_sweep.sh launch > "$log" 2>&1
    ex=$(grep -o "beaker.org/ex/[A-Z0-9]*" "$log" | tail -1 | sed 's#beaker.org/ex/##')
    if [ -n "$ex" ]; then
      run=$(grep -o "olmoe3_275m_emo_learnedd_sweep_[A-Za-z0-9._]*" "$log" | head -1)
      printf "%s\t%s\t%s\t%s\n" "$name" "$ex" "$run" "$(date -u +%FT%TZ)" >> "$OUT/experiments.tsv"
      echo "$name -> $ex ($run)"; break
    fi
    echo "$name: launch attempt $attempt failed ($(grep -m1 -E 'Error|error|BrokenPipe' "$log" | cut -c1-120))"; sleep 20
  done
done
