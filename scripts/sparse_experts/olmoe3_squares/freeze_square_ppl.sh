#!/usr/bin/env bash
# ppl-set evaluation of every individual square of the freeze ablations (+ the unfrozen 4e-4 reference arm) at the matched points, for
# the "squares alone, mean of 8" lines of the freeze section (user request 2026-10-13). One allocated 1-GPU job per checkpoint; idempotent.
#   bash scripts/sparse_experts/olmoe3_squares/freeze_square_ppl.sh
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad; mkdir -p $SP
say() { echo "$(date -u +%m-%d\ %H:%M) [sqppl] $*"; }
launch() { local name=$1; shift; local log=$SP/launch_$name.log; local u=""
  for attempt in 1 2 3 4 5 6; do
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$name" --gpus 1 --allocated -- "$@" > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$name: ${u:-LAUNCH FAILED}"; }
n=0
for arm in frzrouter frzrouter_attn frznon_expert ""; do sqn=olmoe3_squares_emorand8_lr4e-4${arm:+_$arm}; rp=olmoe3_275m_emorand8_lr4e-4${arm:+_$arm}_square
  for g in 0 1 2 3 4 5 6 7; do
    for run in ${rp}$g ${rp}${g}_w2; do d=$S/$run; [ -d $d ] || continue
      for st in $(ls $d | grep -E '^step[0-9]+$' | sed 's/step//' | sort -n); do
        case $run in *_w2) [ $st -ge 2370 ] || continue;; *) { [ $st -ge 2370 ] || { [ $st -ge 1360 ] && [ $st -le 1375 ]; }; } || continue;; esac   # finals (~2383) and the mid-window point (~1365-1372); skip ephemeral leftovers
        [ -f $S/$sqn/ppl_validation/$run/step$st.json ] && continue
        launch "$sqn-ppl-$run-$st" python scripts/debug_validation/eval_ppl_validation.py --checkpoints "$W/$run/step$st" --out-dir "$W/$sqn/ppl_validation" --batch-size 8; n=$((n+1))
      done; done; done; done
say "launched $n square ppl jobs"
