#!/usr/bin/env bash
# ppl-set evaluation of the individual squares of the freeze ablations (+ the unfrozen 4e-4 reference arm) at the matched points, for the
# "squares alone, mean of 8" lines of the freeze section (user request 2026-10-13). BATCHED: one allocated 1-GPU job per arm and point,
# scoring that point's 8 square checkpoints in sequence (eval_ppl_validation.py takes several --checkpoints); checkpoints that already have a
# json are skipped. Idempotent.   bash scripts/sparse_experts/olmoe3_squares/freeze_square_ppl.sh
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad; mkdir -p $SP
say() { echo "$(date -u +%m-%d\ %H:%M) [sqppl] $*"; }
launch() { local name=$1; shift; local log=$SP/launch_$name.log; local u=""
  for attempt in 1 2 3 4 5 6; do
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$name" --gpus 1 --allocated -- "$@" > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$name: ${u:-LAUNCH FAILED}"; }
pick() { local d=$1 point=$2; local st; for st in $(ls $d | grep -E '^step[0-9]+$' | sed 's/step//' | sort -n); do   # the checkpoint of a square dir at a point: mid (~1365-1372) or final (>= 2370)
    case $point in mid) [ $st -ge 1360 ] && [ $st -le 1375 ] && echo $st;; fin) [ $st -ge 2370 ] && echo $st;; esac; done | tail -n 1; }
n=0
for arm in frzrouter frzrouter_attn frznon_expert ""; do sqn=olmoe3_squares_emorand8_lr4e-4${arm:+_$arm}; rp=olmoe3_275m_emorand8_lr4e-4${arm:+_$arm}_square
  for point in w1mid w1fin w2fin; do ckpts=""
    for g in 0 1 2 3 4 5 6 7; do case $point in w1mid) run=${rp}$g; st=$(pick $S/$run mid);; w1fin) run=${rp}$g; st=$(pick $S/$run fin);; w2fin) run=${rp}${g}_w2; st=$(pick $S/$run fin);; esac
      [ -n "$st" ] || continue; [ -f $S/$sqn/ppl_validation/$run/step$st.json ] && continue; ckpts="$ckpts $W/$run/step$st"; done
    [ -n "$ckpts" ] || { say "$sqn $point: nothing to do"; continue; }
    launch "$sqn-ppl-squares-$point" python scripts/debug_validation/eval_ppl_validation.py --checkpoints $ckpts --out-dir "$W/$sqn/ppl_validation" --batch-size 8; n=$((n+1))
  done; done
say "launched $n batched square-ppl jobs"
