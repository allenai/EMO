#!/usr/bin/env bash
# Held-out (300B sample) routing pass of every individual square of the freeze ablations (+ the unfrozen 4e-4 reference arm) at the
# matched points, for the "squares alone, mean of 8" lines of the freeze section's held-out chart (user request 2026-10-13). Batched:
# one allocated 1-GPU job per arm, point and half (4 passes run in sequence inside the job via bash -c); passes with a DONE/meta.json are
# skipped. Idempotent.   bash scripts/sparse_experts/olmoe3_squares/freeze_square_heldout.sh
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SAMPLE=sample_8k_300b.npz; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad; mkdir -p $SP
say() { echo "$(date -u +%m-%d\ %H:%M) [sqho] $*"; }
launch() { local name=$1; shift; local log=$SP/launch_$name.log; local u=""
  for attempt in 1 2 3 4 5 6; do
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$name" --gpus 1 --allocated -- "$@" > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$name: ${u:-LAUNCH FAILED}"; }
have() { [ -f "$1/meta.json" ] || [ -f "$1/rank0/DONE" ]; }
pick() { local d=$1 point=$2; local st; for st in $(ls $d | grep -E '^step[0-9]+$' | sed 's/step//' | sort -n); do
    case $point in mid) [ $st -ge 1360 ] && [ $st -le 1375 ] && echo $st;; fin) [ $st -ge 2370 ] && echo $st;; esac; done | tail -n 1; }
n=0
for arm in frzrouter frzrouter_attn frznon_expert ""; do hr=runs_heldout300b_emorand8_lr4e-4${arm:+_$arm}; rp=olmoe3_275m_emorand8_lr4e-4${arm:+_$arm}_square; mkdir -p $S/olmoe3_routing/$hr
  for point in w1mid:30000 w1fin:38148 w2fin:57221; do pt=${point%%:*}; st_match=${point##*:}
    for half in 0 1; do cmds=""
      for g in $(seq $((half*4)) $((half*4+3))); do case $pt in w1mid) run=${rp}$g; st=$(pick $S/$run mid);; w1fin) run=${rp}$g; st=$(pick $S/$run fin);; w2fin) run=${rp}${g}_w2; st=$(pick $S/$run fin);; esac
        [ -n "$st" ] || continue; have $S/olmoe3_routing/$hr/sub${g}_match$st_match/none && continue
        cmds="$cmds python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint $W/$run/step$st --instances $W/olmoe3_routing/$SAMPLE --out-dir $W/olmoe3_routing/$hr/sub${g}_match$st_match/none/rank0 --restrict none --batch-size 8 --log-every 100;"; done
      [ -n "$cmds" ] || continue
      launch "$hr-squares-$pt-h$half" bash -c "$cmds"; n=$((n+1))
    done; done; done
say "launched $n batched square held-out jobs"
