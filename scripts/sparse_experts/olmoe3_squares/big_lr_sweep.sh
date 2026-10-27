#!/usr/bin/env bash
# Pretraining-LR sweep of the EMO 1000e / 2000e models (user request 2026-10-20; report Q5 boxes 5E / 5G, like std128_lr_sweep.sh): six arms from
# scratch (1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3, 3.2e-3; the 8e-4 arm is re-run because the existing 10B run's step-10000 checkpoint was pruned) to step
# 10,000 = 5.24B tokens, 4 nodes each (allocated; baseline_keeper.sh resumes them), every step-10000 checkpoint evaluated on the SELECTION sample
# (sample_8k_300b_val.npz), the reporting sample (sample_8k_300b.npz) and the v3-small ppl sets.
#   bash scripts/sparse_experts/olmoe3_squares/big_lr_sweep.sh 1000e|2000e      (idempotent; detach; commit + push first)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
ARM="${1:?1000e|2000e}"; case $ARM in 1000e) BSF=4; PPLB=8;; 2000e) BSF=2; PPLB=4;; *) echo "1000e|2000e"; exit 1;; esac
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SAMPLE=sample_8k_300b.npz; VSAMPLE=sample_8k_300b_val.npz
SQN=olmoe3_squares_emo${ARM%e}lr; SQ=$S/$SQN; HR=runs_heldout300b_emo${ARM%e}lr; HRV=${HR}_val; LOG=$SQ/logs; SCRIPT=scripts/sparse_experts/model_scripts/olmoe3_275m_${ARM}_emo_lrsweep_5b.sh
SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad; mkdir -p $SQ $LOG $SP $S/olmoe3_routing/$HR $S/olmoe3_routing/$HRV $S/olmoe3_routing/baseline_keeper
LRS="1e-4 2e-4 4e-4 8e-4 1.6e-3 3.2e-3"
say() { echo "$(date -u +%m-%d\ %H:%M) [lrsweep $ARM] $*"; }
LIVE=""; for attempt in 1 2 3 4 5; do LIVE=$(timeout 600 beaker workspace experiments ai2/flex2 --format=json 2>/dev/null | python -c "
import json,sys
try: es=json.load(sys.stdin)
except Exception: es=[]
for e in es:
    js=e.get('jobs') or []; st=(js[-1] if js else {}).get('status',{})
    if not st.get('exited') and not st.get('finalized') and not st.get('canceled'): print(e.get('name',''))"); [ -n "$LIVE" ] && break; sleep 60; done
[ -n "$LIVE" ] || { say "ABORT: could not list the live Beaker experiments"; exit 1; }
LAUNCHED=" "
launch() { local name=$1; shift; local log=$SP/launch_$name.log; local u=""
  case "$LAUNCHED" in *" $name "*) return 0;; esac
  if grep -q -x -E "$(printf '%s' "$name" | sed 's/[.[\*^$]/\\&/g')(-[0-9a-f]+)*" <<< "$LIVE"; then say "$name: already live, not resubmitted"; LAUNCHED="$LAUNCHED$name "; return 0; fi
  for attempt in 1 2 3 4 5 6; do
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$name" --gpus 1 --allocated -- "$@" > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$name: ${u:-LAUNCH FAILED}"; [ -n "$u" ] && LAUNCHED="$LAUNCHED$name "; }
launch_train() { local marker=$1 log=$2 script=$3; shift 3; [ -f "$marker" ] && return 0; local u=""
  for attempt in $(seq 1 12); do env "$@" OLMOE3_FOLLOW=0 bash $script launch > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 60; done
  say "$(basename $marker): ${u:-LAUNCH FAILED}"; [ -n "$u" ] && echo "$u" > "$marker"; }
have() { [ -f "$1/meta.json" ] || [ -f "$1/rank0/DONE" ]; }
heldout() { local hr=$1 tag=$2 ckpt=$3 sample=$4; have $S/olmoe3_routing/$hr/$tag/none || launch "$SQN-$hr-$tag" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$ckpt" --instances "$W/olmoe3_routing/$sample" --out-dir "$W/olmoe3_routing/$hr/$tag/none/rank0" --restrict none --batch-size $BSF --log-every 100; }
ppl() { local ckpt=$1 json=$2 name=$3; [ -f $SQ/ppl_validation/$json ] || launch "$SQN-ppl-$name" python scripts/debug_validation/eval_ppl_validation.py --checkpoints "$ckpt" --out-dir "$W/$SQN/ppl_validation" --batch-size $PPLB; }
evals() { local lr=$1 run=olmoe3_275m_${ARM}_emo_lr${lr}_5b; heldout $HRV lr${lr}_step10000 $W/$run/step10000 $VSAMPLE; heldout $HR lr${lr}_step10000 $W/$run/step10000 $SAMPLE; ppl $W/$run/step10000 $run/step10000.json lr${lr}; }
for lr in $LRS; do launch_train $LOG/lr${lr}_launched $LOG/launch_lr$lr.log $SCRIPT OLMOE3_LR=$lr; done
pgrep -f "^bash scripts/sparse_experts/olmoe3_squares/baseline_keeper.sh" > /dev/null || { ( setsid nohup bash scripts/sparse_experts/olmoe3_squares/baseline_keeper.sh >> $S/olmoe3_routing/baseline_keeper/log.txt 2>&1 < /dev/null & ); say "baseline keeper started"; }
# evaluate every arm as soon as ITS checkpoint lands (not in list order: 2026-10-27 both drivers sat behind one relaunched arm while 5 finished
# checkpoints waited); evals() is idempotent, so a restarted driver only submits what is missing
pending="$LRS"; while [ -n "$pending" ]; do rest=""; for lr in $pending; do if [ -f $S/olmoe3_275m_${ARM}_emo_lr${lr}_5b/step10000/train/rank0.pt ]; then say "lr $lr: step 10000 present"; evals $lr; else rest="$rest $lr"; fi; done; pending="${rest# }"; [ -n "$pending" ] && sleep 300; done
for lr in $LRS; do for hr in $HRV $HR; do until have $S/olmoe3_routing/$hr/lr${lr}_step10000/none; do sleep 300; done; done; until [ -f $SQ/ppl_validation/olmoe3_275m_${ARM}_emo_lr${lr}_5b/step10000.json ]; do sleep 300; done; done
for d in $S/olmoe3_routing/$HRV/*/none $S/olmoe3_routing/$HR/*/none; do [ -f $d/rank0/DONE ] && [ ! -f $d/meta.json ] && python scripts/sparse_experts/olmoe3_routing/merge_routing.py $d 2>&1 | tail -1; done
say "selection CE at 5.24B (lower is better):"; for lr in $LRS; do echo "  $lr $(python -c "import json; print(round(json.load(open('$S/olmoe3_routing/$HRV/lr${lr}_step10000/none/meta.json'))['mean_ce'], 4))")"; done
say "done"
