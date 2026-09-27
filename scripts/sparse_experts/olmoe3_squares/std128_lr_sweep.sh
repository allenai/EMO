#!/usr/bin/env bash
# Pretraining-LR sweep of the standard-routing 128e model (user request 2026-09-27): five arms from scratch (1e-4, 2e-4, 4e-4, 1.6e-3,
# 3.2e-3; the existing 8e-4 run olmoe3_275m_128e_10b is the sixth) to step 10,000 = 5.24B tokens, one node each; every step-10000
# checkpoint evaluated on the SELECTION sample (sample_8k_300b_val.npz), the reporting sample (sample_8k_300b.npz) and the v3-small ppl
# sets. Mode final <lr>: pretrain that LR from scratch to 130B (4 nodes, allocated) and evaluate every matched checkpoint.
#   bash scripts/sparse_experts/olmoe3_squares/std128_lr_sweep.sh sweep          (idempotent; detach; commit + push first)
#   bash scripts/sparse_experts/olmoe3_squares/std128_lr_sweep.sh final <lr>
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
MODE="${1:?sweep|final}"; S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SAMPLE=sample_8k_300b.npz; VSAMPLE=sample_8k_300b_val.npz
SQN=olmoe3_squares_s128lr; SQ=$S/$SQN; HR=runs_heldout300b_s128lr; HRV=runs_heldout300b_s128lr_val; LOG=$SQ/logs; KEEP=$S/olmoe3_routing/baseline_keeper
SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad; mkdir -p $SQ $LOG $SP $S/olmoe3_routing/$HR $S/olmoe3_routing/$HRV $KEEP
LRS="1e-4 2e-4 4e-4 1.6e-3 3.2e-3"
say() { echo "$(date -u +%m-%d\ %H:%M) [s128lr $MODE] $*"; }
launch() { local name=$1; shift; local log=$SP/launch_$name.log; local u=""
  for attempt in 1 2 3 4 5 6; do
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$name" --gpus 1 --allocated -- "$@" > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$name: ${u:-LAUNCH FAILED}"; }
launch_train() { local marker=$1 log=$2 script=$3; shift 3; [ -f "$marker" ] && return 0; local u=""
  for attempt in $(seq 1 12); do env "$@" OLMOE3_FOLLOW=0 bash $script launch > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 60; done
  say "$(basename $marker): ${u:-LAUNCH FAILED}"; [ -n "$u" ] && echo "$u" > "$marker"; }
have() { [ -f "$1/meta.json" ] || [ -f "$1/rank0/DONE" ]; }
heldout() { local hr=$1 tag=$2 ckpt=$3 sample=$4; have $S/olmoe3_routing/$hr/$tag/none || launch "$SQN-$hr-$tag" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$ckpt" --instances "$W/olmoe3_routing/$sample" --out-dir "$W/olmoe3_routing/$hr/$tag/none/rank0" --restrict none --batch-size 8 --log-every 100; }
ppl() { local ckpt=$1 json=$2 name=$3; [ -f $SQ/ppl_validation/$json ] || launch "$SQN-ppl-$name" python scripts/debug_validation/eval_ppl_validation.py --checkpoints "$ckpt" --out-dir "$W/$SQN/ppl_validation" --batch-size 8; }
evals() { local tag=$1 run=$2 step=$3   # selection + reporting held-out passes and the ppl sets of one checkpoint
  heldout $HRV $tag $W/$run/step$step $VSAMPLE; heldout $HR $tag $W/$run/step$step $SAMPLE; ppl $W/$run/step$step $run/step$step.json $tag; }
if [ $MODE = sweep ]; then
  for lr in $LRS; do launch_train $LOG/lr${lr}_launched $LOG/launch_lr$lr.log scripts/sparse_experts/model_scripts/olmoe3_275m_128e_lrsweep_5b.sh OLMOE3_LR=$lr; done
  evals lr8e-4_step10000 olmoe3_275m_128e_10b 10000   # the existing 8e-4 run at the same point
  for lr in $LRS; do until [ -f $S/olmoe3_275m_128e_lr${lr}_5b/step10000/train/rank0.pt ]; do sleep 300; done; say "lr $lr: step 10000 present"; evals lr${lr}_step10000 olmoe3_275m_128e_lr${lr}_5b 10000; done
  for lr in 8e-4 $LRS; do for hr in $HRV $HR; do until have $S/olmoe3_routing/$hr/lr${lr}_step10000/none; do sleep 300; done; done; until [ -f $SQ/ppl_validation/olmoe3_275m_128e_$([ $lr = 8e-4 ] && echo 10b || echo lr${lr}_5b)/step10000.json ]; do sleep 300; done; done
  for d in $S/olmoe3_routing/$HRV/*/none $S/olmoe3_routing/$HR/*/none; do [ -f $d/rank0/DONE ] && [ ! -f $d/meta.json ] && python scripts/sparse_experts/olmoe3_routing/merge_routing.py $d 2>&1 | tail -1; done
  say "selection CE at 5.24B (lower is better):"; for lr in 1e-4 2e-4 4e-4 8e-4 1.6e-3 3.2e-3; do echo "  $lr $(python -c "import json; print(round(json.load(open('$S/olmoe3_routing/$HRV/lr${lr}_step10000/none/meta.json'))['mean_ce'], 4))")"; done
  say "done"
else
  LR="${2:?lr}"; RUN=olmoe3_275m_128e_lr${LR}_130b
  launch_train $KEEP/s128lr_130b.url $LOG/launch_final.log scripts/sparse_experts/model_scripts/olmoe3_275m_128e_lrsel_130b.sh OLMOE3_LR=$LR
  for s in 10000 19074 20000 25000 30000 35000 38148 39073 44073 49073 54073 57221 66481 116479 166478 216477 247956; do
    until [ -f $S/$RUN/step$s/train/rank0.pt ]; do sleep 600; done; say "step $s present"
    heldout $HR base_step$s $W/$RUN/step$s $SAMPLE; ppl $W/$RUN/step$s $RUN/step$s.json base-$s
  done
  for s in 10000 19074 20000 25000 30000 35000 38148 39073 44073 49073 54073 57221 66481 116479 166478 216477 247956; do until have $S/olmoe3_routing/$HR/base_step$s/none; do sleep 300; done; done
  for d in $S/olmoe3_routing/$HR/*/none; do [ -f $d/rank0/DONE ] && [ ! -f $d/meta.json ] && python scripts/sparse_experts/olmoe3_routing/merge_routing.py $d 2>&1 | tail -1; done
  say "done"
fi
