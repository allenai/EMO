#!/usr/bin/env bash
# Half-size-expert EMO arms (1024 experts x hidden 272, top-16 or top-32; user request 2026-10-15; report Q5 box 5F): the eight-square control
# of squares_emo1000.sh applied to olmoe3_275m_1024e_h272_k<K>_emo_10b/step19074: 8 random sub-models of 128 experts in every layer 2-9 (layer 1
# whole; fresh random groups, seed 2) with EXACTLY the same random document packs as the 512e k = 8 controls (pack / pack2 of
# olmoe3_squares_stdrand8), each trained at LR 4e-4 (hidden 272, top-K, EP=2 sync_1d like the pretraining run: layer 1 keeps all 1024 experts,
# above the grouped-GEMM cap at EP=1) on its documents of the 10B -> 20B window, then of the 20B -> 30B window from its own window-1 final;
# merged at the matched steps 30000 (~16B), 38148 (20B) and 57221 (30B); merged model, every square alone, start model and the joint baseline
# continued 10B -> 30B (olmoe3_275m_1024e_h272_k<K>_emo_30b_baseline.sh, 4 nodes, keeper) evaluated on the 300B held-out sample and the ppl sets.
#   bash scripts/sparse_experts/olmoe3_squares/squares_h272.sh k16|k32      (idempotent; detach it; commit + push first)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
KV="${1:?k16|k32}"; case $KV in k16) TK=16;; k32) TK=32;; *) echo "k16|k32"; exit 1;; esac
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; WR=/weka/oe-training-default/ryanwang/EMO
FULL=olmoe3_275m_1024e_h272_${KV}_emo_10b; SAMPLE=sample_8k_300b.npz; K=8; LR=4e-4; E=1024; PACKSRC=olmoe3_squares_stdrand8
SQN=olmoe3_squares_h272${KV}rand8; RP=olmoe3_275m_h272${KV}rand8_square; HR=runs_heldout300b_h272${KV}rand8; HRB=runs_heldout300b_h272${KV}; START=emo1024h272${KV}_step19074
BASE=olmoe3_275m_1024e_h272_${KV}_emo_30b; BASE_SCRIPT=scripts/sparse_experts/model_scripts/olmoe3_275m_1024e_h272_${KV}_emo_30b_baseline.sh
SQENV="OLMOE3_EXPERT_HIDDEN=272 OLMOE3_TOP_K=$TK OLMOE3_EP_SIZE=2 OLMOE3_EP_PATH=sync_1d"   # the model shape + EP of the pretraining run, for every square
SQ=$S/$SQN; LOG=$SQ/logs; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad
mkdir -p $SQ $LOG $SP $S/olmoe3_routing/$HR $S/olmoe3_routing/$HRB $S/olmoe3_routing/baseline_keeper
say() { echo "$(date -u +%m-%d\ %H:%M) [h272 $KV] $*"; }
# names of the live (not finalized) experiments of the workspace, read once per driver start: a restart must not resubmit a pass that is
# already queued or running (it did, three times on 2026-10-14; concurrent writers of one out-dir)
LIVE=$(beaker workspace experiments ai2/flex2 --format=json 2>/dev/null | python -c "
import json,sys
try: es=json.load(sys.stdin)
except Exception: es=[]
for e in es:
    js=e.get('jobs') or []; st=(js[-1] if js else {}).get('status',{})
    if not st.get('exited') and not st.get('finalized') and not st.get('canceled'): print(e.get('name',''))")
launch() { local name=$1; shift; local log=$SP/launch_$name.log; local u=""
  if grep -q -x -E "$(printf '%s' "$name" | sed 's/[.[\*^$]/\\&/g')-[0-9a-f]+(-[0-9a-f]+)?" <<< "$LIVE"; then say "$name: already live, not resubmitted"; return 0; fi
  for attempt in 1 2 3 4 5 6; do
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$name" --gpus 1 --allocated -- "$@" > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$name: ${u:-LAUNCH FAILED}"; }
launch_train() { local marker=$1 log=$2 script=$3; shift 3; [ -f "$marker" ] && return 0; local u=""
  for attempt in $(seq 1 12); do env "$@" OLMOE3_FOLLOW=0 bash $script launch > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 60; done
  say "$(basename $marker): ${u:-LAUNCH FAILED}"; [ -n "$u" ] && echo "$u" > "$marker"; }
have() { [ -f "$1/meta.json" ] || [ -f "$1/rank0/DONE" ]; }
heldout() { local hr=$1 tag=$2 ckpt=$3 name=$4; have $S/olmoe3_routing/$hr/$tag/none || launch "$name" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$ckpt" --instances "$W/olmoe3_routing/$SAMPLE" --out-dir "$W/olmoe3_routing/$hr/$tag/none/rank0" --restrict none --batch-size 8 --log-every 100; }
ppl() { local ckpt=$1 json=$2 name=$3; [ -f $SQ/ppl_validation/$json ] || launch "$name" python scripts/debug_validation/eval_ppl_validation.py --checkpoints "$ckpt" --out-dir "$W/$SQN/ppl_validation" --batch-size 8; }
shares_of() { python -c "import json; print(','.join(f'{x:.4f}' for x in json.load(open('$1'))['token_share']))"; }
tokens_of() { python -c "import json; print(json.load(open('$1'))['tokens_per_group'][$2])"; }
mid() { python -c "s=$1; print(max(1, round(s*10926/19074)))"; }
merge_k() { local out=$1 weights=$2; shift 2; [ -f $out/merge_info.json ] || PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $SQ/groups.json --full $S/$FULL/step19074 --subs "$(IFS=,; echo "$*")" --weights "$weights" --out $out --overwrite 2>&1 | tail -1 | cut -c1-120; }
# every square alone at a matched point: held-out passes batched 4 per 1-GPU job (sub<g>_match<s>), ppl sets batched 8 per job
squares_eval() { local s=$1; shift; local runs=("$@"); local cmds="" ckpts="" half
  for half in 0 1; do cmds=""
    for g in $(seq $((half*4)) $((half*4+3))); do have $S/olmoe3_routing/$HR/sub${g}_match$s/none && continue
      cmds="$cmds python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint $W/${runs[$g]} --instances $W/olmoe3_routing/$SAMPLE --out-dir $W/olmoe3_routing/$HR/sub${g}_match$s/none/rank0 --restrict none --batch-size 8 --log-every 100;"; done
    [ -n "$cmds" ] && launch "$SQN-squares-heldout-$s-h$half" bash -c "$cmds"; done
  for g in $(seq 0 $((K-1))); do [ -f $SQ/ppl_validation/${runs[$g]}.json ] || ckpts="$ckpts $W/${runs[$g]}"; done
  [ -n "$ckpts" ] && launch "$SQN-squares-ppl-$s" python scripts/debug_validation/eval_ppl_validation.py --checkpoints $ckpts --out-dir "$W/$SQN/ppl_validation" --batch-size 8; return 0; }
GS=$(seq 0 $((K-1)))
# ---- 0. fresh random expert groups of 125 (layer 1 whole); the 512e k=8 controls' random document packs; start model sliced ----
[ -f $SQ/groups.json ] || python scripts/sparse_experts/olmoe3_squares/partition_random.py --out $SQ/groups.json --k $K --seed 2 --num-experts $E
for d in pack pack2; do [ -e $SQ/$d ] || ln -s ../$PACKSRC/$d $SQ/$d; done
# ---- 1. joint baseline 10B -> 30B (4 nodes, allocated; keeper resumes it across the 8 h cut) + start-model evaluations ----
launch_train $LOG/baseline_launched $LOG/launch_baseline.log $BASE_SCRIPT
pgrep -f "scripts/sparse_experts/olmoe3_squares/baseline_keeper.sh" > /dev/null || { ( setsid nohup bash scripts/sparse_experts/olmoe3_squares/baseline_keeper.sh >> $S/olmoe3_routing/baseline_keeper/log.txt 2>&1 < /dev/null & ); say "baseline keeper started"; }   # subshell: not a child of this script, or the bare `wait`s below would wait for it (stuck 3 h on 2026-10-14)
heldout $HRB $START $W/$FULL/step19074 "$SQN-eval-$START"
[ -f claude_outputs/debug_validation/ppl_validation/$FULL/step19074.json ] || launch "$SQN-ppl-start" python scripts/debug_validation/eval_ppl_validation.py --checkpoints $W/$FULL/step19074 --out-dir $WR/claude_outputs/debug_validation/ppl_validation --batch-size 8
slice1() { local g=$1; PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/slice_checkpoint.py --checkpoint $S/$FULL/step19074 --groups $SQ/groups.json --group $g --out $SQ/init/group$g --overwrite > $LOG/slice_group$g.log 2>&1 && say "slice group$g done" || say "slice group$g FAILED (see $LOG/slice_group$g.log)"; }
for half in 0 1; do pids=""; for g in $(seq $((half*4)) $((half*4+3))); do [ -f $SQ/init/group$g/.metadata ] || { slice1 $g & pids="$pids $!"; }; done; [ -z "$pids" ] || wait $pids; done   # 4 slices at a time (CPU, ~10 min each)
for g in $GS; do [ -f $SQ/init/group$g/.metadata ] || { say "ABORT: start slice $SQ/init/group$g missing"; exit 1; }; done
# ---- 2. window 1 (10B -> 20B): 8 squares at LR 4e-4, checkpoints at the mid point (matched 30000) and the final (38148) ----
SH1=$(shares_of $SQ/pack/stats.json); declare -a FIN MID
for g in $GS; do t=$(tokens_of $SQ/pack/stats.json $g); FIN[$g]=$((t / 524288)); MID[$g]=$(mid ${FIN[$g]})
  launch_train $LOG/square${g}_launched $LOG/launch_square$g.log scripts/sparse_experts/model_scripts/olmoe3_275m_emo_square.sh SQUARE_GROUP=$g SQUARES_NAME=$SQN OLMOE3_EMO=1 OLMOE3_NUM_EXPERTS=$E OLMOE3_LR=$LR OLMOE3_FIXED_STEPS="${MID[$g]},${FIN[$g]}" OLMOE3_RUNNAME=${RP}$g OLMOE3_WANDB_TAGS=$SQN,square,halfexperts $SQENV
done
say "window-1 steps: $(for g in $GS; do echo -n "g$g mid ${MID[$g]} final ${FIN[$g]} "; done); shares $SH1"
for point in 30000:MID 38148:FIN; do s=${point%%:*}; arr=${point##*:}; subs=(); runs=()
  for g in $GS; do eval "st=\${$arr[$g]}"; until [ -f $S/${RP}$g/step$st/train/rank0.pt ]; do sleep 300; done; subs+=("$S/${RP}$g/step$st"); runs+=("${RP}$g/step$st"); done; say "point $s: square checkpoints present"
  merge_k $SQ/merged/match$s "$SH1" "${subs[@]}"
  heldout $HR merged_match$s $W/$SQN/merged/match$s "$SQN-eval-merged_match$s"; ppl $W/$SQN/merged/match$s merged/match$s.json "$SQN-ppl-merged-$s"
  squares_eval $s "${runs[@]}"
done
# ---- 3. window 2 (20B -> 30B): finals rewritten as start checkpoints -> squares on pack2 -> merge at 57221 ----
SH2=$(shares_of $SQ/pack2/stats.json); subs=(); runs=()
rw1() { local g=$1; PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/rewrite_checkpoint.py --src $S/${RP}$g/step${FIN[$g]} --out $SQ/init2/group$g --overwrite > $LOG/rewrite_group$g.log 2>&1 && say "init2 group$g done" || say "init2 group$g FAILED"; }
for half in 0 1; do pids=""; for g in $(seq $((half*4)) $((half*4+3))); do [ -f $SQ/init2/group$g/model_and_optim/.metadata ] || { rw1 $g & pids="$pids $!"; }; done; [ -z "$pids" ] || wait $pids; done
for g in $GS; do [ -f $SQ/init2/group$g/model_and_optim/.metadata ] || { say "ABORT: init2 group$g missing"; exit 1; }
  t2=$(tokens_of $SQ/pack2/stats.json $g); fin2=$((t2 / 524288)); subs+=("$S/${RP}${g}_w2/step$fin2"); runs+=("${RP}${g}_w2/step$fin2")
  launch_train $LOG/square${g}_w2_launched $LOG/launch_square${g}_w2.log scripts/sparse_experts/model_scripts/olmoe3_275m_emo_square.sh SQUARE_GROUP=$g SQUARES_NAME=$SQN OLMOE3_EMO=1 OLMOE3_NUM_EXPERTS=$E OLMOE3_LR=$LR OLMOE3_TOKENS=$t2 OLMOE3_FIXED_STEPS="$fin2" OLMOE3_DATA_PATHS="$W/$SQN/pack2/group$g/*.npy" OLMOE3_INIT_FROM="$W/$SQN/init2/group$g/model_and_optim" OLMOE3_RUNNAME=${RP}${g}_w2 OLMOE3_WANDB_TAGS=$SQN,square,halfexperts,w2 $SQENV
done
# baseline points as they appear (the 4-node run is slower than the squares; evaluated here so the squares never wait for it)
basepts() { :; }   # baseline points are launched by ensure_passes.py (monitor, every 30 min; it checks the live Beaker jobs first): launching them here too duplicated the 30000 pass (2026-10-14)
basepts
for d in "${subs[@]}"; do until [ -f "$d/train/rank0.pt" ]; do sleep 300; basepts; done; done; say "window-2 finals present"
merge_k $SQ/merged/match57221 "$SH2" "${subs[@]}"
heldout $HR merged_match57221 $W/$SQN/merged/match57221 "$SQN-eval-merged_match57221"; ppl $W/$SQN/merged/match57221 merged/match57221.json "$SQN-ppl-merged-57221"
squares_eval 57221 "${runs[@]}"
# ---- 4. collect: wait for the baseline's three points and every pass; summarise finished passes ----
until [ -f $S/$BASE/step57221/train/rank0.pt ]; do sleep 600; basepts; done; basepts
expected="$HRB/$START"; for s in 30000 38148 57221; do expected="$expected $HR/merged_match$s $HRB/baseline_step$s"; for g in $GS; do expected="$expected $HR/sub${g}_match$s"; done; done
for tag in $expected; do until have $S/olmoe3_routing/$tag/none; do sleep 300; done; done
for d in $S/olmoe3_routing/$HR/*/none $S/olmoe3_routing/$HRB/*/none; do [ -f $d/rank0/DONE ] && [ ! -f $d/meta.json ] && python scripts/sparse_experts/olmoe3_routing/merge_routing.py $d 2>&1 | tail -1; done
say "done"
