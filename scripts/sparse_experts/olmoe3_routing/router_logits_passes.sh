#!/usr/bin/env bash
# Router-weight / router-logit passes (router_logits.py) for the disjoint-training analysis (user request 2026-11-06): for EMO 512e, 1000e and
# 2000e, the start model, the joint baseline, the merged 4- and 8-square models at sub-model LR 4e-4 and every single square, at every matched
# point whose checkpoint still exists (15.7B = 30000, 20B = 38148, 30B = 57221, 61B = 116479). 64 held-out instances per pass, the same 4,096
# token positions for every checkpoint. Batched: several passes run in sequence inside one allocated 1-GPU job (bash -c); a pass with a
# meta.json is skipped, a job with a marker is not relaunched (pass `relaunch` to ignore markers). Idempotent.
#   bash scripts/sparse_experts/olmoe3_routing/router_logits_passes.sh [dry|relaunch]      (commit + push first: the workers clone HEAD)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
MODE="${1:-run}"
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SAMPLE=sample_8k_300b.npz
R=$S/olmoe3_routing/runs_routerlogits; LOGS=$R/logs; mkdir -p $LOGS
say() { echo "$(date -u +%m-%d\ %H:%M) [rlogits] $*"; }
launch() { local name=$1; shift; local mark=$LOGS/$name.launched log=$LOGS/launch_$name.log u=""
  [ -f $mark ] && [ $MODE != relaunch ] && return 0
  if [ $MODE = dry ]; then say "would launch $name: $(echo "$*" | grep -o 'router_logits.py' | wc -l) passes"; return 0; fi
  for attempt in 1 2 3 4 5 6; do
    if [ "${RLOGITS_LAUNCHER:-cli}" = cli ]; then python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "$name${RLOGITS_UNALLOCATED:+-u1}" --gpus 1 ${RLOGITS_UNALLOCATED:+--unallocated} -- "$@" > "$log" 2>&1   # no olmo_core import (cold hosts); RLOGITS_UNALLOCATED=1 = filler slots
    else PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$name" --gpus 1 --allocated -- "$@" > "$log" 2>&1; fi
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$name: ${u:-LAUNCH FAILED}"; [ -n "$u" ] && echo "$u" > $mark; }
step_of() { local d=$1 which=$2; ls $d 2>/dev/null | grep -E '^step[0-9]+$' | sed 's/step//' | sort -n | { [ $which = min ] && head -n 1 || tail -n 1; }; }
# pass <model> <tag> <checkpoint dir relative to sparse_experts> <batch size>: appends one command to CMDS if the pass is missing
CMDS=""; NP=0
pass() { local model=$1 tag=$2 ck=$3 bs=$4
  [ -f $S/$ck/config.json ] || { say "skip $model/$tag: no checkpoint at $ck"; return 0; }
  [ -f $R/$model/$tag/meta.json ] && return 0
  CMDS="$CMDS python scripts/sparse_experts/olmoe3_routing/router_logits.py --checkpoint $W/$ck --instances $W/olmoe3_routing/$SAMPLE --out-dir $W/olmoe3_routing/runs_routerlogits/$model/$tag --batch-size $bs;"; NP=$((NP+1)); }
flush() { local name=$1; [ -n "$CMDS" ] && launch "$name" bash -c "$CMDS"; CMDS=""; NP=0; }
# squares <model> <K> <run prefix> <batch size> <has mid checkpoint 0|1> <has window 3 0|1>
squares() { local model=$1 K=$2 rp=$3 bs=$4 mid=$5 w3=$6 g
  for g in $(seq 0 $((K-1))); do
    [ $mid = 1 ] && pass $model sq${K}_g${g}_30000 ${rp}$g/step$(step_of $S/${rp}$g min) $bs
    pass $model sq${K}_g${g}_38148 ${rp}$g/step$(step_of $S/${rp}$g max) $bs
    pass $model sq${K}_g${g}_57221 ${rp}${g}_w2/step$(step_of $S/${rp}${g}_w2 max) $bs
    [ $w3 = 1 ] && pass $model sq${K}_g${g}_116479 ${rp}${g}_w3/step$(step_of $S/${rp}${g}_w3 max) $bs
    [ $NP -ge 8 ] && flush "rlogits-$model-sq$K-g$g"
  done; flush "rlogits-$model-sq$K-rest"; }
full() { local model=$1 bs=$2 chunk=$3; shift 3; local i=0 spec   # spec = tag:checkpoint
  for spec in "$@"; do pass $model ${spec%%:*} ${spec#*:} $bs; [ $NP -ge $chunk ] && { flush "rlogits-$model-full-$i"; i=$((i+1)); }; done; flush "rlogits-$model-full-$i"; }

# --- EMO 512e (2.61B): no mid-window square checkpoints; the joint 20B checkpoint is gone
full emo512 8 4 start:olmoe3_275m_emo_10b/step19074 joint_57221:olmoe3_275m_emo_30b_1node/step57221 joint_116479:olmoe3_275m_emo_130b/step116479 \
  merged4_38148:olmoe3_squares_emorand_lr4e-4/merged/match38148 merged4_57221:olmoe3_squares_emorand_lr4e-4/merged/match57221 merged4_116479:olmoe3_squares_emorand_lr4e-4/merged/match116479 \
  merged8_38148:olmoe3_squares_emorand8_lr4e-4/merged/match38148 merged8_57221:olmoe3_squares_emorand8_lr4e-4/merged/match57221
squares emo512 4 olmoe3_275m_emorand_lr4e-4_square 8 0 1
squares emo512 8 olmoe3_275m_emorand8_lr4e-4_square 8 0 0
# --- EMO 1000e (4.90B): full-model passes at batch 4
full emo1000 4 4 start:olmoe3_275m_1000e_emo_10b/step19074 joint_30000:olmoe3_275m_1000e_emo_30b/step30000 joint_38148:olmoe3_275m_1000e_emo_30b/step38148 joint_57221:olmoe3_275m_1000e_emo_30b/step57221 joint_116479:olmoe3_275m_1000e_emo_61b/step116479 \
  merged4_30000:olmoe3_squares_emo1000rand4/merged/match30000 merged4_38148:olmoe3_squares_emo1000rand4/merged/match38148 merged4_57221:olmoe3_squares_emo1000rand4/merged/match57221 merged4_116479:olmoe3_squares_emo1000rand4/merged/match116479 \
  merged8_30000:olmoe3_squares_emo1000rand8/merged/match30000 merged8_38148:olmoe3_squares_emo1000rand8/merged/match38148 merged8_57221:olmoe3_squares_emo1000rand8/merged/match57221
squares emo1000 4 olmoe3_275m_emo1000rand4_square 8 1 1
squares emo1000 8 olmoe3_275m_emo1000rand8_square 8 1 0
# --- EMO 2000e (9.61B): batch 2 everywhere, including the squares
full emo2000 2 3 start:olmoe3_275m_2000e_emo_10b/step19074 joint_30000:olmoe3_275m_2000e_emo_30b/step30000 joint_38148:olmoe3_275m_2000e_emo_30b/step38148 joint_57221:olmoe3_275m_2000e_emo_30b/step57221 joint_116479:olmoe3_275m_2000e_emo_61b/step116479 \
  merged4_30000:olmoe3_squares_emo2000rand4/merged/match30000 merged4_38148:olmoe3_squares_emo2000rand4/merged/match38148 merged4_57221:olmoe3_squares_emo2000rand4/merged/match57221 merged4_116479:olmoe3_squares_emo2000rand4/merged/match116479 \
  merged8_30000:olmoe3_squares_emo2000rand8/merged/match30000 merged8_38148:olmoe3_squares_emo2000rand8/merged/match38148 merged8_57221:olmoe3_squares_emo2000rand8/merged/match57221
squares emo2000 4 olmoe3_275m_emo2000rand4_square 2 1 1
squares emo2000 8 olmoe3_275m_emo2000rand8_square 2 1 0
say "done ($MODE)"
