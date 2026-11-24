#!/usr/bin/env bash
# Launch the four runs of one whole-layer set (user request 2026-11-24): mixed-pool EMO 512e with a sync every 2B (run) and EMO 512e without
# re-merging (nosync), 8 and 12 sub-models each, sub-model LR 4e-4, 10B -> 30B (squares_sync12.sh). With a second set as argument 2, that set
# is launched once every run of the first has finished (16 worker nodes at a time).
#   bash scripts/sparse_experts/olmoe3_squares/launch_wholeset.sh 1,3,5,9 [1,5]      (detach; commit + push first)
set -u; cd "$(git rev-parse --show-toplevel)"; S=sparse_experts
launch() { local ws=$1 t=ws${1//,/} k m mode a
  for m in mix20 emo; do mode=run; pre=sync; [ $m = emo ] && { mode=nosync; pre=fix; }
    for k in 8 12; do a=$S/olmoe3_squares_${m}${pre}${k}${t}_lr4e-4; mkdir -p $a
      ( SYNC_MODEL=$m SYNC_K=$k SYNC_WHOLESET=$ws setsid nohup bash scripts/sparse_experts/olmoe3_squares/squares_sync12.sh $mode 4e-4 >> $a/logs_driver.log 2>&1 < /dev/null & ); done; done
  echo "$(date -u +%m-%d\ %H:%M) launched set $ws"; }
alldone() { local t=ws${1//,/} a; for a in $S/olmoe3_squares_mix20sync8${t}_lr4e-4 $S/olmoe3_squares_mix20sync12${t}_lr4e-4 $S/olmoe3_squares_emofix8${t}_lr4e-4 $S/olmoe3_squares_emofix12${t}_lr4e-4; do tail -n 1 $a/logs_driver.log 2>/dev/null | grep -q '\] done$' || return 1; done; }
alldone "$1" || launch "$1"
if [ -n "${2:-}" ]; then until alldone "$1"; do sleep 300; done; alldone "$2" || launch "$2"; fi
