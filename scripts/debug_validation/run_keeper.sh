#!/usr/bin/env bash
# Keeps the debug_validation 4-node runs alive across the 8-hour multi-node runtime cut and node faults: every 10 min, for each
# run whose final checkpoint is missing, if its current Beaker experiment has ended, relaunch it (the trainer resumes from the
# latest complete checkpoint in the run folder). Generic copy of scripts/sparse_experts/olmoe3_squares/baseline_keeper.sh.
#   bash scripts/debug_validation/run_keeper.sh   (detach it; log: debug_validation/run_keeper.log)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
D=debug_validation; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad; KEEP=$D/run_keeper; mkdir -p $KEEP $SP
# run dir | final step | launch script | file holding the current experiment URL
# RETIRED (2026-11-10): the randsel / randsel64 entries. Their runs finished 2026-09-23 and their checkpoints were deleted 2026-10-13, so a keeper that
# still lists them sees "final checkpoint missing" and RELAUNCHES them from scratch (it did, 2026-11-10 07:50, two 4-node jobs; stopped). Same pitfall as
# baseline_keeper.sh's retired.txt guard: never list a run whose checkpoints were deleted; a run is retired once its final checkpoint is gone on purpose.
JOBS=("olmoe3_275m_emo_mix20_10b|19074|olmoe3_275m_emo_mix20_10b.sh|$KEEP/mix20.url")
RETIRED="olmoe3_275m_emo_randsel_10b olmoe3_275m_emo_randsel64_10b"
[ -f $KEEP/mix20.url ] || { echo "mix20: no experiment URL yet ($KEEP/mix20.url); launch the run first"; exit 1; }
say() { echo "$(date -u +%m-%d\ %H:%M) $*"; }
state() { beaker experiment get "$1" --format=json 2>/dev/null | python -c "
import json,sys; d=json.load(sys.stdin)[0]; js=d.get('jobs') or []; s=(js[-1] if js else {}).get('status',{}); print([k for k in ('created','scheduled','started','exited','canceled','finalized') if s.get(k)][-1] if js else 'none')"; }
while true; do
  alldone=1
  for spec in "${JOBS[@]}"; do IFS='|' read -r run final script urlf <<< "$spec"
    case " $RETIRED " in *" $run "*) say "$run is retired: never relaunched"; continue;; esac
    [ -f $D/$run/step$final/train/rank0.pt ] && continue; alldone=0
    id=$(sed 's|.*/||' $urlf); st=$(state $id)
    if [ "$st" = exited ] || [ "$st" = finalized ] || [ "$st" = canceled ]; then
      rm -rf $D/$run/*-tmp 2>/dev/null; u=""
      for attempt in 1 2 3 4 5 6; do OLMOE3_FOLLOW=0 bash scripts/debug_validation/model_scripts/$script launch > $SP/keeper_$run.log 2>&1; u=$(sed 's/\x1b\[[0-9;]*m//g' $SP/keeper_$run.log | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 60; done
      say "$run: experiment $id $st -> relaunched: ${u:-FAILED} (latest checkpoint $(ls $D/$run 2>/dev/null | grep -E '^step[0-9]+$' | sed 's/step//' | sort -n | tail -1))"; [ -n "$u" ] && echo "$u" > $urlf
    fi
  done
  [ $alldone = 1 ] && { say "all runs complete"; exit 0; }
  sleep 600
done
