#!/usr/bin/env bash
# Keeps the debug_validation 4-node runs alive across the 8-hour multi-node runtime cut and node faults: every 10 min, for each
# run whose final checkpoint is missing, if its current Beaker experiment has ended, relaunch it (the trainer resumes from the
# latest complete checkpoint in the run folder). Generic copy of scripts/sparse_experts/olmoe3_squares/baseline_keeper.sh.
#   bash scripts/debug_validation/run_keeper.sh   (detach it; log: debug_validation/run_keeper.log)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
D=debug_validation; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad; KEEP=$D/run_keeper; mkdir -p $KEEP $SP
# run dir | final step | launch script | file holding the current experiment URL
JOBS=("olmoe3_275m_emo_randsel_10b|19074|olmoe3_275m_emo_randsel_10b.sh|$KEEP/randsel.url"
      "olmoe3_275m_emo_randsel64_10b|19074|olmoe3_275m_emo_randsel64_10b.sh|$KEEP/randsel64.url")
[ -f $KEEP/randsel.url ] || echo "beaker.org/ex/01M35S4J87E8N4KTJ4B2ZV0Z1F" > $KEEP/randsel.url
[ -f $KEEP/randsel64.url ] || echo "beaker.org/ex/01M35S5E6RWQBJKBVTBM5GHWZM" > $KEEP/randsel64.url
say() { echo "$(date -u +%m-%d\ %H:%M) $*"; }
state() { beaker experiment get "$1" --format=json 2>/dev/null | python -c "
import json,sys; d=json.load(sys.stdin)[0]; js=d.get('jobs') or []; s=(js[-1] if js else {}).get('status',{}); print([k for k in ('created','scheduled','started','exited','canceled','finalized') if s.get(k)][-1] if js else 'none')"; }
while true; do
  alldone=1
  for spec in "${JOBS[@]}"; do IFS='|' read -r run final script urlf <<< "$spec"
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
