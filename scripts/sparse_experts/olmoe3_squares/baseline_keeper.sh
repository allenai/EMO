#!/usr/bin/env bash
# Keeps the long multi-node baselines alive across the 8-hour multi-node runtime cut (observed 2026-09-21: the EMO 130B baseline
# was killed exactly 8 h after start) and node faults: every 10 min, for each baseline whose final checkpoint is missing, if its
# current Beaker experiment has ended, relaunch it (the trainer resumes from the latest complete checkpoint in the run folder).
#   bash scripts/sparse_experts/olmoe3_squares/baseline_keeper.sh   (detach it)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
S=sparse_experts; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad; KEEP=$S/olmoe3_routing/baseline_keeper; mkdir -p $KEEP $SP
# run dir | final step | launch script | file holding the current experiment URL
JOBS=("olmoe3_275m_emo_130b|247956|olmoe3_275m_emo_130b_baseline.sh|$KEEP/emo_130b.url"
      "olmoe3_275m_128e_130b|247956|olmoe3_275m_128e_130b_baseline.sh|$KEEP/128e_130b.url")
[ -f $KEEP/emo_130b.url ] || echo "beaker.org/ex/01M31MR55FQVJC7CTRVFBBZ0WW" > $KEEP/emo_130b.url
[ -f $KEEP/128e_130b.url ] || cp $S/olmoe3_squares_s128rand4/logs/baseline_launched $KEEP/128e_130b.url
say() { echo "$(date -u +%m-%d\ %H:%M) $*"; }
state() { beaker experiment get "$1" --format=json 2>/dev/null | python -c "
import json,sys; d=json.load(sys.stdin)[0]; js=d.get('jobs') or []; s=(js[-1] if js else {}).get('status',{}); print([k for k in ('created','scheduled','started','exited','canceled','finalized') if s.get(k)][-1] if s else 'none')"; }
while true; do
  alldone=1
  for spec in "${JOBS[@]}"; do IFS='|' read -r run final script urlf <<< "$spec"
    [ -f $S/$run/step$final/train/rank0.pt ] && continue; alldone=0
    id=$(sed 's|.*/||' $urlf); st=$(state $id)
    if [ "$st" = exited ] || [ "$st" = finalized ] || [ "$st" = canceled ]; then
      rm -rf $S/$run/*-tmp 2>/dev/null; u=""
      for attempt in 1 2 3; do OLMOE3_FOLLOW=0 bash scripts/sparse_experts/model_scripts/$script launch > $SP/keeper_$run.log 2>&1; u=$(sed 's/\x1b\[[0-9;]*m//g' $SP/keeper_$run.log | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 60; done
      say "$run: experiment $id $st -> relaunched: ${u:-FAILED} (latest checkpoint $(ls $S/$run 2>/dev/null | grep -E '^step[0-9]+$' | sed 's/step//' | sort -n | tail -1))"; [ -n "$u" ] && echo "$u" > $urlf
    fi
  done
  [ $alldone = 1 ] && { say "all baselines complete"; exit 0; }
  sleep 600
done
