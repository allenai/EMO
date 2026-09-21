#!/usr/bin/env bash
# Twin rule for the two 4-node baselines (EMO 130B, 128e 130B): an unallocated twin (launcher with OLMOE3_PREEMPTIBLE=filler) competes
# with the allocated job; whichever starts first is kept, the other stopped; the keeper (baseline_keeper.sh) relaunches whatever the
# url file points at if it ends before the final checkpoint.   bash scripts/sparse_experts/olmoe3_squares/baseline_twins.sh   (detach)
set -u; cd "$(git rev-parse --show-toplevel)"; K=sparse_experts/olmoe3_routing/baseline_keeper
state() { beaker experiment get "$1" --format=json 2>/dev/null | python -c "
import json,sys; d=json.load(sys.stdin)[0]; js=d.get('jobs') or []; s=(js[-1] if js else {}).get('status',{}); print([k for k in ('created','scheduled','started','exited','canceled','finalized') if s.get(k)][-1] if s else 'none')"; }
say() { echo "$(date -u +%m-%d\ %H:%M) $*"; }
while true; do
  for name in emo_130b 128e_130b; do
    [ -f $K/$name.twin.url ] || continue
    a=$(sed 's|.*/||' $K/$name.url); t=$(sed 's|.*/||' $K/$name.twin.url); sa=$(state $a); st=$(state $t)
    if [ "$st" = started ] && [ "$sa" != started ]; then beaker experiment stop $a >/dev/null 2>&1; say "$name: unallocated twin $t started -> stopped allocated $a"; cp $K/$name.twin.url $K/$name.url; rm -f $K/$name.twin.url
    elif [ "$sa" = started ] && [ "$st" != started ]; then beaker experiment stop $t >/dev/null 2>&1; say "$name: allocated $a started -> stopped twin $t"; rm -f $K/$name.twin.url; fi
  done
  ls $K/*.twin.url >/dev/null 2>&1 || { say "no twins left"; exit 0; }
  sleep 300
done
