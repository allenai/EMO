#!/usr/bin/env bash
# Unallocated twins for the debug_validation 4-node runs while their allocated experiments queue (allocated 4-node jobs have queued
# for hours; unallocated ones start in minutes, see beaker-allocated-vs-preemptible memory). For each run in JOBS whose keeper URL
# is still queued and that has no twin yet: clone its spec (drop minRuntime/autoResume, set preemptible) and submit it as <run>-u1.
# Then every 5 min: whichever of the pair starts first wins, the other is stopped; a winning twin becomes the keeper's current URL
# (run_keeper.sh relaunches allocated if it later ends without the final checkpoint).
#   bash scripts/debug_validation/run_twins.sh   (detach it; log: debug_validation/run_twins.log)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
D=debug_validation; K=$D/run_keeper; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad
JOBS=("olmoe3_275m_emo_randsel_10b|randsel" "olmoe3_275m_emo_randsel64_10b|randsel64")
say() { echo "$(date -u +%m-%d\ %H:%M) $*"; }
state() { beaker experiment get "$1" --format=json 2>/dev/null | python -c "
import json,sys; d=json.load(sys.stdin)[0]; js=d.get('jobs') or []; s=(js[-1] if js else {}).get('status',{}); print([k for k in ('created','scheduled','started','exited','canceled','finalized') if s.get(k)][-1] if js else 'none')"; }
twin() { local a=$1 name=$2; python - "$a" "$name" "$SP" <<'PY'
import subprocess, sys, yaml
a, name, sp = sys.argv[1:]
spec = yaml.safe_load(subprocess.run(["beaker", "experiment", "spec", a], capture_output=True, text=True).stdout)
for t in spec["tasks"]:
    ctx = t.setdefault("context", {}); ctx.pop("minRuntime", None); ctx.pop("autoResume", None); ctx["preemptible"] = True
f = f"{sp}/spec_{name}.yaml"; yaml.safe_dump(spec, open(f, "w"))
r = subprocess.run(["beaker", "experiment", "create", "-w", "ai2/flex2", "-n", name, f], capture_output=True, text=True)
import re; m = re.search(r"beaker\.org/ex/([A-Z0-9]+)", r.stdout + r.stderr); print(f"beaker.org/ex/{m.group(1)}" if m else "")
PY
}
while true; do
  left=0
  for spec in "${JOBS[@]}"; do IFS='|' read -r run name <<< "$spec"
    [ -f $D/$run/step19074/train/rank0.pt ] && continue
    a=$(sed 's|.*/||' $K/$name.url); sa=$(state $a)
    if [ ! -f $K/$name.twin.url ]; then
      [ "$sa" = started ] && continue   # allocated already running: no twin needed
      [ "$sa" = exited ] || [ "$sa" = finalized ] || [ "$sa" = canceled ] && continue   # keeper's job
      u=$(twin $a $run-u1); [ -n "$u" ] && { echo "$u" > $K/$name.twin.url; say "$name: allocated $a $sa -> twin submitted $u"; } || say "$name: twin submission failed"
      left=1; continue
    fi
    t=$(sed 's|.*/||' $K/$name.twin.url); st=$(state $t); left=1
    if [ "$st" = started ] && [ "$sa" != started ]; then beaker experiment stop $a >/dev/null 2>&1; say "$name: twin $t started -> stopped allocated $a"; cp $K/$name.twin.url $K/$name.url; rm -f $K/$name.twin.url
    elif [ "$sa" = started ] && [ "$st" != started ]; then beaker experiment stop $t >/dev/null 2>&1; say "$name: allocated $a started -> stopped twin $t"; rm -f $K/$name.twin.url
    elif [ "$st" = exited ] || [ "$st" = finalized ] || [ "$st" = canceled ]; then say "$name: twin $t $st before starting -> dropped"; rm -f $K/$name.twin.url; fi
  done
  [ $left = 0 ] && { say "no twins needed"; exit 0; }
  sleep 300
done
