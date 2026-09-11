#!/usr/bin/env bash
# Merge every finished held-out routing pass under runs_heldout20b (rank0/DONE present, meta.json absent)
# and print the olmoe3_squares evaluation table (held-out CE / oracle CE / v3-small mean CE).
cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
for d in sparse_experts/olmoe3_routing/runs_heldout20b/*/*; do
  [ -f "$d/rank0/DONE" ] && [ ! -f "$d/meta.json" ] && python scripts/sparse_experts/olmoe3_routing/merge_routing.py "$d" 2>&1 | tail -1 | sed "s|^|$(basename $(dirname $d))/$(basename $d): |"
done
python - <<'PY'
import json, glob, os
H="sparse_experts/olmoe3_routing/runs_heldout20b"; P="sparse_experts/olmoe3_squares/ppl_validation"
def ce(d):
    f=f"{d}/meta.json"; return round(json.load(open(f))["mean_ce"],4) if os.path.exists(f) else None
def ppl(f):
    if not os.path.exists(f): return None
    d=json.load(open(f))["per_set"]; return round(sum(v["CE loss"] for v in d.values())/len(d),4)
rows=[("start19074", ce(f"{H}/emo_step19074/none"), ce(f"{H}/emo_step19074/oracle"), ppl("claude_outputs/debug_validation/ppl_validation/olmoe3_275m_emo_10b/step19074.json"))]
for s in (20000,25000,30000,35000,38148):
    bp=next((ppl(f"{P}/{r}/step{s}.json") for r in ("olmoe3_275m_emo_20b","olmoe3_275m_emo_20b_filler","olmoe3_275m_emo_20b_1node") if os.path.exists(f"{P}/{r}/step{s}.json")), None)
    rows.append((f"baseline{s}", ce(f"{H}/baseline_step{s}/none"), ce(f"{H}/baseline_step{s}/oracle"), bp))
    rows.append((f"merged{s}", ce(f"{H}/merged_match{s}/none"), ce(f"{H}/merged_match{s}/oracle"), ppl(f"{P}/merged/match{s}.json")))
print(f"{'model':14s} {'heldout':>8s} {'oracle':>8s} {'ppl':>8s}")
for r in rows: print(f"{r[0]:14s} " + " ".join(f"{(x if x is not None else '-'):>8}" for x in r[1:]))
PY
