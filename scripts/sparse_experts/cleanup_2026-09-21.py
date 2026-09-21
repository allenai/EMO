#!/usr/bin/env python3
"""Checkpoint cleanup approved by the user on 2026-09-21 (rows A + B + C of the proposal):
  A  ephemeral resume checkpoints of FINISHED runs (10B runs' step19000, baselines' step38000/57000/247000/247500, finetunes'
     non-final 500-multiples, squares' non-final 500-multiples that no merge references)
  B  learnedd_sweep/ and smoke/
  C  intermediate checkpoints of the expert-count arms (sparse_8of1024/8of512: keep the final and step2384; 1000e/2000e: keep step19074)
Runs still in flight (EMO 130B baseline + window-3 squares, re-merge squares, 8-way and 128e arms, 128e baseline) are never touched.
  python scripts/sparse_experts/cleanup_2026-09-21.py [--delete]
"""
import glob, json, os, re, shutil, subprocess, sys
from pathlib import Path

S = Path("sparse_experts"); DELETE = "--delete" in sys.argv
IN_FLIGHT = ("olmoe3_275m_emo_130b", "olmoe3_275m_emorand_square1_w3", "olmoe3_275m_emorand_square2_w3", "olmoe3_275m_emorand_square3_w3",
             "olmoe3_275m_stdremerge_square", "olmoe3_275m_stdrand8_square", "olmoe3_275m_s128rand4_square", "olmoe3_275m_s128rand8_square", "olmoe3_275m_128e_130b")
def in_flight(p): return any(str(p).startswith(str(S / x)) for x in IN_FLIGHT)
def step_of(d):
    m = re.fullmatch(r"step(\d+)", d.name); return int(m.group(1)) if m else None
# checkpoints referenced by any merge (sub-model inputs) are protected
protected = set()
for f in glob.glob("sparse_experts/olmoe3_squares*/merged*/match*/merge_info.json"):
    try: protected |= {os.path.normpath(p) for p in json.load(open(f)).get("subs", [])}
    except Exception: pass
targets = []
# A1: 10B runs' step19000 (+ debug_validation beta arms)
for run in glob.glob("sparse_experts/olmoe3_275m_*_10b") + glob.glob("sparse_experts/olmoe3_275m_10b") + glob.glob("debug_validation/olmoe3_275m_*_10b"):
    d = Path(run) / "step19000"
    if d.is_dir(): targets.append(("A", d))
# A2: baselines' ephemeral leftovers (explicit; their fixed points are 500-multiples too)
for run, steps in {"olmoe3_275m_20b_1node": (38000,), "olmoe3_275m_emo_20b_1node": (38000,), "olmoe3_275m_pool64or512_20b_1node": (38000,), "olmoe3_275m_learnedd_20b_1node": (38000,),
                   "olmoe3_275m_30b_1node": (57000,), "olmoe3_275m_emo_30b_1node": (57000,), "olmoe3_275m_130b": (247000, 247500)}.items():
    for st in steps:
        d = S / run / f"step{st}"
        if d.is_dir(): targets.append(("A", d))
# A3: finetunes (any *_ft*, *rft*, *30_ft*): 500-multiples that are not the run's last step
for run in glob.glob("sparse_experts/olmoe3_275m_*_ft") + glob.glob("sparse_experts/olmoe3_275m_*_ft2") + glob.glob("sparse_experts/olmoe3_275m_*_rft*"):
    steps = sorted(s for d in Path(run).iterdir() if (s := step_of(d)) is not None)
    for st in steps:
        if st % 500 == 0 and st != max(steps): targets.append(("A", Path(run) / f"step{st}"))
# A4: squares (all variants/windows) that are finished: 500-multiples that are not the last step and not a merge input
for run in glob.glob("sparse_experts/olmoe3_275m_*square*"):
    if in_flight(run): continue
    steps = sorted(s for d in Path(run).iterdir() if (s := step_of(d)) is not None)
    if not steps or not (Path(run) / f"step{max(steps)}" / "train" / "rank0.pt").exists(): continue  # not finished
    for st in steps:
        d = Path(run) / f"step{st}"
        if st % 500 == 0 and st != max(steps) and os.path.normpath(d) not in protected: targets.append(("A", d))
# B
for d in ("sparse_experts/learnedd_sweep", "sparse_experts/smoke"):
    if Path(d).is_dir(): targets.append(("B", Path(d)))
# C
for run, keep in {"sparse_experts/sparse_8of1024_10b": {2385, 2384}, "sparse_experts/sparse_8of512_10b": {2385, 2384},
                  "sparse_experts/olmoe3_275m_2000e_10b": {19074}, "sparse_experts/olmoe3_275m_2000e_emo_10b": {19074},
                  "sparse_experts/olmoe3_275m_1000e_10b": {19074}, "sparse_experts/olmoe3_275m_1000e_emo_10b": {19074}}.items():
    for d in sorted(Path(run).iterdir()):
        st = step_of(d)
        if st is not None and st not in keep and (row := ("C", d)) not in targets: targets.append(row)
targets = [(r, d) for r, d in targets if not in_flight(d)]
seen = set(); targets = [t for t in targets if not (str(t[1]) in seen or seen.add(str(t[1])))]
sizes = {}
out = subprocess.run(["du", "-s", "-BG", *[str(d) for _, d in targets]], capture_output=True, text=True).stdout
for line in out.strip().split("\n"):
    if line: g, p = line.split("\t"); sizes[p] = int(g.rstrip("G"))
tot = {}
for r, d in targets: tot[r] = tot.get(r, 0) + sizes.get(str(d), 0)
for r, d in targets: print(f"{r} {sizes.get(str(d), 0):5d}G  {d}")
print("TOTAL:", {r: f"{v/1024:.2f} TB" for r, v in tot.items()}, f"= {sum(tot.values())/1024:.2f} TB in {len(targets)} paths")
if DELETE:
    for r, d in targets: shutil.rmtree(d, ignore_errors=False)
    print("deleted")
