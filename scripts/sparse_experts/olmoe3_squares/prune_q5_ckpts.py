#!/usr/bin/env python3
"""Q5 checkpoint rule (user proposal 2026-09-27): of every Q5 model (random-partition controls, k = 4 and 8, all windows; their merged models;
their baselines; the sub-model LR-sweep arms), keep only the checkpoints at 10B, 20B, 30B, 61B, 87B, 113B and 130B (steps 19074, 38148
[38147 in the older 20B baselines], 57221, 116479, 166478, 216477, 247956); delete every other permanent checkpoint and every ephemeral
leftover. A checkpoint at a matched point is only listed for deletion if its evaluation exists (held-out pass meta.json; for merged models
also the ppl json), so nothing un-evaluated is lost. DRY RUN unless --delete. Running experiments (re-partition cycles, 128e sweep) are not touched.
  python scripts/sparse_experts/olmoe3_squares/prune_q5_ckpts.py [--delete]
"""
import json, os, re, shutil, sys
from pathlib import Path

S = Path("sparse_experts"); R = S / "olmoe3_routing"; DELETE = "--delete" in sys.argv
KEEP = {19074, 38147, 38148, 57221, 116479, 166478, 216477, 247956}
FR = (926, 5926, 10926, 15926)
ARMS = [  # squares dir, run prefix, K, held-out dir, windows, baseline held-out dir
    ("olmoe3_squares_stdrand", "olmoe3_275m_stdrand_square", 4, "runs_heldout300b_stdrand", 3), ("olmoe3_squares_stdrand8", "olmoe3_275m_stdrand8_square", 8, "runs_heldout300b_stdrand8", 3),
    ("olmoe3_squares_emorand", "olmoe3_275m_emorand_square", 4, "runs_heldout300b_emorand", 3), ("olmoe3_squares_emorand8", "olmoe3_275m_emorand8_square", 8, "runs_heldout300b_emorand8", 3),
    ("olmoe3_squares_s128rand4", "olmoe3_275m_s128rand4_square", 4, "runs_heldout300b_s128rand4", 3), ("olmoe3_squares_s128rand8", "olmoe3_275m_s128rand8_square", 8, "runs_heldout300b_s128rand8", 3),
    ("olmoe3_squares_randsel4", "olmoe3_275m_randsel4_square", 4, "runs_heldout300b_randsel4", 3), ("olmoe3_squares_randsel8", "olmoe3_275m_randsel8_square", 8, "runs_heldout300b_randsel8", 3),
    ("olmoe3_squares_randsel64k4", "olmoe3_275m_randsel64k4_square", 4, "runs_heldout300b_randsel64k4", 3), ("olmoe3_squares_randsel64k8", "olmoe3_275m_randsel64k8_square", 8, "runs_heldout300b_randsel64k8", 3),
]
LR_ARMS = [(f"olmoe3_squares_emorand{k8}_lr{lr}", f"olmoe3_275m_emorand{k8}_lr{lr}_square", 8 if k8 else 4, f"runs_heldout300b_emorand{k8}_lr{lr}") for k8 in ("", "8") for lr in ("1e-4", "2e-4", "4e-4", "1.6e-3", "3.2e-3")]
BASELINES = [  # run, held-out dir of its passes
    ("olmoe3_275m_130b", "runs_heldout300b_std"), ("olmoe3_275m_20b_1node", "runs_heldout300b_std"), ("olmoe3_275m_30b_1node", "runs_heldout300b_std"),
    ("olmoe3_275m_emo_130b", "runs_heldout300b_emo"), ("olmoe3_275m_emo_20b_1node", "runs_heldout300b_emo"), ("olmoe3_275m_emo_30b_1node", "runs_heldout300b_emo"), ("olmoe3_275m_emo_20b_step20000", "runs_heldout300b_emo"),
    ("olmoe3_275m_128e_130b", "runs_heldout300b_s128"), ("olmoe3_275m_randsel_30b", "runs_heldout300b_randsel"), ("olmoe3_275m_randsel_130b", "runs_heldout300b_randsel"),
    ("olmoe3_275m_randsel64_30b", "runs_heldout300b_randsel64"), ("olmoe3_275m_randsel64_130b", "runs_heldout300b_randsel64"), ("olmoe3_275m_stdremerge_square", None)]
STDREMERGE = ("olmoe3_squares_stdremerge", "olmoe3_275m_stdremerge_square", 4, "runs_heldout300b_stdremerge")


def size(d): return sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
def steps_in(d): return sorted((int(m.group(1)), p) for p in d.iterdir() if (m := re.fullmatch(r"step(\d+)(-tmp)?", p.name)))
def evaluated(hr, tag): return (R / hr / tag / "none" / "meta.json").exists()
def steps_of(stats, g): t = json.load(open(stats))["tokens_per_group"][g]; s = t // 524288; return [max(1, round(s * f / 19074)) for f in FR] + [s]
def w3_steps(k, g): sq3 = -(-190735 // k); b = 57221 + g * sq3; return [b + max(1, round(sq3 * f / 19074)) for f in FR] + [b + sq3]

todo, blocked, kept = [], [], 0   # (path, bytes, why)
def consider(path, keep, why, need=None):
    global kept
    if not path.exists(): return
    b = size(path)
    if keep: kept += b; return
    if need and not need(): blocked.append((str(path), b, why)); return
    todo.append((str(path), b, why))

W1 = [20000, 25000, 30000, 35000, 38148]; W2 = [39073, 44073, 49073, 54073, 57221]; W3 = [66481, 116479, 166478, 216477, 247956]
for sqn, rp, k, hr, windows in ARMS + [STDREMERGE[:4] + (0,)]:
    SQ = S / sqn
    if sqn != "olmoe3_squares_stdremerge":
        for g in range(k):
            for w, run, pts, st in ((1, S / f"{rp}{g}", W1, steps_of(SQ / "pack/stats.json", g)), (2, S / f"{rp}{g}_w2", W2, steps_of(SQ / "pack2/stats.json", g)), (3, S / f"{rp}{g}_w3", W3, w3_steps(k, g))):
                if not run.exists(): continue
                match = dict(zip(st, pts))
                for s_, p in steps_in(run):
                    m = match.get(s_); keep = m in KEEP
                    consider(p, keep, f"square w{w} {'matched ' + str(m) if m else 'ephemeral'}", (lambda m=m, g=g, hr=hr: evaluated(hr, f"sub{g}_match{m}")) if m else None)
    for name in ("merged", "merged_optim"):
        md = SQ / name
        if not md.exists(): continue
        for p in md.iterdir():
            m = re.fullmatch(r"match(\d+)", p.name)
            if not m: continue
            s_ = int(m.group(1)); consider(p, s_ in KEEP, f"{name} match{s_}", lambda s_=s_, hr=hr, SQ=SQ: evaluated(hr, f"merged_match{s_}") and (SQ / "ppl_validation/merged" / f"match{s_}.json").exists())
    for name in ("init", "init2", "init3", "ft_start", "init_ckpt"):
        consider(SQ / name, False, f"{name} slices (regenerable)")
# stdremerge squares: their own step layout (offsets 12500/25000/32870 from 116479 + g*32870 -> points 166478/216477/247956)
for g in range(4):
    run = S / f"olmoe3_275m_stdremerge_square{g}"
    if run.exists():
        b = 116479 + g * 32870; match = {b + 12500: 166478, b + 25000: 216477, b + 32870: 247956}
        for s_, p in steps_in(run):
            m = match.get(s_); consider(p, m in KEEP, f"stdremerge square {'matched ' + str(m) if m else 'ephemeral'}", (lambda m=m, g=g: evaluated("runs_heldout300b_stdremerge", f"sub{g}_match{m}")) if m else None)
for sqn, rp, k, hr in LR_ARMS:
    SQ = S / sqn
    for g in range(k):
        for run in (S / f"{rp}{g}", S / f"{rp}{g}_w2"):
            if not run.exists(): continue
            st = steps_in(run); final = max(s for s, _ in st if "-tmp" not in _.name) if st else None
            for s_, p in st: consider(p, s_ == final and "-tmp" not in p.name, "LR-sweep square " + ("final" if s_ == final else "mid-window / ephemeral"))
    for p in (SQ / "merged").iterdir() if (SQ / "merged").exists() else []:
        m = re.fullmatch(r"match(\d+)", p.name)
        if m: s_ = int(m.group(1)); consider(p, s_ in KEEP, f"LR-sweep merged match{s_}", lambda s_=s_, hr=hr, SQ=SQ: evaluated(hr, f"merged_match{s_}") and (SQ / "ppl_validation/merged" / f"match{s_}.json").exists())
    consider(SQ / "init2", False, "init2 slices (regenerable)")
for run, hr in BASELINES:
    d = S / run
    if not d.exists() or hr is None: continue
    for s_, p in steps_in(d):
        keep = s_ in KEEP and "-tmp" not in p.name
        consider(p, keep, f"baseline {'matched ' + str(s_) if s_ in W1 + W2 + W3 else 'ephemeral'}", (lambda s_=s_, hr=hr: evaluated(hr, f"baseline_step{s_}")) if s_ in W1 + W2 + W3 else None)
by = {}
for path, b, why in todo: by[why.split(" ")[0] + (" " + why.split(" ")[1] if why.startswith(("square", "LR-sweep", "stdremerge", "baseline")) else "")] = by.get(why.split(" ")[0] + (" " + why.split(" ")[1] if why.startswith(("square", "LR-sweep", "stdremerge", "baseline")) else ""), 0) + b
T = 1e12
print(f"would delete {sum(b for _, b, _ in todo) / T:.2f} TB in {len(todo)} dirs; keep {kept / T:.2f} TB; blocked (no evaluation yet) {sum(b for _, b, _ in blocked) / T:.2f} TB in {len(blocked)} dirs")
for k_, v in sorted(by.items(), key=lambda x: -x[1]): print(f"  {v / T:6.2f} TB  {k_}")
for path, b, why in blocked[:20]: print("  BLOCKED", path, why)
if DELETE:
    for path, b, why in todo: shutil.rmtree(path)
    print(f"deleted {sum(b for _, b, _ in todo) / T:.2f} TB")
