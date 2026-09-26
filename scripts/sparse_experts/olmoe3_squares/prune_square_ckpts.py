#!/usr/bin/env python3
"""Delete the training checkpoints of the random-control SQUARES whose arm is finished (every merge and evaluation done); the merged
models, held-out summaries and ppl results stay. Two modes (user decision 2026-09-24):
  --mode all      delete every checkpoint of every finished square (~6.4 TB)
  --mode finals   keep each square's window-final checkpoint (the last fixed step of each window), delete the rest (~5 TB)
A square run dir is <prefix><g>[_w2|_w3]; its window-final step = the largest step<N> dir it holds. Arms still training are skipped.
  python scripts/sparse_experts/olmoe3_squares/prune_square_ckpts.py --mode finals --dry-run
"""
import re, shutil, sys
from pathlib import Path

S = Path("sparse_experts"); DRY = "--dry-run" in sys.argv
mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "finals"; assert mode in ("all", "finals")
# finished arms: run-dir prefix, K, windows trained
ARMS = [("olmoe3_275m_stdrand_square", 4, 3), ("olmoe3_275m_stdrand8_square", 8, 3), ("olmoe3_275m_emorand_square", 4, 3),
        ("olmoe3_275m_s128rand4_square", 4, 3), ("olmoe3_275m_s128rand8_square", 8, 3), ("olmoe3_275m_stdremerge_square", 4, 1),
        ("olmoe3_275m_randsel4_square", 4, 3), ("olmoe3_275m_randsel8_square", 8, 3), ("olmoe3_275m_randsel64k4_square", 4, 3), ("olmoe3_275m_randsel64k8_square", 8, 3), ("olmoe3_275m_emorand8_square", 8, 3), ("olmoe3_275m_emofrz_square", 4, 1)]
RUNNING = set()   # nothing in flight (random-pool arms finished window 3 and the frozen-router squares finished, 2026-09-26)

def size(d): return sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
freed = kept = 0
for prefix, k, windows in ARMS:
    if prefix in RUNNING: continue
    for g in range(k):
        for suf in ["", "_w2", "_w3"][:windows]:
            run = S / f"{prefix}{g}{suf}"
            if not run.exists(): continue
            steps = sorted(int(m.group(1)) for d in run.iterdir() if (m := re.fullmatch(r"step(\d+)", d.name)))
            if not steps: continue
            final = steps[-1]
            for st in steps:
                d = run / f"step{st}"; sz = size(d)
                if mode == "finals" and st == final: kept += sz; continue
                freed += sz
                if not DRY: shutil.rmtree(d)
            for d in run.glob("step*-tmp"):   # leftover partial saves
                freed += size(d)
                if not DRY: shutil.rmtree(d)
print(f"prune_square_ckpts --mode {mode}: {'would free' if DRY else 'freed'} {freed / 1e12:.2f} TB, keeping {kept / 1e12:.2f} TB of window-final checkpoints")
