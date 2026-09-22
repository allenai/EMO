#!/usr/bin/env python3
"""Checkpoint cleanup approved by the user on 2026-09-22 (rows M1, M2, M3, V1, V2, X1, X2, F1 of the proposal), outside the squares work.
  M1  meta_learning: the three stopped meta arms keep only step4768 (+ the one HF export)
  M2  meta_learning/meta128_vanilla: keep step4768 (+HF) and step9537
  M3  meta_learning/cluster and meta_learning/data entirely
  V1  models_v2 trunk: early 50B saves except step11921; the 200-800B intermediates
  V2  models_v2 trunk: anneals/ and the extend1t resume points step198500 / step199000
  X1  modular_extension/k32_cpt_runs: fresh / carry / carry_shuf (anchors + evals kept)
  X2  modular_extension/emo64_100b130b_baseline: ephemerals step30000 / step30500
  F1  models_fullextend: ephemerals step11000 / step11500 of both runs
Kept: meta arms' step4768 (+hf), vanilla 4768/9537, models_v2 step23842 (+hf) and step11921, k32_cpt anchors/evals, baseline step30995,
fullextend step11921 (+hf). A README_DELETED_2026-09-22.md tombstone is written in every touched directory.
  python scripts/sparse_experts/cleanup_2026-09-22_other_experiments.py [--delete]
"""
import subprocess, sys, time
from pathlib import Path

DELETE = "--delete" in sys.argv
ROWS = {
    "M1": [f"meta_learning/{r}/{s}" for r in ("meta128_sametok_ws_lam05", "meta128_sametok_ws_lam05_adam", "meta128_sametok_ws_lam05_adam_scale1") for s in ("step2384", "step4000", "step4500", "step4769")],
    "M2": [f"meta_learning/meta128_vanilla/{s}" for s in ("step2384", "step7153", "step9000", "step9500")],
    "M3": ["meta_learning/cluster", "meta_learning/data"],
    "V1": [f"models_v2/emo_64exp_50b_wsd_lr2e-3/{s}" for s in ("step1192", "step2384", "step3576", "step4768", "step5960", "step7153", "step8345", "step9537", "step10729",
                                                             "step47684", "step71526", "step95368", "step119210", "step143052", "step166894", "step190736")],
    "V2": [f"models_v2/emo_64exp_50b_wsd_lr2e-3/{s}" for s in ("anneals", "step198500", "step199000")],
    "X1": [f"modular_extension/k32_cpt_runs/{s}" for s in ("fresh", "carry", "carry_shuf")],
    "X2": [f"modular_extension/emo64_100b130b_baseline/{s}" for s in ("step30000", "step30500")],
    "F1": [f"models_fullextend/{r}/{s}" for r in ("stdmoe_1b14b_50bof130b", "emo_1b14b_50bof130b") for s in ("step11000", "step11500")],
}
KEEP_NOTE = {"meta_learning": "kept: each meta arm's step4768 (+ HF export), meta128_vanilla step4768 (+HF) and step9537, wandb/",
             "models_v2": "kept: step23842 (100B anchor) + step23842-hf, step11921 (50B), merged_evals; the extend1t run can no longer be resumed",
             "modular_extension": "kept: k32_cpt_runs/anchors + evals, emo64_100b130b_baseline/step30995 (130B), cluster/, data/ (frozen partition)",
             "models_fullextend": "kept: step11921 (+ HF) of both runs, evals"}
targets = [(row, Path(p)) for row, ps in ROWS.items() for p in ps if Path(p).exists()]
missing = [(row, p) for row, ps in ROWS.items() for p in ps if not Path(p).exists()]
out = subprocess.run(["du", "-s", "-BG", *[str(p) for _, p in targets]], capture_output=True, text=True).stdout
sizes = {l.split("\t")[1]: int(l.split("\t")[0].rstrip("G")) for l in out.strip().split("\n") if l}
tot = {}
for row, p in targets: tot[row] = tot.get(row, 0) + sizes.get(str(p), 0); print(f"{row} {sizes.get(str(p), 0):5d}G  {p}")
for row, p in missing: print(f"{row} (absent) {p}")
print("TOTAL:", {r: f"{v/1024:.2f} TB" for r, v in tot.items()}, f"= {sum(tot.values())/1024:.2f} TB in {len(targets)} paths")
if DELETE:
    import shutil
    touched = {}
    for row, p in targets:
        shutil.rmtree(p); touched.setdefault(p.parent, []).append(f"{row}: {p.name} ({sizes.get(str(p), 0)} G)")
    for d, items in touched.items():
        root = str(d).split("/")[0]
        with open(d / "README_DELETED_2026-09-22.md", "a") as f:
            f.write(f"# Deleted {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} (user-approved cleanup; weka full)\n" + "\n".join(f"- {x}" for x in items) + f"\n\n{KEEP_NOTE.get(root, '')}\nLoss curves live in W&B (project emo-extension); see scripts/sparse_experts/cleanup_2026-09-22_other_experiments.py.\n")
    print("deleted")
