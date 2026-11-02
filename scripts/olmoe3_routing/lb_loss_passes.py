#!/usr/bin/env python3
"""Fixed-granularity load-balancing loss of the Q5 joint baselines and merged four-/eight-square models (sub-model LR 4e-4) on the held-out
300B sample (box 5H, user request 2026-11-01): one 1-GPU Beaker job per checkpoint running extract_routing.py --lb-only, which scores the
model's free routing on the same 65.5M tokens in blocks of 16k / 131k / 524k tokens (= 1 / 8 / 32 ranks' worth of a training step) and whole-sample.
Differences between models are then real balance differences: same tokens, same block size, same formula, independent of how many GPUs trained them.
The 10B point of a merged model is the start model itself (untrained slices merge back), so only the joint checkpoint is scored there.
Outputs: $W/olmoe3_routing/runs_lbloss/<model>/<joint|merged4|merged8>_step<N>/lb_loss.json; marker <out>.launched holds the experiment URL.
  python scripts/olmoe3_routing/lb_loss_passes.py [--dry-run] [--smoke] [--relaunch]      (commit + push first: the workers clone HEAD)
"""
import argparse, json, re, subprocess, time
from pathlib import Path

W = "/weka/oe-training-default/ryanwang/EMO/sparse_experts"
S = Path("sparse_experts"); R = S / "olmoe3_routing" / "runs_lbloss"
SAMPLE = f"{W}/olmoe3_routing/sample_8k_300b.npz"
POINTS = [19074, 30000, 38148, 57221, 116479]
MODELS = {  # tag -> batch size (full model), joint checkpoints by matched step, merged dirs of the 4- and 8-square LR-4e-4 arms
    "emo512": dict(bs=8, m4="olmoe3_squares_emorand_lr4e-4", m8="olmoe3_squares_emorand8_lr4e-4",
                   joint={19074: "olmoe3_275m_emo_10b/step19074", 38148: "olmoe3_275m_emo_20b_1node/step38147", 57221: "olmoe3_275m_emo_30b_1node/step57221", 116479: "olmoe3_275m_emo_130b/step116479"}),
    "emo1000": dict(bs=4, m4="olmoe3_squares_emo1000rand4", m8="olmoe3_squares_emo1000rand8",
                    joint={19074: "olmoe3_275m_1000e_emo_10b/step19074", 30000: "olmoe3_275m_1000e_emo_30b/step30000", 38148: "olmoe3_275m_1000e_emo_30b/step38148", 57221: "olmoe3_275m_1000e_emo_30b/step57221", 116479: "olmoe3_275m_1000e_emo_61b/step116479"}),
    "emo2000": dict(bs=2, m4="olmoe3_squares_emo2000rand4", m8="olmoe3_squares_emo2000rand8",
                    joint={19074: "olmoe3_275m_2000e_emo_10b/step19074", 30000: "olmoe3_275m_2000e_emo_30b/step30000", 38148: "olmoe3_275m_2000e_emo_30b/step38148", 57221: "olmoe3_275m_2000e_emo_30b/step57221", 116479: "olmoe3_275m_2000e_emo_61b/step116479"})}


def plan():
    jobs = []
    for tag, m in MODELS.items():
        for st, ck in m["joint"].items():
            if (S / ck / "model_and_optim").exists() or (S / ck / "model").exists(): jobs.append((tag, f"joint_step{st}", ck, m["bs"]))
            else: print(f"MISSING joint checkpoint {ck}")
        for K in (4, 8):
            for st in POINTS[1:]:   # 10B = start model
                d = Path(m[f"m{K}"]) / "merged" / f"match{st}"
                if (S / d / "merge_info.json").exists(): jobs.append((tag, f"merged{K}_step{st}", str(d), m["bs"]))
    return jobs


def reap():
    """For every finished checkpoint, stop the copies still live on Beaker (the allocated original and/or the unallocated twin)."""
    import os; env = dict(os.environ, PYTHONPATH="external/OLMo-core/src" + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
    n = 0
    for f in sorted(R.glob("*/*/lb_loss.json")):
        for mark in (f.parent.parent / f"{f.parent.name}.launched", f.parent.parent / f"{f.parent.name}.launched_u"):
            if not mark.exists() or mark.with_suffix(mark.suffix + ".reaped").exists(): continue
            eid = mark.read_text().strip().split("/")[-1]
            r = subprocess.run(["beaker", "experiment", "get", eid, "--format", "json"], capture_output=True, text=True, env=env)
            try: st = json.loads(r.stdout)[0]["jobs"][-1]["status"]
            except Exception: continue
            if not (st.get("exited") or st.get("finalized") or st.get("canceled")):
                subprocess.run(["beaker", "experiment", "stop", eid], capture_output=True, text=True, env=env); print(f"stopped {eid} ({mark.name}: result exists)"); n += 1
            mark.with_suffix(mark.suffix + ".reaped").touch()
    print(f"{n} live copies stopped")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--smoke", action="store_true", help="one short job (512 instances) into runs_lbloss_smoke")
    ap.add_argument("--relaunch", action="store_true", help="ignore .launched markers (a job died)")
    ap.add_argument("--twin", action="store_true", help="launch UNALLOCATED twins (-u1) of the jobs without a result yet (2026-11-02: 34 allocated 1-GPU jobs sat >2 h in 'created'); marker <name>.launched_u")
    ap.add_argument("--reap", action="store_true", help="stop the still-live copy (allocated or twin) of every checkpoint that has its lb_loss.json"); a = ap.parse_args()
    if a.reap: return reap()
    jobs = plan(); print(f"{len(jobs)} checkpoints to score")
    if a.smoke: jobs = jobs[:1]
    n = 0
    for tag, name, ck, bs in jobs:
        root = R.parent / ("runs_lbloss_smoke" if a.smoke else "runs_lbloss"); out = root / tag / name; mark = out.parent / (f"{name}.launched_u" if a.twin else f"{name}.launched")
        if (out / "lb_loss.json").exists(): continue
        if mark.exists() and not a.relaunch: continue
        cmd = ["python", "scripts/sparse_experts/olmoe3_beaker_cmd.py", "--name", f"lbloss-{tag}-{name}" + ("-u1" if a.twin else ""), "--gpus", "1"] + ([] if a.twin else ["--allocated"]) + ["--",
               "python", "scripts/sparse_experts/olmoe3_routing/extract_routing.py", "--checkpoint", f"{W}/{ck}", "--instances", SAMPLE, "--out-dir", f"{W}/olmoe3_routing/{root.name}/{tag}/{name}",
               "--restrict", "none", "--batch-size", str(bs), "--log-every", "100", "--lb-only"] + (["--max-instances", "512"] if a.smoke else [])
        if a.dry_run: print("would launch:", " ".join(cmd[3:5]), "...", ck); continue
        import os; env = dict(os.environ, PYTHONPATH="external/OLMo-core/src" + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
        for attempt in range(4):
            r = subprocess.run(cmd, capture_output=True, text=True, env=env); txt = re.sub(r"\x1b\[[0-9;]*m", "", r.stdout + r.stderr); mm = re.search(r"beaker\.org/ex/[A-Z0-9]+", txt)
            if mm: out.parent.mkdir(parents=True, exist_ok=True); mark.write_text(mm.group(0) + "\n"); print(f"{time.strftime('%m-%d %H:%M')} launched {tag}/{name}: {mm.group(0)}", flush=True); n += 1; break
            print(f"launch attempt {attempt + 1} failed for {tag}/{name}: {txt[-300:]}", flush=True); time.sleep(45)
    print(f"{n} jobs launched")


if __name__ == "__main__":
    main()
