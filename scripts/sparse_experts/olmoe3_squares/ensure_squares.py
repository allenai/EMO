#!/usr/bin/env python3
"""Make sure every square training run of the random-control arms is finished or in flight; (re)launch the ones whose submission
failed or whose job died without the final checkpoint (the trainer resumes from the run folder). Idempotent; run periodically.
  python scripts/sparse_experts/olmoe3_squares/ensure_squares.py [--dry-run]
"""
import json, math, re, subprocess, sys, time
from pathlib import Path

S = Path("sparse_experts"); W = "/weka/oe-training-default/ryanwang/EMO/sparse_experts"; DRY = "--dry-run" in sys.argv
ARMS = [dict(v="emo_k8", sqn="olmoe3_squares_emorand8", rp="olmoe3_275m_emorand8_square", k=8, E=512, emo=1, windows=1),
        dict(v="std_k8", sqn="olmoe3_squares_stdrand8", rp="olmoe3_275m_stdrand8_square", k=8, E=512, emo=0, windows=3),
        dict(v="s128_k4", sqn="olmoe3_squares_s128rand4", rp="olmoe3_275m_s128rand4_square", k=4, E=128, emo=0, windows=3),
        dict(v="s128_k8", sqn="olmoe3_squares_s128rand8", rp="olmoe3_275m_s128rand8_square", k=8, E=128, emo=0, windows=3),
        dict(v="randsel_k4", sqn="olmoe3_squares_randsel4", rp="olmoe3_275m_randsel4_square", k=4, E=512, emo=1, windows=2, psel="random"),
        dict(v="randsel_k8", sqn="olmoe3_squares_randsel8", rp="olmoe3_275m_randsel8_square", k=8, E=512, emo=1, windows=2, psel="random"),
        dict(v="randsel64_k4", sqn="olmoe3_squares_randsel64k4", rp="olmoe3_275m_randsel64k4_square", k=4, E=512, emo=1, windows=2, psel="random", minpool=64),
        dict(v="randsel64_k8", sqn="olmoe3_squares_randsel64k8", rp="olmoe3_275m_randsel64k8_square", k=8, E=512, emo=1, windows=2, psel="random", minpool=64)]


def sh(*cmd, env=None): return subprocess.run(cmd, capture_output=True, text=True, env=env)


def live():
    r = sh("beaker", "workspace", "experiments", "ai2/flex2", "--format=json")
    if r.returncode != 0: return set()
    out = set()
    for e in json.loads(r.stdout):
        js = e.get("jobs") or []; st = (js[-1] if js else {}).get("status", {})
        if not st.get("exited") and not st.get("finalized") and not st.get("canceled"): out.add(e.get("name", ""))
    return out


def in_flight(run, lv):
    """Beaker names are <run>-<suffix> (launcher) or <run>-u<n> (twins); a launched marker younger than 45 min also counts."""
    return any(nm == run or nm.startswith(run + "-") for nm in lv)


def final_present(run, final): return (S / run / f"step{final}" / "train" / "rank0.pt").exists()


def settled(path, minutes=45):
    """The driver launches right after it writes a prerequisite; only step in once the prerequisite is old and still unlaunched."""
    return path.exists() and time.time() - path.stat().st_mtime > minutes * 60


def launch(script, run, marker, env_extra):
    if DRY: print("  would launch", run); return
    import os
    env = dict(os.environ, OLMOE3_FOLLOW="0", **env_extra)
    for attempt in range(8):
        r = sh("bash", f"scripts/sparse_experts/model_scripts/{script}", "launch", env=env)
        m = re.search(r"beaker\.org/ex/([A-Z0-9]+)", re.sub(r"\x1b\[[0-9;]*m", "", r.stdout + r.stderr))
        if m: Path(marker).write_text(f"beaker.org/ex/{m.group(1)}\n"); print(f"  launched {run}: {m.group(1)}", flush=True); return
        time.sleep(60)
    print(f"  LAUNCH FAILED {run}", flush=True)


def main():
    lv = live(); n = 0
    for a in ARMS:
        SQ = S / a["sqn"]; LOG = SQ / "logs"; rp, k = a["rp"], a["k"]
        if not (SQ / "groups.json").exists() or not (SQ / "pack/stats.json").exists(): continue
        base = dict(SQUARE_GROUP="", SQUARES_NAME=a["sqn"], OLMOE3_EMO=str(a["emo"]), OLMOE3_NUM_EXPERTS=str(a["E"]), OLMOE3_EMO_POOL_SELECT=a.get("psel", "relevance"), OLMOE3_EMO_MIN_POOL=str(a.get("minpool", 16)))
        tok1 = json.load(open(SQ / "pack/stats.json"))["tokens_per_group"]
        for g in range(k):  # window 1 (the driver launches all k at once after slicing every init)
            run = f"{rp}{g}"; final = tok1[g] // 524288
            if final_present(run, final) or in_flight(run, lv) or not all(settled(SQ / f"init/group{h}/.metadata") for h in range(k)): continue
            n += 1; launch("olmoe3_275m_emo_square.sh", run, LOG / f"square{g}_launched", dict(base, SQUARE_GROUP=str(g), OLMOE3_RUNNAME=run, OLMOE3_WANDB_TAGS=f"{a['sqn']},square"))
        if a["windows"] < 2 or not (SQ / "pack2/stats.json").exists(): continue
        tok2 = json.load(open(SQ / "pack2/stats.json"))["tokens_per_group"]
        for g in range(k):  # window 2 (needs init2 = the rewritten window-1 final, made by the driver)
            run = f"{rp}{g}_w2"; final = tok2[g] // 524288
            if final_present(run, final) or in_flight(run, lv) or not settled(SQ / f"init2/group{g}/model_and_optim/.metadata"): continue
            n += 1; launch("olmoe3_275m_emo_square.sh", run, LOG / f"square{g}_w2_launched", dict(base, SQUARE_GROUP=str(g), OLMOE3_TOKENS=str(tok2[g]), OLMOE3_DATA_PATHS=f"{W}/{a['sqn']}/pack2/group{g}/*.npy",
                                                                                           OLMOE3_INIT_FROM=f"{W}/{a['sqn']}/init2/group{g}/model_and_optim", OLMOE3_RUNNAME=run, OLMOE3_WANDB_TAGS=f"{a['sqn']},square,w2"))
        if a["windows"] < 3: continue
        sq3 = math.ceil(190735 / k)
        for g in range(k):  # window 3 (needs the finetune-start dir made by the driver)
            run = f"{rp}{g}_w3"; start = 57221 + g * sq3; final = start + sq3
            if final_present(run, final) or in_flight(run, lv) or not settled(SQ / f"ft_start/w3_group{g}/step{start}/train/rank0.pt"): continue
            n += 1; launch("olmoe3_275m_stdrand_square_w3.sh", run, LOG / f"square{g}_w3_launched", dict(base, SQUARE_GROUP=str(g), RUN_PREFIX=rp, FT_START=f"{W}/{a['sqn']}/ft_start/w3_group{g}", FT_START_STEP=str(start), FT_STEPS=str(sq3)))
    print(f"{time.strftime('%m-%d %H:%M')} ensure_squares: {n} runs {'would be ' if DRY else ''}launched")


if __name__ == "__main__":
    import fcntl
    _LOCK = open(f"/tmp/olmoe3_ensure_squares.lock", "w")   # one instance at a time: two overlapping runs would launch the same jobs twice
    try: fcntl.flock(_LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError: print(f"ensure_squares: another instance is running, skipping"); sys.exit(0)
    main()
