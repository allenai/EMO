#!/usr/bin/env python3
"""Make sure every expected held-out pass / v3-small evaluation of the random-control arms exists or is in flight; launch the
missing ones (covers launches the drivers dropped on gantry's flaky git check and jobs that died). Idempotent; run periodically.
  python scripts/sparse_experts/olmoe3_squares/ensure_passes.py [--dry-run]
"""
import json, math, re, subprocess, sys, time
from pathlib import Path

S = Path("sparse_experts"); W = "/weka/oe-training-default/ryanwang/EMO/sparse_experts"; SP = Path("/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad")
W1 = [20000, 25000, 30000, 35000, 38148]; W2 = [39073, 44073, 49073, 54073, 57221]; W3 = [66481, 116479, 166478, 216477, 247956]; FR = (926, 5926, 10926, 15926)
DRY = "--dry-run" in sys.argv; SP.mkdir(parents=True, exist_ok=True)
ARMS = [  # sqn, run prefix, K, held-out dir, baseline held-out dir, start tag, windows, baseline run per window (None = evaluated elsewhere)
    dict(sqn="olmoe3_squares_emorand", rp="olmoe3_275m_emorand_square", k=4, hr="runs_heldout300b_emorand", hrb="runs_heldout300b_emo", windows=3, base={3: "olmoe3_275m_emo_130b"}),
    dict(sqn="olmoe3_squares_emorand8", rp="olmoe3_275m_emorand8_square", k=8, hr="runs_heldout300b_emorand8", hrb="runs_heldout300b_emo", windows=3, base={}),
    dict(sqn="olmoe3_squares_stdrand8", rp="olmoe3_275m_stdrand8_square", k=8, hr="runs_heldout300b_stdrand8", hrb="runs_heldout300b_std", windows=3, base={}),
    dict(sqn="olmoe3_squares_s128rand4", rp="olmoe3_275m_s128rand4_square", k=4, hr="runs_heldout300b_s128rand4", hrb="runs_heldout300b_s128", windows=3, base={1: "olmoe3_275m_128e_130b", 2: "olmoe3_275m_128e_130b", 3: "olmoe3_275m_128e_130b"}),
    dict(sqn="olmoe3_squares_s128rand8", rp="olmoe3_275m_s128rand8_square", k=8, hr="runs_heldout300b_s128rand8", hrb="runs_heldout300b_s128", windows=3, base={}),
    dict(sqn="olmoe3_squares_randsel4", rp="olmoe3_275m_randsel4_square", k=4, hr="runs_heldout300b_randsel4", hrb="runs_heldout300b_randsel", windows=2, base={1: "olmoe3_275m_randsel_30b", 2: "olmoe3_275m_randsel_30b"}),
    dict(sqn="olmoe3_squares_randsel8", rp="olmoe3_275m_randsel8_square", k=8, hr="runs_heldout300b_randsel8", hrb="runs_heldout300b_randsel", windows=2, base={}),
    dict(sqn="olmoe3_squares_randsel64k4", rp="olmoe3_275m_randsel64k4_square", k=4, hr="runs_heldout300b_randsel64k4", hrb="runs_heldout300b_randsel64", windows=2, base={1: "olmoe3_275m_randsel64_30b", 2: "olmoe3_275m_randsel64_30b"}),
    dict(sqn="olmoe3_squares_randsel64k8", rp="olmoe3_275m_randsel64k8_square", k=8, hr="runs_heldout300b_randsel64k8", hrb="runs_heldout300b_randsel64", windows=2, base={}),
]


def sh(*cmd): return subprocess.run(cmd, capture_output=True, text=True)


def live_names():
    r = sh("beaker", "workspace", "experiments", "ai2/flex2", "--format=json")
    if r.returncode != 0: return set()
    out = set()
    for e in json.loads(r.stdout):
        js = e.get("jobs") or []; st = (js[-1] if js else {}).get("status", {})
        if not st.get("exited") and not st.get("finalized") and not st.get("canceled"): out.add(e.get("name", ""))
    return out


def have(d): return (d / "meta.json").exists() or (d / "rank0" / "DONE").exists()


def launch(name, *args):
    if DRY: print("  would launch", name); return
    for attempt in range(6):
        r = sh("python", "scripts/sparse_experts/olmoe3_beaker_cmd.py", "--name", name, "--gpus", "1", "--allocated", "--", *args)
        m = re.search(r"beaker\.org/ex/([A-Z0-9]+)", re.sub(r"\x1b\[[0-9;]*m", "", r.stdout + r.stderr))
        if m: (SP / f"launch_{name}.log").write_text(r.stdout + r.stderr); print(f"  launched {name}: {m.group(1)}", flush=True); return
        time.sleep(45)
    print(f"  LAUNCH FAILED {name}", flush=True)


def steps_w(sqn, k, pack):
    st = json.load(open(S / sqn / pack / "stats.json"))["tokens_per_group"]
    return [[max(1, round((t // 524288) * f / 19074)) for f in FR] + [t // 524288] for t in st[:k]]


def main():
    live = live_names(); n = 0
    for a in ARMS:
        sqn, rp, k, hr = a["sqn"], a["rp"], a["k"], a["hr"]; SQ = S / sqn
        if not (SQ / "groups.json").exists(): continue
        plan = []  # (tag, held-out dir, checkpoint path, ppl json name or None)
        wins = []
        if a["windows"] >= 1 and (SQ / "pack/stats.json").exists(): wins.append((1, W1, steps_w(sqn, k, "pack"), lambda g, i: f"{rp}{g}"))
        if a["windows"] >= 2 and (SQ / "pack2/stats.json").exists(): wins.append((2, W2, steps_w(sqn, k, "pack2"), lambda g, i: f"{rp}{g}_w2"))
        if a["windows"] >= 3:
            sq3 = math.ceil(190735 / k); wins.append((3, W3, [[57221 + g * sq3 + max(1, round(sq3 * f / 19074)) for f in FR] + [57221 + g * sq3 + sq3] for g in range(k)], lambda g, i: f"{rp}{g}_w3"))
        for wn, pts, st, run_of in wins:
            for i, s in enumerate(pts):
                subs = [S / run_of(g, i) / f"step{st[g][i]}" for g in range(k)]
                if all((p / "train" / "rank0.pt").exists() for p in subs):
                    if (SQ / "merged" / f"match{s}" / "merge_info.json").exists(): plan.append((f"merged_match{s}", hr, f"{W}/{sqn}/merged/match{s}", f"merged/match{s}.json"))
                    for g in range(k): plan.append((f"sub{g}_match{s}", hr, f"{W}/{run_of(g, i)}/step{st[g][i]}", None))
                if wn in a["base"]:
                    b = a["base"][wn]
                    if (S / b / f"step{s}" / "train" / "rank0.pt").exists(): plan.append((f"baseline_step{s}", a["hrb"], f"{W}/{b}/step{s}", f"{b}/step{s}.json"))
        for tag, hd, ckpt, ppl in plan:
            d = S / "olmoe3_routing" / hd / tag / "none"
            if not have(d) and not any(tag in nm and (hd in nm or sqn in nm) for nm in live):
                n += 1; launch(f"{sqn}-ensure-eval-{tag}", "python", "scripts/sparse_experts/olmoe3_routing/extract_routing.py", "--checkpoint", ckpt, "--instances", f"{W}/olmoe3_routing/sample_8k_300b.npz",
                                "--out-dir", f"{W}/olmoe3_routing/{hd}/{tag}/none/rank0", "--restrict", "none", "--batch-size", "8", "--log-every", "100")
            if ppl and not (SQ / "ppl_validation" / ppl).exists() and not any(("ppl" in nm) and (ppl.split("/")[-1].replace(".json", "") in nm or tag in nm) and sqn in nm for nm in live):
                n += 1; launch(f"{sqn}-ensure-ppl-{tag}", "python", "scripts/debug_validation/eval_ppl_validation.py", "--checkpoints", ckpt, "--out-dir", f"{W}/{sqn}/ppl_validation", "--batch-size", "8")
    print(f"{time.strftime('%m-%d %H:%M')} ensure_passes: {n} evaluations {'would be ' if DRY else ''}launched")


if __name__ == "__main__":
    import fcntl
    _LOCK = open(f"/tmp/olmoe3_ensure_passes.lock", "w")   # one instance at a time: two overlapping runs would launch the same jobs twice
    try: fcntl.flock(_LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError: print(f"ensure_passes: another instance is running, skipping"); sys.exit(0)
    main()
