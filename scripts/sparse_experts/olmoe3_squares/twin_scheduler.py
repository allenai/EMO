#!/usr/bin/env python3
"""Twin scheduler for queued single-node training jobs (user request 2026-09-21): every allocated training experiment that is
still queued gets an UNALLOCATED (preemptible) clone of its Beaker spec; whichever of the two starts first wins and the other is
stopped; a job that ends without the run's final checkpoint (preemption, node fault) is resubmitted as both kinds again. The trainer
resumes from the latest checkpoint in the run folder, so a restart loses at most 500 steps.
Jobs come from the drivers' marker files (sparse_experts/olmoe3_squares_*/logs/*_launched, written after 2026-09-20); multi-node
jobs (baselines: baseline_keeper.sh) and evaluation jobs are ignored. Twins already in the workspace (named <run>-u<n> / -a<n>) are
adopted. State + log: sparse_experts/olmoe3_routing/twin_scheduler/.
  python scripts/sparse_experts/olmoe3_squares/twin_scheduler.py [--once] [--interval 120]
"""
import argparse, copy, glob, json, math, os, re, subprocess, time
from pathlib import Path
import yaml

S = Path("sparse_experts"); ST = S / "olmoe3_routing" / "twin_scheduler"; ST.mkdir(parents=True, exist_ok=True)
MAX_SUBMITS = 4; WORKSPACE = "ai2/flex2"; TERMINAL = ("exited", "finalized", "canceled", "failed"); CUTOFF = time.mktime(time.strptime("2026-09-20 00:00", "%Y-%m-%d %H:%M"))


def sh(*cmd): return subprocess.run(cmd, capture_output=True, text=True)


def log(msg):
    line = f"{time.strftime('%m-%d %H:%M', time.gmtime())} {msg}"; print(line, flush=True)
    with open(ST / "log.txt", "a") as f: f.write(line + "\n")


def status_of(d):
    js = d.get("jobs") or []
    if not js: return ("created", None, None)
    st = js[-1].get("status", {}); keys = [k for k in ("created", "scheduled", "started", "exited", "canceled", "finalized") if st.get(k)]
    state = "canceled" if st.get("canceled") else (keys[-1] if keys else "none")
    return (state, st.get("started"), st.get("exitCode"))


def exp_state(eid):
    r = sh("beaker", "experiment", "get", eid, "--format=json")
    return status_of(json.loads(r.stdout)[0]) if r.returncode == 0 else ("unknown", None, None)


def workspace_twins():
    """{name: (id, state)} of experiments named <run>-u<n>/-a<n> in the workspace."""
    r = sh("beaker", "workspace", "experiments", WORKSPACE, "--format=json")
    if r.returncode != 0: return {}
    out = {}
    for e in json.loads(r.stdout):
        n = e.get("name", "")
        if re.search(r"-(a|u)\d+$", n) and (n.startswith("olmoe3_275m_") or n.startswith("eval_") or n.startswith("ppl_")): out.setdefault(n, (e["id"], status_of(e)))
    return out


def get_spec(eid):
    r = sh("beaker", "experiment", "spec", eid); return yaml.safe_load(r.stdout) if r.returncode == 0 else None


WEKA = "/weka/oe-training-default/ryanwang/EMO/"


def eval_from_spec(spec):
    """1-GPU evaluation jobs (extract_routing / eval_ppl_validation launched by the drivers): run key + the output that marks completion."""
    t = spec["tasks"][0]; args = t.get("arguments") or []
    if t.get("resources", {}).get("gpuCount") != 1 or len(args) < 2: return None
    def local(p): return Path(p.replace(WEKA, "")) if p.startswith(WEKA) else Path(p)
    if args[1] == "scripts/sparse_experts/olmoe3_routing/extract_routing.py":
        out = local(args[args.index("--out-dir") + 1]); return dict(run="eval:" + str(out.parent.parent.parent.name + "/" + out.parent.parent.name), final=0, done_path=str(out / "DONE"))
    if args[1] == "scripts/debug_validation/eval_ppl_validation.py":
        ck = local(args[args.index("--checkpoints") + 1]); od = local(args[args.index("--out-dir") + 1])
        return dict(run="ppl:" + str(od.parent.name + "/" + ck.parent.name + "/" + ck.name), final=0, done_path=str(od / ck.parent.name / f"{ck.name}.json"))
    return None


def job_from_spec(spec):
    t = spec["tasks"][0]
    if t.get("resources", {}).get("gpuCount") == 1: return eval_from_spec(spec)
    if t.get("replicas", 1) != 1 or t.get("resources", {}).get("gpuCount") != 8: return None
    args = t.get("arguments") or []
    if len(args) < 4 or args[1] != "scripts/sparse_experts/olmoe3_275m.py" or args[2] != "train": return None
    env = {e["name"]: e.get("value") for e in t.get("envVars", []) if "value" in e}
    run = args[3]; fixed = env.get("OLMOE3_FIXED_STEPS"); tokens = env.get("OLMOE3_TOKENS")
    if "_square" not in run or env.get("OLMOE3_NUM_NODES", "1") != "1": return None  # squares only; the multi-node baselines belong to baseline_keeper.sh
    if fixed: final = int(fixed.split(",")[-1])
    elif tokens: final = int(int(float(tokens)) // 524288) if env.get("OLMOE3_DATA_PATHS") else math.ceil(int(float(tokens)) / 524288)
    else: return None
    return dict(run=run, final=final)


def done(job, alloc_state=None):
    """Final checkpoint present, the run already at its (possibly ceil-era) final, or the tracked job succeeded."""
    if job.get("done_path"): return Path(job["done_path"]).exists() or (alloc_state is not None and alloc_state[0] in ("finalized", "exited") and alloc_state[2] == 0)
    if (S / job["run"] / f"step{job['final']}" / "train" / "rank0.pt").exists(): return True
    steps = [int(m.group(1)) for d in (S / job["run"]).glob("step*") if (m := re.fullmatch(r"step(\d+)", d.name)) and (d / "train" / "rank0.pt").exists()]
    if steps and max(steps) >= job["final"] - 1: return True
    return alloc_state is not None and alloc_state[0] in ("finalized", "exited") and alloc_state[2] == 0


def submit(spec, name, allocated):
    sp = copy.deepcopy(spec)
    for t in sp["tasks"]:
        ctx = t.setdefault("context", {}); ctx.pop("autoResume", None)
        if allocated: ctx.pop("preemptible", None); ctx["minRuntime"] = "8h0m0s"   # the launcher's allocated form
        else: ctx.pop("minRuntime", None); ctx["preemptible"] = True                # unallocated: preemptible only
    sp.pop("description", None)
    f = ST / f"spec_{name}.yaml"; yaml.safe_dump(sp, open(f, "w"))
    r = sh("beaker", "experiment", "create", "-w", WORKSPACE, "-n", name, str(f))
    m = re.search(r"beaker\.org/ex/([A-Z0-9]+)", r.stdout + r.stderr) or re.search(r"\b(01[A-Z0-9]{24})\b", r.stdout + r.stderr)
    if not m: log(f"SUBMIT FAILED {name}: {(r.stdout + r.stderr)[-300:]}"); return None
    return m.group(1)


def stop(eid, why):
    r = sh("beaker", "experiment", "stop", eid); log(f"stop {eid} ({why}): {'ok' if r.returncode == 0 else r.stderr.strip()[:120]}")


SP = "/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad"


def discover(state, twins):
    markers = glob.glob("sparse_experts/olmoe3_squares_*/logs/*_launched") + [f for f in glob.glob(f"{SP}/launch_*.log") if os.path.getmtime(f) >= CUTOFF]
    for marker in markers:
        if os.path.getmtime(marker) < CUTOFF or any(v.get("marker") == marker for v in state.values()): continue
        txt = re.sub(r"\x1b\[[0-9;]*m", "", open(marker, errors="ignore").read()); m = re.search(r"beaker\.org/ex/([A-Z0-9]+)", txt)
        if not m: continue
        eid = m.group(1)
        spec = get_spec(eid); job = job_from_spec(spec) if spec else None
        if not job: continue
        st = exp_state(eid)
        if done(job, st): continue
        j = dict(job, marker=marker, spec=spec, alloc=None, unalloc=None, n=0)
        # the marker's experiment may itself be an adopted unallocated twin
        kind = "unalloc" if spec["tasks"][0].get("context", {}).get("preemptible") else "alloc"; j[kind] = eid
        for name, (tid, tst) in twins.items():  # adopt live twins by name
            m = re.fullmatch(re.escape(re.sub(r"[^A-Za-z0-9_.-]", "_", job["run"])[:100]) + r"-(a|u)(\d+)", name)
            if not m or tid == eid: continue
            j["n"] = max(j["n"], int(m.group(2)))   # every twin name ever used counts (names must be unique; a relaunched run must not reuse a dead twin's name)
            if tst[0] not in TERMINAL:
                k2 = "alloc" if m.group(1) == "a" else "unalloc"
                if not j.get(k2): j[k2] = tid; log(f"{job['run']}: adopted {k2} twin {name} {tid} [{tst[0]}]")
        j["safe"] = re.sub(r"[^A-Za-z0-9_.-]", "_", job["run"])[:100]
        state[job["run"]] = j; log(f"tracking {job['run']} {kind} {eid} [{st[0]}]")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--once", action="store_true"); ap.add_argument("--interval", type=int, default=120); a = ap.parse_args()
    sf = ST / "state.json"
    try: state = json.load(open(sf)) if sf.exists() else {}
    except Exception as e: log(f"state.json unreadable ({e}); starting from the markers and adopting existing twins"); state = {}
    while True:
        try:
            twins = workspace_twins(); discover(state, twins)
            for run, j in list(state.items()):
                sa = exp_state(j["alloc"]) if j.get("alloc") else ("none", None, None)
                su = exp_state(j["unalloc"]) if j.get("unalloc") else ("none", None, None)
                if done(j, sa if j.get("alloc") else su):
                    for kind, st in (("alloc", sa), ("unalloc", su)):
                        if j.get(kind) and st[0] not in TERMINAL and st[0] != "none": stop(j[kind], f"{run} finished")
                    log(f"{run}: done"); state.pop(run); continue
                ra, ru = sa[0] == "started", su[0] == "started"
                if ra and ru:  # both started: keep the earlier one
                    if (sa[1] or "") <= (su[1] or ""): stop(j["unalloc"], f"{run}: allocated started first"); j["unalloc"] = None
                    else: stop(j["alloc"], f"{run}: unallocated started first"); j["alloc"] = None
                    continue
                if ra and j.get("unalloc") and su[0] not in TERMINAL: stop(j["unalloc"], f"{run}: allocated running"); j["unalloc"] = None; continue
                if ru and j.get("alloc") and sa[0] not in TERMINAL: stop(j["alloc"], f"{run}: unallocated running"); j["alloc"] = None; (open(j["marker"], "w").write(f"beaker.org/ex/{j['unalloc']}\n") if j["marker"].endswith("_launched") else None); continue
                if ra or ru: continue
                if j.get("submits", 0) >= MAX_SUBMITS:  # a job that keeps failing must not be resubmitted forever (counts this scheduler's own submissions, not adopted names)
                    if not j.get("gave_up"): log(f"{run}: {j['n']} submissions, giving up (fix the job and delete it from state.json to retry)"); j["gave_up"] = True
                    continue
                # nothing running: (re)submit what is missing or ended (preemption / fault); both kinds compete again
                if not j.get("alloc") or sa[0] in TERMINAL:
                    j["n"] += 1; j["submits"] = j.get("submits", 0) + 1; new = submit(j["spec"], f"{j.get('safe', run)}-a{j['n']}", allocated=True)
                    if new: log(f"{run}: allocated {j.get('alloc')} [{sa[0]}] -> resubmitted {new}"); j["alloc"] = new; (open(j["marker"], "w").write(f"beaker.org/ex/{new}\n") if j["marker"].endswith("_launched") else None)
                if not j.get("unalloc") or su[0] in TERMINAL:
                    j["n"] += 1; j["submits"] = j.get("submits", 0) + 1; new = submit(j["spec"], f"{j.get('safe', run)}-u{j['n']}", allocated=False)
                    if new: log(f"{run}: unallocated {j.get('unalloc')} [{su[0]}] -> submitted {new}"); j["unalloc"] = new
            tmp = sf.with_suffix(".tmp"); json.dump(state, open(tmp, "w")); os.replace(tmp, sf)  # atomic: a crash mid-write must not corrupt the state
        except Exception as e:
            log(f"ERROR {type(e).__name__}: {e}")
        if a.once: break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
