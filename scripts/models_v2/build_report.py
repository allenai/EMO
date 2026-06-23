#!/usr/bin/env python3
"""Build the models_v2 experiment report.

Pulls six training/eval curves from W&B for the models_v2 stdMoE runs and renders
a concise multi-chart HTML report to claude_outputs/models_v2/report.html.

Run:  python scripts/models_v2/build_report.py            # pull from W&B + render
      python scripts/models_v2/build_report.py --no-wandb # render from cached metrics.json

Registered in scripts/publish_reports.sh (deploys to https://emo-reports.pages.dev/).
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

ENTITY_PROJECT = "ryanyxw/emo-extension"
TOK_PER_STEP = 1024 * 4096  # global_batch_size(1024) * seq_len(4096) = 4,194,304 tokens/step

# (label, W&B run id, colour) — pinned to the healthy/current runs (crashed restarts excluded).
RUNS = [
    ("stdmoe_64exp_25b",  "lsq79eb5", "#d62728"),
    ("stdmoe_64exp_50b",  "r5kyiexy", "#1f77b4"),
    ("stdmoe_128exp_50b", "yuafg0dw", "#2ca02c"),
]

# (chart title, W&B metric key)
METRICS = [
    ("CE loss",                      "train/CE loss"),
    ("Grad norm",                    "optim/total grad norm"),
    ("Load balancing loss",          "train/load balancing loss"),
    ("Unique experts used / batch",  "train/unique experts used per batch"),
    ("HellaSwag (soft loss v2)",     "eval/downstream/hellaswag (soft loss v2)"),
    ("ARC-Challenge (soft loss v2)", "eval/downstream/arc_challenge (soft loss v2)"),
]

BASE = Path(__file__).resolve().parents[2] / "claude_outputs" / "models_v2"
CACHE = BASE / "metrics.json"


def fetch_from_wandb() -> dict:
    import wandb
    api = wandb.Api()
    keys = [k for _, k in METRICS]
    data = {k: {} for k in keys}
    meta = {}
    for label, rid, _ in RUNS:
        run = api.run(f"{ENTITY_PROJECT}/{rid}")
        meta[label] = {"state": run.state, "id": rid}
        for k in keys:
            pts = []
            for row in run.history(keys=[k], samples=600, pandas=False):
                v, s = row.get(k), row.get("_step")
                if v is None or s is None:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isnan(fv) or math.isinf(fv):
                    continue
                pts.append([round(s * TOK_PER_STEP / 1e9, 4), fv])  # x = tokens (B)
            data[k][label] = pts
        # latest CE for the summary table
        ce = data["train/CE loss"].get(label) or []
        meta[label]["last_ce"] = round(ce[-1][1], 3) if ce else None
        meta[label]["last_tok"] = ce[-1][0] if ce else None
    return {"data": data, "meta": meta}


def render(payload: dict) -> str:
    data, meta = payload["data"], payload["meta"]
    runs_js = json.dumps([{"label": l, "color": c} for l, _, c in RUNS])
    metrics_js = json.dumps([{"title": t, "key": k} for t, k in METRICS])
    data_js = json.dumps(data)

    rows = ""
    for label, _, color in RUNS:
        m = meta.get(label, {})
        rows += (
            f"<tr><td><span class='dot' style='background:{color}'></span>{label}</td>"
            f"<td>{m.get('state','?')}</td>"
            f"<td>{m.get('last_tok','?')}B</td>"
            f"<td>{m.get('last_ce','?')}</td></tr>"
        )

    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>models_v2 report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
 body{{font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;background:#fafafa;color:#1a1a1a}}
 .wrap{{max-width:1100px;margin:0 auto;padding:28px}}
 h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:15px;color:#555;font-weight:500;margin:0 0 18px}}
 p{{color:#333}} code{{background:#eee;padding:1px 5px;border-radius:4px;font-size:12px}}
 table{{border-collapse:collapse;margin:14px 0 26px;font-size:13px}}
 th,td{{text-align:left;padding:6px 14px 6px 0;border-bottom:1px solid #e3e3e3}}
 .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px;vertical-align:middle}}
 .grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:20px}}
 .card{{background:#fff;border:1px solid #e6e6e6;border-radius:10px;padding:12px 14px}}
 .card h3{{font-size:14px;margin:0 0 8px}}
 @media(max-width:720px){{.grid{{grid-template-columns:1fr}}}}
 .foot{{color:#888;font-size:12px;margin-top:26px}}
</style></head><body><div class="wrap">
<h1>models_v2 &mdash; stdMoE token-budget / expert-count sweep</h1>
<h2>Standard top-k MoE (<code>moe_lbreducedp_sharedexp</code>), trained with the LR cosine decayed directly over the true token budget. Curves below pulled live from W&amp;B (x-axis = tokens, B).</h2>
<p>Three runs: 64 experts at a true 25B budget, 64 experts at 50B, and 128 experts at 50B &mdash; all 8 nodes / 64 GPUs, OLMoE-mix-0824, lr 4e-3, lb 1e-1, 1 shared expert.</p>
<table><thead><tr><th>run</th><th>state</th><th>tokens</th><th>latest CE</th></tr></thead><tbody>{rows}</tbody></table>
<div class="grid" id="grid"></div>
<p class="foot">Generated by <code>scripts/models_v2/build_report.py</code> from W&amp;B project <code>{ENTITY_PROJECT}</code>. Grad norm uses a log y-axis.</p>
</div>
<script>
const RUNS={runs_js}, METRICS={metrics_js}, DATA={data_js};
const grid=document.getElementById('grid');
METRICS.forEach(m=>{{
  const card=document.createElement('div'); card.className='card';
  card.innerHTML=`<h3>${{m.title}}</h3><canvas></canvas>`;
  grid.appendChild(card);
  const ds=RUNS.map(r=>({{label:r.label,borderColor:r.color,backgroundColor:r.color,
    data:(DATA[m.key]||{{}})[r.label]||[],
    pointRadius:0,borderWidth:1.6,tension:.15,spanGaps:true}}))
    .filter(d=>d.data.length);
  const logy=m.title==='Grad norm';
  new Chart(card.querySelector('canvas'),{{type:'line',
    data:{{datasets:ds.map(d=>({{...d,data:d.data.map(p=>({{x:p[0],y:p[1]}}))}}))}},
    options:{{responsive:true,animation:false,interaction:{{mode:'nearest',intersect:false}},
      plugins:{{legend:{{labels:{{boxWidth:10,font:{{size:11}}}}}}}},
      scales:{{x:{{type:'linear',title:{{display:true,text:'tokens (B)'}},ticks:{{font:{{size:10}}}}}},
        y:{{type:logy?'logarithmic':'linear',ticks:{{font:{{size:10}}}}}}}}}}}});
}});
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wandb", action="store_true", help="render from cached metrics.json")
    args = ap.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    if args.no_wandb:
        payload = json.loads(CACHE.read_text())
    else:
        payload = fetch_from_wandb()
        CACHE.write_text(json.dumps(payload))
    (BASE / "report.html").write_text(render(payload))
    print(f"Wrote {BASE/'report.html'}  (cache: {CACHE})")


if __name__ == "__main__":
    main()
