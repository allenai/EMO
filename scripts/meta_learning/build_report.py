# PARENT: "scripts/models_routerfixed/build_report.py" (structure only; content skeleton for now)
# DESCRIPTION:
#     Builds the meta_learning experiment report at claude_outputs/meta_learning/report.html.
#     Currently a skeleton: motivation, mechanism, arm design, and a status section. Result
#     cards (selective-vs-full gap curves, W&B train curves, meta diagnostics) get folded in as
#     the smoke/pilot stages produce data.
#
#   python scripts/meta_learning/build_report.py
##############################################################

import datetime
import html
import pathlib

OUT_DIR = pathlib.Path("claude_outputs/meta_learning")
OUT_PATH = OUT_DIR / "report.html"

CSS = """
:root { --fg: #1a1a1a; --bg: #ffffff; --muted: #666; --card: #f6f7f9; --accent: #2563eb; }
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; color: var(--fg);
       background: var(--bg); max-width: 980px; margin: 2rem auto; padding: 0 1rem; line-height: 1.55; }
h1 { font-size: 1.6rem; } h2 { font-size: 1.25rem; margin-top: 2rem; }
.card { background: var(--card); border-radius: 10px; padding: 1rem 1.25rem; margin: 1rem 0; }
.status { color: var(--muted); font-size: 0.9rem; }
code { background: #eef; padding: 0.1em 0.3em; border-radius: 4px; font-size: 0.9em; }
table { border-collapse: collapse; } td, th { padding: 0.3em 0.8em; border: 1px solid #ddd; }
"""

ARMS = [
    (
        "meta128_vanilla_20b",
        "vanilla",
        "baseline 128e EMO; 20B (10B = matched tokens, 20B = matched compute vs same_tokens)",
    ),
    (
        "meta128_sametok_10b",
        "same_tokens",
        "FOMAML, outer pass on the same tokens (~2.1x step cost)",
    ),
    (
        "meta128_heldout_10b",
        "heldout",
        "FOMAML, outer pass on a held-out half of the rank batch (~1x step cost)",
    ),
]


def build() -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    arm_rows = "\n".join(
        f"<tr><td><code>{html.escape(r)}</code></td><td>{html.escape(m)}</td><td>{html.escape(d)}</td></tr>"
        for r, m, d in ARMS
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>meta_learning — FOMAML EMO pretraining</title>
<style>{CSS}</style></head><body>
<h1>meta_learning — FOMAML-style EMO pretraining</h1>
<p class="status">Skeleton report — generated {now}. Result cards land here as the pilot produces data.</p>

<h2>Motivation</h2>
<div class="card">
<p>The modular_extension <i>k=32 CPT (no extension)</i> arm showed that post-hoc selective-expert
CPT on a vanilla-EMO model buys almost nothing on its own cluster and degrades the pool broadly.
This experiment changes EMO's own pretraining objective (from scratch, 128 experts) so that
<b>updates made in 32-expert selective mode also improve the full model</b>.</p>
</div>

<h2>Mechanism (first-order MAML)</h2>
<div class="card">
<p>Per step: (1) inner forward+backward with each document restricted to its router top-32
experts; (2) temporary SGD pseudo-step on expert weights &theta;' = &theta; &minus;
&alpha;&middot;g<sub>inner</sub>; (3) outer forward+backward with all 128 experts at &theta;';
(4) restore &theta;, AdamW consumes <code>&lambda;&middot;g_inner + g_outer(&theta;')</code>
(default &lambda;=0, pure FOMAML). Implementation:
<code>src/olmo_core/train/train_module/transformer/meta_learning.py</code>.</p>
</div>

<h2>Pilot arms (10B tokens, 8 nodes each)</h2>
<div class="card"><table>
<tr><th>run</th><th>meta_mode</th><th>description</th></tr>
{arm_rows}
</table>
<p>Headline metric: the <b>selective-vs-full CE gap</b> — <code>eval/lm-pool32/*</code> minus
<code>eval/lm-full/*</code> on the v3-small ppl validation mix, per arm vs the vanilla baseline.
Go/no-go: the meta arms shrink the gap without materially regressing full-routing CE.</p></div>

<h2>Status</h2>
<div class="card"><ul>
<li>Mechanism gate (<code>verify_meta_step.py</code>): <b>7/7 PASS</b> (2026-08-16, 1&times;A100):
&alpha;=0 oracle matches <code>outer_only</code> bitwise; finite-difference
(L(0)&minus;L(&epsilon;))/&epsilon; vs &lang;g<sub>inner</sub>, g<sub>outer</sub>&rang; agrees to
0.75%; module gradients match a hand-rolled reference to ~5e-8 at &lambda;&isin;{{0, 0.5}};
expert weights restored bitwise every step.</li>
<li>Local smoke (tiny model, real data, all 4 modes &times; 30 steps): <b>PASS</b>, zero errors,
115/115 restore asserts.</li>
<li>Inner-lr sweep (same_tokens): <b>&alpha; = 3e-2 picked</b> — best CE (7.59 vs 7.78 @3e-3 /
7.98 @3e-1), inner/outer grad cosine 0.96, delta/weight ~7e-5; &alpha;=3e-1 was clip-saturated
with unstable cosine. To re-check at pilot scale (bf16 quantization if delta/weight stays
&lt;~1e-4).</li>
<li>Compile-on trial: <b>PASS</b> — 15/15 steps, steady ~2.55 s/step after warmup (faster than compile-off ~3.3 s); one warmup-time dynamo recompile-limit fallback to eager on the router pool-flip frame, no ongoing churn. Compile stays ON for the pilots.</li>
<li>Pilot launches (3 arms, 8 nodes each): pending approval.</li>
</ul></div>

</body></html>
"""


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(build())
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
