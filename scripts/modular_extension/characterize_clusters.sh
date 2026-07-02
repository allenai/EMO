#!/usr/bin/env bash
# Characterize what distinguishes each of the 64 document-router clusters, for both the
# soft (doc_probs) and hard (doc_topk_freq) routing signals, then emit the cross-embedding
# hard-vs-soft comparison + a markdown writeup. CPU-only (numpy/matplotlib).
#
#   bash scripts/modular_extension/characterize_clusters.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

BASE="modular_extension/cluster/emo100b_step23842"
OUT="${BASE}/expert_attribution"
K="${K:-64}"

for EMB in doc_probs doc_topk_freq; do
    echo "=== characterize ${EMB} ==="
    PYTHONPATH=.:src python -u -m src.scripts.clustering.characterize_cluster_expert_usage \
        --data-dir "${BASE}" --embedding "${EMB}" --k "${K}" \
        --output-dir "${OUT}/${EMB}"
done

echo "=== cross-embedding hard-vs-soft + writeup ==="
OUT="${OUT}" PYTHONPATH=.:src python -u - <<'PY'
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = os.environ["OUT"]
soft = json.load(open(f"{out}/doc_probs/metrics.json"))
hard = json.load(open(f"{out}/doc_topk_freq/metrics.json"))

# per-cluster best-single-dim AUC: hard (selection) vs soft (affinity)
sb = np.array([c["best_dim"]["auc"] for c in soft["per_cluster"]])
hb = np.array([c["best_dim"]["auc"] for c in hard["per_cluster"]])
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.scatter(hb, sb, s=40, alpha=0.7, edgecolors="k", linewidths=0.4)
lim = [0.5, 1.02]
ax.plot(lim, lim, "--", color="gray", lw=1)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("best single-expert AUC — HARD selection (doc_topk_freq)")
ax.set_ylabel("best single-expert AUC — SOFT affinity (doc_probs)")
ax.set_title("Is a cluster identifiable from which experts it actually selects,\nor only from soft affinity?")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{out}/hard_vs_soft.png", dpi=150)
plt.close(fig)

def line(s):
    vc = s["verdict_counts"]
    return (f"- **{s['embedding']}**: {vc['few-expert']} few-expert / {vc['broad-redundant']} "
            f"broad-redundant / {vc['subtle']} subtle clusters; median effective deviation dims "
            f"{s['median_effective_dims']:.1f}/{s['n_dims']} (top-5 experts carry only "
            f"{s['median_top5_mass']*100:.0f}% of the deviation); median best-single vs full-pattern "
            f"AUC {s['median_best_single_dim_auc']:.3f} vs {s['median_full_pattern_auc']:.3f}; median "
            f"experts used/layer {s['median_effective_experts_used']:.1f}/{s['n_experts_per_layer']} "
            f"(global {s['global_effective_experts']:.1f}).")

with open(f"{out}/characterization.md", "w") as f:
    f.write("# What distinguishes the 64 document-router clusters\n\n")
    f.write("Signals: **doc_probs** = soft router affinity; **doc_topk_freq** = hard expert "
            "selection (which experts tokens actually route to).\n\n## Headline\n\n")
    f.write(line(soft) + "\n" + line(hard) + "\n\n")
    f.write("**Neither a few-expert code nor a subtle one.** The distinguishing signal is spread "
            f"over most of the {soft['n_dims']} layer-expert dimensions (top-5 carry ~"
            f"{soft['median_top5_mass']*100:.0f}%), so clusters are *not* defined by a handful of "
            "experts. Yet the per-expert differences are large, not subtle: a single well-chosen "
            f"expert separates a cluster with AUC ~{soft['median_best_single_dim_auc']:.2f}, nearly "
            f"the full-pattern {soft['median_full_pattern_auc']:.2f}. Clusters concentrate routing "
            f"onto ~{soft['median_effective_experts_used']:.0f}/{soft['n_experts_per_layer']} experts "
            f"per layer vs the corpus average of ~{soft['global_effective_experts']:.0f}. So each "
            "cluster is a **broad, redundant expert-usage signature**: many experts each shift "
            "consistently and substantially, and any one of them already flags the cluster.\n\n")
    f.write("## Reading the figures\n\n"
            "- `signature_concentration.png` — fraction of each cluster's deviation carried by its top-m experts.\n"
            "- `single_dim_vs_full_auc.png` — best single expert vs whole-pattern separability (gap = subtlety).\n"
            "- `deviation_heatmap.png` — per-cluster expert deviation (z-scored); blocky = few experts, diffuse = subtle.\n"
            "- `verdict_scatter.png` — effective deviation dims vs routing peakedness, sized by #docs.\n"
            "- `example_profiles.png` — 16×63 routing profiles, most concentrated → most distributed.\n"
            "- `hard_vs_soft.png` — single-expert identifiability under hard selection vs soft affinity.\n")
print("wrote", f"{out}/characterization.md and hard_vs_soft.png")
PY
echo "DONE: ${OUT}/"
