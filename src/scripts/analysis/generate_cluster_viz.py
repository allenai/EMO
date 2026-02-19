"""
Generate cluster analysis report + interactive HTML visualizer.

Usage:
    python -m src.scripts.analysis.generate_cluster_viz \
        --output-dir claude_outputs/analysis/router_clustering \
        --k 64
"""

import argparse
import gzip
import json
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import umap as umap_lib

import logging
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cluster metadata: hand-labeled names + categories
# ---------------------------------------------------------------------------

CLUSTER_LABELS = {
    0:  ("Industrial & Manufacturing Products", "reference"),
    1:  ("Parenting & Family Blogs", "personal"),
    2:  ("Software Development Q&A", "code"),
    3:  ("Environmental & Energy Journalism", "news"),
    4:  ("International Conflict News", "news"),
    5:  ("Business & HR Advice", "business"),
    6:  ("Biomedical / Clinical Medicine", "science"),
    7:  ("Gaming & RPG", "personal"),
    8:  ("Digital Marketing & E-Commerce", "business"),
    9:  ("Food & Recipe Blogs", "personal"),
    10: ("Personal Lifestyle Blogs", "personal"),
    11: ("Celebrity & Entertainment Gossip", "personal"),
    12: ("Pet Care", "personal"),
    13: ("Online Forums & Game FAQs", "reference"),
    14: ("Consumer Health & Wellness", "health"),
    15: ("US Political Commentary", "news"),
    16: ("Sports Training & Fitness", "personal"),
    17: ("General Programming Code", "code"),
    18: ("Web Forms & Registration Pages", "reference"),
    19: ("Personal Finance & Mortgages", "business"),
    20: ("Music Reviews & Blogs", "arts"),
    21: ("Religion & Christianity", "arts"),
    22: ("Multilingual / Non-English Web", "reference"),
    23: ("Travel & Tourism", "personal"),
    24: ("Math & Physics Q&A Forums", "science"),
    25: ("Self-Help & Personal Development", "personal"),
    26: ("Movies & Entertainment Reviews", "arts"),
    27: ("Chinese Manufacturing Listings", "business"),
    28: ("Web Hosting & Tech Support", "code"),
    29: ("Earth Science & Climate", "science"),
    30: ("US Law & Legal Commentary", "news"),
    31: ("Geopolitical / Foreign Policy News", "news"),
    32: ("Humanities & Literary Analysis", "arts"),
    33: ("Professional Sports News", "personal"),
    34: ("Cooking Recipes", "personal"),
    35: ("Education & Pedagogy", "science"),
    36: ("Philosophy & Secular Blogs", "arts"),
    37: ("Spam / Low-Quality Web", "spam"),
    38: ("Crafts, DIY & Handmade Arts", "personal"),
    39: ("Finance & Investment Analysis", "business"),
    40: ("Crime & Violence News", "news"),
    41: ("Generic Listicle / Boilerplate", "spam"),
    42: ("School & Community News", "education"),
    43: ("South Asian / Indian News", "news"),
    44: ("Wildlife & Animal Biology", "science"),
    45: ("Programming Tutorials & Docs", "code"),
    46: ("Astronomy & Space Science", "science"),
    47: ("Slang & Quotations (Urban Dict.)", "reference"),
    48: ("Automotive Reviews", "reference"),
    49: ("Fan Fiction & Creative Fiction", "personal"),
    50: ("Biomedical Research (Immunology)", "science"),
    51: ("Consumer Tech & Gadgets", "reference"),
    52: ("Applied Engineering Research", "science"),
    53: ("Tech Stack Exchange Q&A", "code"),
    54: ("Poetry & Creative Writing", "arts"),
    55: ("Legal & Regulatory Compliance", "business"),
    56: ("Consumer Electronics & Hardware", "reference"),
    57: ("SEO Spam / Dating Ads", "spam"),
    58: ("Miscellaneous Personal Blogs", "personal"),
    59: ("Academic Research Citations", "science"),
    60: ("Reference & Dictionary Sites", "reference"),
    61: ("Alternative Health & Supplements", "health"),
    62: ("Systems & Infrastructure Code", "code"),
    63: ("International Trade & Economics", "news"),
}

CATEGORY_COLORS = {
    "code":      "#4A90E2",
    "science":   "#27AE60",
    "news":      "#E67E22",
    "personal":  "#E91E8C",
    "business":  "#9B59B6",
    "health":    "#E74C3C",
    "arts":      "#F39C12",
    "education": "#1ABC9C",
    "reference": "#7F8C8D",
    "spam":      "#BDC3C7",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="claude_outputs/analysis/router_clustering")
    parser.add_argument("--k", type=int, default=64)
    args = parser.parse_args()

    cluster_dir = os.path.join(args.output_dir, f"clusters_k{args.k}")

    # Load everything
    logger.info("Loading embeddings...")
    emb = np.load(os.path.join(args.output_dir, "embeddings_optA_avgprob.npy")).astype(np.float32)
    labels = np.load(os.path.join(cluster_dir, "assignments.npy"))

    logger.info("Loading metadata...")
    meta = []
    with gzip.open(os.path.join(args.output_dir, "metadata.jsonl.gz"), "rt") as f:
        for line in f:
            meta.append(json.loads(line))

    with open(os.path.join(args.output_dir, "info.json")) as f:
        info = json.load(f)

    with open(os.path.join(cluster_dir, "summary.json")) as f:
        summary = json.load(f)

    # UMAP
    umap_path = os.path.join(cluster_dir, "umap_coords.npy")
    if os.path.exists(umap_path):
        logger.info("Loading cached UMAP coords...")
        coords_2d = np.load(umap_path)
    else:
        logger.info("PCA-50...")
        pca = PCA(n_components=50, random_state=42)
        reduced = pca.fit_transform(emb)
        reduced_normed = normalize(reduced, norm="l2")

        logger.info("Running UMAP...")
        reducer = umap_lib.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                                metric="euclidean", random_state=42, verbose=False)
        coords_2d = reducer.fit_transform(reduced_normed)
        np.save(umap_path, coords_2d)
        logger.info(f"Saved UMAP coords → {umap_path}")

    # Build per-document data for JS (clip to keep file size manageable)
    logger.info("Building document data...")
    docs_js = []
    for i, (m, lbl, xy) in enumerate(zip(meta, labels, coords_2d)):
        docs_js.append({
            "i": i,
            "c": int(lbl),
            "s": m["source"],
            "l": m["doc_len"],
            "p": m["preview"],   # full 400-char preview
            "x": round(float(xy[0]), 3),
            "y": round(float(xy[1]), 3),
        })

    # Build cluster data for JS
    clusters_js = []
    for c in summary:
        cid = c["cluster"]
        label, cat = CLUSTER_LABELS.get(cid, (f"Cluster {cid}", "reference"))
        clusters_js.append({
            "id": cid,
            "label": label,
            "category": cat,
            "color": CATEGORY_COLORS[cat],
            "size": c["size"],
            "source_counts": c["source_counts"],
            "top_experts": c["top10_experts_global"],
            "rep_docs": c["representative_docs"],
        })

    # Write markdown report
    report_path = os.path.join(args.output_dir, "cluster_report.md")
    write_report(clusters_js, info, args.k, report_path)
    logger.info(f"Saved report → {report_path}")

    # Write HTML visualizer
    html_path = os.path.join(args.output_dir, "cluster_explorer.html")
    write_html(clusters_js, docs_js, info, args.k, html_path)
    logger.info(f"Saved HTML visualizer → {html_path}")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_report(clusters_js, info, k, path):
    lines = []
    lines.append(f"# FlexMoE Router Cluster Analysis (k={k})")
    lines.append("")
    lines.append(f"**Model:** `{info['model_path']}`")
    lines.append(f"**Documents:** {info['num_docs']:,}  |  **Tokens:** {info['total_tokens']:,}")
    lines.append(f"**Layers:** {info['num_layers']}  |  **Standard experts:** {info['num_standard_experts']}")
    lines.append(f"**Embedding:** Option A — avg softmax prob per expert per layer → {info['emb_dim']}-dim float16")
    lines.append(f"**Preprocessing:** PCA-50 + L2 norm → MiniBatchKMeans k={k}")
    lines.append(f"**Silhouette (k=64):** 0.1327")
    lines.append("")

    lines.append("## Data Composition")
    lines.append("")
    lines.append("| Source | Docs | % |")
    lines.append("|--------|------|---|")
    for src, cnt in sorted(info["source_doc_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"| {src} | {cnt:,} | {cnt/info['num_docs']:.1%} |")
    lines.append("")

    lines.append("## Key Findings")
    lines.append("")
    lines.append("The model's router has learned semantic domains that **transcend data source boundaries**.")
    lines.append("The most diagnostic clusters are those that mix sources — these reveal domains")
    lines.append("the model considers equivalent despite coming from different data pipelines.")
    lines.append("")
    lines.append("### Cross-Source Clusters (most informative)")
    lines.append("")
    lines.append("| Cluster | Label | Size | Notable source mix |")
    lines.append("|---------|-------|------|-------------------|")
    for c in clusters_js:
        src = c["source_counts"]
        total = c["size"]
        dclm_frac = src.get("dclm", 0) / total
        if dclm_frac < 0.92:
            non_dclm = {k: v for k, v in src.items() if k != "dclm"}
            mix_str = ", ".join(f"{k} {v/total:.0%}" for k, v in
                                sorted(non_dclm.items(), key=lambda x: -x[1]))
            lines.append(f"| {c['id']} | {c['label']} | {c['size']} | {mix_str} |")
    lines.append("")

    lines.append("### Category breakdown")
    lines.append("")
    cat_totals = {}
    for c in clusters_js:
        cat = c["category"]
        cat_totals[cat] = cat_totals.get(cat, 0) + c["size"]
    lines.append("| Category | Clusters | Docs | % |")
    lines.append("|----------|----------|------|---|")
    total_docs = sum(cat_totals.values())
    for cat, cnt in sorted(cat_totals.items(), key=lambda x: -x[1]):
        n_clusters = sum(1 for c in clusters_js if c["category"] == cat)
        lines.append(f"| {cat} | {n_clusters} | {cnt:,} | {cnt/total_docs:.1%} |")
    lines.append("")

    lines.append("## All Clusters")
    lines.append("")

    by_cat = {}
    for c in clusters_js:
        by_cat.setdefault(c["category"], []).append(c)

    for cat in sorted(by_cat):
        lines.append(f"### {cat.title()}")
        lines.append("")
        for c in sorted(by_cat[cat], key=lambda x: -x["size"]):
            src = c["source_counts"]
            total = c["size"]
            src_str = " · ".join(
                f"{s} {v/total:.0%}"
                for s, v in sorted(src.items(), key=lambda x: -x[1])
                if v / total > 0.01
            )
            lines.append(f"#### Cluster {c['id']}: {c['label']}")
            lines.append("")
            lines.append(f"**Size:** {c['size']:,} docs ({c['size']/total_docs:.1%} of corpus)")
            lines.append(f"**Sources:** {src_str}")
            lines.append(f"**Top experts (summed across layers):** {c['top_experts'][:5]}")
            lines.append("")
            lines.append("**Representative documents:**")
            lines.append("")
            for i, doc in enumerate(c["rep_docs"]):
                lines.append(f"{i+1}. `[{doc['source']}]` ({doc['doc_len']} tokens)")
                lines.append(f"   > {doc['preview'][:250]}")
                lines.append("")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# HTML visualizer
# ---------------------------------------------------------------------------

def write_html(clusters_js, docs_js, info, k, path):
    clusters_json = json.dumps(clusters_js)
    docs_json = json.dumps(docs_js)
    cat_colors_json = json.dumps(CATEGORY_COLORS)
    model_path = info["model_path"]
    num_docs = info["num_docs"]
    num_tokens = info["total_tokens"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FlexMoE Cluster Explorer — k={k}</title>
<style>
  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #232635;
    --border: #2e3347;
    --text: #e2e8f0;
    --text-dim: #8892a4;
    --accent: #4A90E2;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 13px; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}

  /* Header */
  #header {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 10px 16px; display: flex; align-items: center; gap: 16px; flex-shrink: 0; }}
  #header h1 {{ font-size: 15px; font-weight: 600; color: var(--text); }}
  #header .meta {{ color: var(--text-dim); font-size: 11px; }}
  #view-tabs {{ margin-left: auto; display: flex; gap: 4px; }}
  .view-tab {{ background: var(--surface2); border: 1px solid var(--border); color: var(--text-dim); padding: 5px 14px; border-radius: 6px; cursor: pointer; font-size: 12px; }}
  .view-tab.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}

  /* Filter bar */
  #filterbar {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 8px 16px; display: flex; gap: 6px; flex-wrap: wrap; align-items: center; flex-shrink: 0; }}
  #search {{ background: var(--surface2); border: 1px solid var(--border); color: var(--text); padding: 5px 10px; border-radius: 6px; font-size: 12px; width: 200px; outline: none; }}
  #search:focus {{ border-color: var(--accent); }}
  .cat-btn {{ border: 1px solid var(--border); background: var(--surface2); color: var(--text-dim); padding: 4px 10px; border-radius: 12px; cursor: pointer; font-size: 11px; transition: all .15s; }}
  .cat-btn:hover {{ border-color: #555; color: var(--text); }}
  .cat-btn.active {{ color: #fff; }}
  #sort-select {{ background: var(--surface2); border: 1px solid var(--border); color: var(--text-dim); padding: 4px 8px; border-radius: 6px; font-size: 11px; outline: none; margin-left: auto; }}

  /* Main layout */
  #main {{ display: flex; flex: 1; overflow: hidden; }}

  /* Sidebar */
  #sidebar {{ width: 280px; flex-shrink: 0; background: var(--surface); border-right: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; }}
  #cluster-list {{ overflow-y: auto; flex: 1; }}
  .cluster-item {{ padding: 8px 12px; cursor: pointer; border-bottom: 1px solid var(--border); transition: background .1s; }}
  .cluster-item:hover {{ background: var(--surface2); }}
  .cluster-item.selected {{ background: var(--surface2); border-left: 3px solid var(--accent); padding-left: 9px; }}
  .cluster-item.hidden {{ display: none; }}
  .ci-header {{ display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }}
  .ci-dot {{ width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }}
  .ci-id {{ color: var(--text-dim); font-size: 10px; width: 24px; }}
  .ci-label {{ font-size: 12px; font-weight: 500; flex: 1; line-height: 1.3; }}
  .ci-size {{ color: var(--text-dim); font-size: 10px; }}
  .ci-srcbar {{ height: 4px; border-radius: 2px; background: var(--border); overflow: hidden; display: flex; }}
  .ci-srcbar-seg {{ height: 100%; }}
  #sidebar-count {{ padding: 6px 12px; color: var(--text-dim); font-size: 11px; border-top: 1px solid var(--border); flex-shrink: 0; }}

  /* Content panel */
  #content {{ flex: 1; overflow: hidden; display: flex; flex-direction: column; }}

  /* ── Cluster detail view ── */
  #detail-view {{ flex: 1; overflow-y: auto; display: none; flex-direction: column; }}
  #detail-view.active {{ display: flex; }}

  #detail-header {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 14px 20px; flex-shrink: 0; }}
  #detail-title {{ font-size: 18px; font-weight: 700; margin-bottom: 2px; }}
  #detail-meta {{ color: var(--text-dim); font-size: 12px; }}

  #detail-body {{ padding: 16px 20px; display: flex; flex-direction: column; gap: 16px; }}

  .section-title {{ font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; color: var(--text-dim); margin-bottom: 8px; }}

  /* Source breakdown */
  #src-breakdown {{ display: flex; flex-direction: column; gap: 6px; }}
  .src-row {{ display: flex; align-items: center; gap: 8px; }}
  .src-name {{ width: 160px; font-size: 12px; color: var(--text-dim); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .src-bar-wrap {{ flex: 1; height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; }}
  .src-bar-fill {{ height: 100%; border-radius: 4px; }}
  .src-count {{ width: 80px; text-align: right; font-size: 11px; color: var(--text-dim); }}

  /* Top experts */
  #experts-list {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .expert-tag {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 4px; padding: 3px 8px; font-size: 12px; font-family: monospace; }}

  /* Rep docs */
  .rep-doc {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; cursor: pointer; transition: border-color .15s; }}
  .rep-doc:hover {{ border-color: var(--accent); }}
  .rep-doc-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
  .source-badge {{ font-size: 10px; padding: 2px 7px; border-radius: 10px; font-weight: 600; letter-spacing: .03em; }}
  .rep-doc-tokens {{ color: var(--text-dim); font-size: 11px; }}
  .rep-doc-preview {{ color: var(--text-dim); font-size: 12px; line-height: 1.6; white-space: pre-wrap; }}

  /* Document browser */
  #doc-browser {{ border-top: 1px solid var(--border); flex-shrink: 0; }}
  #doc-browser-header {{ display: flex; align-items: center; gap: 8px; padding: 8px 20px; background: var(--surface); border-bottom: 1px solid var(--border); cursor: pointer; user-select: none; }}
  #doc-browser-header h3 {{ font-size: 12px; font-weight: 600; }}
  #doc-browser-toggle {{ margin-left: auto; color: var(--text-dim); font-size: 11px; }}
  #doc-search {{ background: var(--surface2); border: 1px solid var(--border); color: var(--text); padding: 4px 8px; border-radius: 5px; font-size: 11px; outline: none; width: 180px; }}
  #doc-search:focus {{ border-color: var(--accent); }}
  #doc-table-wrap {{ height: 240px; overflow-y: auto; display: none; }}
  #doc-table-wrap.open {{ display: block; }}
  #doc-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  #doc-table th {{ position: sticky; top: 0; background: var(--surface); padding: 6px 12px; text-align: left; color: var(--text-dim); font-weight: 500; border-bottom: 1px solid var(--border); font-size: 11px; cursor: pointer; user-select: none; }}
  #doc-table th:hover {{ color: var(--text); }}
  #doc-table td {{ padding: 5px 12px; border-bottom: 1px solid var(--border); vertical-align: top; }}
  #doc-table tr:hover td {{ background: var(--surface2); cursor: pointer; }}
  .doc-preview-cell {{ color: var(--text-dim); max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}

  /* Document modal */
  #doc-modal {{ display: none; position: fixed; inset: 0; z-index: 100; background: rgba(0,0,0,.7); align-items: center; justify-content: center; }}
  #doc-modal.open {{ display: flex; }}
  #doc-modal-box {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; width: 740px; max-width: 90vw; max-height: 80vh; display: flex; flex-direction: column; overflow: hidden; }}
  #doc-modal-header {{ display: flex; align-items: center; gap: 10px; padding: 14px 18px; border-bottom: 1px solid var(--border); flex-shrink: 0; }}
  #doc-modal-title {{ font-size: 14px; font-weight: 600; flex: 1; }}
  #doc-modal-close {{ color: var(--text-dim); cursor: pointer; font-size: 18px; line-height: 1; }}
  #doc-modal-close:hover {{ color: var(--text); }}
  #doc-modal-body {{ padding: 16px 18px; overflow-y: auto; }}
  #doc-modal-meta {{ color: var(--text-dim); font-size: 11px; margin-bottom: 12px; }}
  #doc-modal-content {{ font-size: 13px; line-height: 1.8; white-space: pre-wrap; color: var(--text); }}

  /* ── UMAP view ── */
  #umap-view {{ flex: 1; overflow: hidden; display: none; flex-direction: column; }}
  #umap-view.active {{ display: flex; }}
  #umap-controls {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 8px 16px; display: flex; gap: 12px; align-items: center; flex-shrink: 0; font-size: 12px; }}
  .umap-radio label {{ cursor: pointer; color: var(--text-dim); }}
  .umap-radio input {{ margin-right: 4px; }}
  #umap-tooltip {{ position: fixed; background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; font-size: 12px; pointer-events: none; z-index: 50; display: none; max-width: 260px; }}
  #umap-canvas-wrap {{ flex: 1; position: relative; overflow: hidden; }}
  #umap-canvas {{ display: block; width: 100%; height: 100%; cursor: crosshair; }}
  #umap-legend {{ position: absolute; top: 12px; right: 12px; background: rgba(15,17,23,.85); border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; max-height: 80vh; overflow-y: auto; font-size: 11px; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; margin-bottom: 5px; cursor: pointer; }}
  .legend-dot {{ width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }}
  .legend-label {{ color: var(--text-dim); }}
  .legend-item:hover .legend-label {{ color: var(--text); }}

  /* Scrollbar */
  ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
  ::-webkit-scrollbar-track {{ background: transparent; }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}

  /* Source badge colors */
  .badge-dclm {{ background: #1e3a5f; color: #64b0f4; }}
  .badge-starcoder {{ background: #1e3a2f; color: #4caf78; }}
  .badge-pes2o {{ background: #3a2b1e; color: #f4a04a; }}
  .badge-proofpile-2-arxiv {{ background: #2e1e3a; color: #c084f5; }}
  .badge-proofpile-2-open-web-math {{ background: #3a2e1e; color: #f5c842; }}
  .badge-proofpile-2-stack {{ background: #1e2e3a; color: #42c8f5; }}
  .badge-wikipedia {{ background: #2a2a2a; color: #aaaaaa; }}
</style>
</head>
<body>

<div id="header">
  <h1>FlexMoE Router Cluster Explorer</h1>
  <span class="meta">k={k} · {num_docs:,} docs · {num_tokens:,} tokens · {model_path.split('/')[-2]}</span>
  <div id="view-tabs">
    <button class="view-tab active" onclick="setView('detail')">Clusters</button>
    <button class="view-tab" onclick="setView('umap')">UMAP</button>
  </div>
</div>

<div id="filterbar">
  <input id="search" type="text" placeholder="Search clusters…" oninput="filterClusters()">
  <button class="cat-btn active" data-cat="all" onclick="setCat('all', this)" style="border-color:#4A90E2;color:#4A90E2;">All</button>
</div>

<div id="main">
  <div id="sidebar">
    <div id="cluster-list"></div>
    <div id="sidebar-count"></div>
  </div>

  <div id="content">
    <!-- Detail view -->
    <div id="detail-view" class="active">
      <div id="detail-header">
        <div id="detail-title">Select a cluster</div>
        <div id="detail-meta"></div>
      </div>
      <div id="detail-body">
        <div id="detail-placeholder" style="color:var(--text-dim);padding:20px 0;">
          Click a cluster in the sidebar to explore its documents and expert routing patterns.
        </div>
        <div id="detail-content" style="display:none">
          <div>
            <div class="section-title">Source Breakdown</div>
            <div id="src-breakdown"></div>
          </div>
          <div>
            <div class="section-title">Top Experts (summed across all 16 layers)</div>
            <div id="experts-list"></div>
          </div>
          <div>
            <div class="section-title">Representative Documents (closest to centroid)</div>
            <div id="rep-docs-list"></div>
          </div>
        </div>
      </div>
      <div id="doc-browser">
        <div id="doc-browser-header" onclick="toggleDocBrowser()">
          <h3 id="doc-browser-title">All Documents in Cluster</h3>
          <input id="doc-search" type="text" placeholder="Filter docs…" onclick="event.stopPropagation()" oninput="filterDocTable()">
          <span id="doc-browser-toggle">▲ Show</span>
        </div>
        <div id="doc-table-wrap">
          <table id="doc-table">
            <thead>
              <tr>
                <th onclick="sortDocTable('i')"># <span id="sort-i"></span></th>
                <th onclick="sortDocTable('s')">Source <span id="sort-s"></span></th>
                <th onclick="sortDocTable('l')">Tokens <span id="sort-l"></span></th>
                <th>Preview</th>
              </tr>
            </thead>
            <tbody id="doc-tbody"></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- UMAP view -->
    <div id="umap-view">
      <div id="umap-controls">
        <span style="color:var(--text-dim)">Color by:</span>
        <label class="umap-radio"><input type="radio" name="umap-color" value="cluster" checked onchange="redrawUmap()"> Cluster</label>
        <label class="umap-radio"><input type="radio" name="umap-color" value="category" onchange="redrawUmap()"> Category</label>
        <label class="umap-radio"><input type="radio" name="umap-color" value="source" onchange="redrawUmap()"> Data Source</label>
        <span id="umap-status" style="color:var(--text-dim);margin-left:auto;"></span>
      </div>
      <div id="umap-canvas-wrap">
        <canvas id="umap-canvas"></canvas>
        <div id="umap-legend"></div>
        <div id="umap-tooltip"></div>
      </div>
    </div>
  </div>
</div>

<!-- Document modal -->
<div id="doc-modal" onclick="if(event.target===this)closeModal()">
  <div id="doc-modal-box">
    <div id="doc-modal-header">
      <div id="doc-modal-title"></div>
      <span id="doc-modal-close" onclick="closeModal()">✕</span>
    </div>
    <div id="doc-modal-body">
      <div id="doc-modal-meta"></div>
      <div id="doc-modal-content"></div>
    </div>
  </div>
</div>

<script>
// ── Data ──
const CLUSTERS = {clusters_json};
const DOCS = {docs_json};
const CAT_COLORS = {cat_colors_json};

const SOURCE_COLORS = {{
  'dclm': '#64b0f4',
  'starcoder': '#4caf78',
  'pes2o': '#f4a04a',
  'proofpile-2-arxiv': '#c084f5',
  'proofpile-2-open-web-math': '#f5c842',
  'proofpile-2-stack': '#42c8f5',
  'wikipedia': '#aaaaaa',
}};

// ── State ──
let selectedCluster = null;
let currentView = 'detail';
let activeCat = 'all';
let docSortCol = 'i';
let docSortDir = 1;
let clusterDocs = [];
let filteredClusterDocs = [];
let docBrowserOpen = false;
let umapInitialized = false;

// ── Init ──
function init() {{
  buildCatButtons();
  buildSidebar();
  filterClusters();
  if (CLUSTERS.length > 0) selectCluster(CLUSTERS[0].id);
}}

function buildCatButtons() {{
  const bar = document.getElementById('filterbar');
  const cats = [...new Set(CLUSTERS.map(c => c.category))].sort();
  cats.forEach(cat => {{
    const btn = document.createElement('button');
    btn.className = 'cat-btn';
    btn.dataset.cat = cat;
    btn.textContent = cat.charAt(0).toUpperCase() + cat.slice(1);
    btn.style.cssText = `border-color:${{CAT_COLORS[cat]}}30`;
    btn.onclick = () => setCat(cat, btn);
    bar.appendChild(btn);
  }});
  // Sort dropdown
  const sel = document.createElement('select');
  sel.id = 'sort-select';
  sel.innerHTML = '<option value="id">Sort: ID</option><option value="size">Sort: Size</option><option value="label">Sort: Name</option>';
  sel.onchange = () => buildSidebar();
  bar.appendChild(sel);
}}

function buildSidebar() {{
  const sortVal = document.getElementById('sort-select')?.value || 'id';
  const sorted = [...CLUSTERS].sort((a, b) => {{
    if (sortVal === 'size') return b.size - a.size;
    if (sortVal === 'label') return a.label.localeCompare(b.label);
    return a.id - b.id;
  }});

  const list = document.getElementById('cluster-list');
  list.innerHTML = '';
  sorted.forEach(c => {{
    const item = document.createElement('div');
    item.className = 'cluster-item' + (c.id === selectedCluster ? ' selected' : '');
    item.dataset.id = c.id;
    item.dataset.cat = c.category;
    item.dataset.label = c.label.toLowerCase();

    const total = c.size;
    const srcSegs = Object.entries(c.source_counts)
      .sort((a,b) => b[1]-a[1])
      .map(([s,n]) => `<div class="ci-srcbar-seg" style="width:${{(n/total*100).toFixed(1)}}%;background:${{SOURCE_COLORS[s] || '#666'}}"></div>`)
      .join('');

    item.innerHTML = `
      <div class="ci-header">
        <div class="ci-dot" style="background:${{c.color}}"></div>
        <div class="ci-id">${{c.id}}</div>
        <div class="ci-label">${{c.label}}</div>
        <div class="ci-size">${{c.size.toLocaleString()}}</div>
      </div>
      <div class="ci-srcbar">${{srcSegs}}</div>`;
    item.onclick = () => selectCluster(c.id);
    list.appendChild(item);
  }});
  updateSidebarCount();
}}

function filterClusters() {{
  const q = document.getElementById('search').value.toLowerCase();
  const items = document.querySelectorAll('.cluster-item');
  let vis = 0;
  items.forEach(el => {{
    const matchCat = activeCat === 'all' || el.dataset.cat === activeCat;
    const matchQ = !q || el.dataset.label.includes(q) || el.dataset.id.includes(q);
    el.classList.toggle('hidden', !(matchCat && matchQ));
    if (matchCat && matchQ) vis++;
  }});
  document.getElementById('sidebar-count').textContent = `${{vis}} of ${{CLUSTERS.length}} clusters`;
}}

function updateSidebarCount() {{
  const vis = document.querySelectorAll('.cluster-item:not(.hidden)').length;
  document.getElementById('sidebar-count').textContent = `${{vis}} of ${{CLUSTERS.length}} clusters`;
}}

function setCat(cat, btn) {{
  activeCat = cat;
  document.querySelectorAll('.cat-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  filterClusters();
}}

function setView(view) {{
  currentView = view;
  document.querySelectorAll('.view-tab').forEach((t,i) => t.classList.toggle('active', ['detail','umap'][i] === view));
  document.getElementById('detail-view').classList.toggle('active', view === 'detail');
  document.getElementById('umap-view').classList.toggle('active', view === 'umap');
  if (view === 'umap' && !umapInitialized) {{ initUmap(); umapInitialized = true; }}
  if (view === 'umap') redrawUmap();
}}

// ── Cluster detail ──
function selectCluster(id) {{
  selectedCluster = id;
  const c = CLUSTERS.find(x => x.id === id);
  if (!c) return;

  // Sidebar highlight
  document.querySelectorAll('.cluster-item').forEach(el => {{
    el.classList.toggle('selected', parseInt(el.dataset.id) === id);
  }});

  // Header
  document.getElementById('detail-title').textContent = `Cluster ${{c.id}} — ${{c.label}}`;
  const catColor = c.color;
  document.getElementById('detail-meta').innerHTML =
    `<span style="background:${{catColor}}22;color:${{catColor}};padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600">${{c.category}}</span>  ` +
    `${{c.size.toLocaleString()}} documents`;

  document.getElementById('detail-placeholder').style.display = 'none';
  document.getElementById('detail-content').style.display = '';

  // Sources
  const total = c.size;
  const srcEl = document.getElementById('src-breakdown');
  srcEl.innerHTML = '';
  Object.entries(c.source_counts).sort((a,b)=>b[1]-a[1]).forEach(([src,cnt]) => {{
    const pct = (cnt/total*100).toFixed(1);
    const color = SOURCE_COLORS[src] || '#aaa';
    srcEl.innerHTML += `
      <div class="src-row">
        <div class="src-name">${{src}}</div>
        <div class="src-bar-wrap"><div class="src-bar-fill" style="width:${{pct}}%;background:${{color}}"></div></div>
        <div class="src-count">${{cnt.toLocaleString()}} (${{pct}}%)</div>
      </div>`;
  }});

  // Experts
  const expEl = document.getElementById('experts-list');
  expEl.innerHTML = c.top_experts.map((e,i) =>
    `<span class="expert-tag" style="opacity:${{1 - i*0.07}}">#${{e}}</span>`).join('');

  // Rep docs
  const repEl = document.getElementById('rep-docs-list');
  repEl.innerHTML = '';
  c.rep_docs.forEach((doc, i) => {{
    const badgeClass = 'badge-' + doc.source.replace(/[^a-z0-9]/g, '-');
    const div = document.createElement('div');
    div.className = 'rep-doc';
    div.innerHTML = `
      <div class="rep-doc-header">
        <span class="source-badge ${{badgeClass}}">${{doc.source}}</span>
        <span class="rep-doc-tokens">${{doc.doc_len.toLocaleString()}} tokens</span>
        <span style="margin-left:auto;color:var(--text-dim);font-size:11px">rep #${{i+1}}</span>
      </div>
      <div class="rep-doc-preview">${{escHtml(doc.preview)}}</div>`;
    div.onclick = () => openDocModal(null, doc.source, doc.doc_len, doc.preview);
    repEl.appendChild(div);
  }});

  // Load cluster docs
  clusterDocs = DOCS.filter(d => d.c === id);
  filteredClusterDocs = [...clusterDocs];
  document.getElementById('doc-browser-title').textContent =
    `All Documents in Cluster (${{clusterDocs.length.toLocaleString()}} total)`;
  if (docBrowserOpen) renderDocTable();
}}

// ── Document browser ──
function toggleDocBrowser() {{
  docBrowserOpen = !docBrowserOpen;
  document.getElementById('doc-table-wrap').classList.toggle('open', docBrowserOpen);
  document.getElementById('doc-browser-toggle').textContent = docBrowserOpen ? '▼ Hide' : '▲ Show';
  if (docBrowserOpen) renderDocTable();
}}

function filterDocTable() {{
  const q = document.getElementById('doc-search').value.toLowerCase();
  filteredClusterDocs = q ? clusterDocs.filter(d =>
    d.s.includes(q) || d.p.toLowerCase().includes(q)) : [...clusterDocs];
  renderDocTable();
}}

function sortDocTable(col) {{
  if (docSortCol === col) docSortDir *= -1;
  else {{ docSortCol = col; docSortDir = 1; }}
  filteredClusterDocs.sort((a,b) => {{
    const av = a[col], bv = b[col];
    return (typeof av === 'string' ? av.localeCompare(bv) : av - bv) * docSortDir;
  }});
  renderDocTable();
}}

function renderDocTable() {{
  const tbody = document.getElementById('doc-tbody');
  tbody.innerHTML = '';
  // Render up to 500 rows for performance
  filteredClusterDocs.slice(0, 500).forEach(d => {{
    const tr = document.createElement('tr');
    const badgeClass = 'badge-' + d.s.replace(/[^a-z0-9]/g, '-');
    tr.innerHTML = `
      <td style="color:var(--text-dim)">${{d.i}}</td>
      <td><span class="source-badge ${{badgeClass}}">${{d.s}}</span></td>
      <td style="color:var(--text-dim)">${{d.l.toLocaleString()}}</td>
      <td class="doc-preview-cell">${{escHtml(d.p)}}</td>`;
    tr.onclick = () => openDocModal(d.i, d.s, d.l, d.p);
    tbody.appendChild(tr);
  }});
  if (filteredClusterDocs.length > 500) {{
    const tr = document.createElement('tr');
    tr.innerHTML = `<td colspan="4" style="color:var(--text-dim);text-align:center;padding:8px">
      … ${{(filteredClusterDocs.length-500).toLocaleString()}} more docs (use filter to narrow)</td>`;
    tbody.appendChild(tr);
  }}
}}

// ── Document modal ──
function openDocModal(idx, source, tokens, preview) {{
  document.getElementById('doc-modal-title').textContent = source;
  document.getElementById('doc-modal-meta').innerHTML =
    `<span class="source-badge badge-${{source.replace(/[^a-z0-9]/g,'-')}}">${{source}}</span>  ` +
    `${{tokens.toLocaleString()}} tokens` +
    (idx !== null ? `  ·  doc #${{idx}}` : '');
  document.getElementById('doc-modal-content').textContent = preview;
  document.getElementById('doc-modal').classList.add('open');
}}
function closeModal() {{
  document.getElementById('doc-modal').classList.remove('open');
}}
document.addEventListener('keydown', e => {{ if (e.key==='Escape') closeModal(); }});

// ── UMAP ──
let umapCanvas, umapCtx, umapPoints = [], umapScale = {{}};

function initUmap() {{
  umapCanvas = document.getElementById('umap-canvas');
  umapCtx = umapCanvas.getContext('2d');
  umapPoints = DOCS;
  buildUmapLegend();

  // Resize observer
  new ResizeObserver(() => redrawUmap()).observe(document.getElementById('umap-canvas-wrap'));

  // Mouse interaction
  umapCanvas.addEventListener('mousemove', onUmapMouseMove);
  umapCanvas.addEventListener('click', onUmapClick);
  umapCanvas.addEventListener('mouseleave', () => document.getElementById('umap-tooltip').style.display = 'none');
}}

function getUmapColor(d) {{
  const mode = document.querySelector('input[name="umap-color"]:checked').value;
  if (mode === 'cluster') {{
    const c = CLUSTERS.find(x => x.id === d.c);
    return c ? c.color : '#555';
  }} else if (mode === 'category') {{
    const c = CLUSTERS.find(x => x.id === d.c);
    return c ? CAT_COLORS[c.category] : '#555';
  }} else {{
    return SOURCE_COLORS[d.s] || '#555';
  }}
}}

function redrawUmap() {{
  const wrap = document.getElementById('umap-canvas-wrap');
  const W = wrap.clientWidth, H = wrap.clientHeight;
  umapCanvas.width = W; umapCanvas.height = H;

  const xs = DOCS.map(d => d.x), ys = DOCS.map(d => d.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const pad = 30;
  umapScale = {{ xMin, xMax, yMin, yMax, W, H, pad }};

  umapCtx.fillStyle = '#0f1117';
  umapCtx.fillRect(0, 0, W, H);

  // Draw all points
  DOCS.forEach(d => {{
    const [px, py] = toCanvas(d.x, d.y);
    const isSelected = d.c === selectedCluster;
    umapCtx.beginPath();
    umapCtx.arc(px, py, isSelected ? 3 : 2, 0, Math.PI*2);
    umapCtx.fillStyle = getUmapColor(d) + (isSelected ? 'ff' : '99');
    umapCtx.fill();
  }});
  buildUmapLegend();
}}

function toCanvas(x, y) {{
  const {{xMin,xMax,yMin,yMax,W,H,pad}} = umapScale;
  return [
    pad + (x - xMin) / (xMax - xMin) * (W - 2*pad),
    pad + (y - yMin) / (yMax - yMin) * (H - 2*pad)
  ];
}}

function fromCanvas(px, py) {{
  const {{xMin,xMax,yMin,yMax,W,H,pad}} = umapScale;
  return [
    xMin + (px - pad) / (W - 2*pad) * (xMax - xMin),
    yMin + (py - pad) / (H - 2*pad) * (yMax - yMin)
  ];
}}

function onUmapMouseMove(e) {{
  const rect = umapCanvas.getBoundingClientRect();
  const px = (e.clientX - rect.left) * (umapCanvas.width / rect.width);
  const py = (e.clientY - rect.top) * (umapCanvas.height / rect.height);
  const [ux, uy] = fromCanvas(px, py);

  // Find nearest point within threshold
  let best = null, bestD = 0.4;
  for (const d of DOCS) {{
    const dx = d.x - ux, dy = d.y - uy;
    const dist = Math.sqrt(dx*dx + dy*dy);
    if (dist < bestD) {{ bestD = dist; best = d; }}
  }}

  const tooltip = document.getElementById('umap-tooltip');
  if (best) {{
    const c = CLUSTERS.find(x => x.id === best.c);
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX + 14) + 'px';
    tooltip.style.top = (e.clientY - 10) + 'px';
    tooltip.innerHTML = `<b>Cluster ${{best.c}}</b>: ${{c ? c.label : ''}}<br>
      <span style="color:var(--text-dim)">${{best.s}} · ${{best.l}} tokens</span><br>
      <span style="color:var(--text-dim)">${{escHtml(best.p.slice(0,100))}}</span>`;
  }} else {{
    tooltip.style.display = 'none';
  }}
}}

function onUmapClick(e) {{
  const rect = umapCanvas.getBoundingClientRect();
  const px = (e.clientX - rect.left) * (umapCanvas.width / rect.width);
  const py = (e.clientY - rect.top) * (umapCanvas.height / rect.height);
  const [ux, uy] = fromCanvas(px, py);

  let best = null, bestD = 0.4;
  for (const d of DOCS) {{
    const dx = d.x - ux, dy = d.y - uy;
    const dist = Math.sqrt(dx*dx + dy*dy);
    if (dist < bestD) {{ bestD = dist; best = d; }}
  }}
  if (best) {{
    selectCluster(best.c);
    setView('detail');
    // Scroll sidebar to selected
    setTimeout(() => {{
      const el = document.querySelector(`.cluster-item[data-id="${{best.c}}"]`);
      if (el) el.scrollIntoView({{block:'nearest'}});
    }}, 50);
  }}
}}

function buildUmapLegend() {{
  const mode = document.querySelector('input[name="umap-color"]:checked')?.value || 'cluster';
  const leg = document.getElementById('umap-legend');
  leg.innerHTML = '';
  if (mode === 'cluster') {{
    CLUSTERS.forEach(c => {{
      const item = document.createElement('div');
      item.className = 'legend-item';
      item.innerHTML = `<div class="legend-dot" style="background:${{c.color}}"></div><div class="legend-label">${{c.id}}: ${{c.label}}</div>`;
      item.onclick = () => {{ selectCluster(c.id); setView('detail'); }};
      leg.appendChild(item);
    }});
  }} else if (mode === 'category') {{
    Object.entries(CAT_COLORS).forEach(([cat, color]) => {{
      const item = document.createElement('div');
      item.className = 'legend-item';
      item.innerHTML = `<div class="legend-dot" style="background:${{color}}"></div><div class="legend-label">${{cat}}</div>`;
      leg.appendChild(item);
    }});
  }} else {{
    Object.entries(SOURCE_COLORS).forEach(([src, color]) => {{
      const item = document.createElement('div');
      item.className = 'legend-item';
      item.innerHTML = `<div class="legend-dot" style="background:${{color}}"></div><div class="legend-label">${{src}}</div>`;
      leg.appendChild(item);
    }});
  }}
}}

function escHtml(s) {{
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

init();
</script>
</body>
</html>"""

    with open(path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
