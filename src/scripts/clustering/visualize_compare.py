"""
Single HTML comparison tool for two models' token-level clusterings.

Two tabs:
  - Clusters: two cluster lists (one per model) side by side. Click a cluster
              to see a list of documents with the cluster's tokens highlighted.
  - Documents: one doc displayed twice side-by-side, each token colored by
              its cluster assignment under the respective model.

Matches the editorial/infographic style of
scripts/ryanwang/other_figures/figure_token_cluster_comparison.py.

Usage:
    python -m src.scripts.clustering.visualize_compare \\
        --cluster-dir-1 claude_outputs/clustering/pretraining/<modmoe>/<run> \\
        --cluster-dir-2 claude_outputs/clustering/pretraining/<stdmoe>/<run> \\
        --label-1 "ModMoE" \\
        --label-2 "Standard MoE" \\
        --output claude_outputs/clustering/pretraining/comparison.html
"""

import argparse
import colorsys
import gzip
import json
import logging
import os

import numpy as np

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Editorial palette (matches figure_token_cluster_comparison.py) ---
M1_ACCENT = "#0e7c7b"  # deep teal
M1_ACCENT_SOFT = "#d3e9e8"
M1_ACCENT_DARK = "#054e4c"
M2_ACCENT = "#b45309"  # burnt amber
M2_ACCENT_SOFT = "#f5e6d0"
M2_ACCENT_DARK = "#5c2a00"

FG = "#1a202c"
FG_MUTED = "#556070"
FG_FAINT = "#9aa4b2"
DOT_DIM = "#cbd5e0"
BG_CARD = "#ffffff"
BG_PAGE = "#fafbfc"
BORDER = "#e5e8ec"
DIVIDER = "#eef0f3"


def safe_json(obj):
    return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")


def palette(n, sat=0.58, val=0.86):
    """Generate n visually distinct hex colors via evenly-spaced HSV hues."""
    out = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i / n, sat, val)
        out.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return out


def load_model(cluster_dir, data_dir=None, name="model"):
    data_dir = data_dir or os.path.dirname(os.path.normpath(cluster_dir))
    logger.info(f"[{name}] cluster-dir: {cluster_dir}")
    logger.info(f"[{name}] data-dir:    {data_dir}")

    assignments = np.load(os.path.join(cluster_dir, "assignments.npy"))
    with open(os.path.join(cluster_dir, "run_info.json")) as f:
        run_info = json.load(f)
    with open(os.path.join(cluster_dir, "summary.json")) as f:
        summary = json.load(f)
    with open(os.path.join(data_dir, "info.json")) as f:
        info = json.load(f)

    labels_path = os.path.join(cluster_dir, "cluster_labels.json")
    ext_labels = {}
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            ext_labels = json.load(f)
    else:
        logger.warning(f"[{name}] no cluster_labels.json; falling back to 'Cluster N'")

    k = run_info["k"]
    clusters = []
    for c in summary:
        cid = c["cluster"]
        ext = ext_labels.get(str(cid), {})
        clusters.append(
            {
                "id": cid,
                "label": ext.get("label", f"Cluster {cid}"),
                "category": ext.get("category", "reference"),
                "size": c["size"],
                "sourceCounts": c["source_counts"],
            }
        )

    return {
        "name": name,
        "cluster_dir": cluster_dir,
        "data_dir": data_dir,
        "assignments": assignments,
        "run_info": run_info,
        "info": info,
        "clusters": clusters,
        "k": k,
    }


def build_shared_doc_data(m1, m2, max_docs=None):
    """Decode docs once and attach per-model cluster assignments.

    Both extraction runs share the shuffle seed, so documents.npy and
    doc_boundaries.npy must match. We verify this.
    """
    logger.info("Loading documents from both data dirs...")
    documents = np.load(os.path.join(m1["data_dir"], "documents.npy"))
    boundaries = np.load(os.path.join(m1["data_dir"], "doc_boundaries.npy"))
    documents2 = np.load(os.path.join(m2["data_dir"], "documents.npy"))
    boundaries2 = np.load(os.path.join(m2["data_dir"], "doc_boundaries.npy"))
    if not np.array_equal(documents, documents2):
        raise RuntimeError(
            "documents.npy differs between the two models' data dirs — "
            "they were not extracted with the same shuffle seed."
        )
    if not np.array_equal(boundaries, boundaries2):
        raise RuntimeError("doc_boundaries.npy differs between the two models.")

    logger.info("Loading token-level metadata from both runs...")
    meta1 = []
    with gzip.open(os.path.join(m1["data_dir"], "metadata_tokens.jsonl.gz"), "rt") as f:
        for line in f:
            meta1.append(json.loads(line))
    meta2 = []
    with gzip.open(os.path.join(m2["data_dir"], "metadata_tokens.jsonl.gz"), "rt") as f:
        for line in f:
            meta2.append(json.loads(line))
    if len(meta1) != len(meta2):
        raise RuntimeError("Token metadata length differs between models.")

    # Doc-level metadata (source labels)
    meta_docs = []
    with gzip.open(os.path.join(m1["data_dir"], "metadata_docs.jsonl.gz"), "rt") as f:
        for line in f:
            meta_docs.append(json.loads(line))

    num_docs = len(boundaries) - 1
    if max_docs is not None:
        num_docs = min(num_docs, max_docs)

    # Build per-doc positional cluster assignment arrays for each model
    logger.info("Building per-doc cluster assignment arrays...")
    c1_by_doc = [[-1] * (int(boundaries[i + 1]) - int(boundaries[i])) for i in range(num_docs)]
    c2_by_doc = [[-1] * (int(boundaries[i + 1]) - int(boundaries[i])) for i in range(num_docs)]
    for i, m in enumerate(meta1):
        di = m["doc_index"]
        if di >= num_docs:
            continue
        c1_by_doc[di][m["token_position"]] = int(m1["assignments"][i])
    for i, m in enumerate(meta2):
        di = m["doc_index"]
        if di >= num_docs:
            continue
        c2_by_doc[di][m["token_position"]] = int(m2["assignments"][i])

    # Tokenizer (both models use the same dolma2 tokenizer in practice, but
    # we use model 1's path to be safe).
    from transformers import AutoTokenizer

    tokenizer_path = m1["info"].get("model_path", "")
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Decode per-doc token text
    logger.info(f"Decoding {num_docs:,} documents...")
    docs = []
    for di in range(num_docs):
        start = int(boundaries[di])
        end = int(boundaries[di + 1])
        token_ids = documents[start:end].tolist()
        tokens = [tokenizer.decode([int(tid)], skip_special_tokens=False) for tid in token_ids]
        source = meta_docs[di].get("source", "") if di < len(meta_docs) else ""
        docs.append(
            {
                "di": di,
                "s": source,
                "n": len(tokens),
                "t": tokens,
                "c1": c1_by_doc[di],
                "c2": c2_by_doc[di],
            }
        )
        if (di + 1) % 2000 == 0:
            logger.info(f"  decoded {di + 1:,}/{num_docs:,}")
    logger.info(f"  decoded {num_docs:,}/{num_docs:,}")
    return docs


# =============================================================================
# HTML template
# =============================================================================


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>__TITLE__</title>
<style>
:root {
  --fg: #1a202c;
  --fg-muted: #556070;
  --fg-faint: #9aa4b2;
  --dot-dim: #cbd5e0;
  --bg-page: #fafbfc;
  --bg-card: #ffffff;
  --border: #e5e8ec;
  --divider: #eef0f3;
  --m1-accent: __M1_ACCENT__;
  --m1-accent-soft: __M1_ACCENT_SOFT__;
  --m1-accent-dark: __M1_ACCENT_DARK__;
  --m2-accent: __M2_ACCENT__;
  --m2-accent-soft: __M2_ACCENT_SOFT__;
  --m2-accent-dark: __M2_ACCENT_DARK__;
  --mono: "SF Mono", "Monaco", "Inconsolata", "Fira Code", "Source Code Pro", monospace;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
  font-size: 14px;
  color: var(--fg);
  background: var(--bg-page);
  line-height: 1.45;
}
.page {
  max-width: 1500px;
  margin: 0 auto;
  padding: 28px 32px 64px;
}
.page-header {
  text-align: center;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--divider);
  margin-bottom: 20px;
}
.page-header h1 {
  margin: 0 0 4px;
  font-size: 26px;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: var(--fg);
}
.page-header .accent-m1 { color: var(--m1-accent); }
.page-header .accent-m2 { color: var(--m2-accent); }
.page-header .subtitle {
  margin: 0;
  color: var(--fg-muted);
  font-size: 13px;
  font-style: italic;
}

.tabs {
  display: flex;
  justify-content: center;
  gap: 4px;
  margin-bottom: 24px;
}
.tab {
  background: none;
  border: none;
  padding: 10px 24px;
  font-size: 14px;
  font-weight: 600;
  letter-spacing: 0.02em;
  color: var(--fg-muted);
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: color 0.15s, border-color 0.15s;
}
.tab:hover { color: var(--fg); }
.tab.active {
  color: var(--fg);
  border-bottom-color: var(--fg);
}
.tab-panel[hidden] { display: none; }

/* ========== Clusters tab ========== */
.clusters-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1.6fr;
  gap: 18px;
}
.model-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  min-height: 520px;
}
.model-card-header {
  padding: 12px 16px 10px;
  border-bottom: 1px solid var(--border);
}
.model-card-header .stripe {
  height: 4px;
  margin: -12px -16px 10px;
}
.model-card .name-row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
}
.model-card .name {
  font-size: 18px;
  font-weight: 700;
  letter-spacing: -0.01em;
}
.model-card .sublabel {
  font-size: 11px;
  color: var(--fg-muted);
  margin-left: 6px;
  font-weight: 500;
}
.model-card .subtitle {
  margin: 2px 0 0;
  font-size: 11.5px;
  color: var(--fg-muted);
  font-style: italic;
}
.model-card .section-label {
  font-size: 9.5px;
  font-weight: 700;
  letter-spacing: 0.08em;
  color: var(--fg-faint);
  padding: 10px 16px 4px;
  text-transform: uppercase;
}
.cluster-list {
  padding: 0 6px 8px;
  overflow-y: auto;
  max-height: 700px;
  flex: 1;
}
.cluster-item {
  display: grid;
  grid-template-columns: 10px 30px 1fr auto;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.12s;
  position: relative;
}
.cluster-item:hover { background: #f7f9fb; }
.cluster-item.selected {
  background: var(--hl-soft, var(--m1-accent-soft));
}
.cluster-item.selected::before {
  content: "";
  position: absolute;
  left: 0;
  top: 6px;
  bottom: 6px;
  width: 3px;
  border-radius: 2px;
  background: var(--hl-accent, var(--m1-accent));
}
.ci-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}
.ci-id {
  font-family: var(--mono);
  font-size: 10.5px;
  color: var(--fg-faint);
  letter-spacing: 0.02em;
}
.ci-label {
  font-size: 13px;
  color: var(--fg);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.cluster-item.selected .ci-label {
  font-weight: 700;
  color: var(--hl-dark, var(--m1-accent-dark));
}
.ci-pct {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--fg-muted);
}
.cluster-item.selected .ci-pct {
  font-weight: 700;
  color: var(--hl-dark, var(--m1-accent-dark));
}
.cluster-footer {
  font-size: 10.5px;
  font-style: italic;
  color: var(--fg-faint);
  text-align: center;
  padding: 10px 16px 12px;
  border-top: 1px solid var(--divider);
}

/* Doc panel (right column of clusters tab) */
.doc-panel {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  min-height: 520px;
}
.doc-panel-header {
  padding: 14px 20px 12px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-card);
  min-height: 60px;
}
.doc-panel-title {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 15px;
  font-weight: 700;
}
.dp-dot {
  width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0;
}
.doc-panel-sub {
  margin-top: 4px;
  font-size: 12px;
  color: var(--fg-muted);
}
.doc-panel-placeholder {
  color: var(--fg-faint);
  padding: 28px 20px;
  font-size: 13px;
  font-style: italic;
  text-align: center;
}
.doc-panel-body {
  padding: 14px 16px 18px;
  overflow-y: auto;
  max-height: 720px;
  flex: 1;
}
.doc-card {
  background: #fbfcfd;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 14px 12px;
  margin-bottom: 10px;
  cursor: pointer;
  transition: border-color 0.12s, box-shadow 0.12s;
}
.doc-card:hover {
  border-color: var(--hl-accent, var(--m1-accent));
  box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.doc-card-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
  font-size: 11px;
  color: var(--fg-muted);
}
.doc-card-header .doc-id {
  font-family: var(--mono);
}
.doc-card-header .source-badge {
  padding: 2px 7px;
  border-radius: 9px;
  font-size: 9.5px;
  font-weight: 600;
  letter-spacing: 0.02em;
  color: #fff;
  background: #94a3b8;
}
.doc-card-header .source-dclm { background: #4A90E2; }
.doc-card-header .source-starcoder { background: #27AE60; }
.doc-card-header .source-pes2o { background: #E67E22; }
.doc-card-header .source-proofpile-2-arxiv { background: #9B59B6; }
.doc-card-header .source-proofpile-2-open-web-math { background: #F39C12; }
.doc-card-header .source-proofpile-2-stack { background: #1ABC9C; }
.doc-card-header .source-wikipedia { background: #64748b; }
.doc-card-header .doc-pct {
  margin-left: auto;
  font-family: var(--mono);
  font-weight: 700;
}
.doc-body {
  font-family: var(--mono);
  font-size: 12.5px;
  line-height: 1.75;
  white-space: pre-wrap;
  word-break: break-word;
  color: #b4bac4;
  max-height: 260px;
  overflow-y: auto;
}
.doc-body .hl {
  padding: 1px 2px;
  border-radius: 3px;
  font-weight: 700;
  color: var(--hl-dark);
  background: var(--hl-soft);
  box-shadow: inset 0 -1px 0 var(--hl-accent);
}
.load-more {
  display: block;
  width: 100%;
  background: var(--hl-soft, #eef2f6);
  border: 1px dashed var(--hl-accent, var(--border));
  color: var(--hl-dark, var(--fg-muted));
  padding: 10px 16px;
  border-radius: 8px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  margin-top: 4px;
  transition: background 0.12s, border-color 0.12s;
}
.load-more:hover {
  background: var(--hl-accent);
  border-style: solid;
  color: #fff;
}
.load-more-done {
  padding: 12px 10px 6px;
  font-size: 11.5px;
  color: var(--fg-faint);
  font-style: italic;
  text-align: center;
}

/* ========== Documents tab ========== */
.docs-controls {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  margin-bottom: 14px;
}
.docs-controls label {
  color: var(--fg-muted);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.docs-controls .doc-nav-btn {
  background: var(--bg-page);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 5px 12px;
  font-size: 13px;
  cursor: pointer;
  font-weight: 500;
}
.docs-controls .doc-nav-btn:hover { background: #f0f2f5; }
.docs-controls input[type="number"] {
  width: 100px;
  padding: 5px 8px;
  border: 1px solid var(--border);
  border-radius: 6px;
  font-family: var(--mono);
  font-size: 13px;
}
.docs-controls select {
  padding: 5px 8px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg-card);
  font-size: 13px;
}
.docs-controls .doc-meta {
  margin-left: auto;
  font-size: 12px;
  color: var(--fg-muted);
  font-family: var(--mono);
}
.docs-compare-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
}
.doc-view {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.doc-view-header {
  padding: 10px 16px;
  color: #fff;
  font-weight: 700;
  font-size: 15px;
  letter-spacing: -0.005em;
}
.doc-view-header .inline-sublabel {
  margin-left: 8px;
  font-weight: 500;
  font-size: 11px;
  opacity: 0.85;
}
.doc-view-text {
  padding: 16px 20px 18px;
  font-family: var(--mono);
  font-size: 13px;
  line-height: 2.0;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--fg);
  max-height: 780px;
  overflow-y: auto;
}
.doc-view-text .tok {
  padding: 1px 0;
  border-radius: 2px;
  transition: outline 0.1s;
}
.doc-view-text .tok.dimmed { opacity: 0.25; }
.doc-view-text .tok.focused { outline: 1.5px solid #000; outline-offset: 0; }

.legends {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
  margin-top: 14px;
}
.legend-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px 14px 12px;
}
.legend-card-header {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.06em;
  color: var(--fg-faint);
  text-transform: uppercase;
  margin-bottom: 8px;
}
.legend-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}
.legend-chip {
  padding: 3px 9px;
  border-radius: 999px;
  font-size: 11px;
  border: 1px solid;
  cursor: pointer;
  user-select: none;
  transition: transform 0.08s, box-shadow 0.08s;
}
.legend-chip:hover { transform: translateY(-1px); }
.legend-chip.active { box-shadow: 0 0 0 2px rgba(0,0,0,0.55); }
.legend-chip .lc-id {
  font-family: var(--mono);
  font-size: 10px;
  opacity: 0.8;
  margin-right: 4px;
}
.legend-chip .lc-count {
  font-family: var(--mono);
  font-size: 10px;
  opacity: 0.75;
  margin-left: 5px;
}
.legend-clear {
  font-size: 10.5px;
  color: var(--fg-faint);
  margin-left: 6px;
  background: none;
  border: none;
  cursor: pointer;
  text-decoration: underline;
}

.token-tooltip {
  position: fixed;
  background: #1a202c;
  color: #fff;
  padding: 6px 10px;
  border-radius: 5px;
  font-size: 11.5px;
  font-family: var(--mono);
  pointer-events: none;
  z-index: 1000;
  display: none;
  max-width: 320px;
  line-height: 1.5;
}
.token-tooltip .tt-label {
  font-family: -apple-system, BlinkMacSystemFont, sans-serif;
  font-weight: 600;
  margin-bottom: 2px;
}

/* Scrollbars (webkit) */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #d8dde3; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #b8c0c8; }
</style>
</head>
<body>
<div class="page">

  <header class="page-header">
    <h1><span class="accent-m1">__M1_NAME__</span> vs <span class="accent-m2">__M2_NAME__</span></h1>
    <p class="subtitle">Token-level router cluster comparison &middot; k=__K1__ / k=__K2__ &middot; __N_DOCS__ documents, __N_TOKENS__ tokens</p>
  </header>

  <nav class="tabs">
    <button class="tab active" data-tab="clusters">Clusters</button>
    <button class="tab" data-tab="documents">Documents</button>
  </nav>

  <main>
    <section id="tab-clusters" class="tab-panel">
      <div class="clusters-grid">
        <div class="model-card">
          <div class="model-card-header">
            <div class="stripe" style="background: var(--m1-accent);"></div>
            <div class="name-row">
              <div>
                <span class="name">__M1_NAME__</span>
                <span class="sublabel">__M1_SUBLABEL__</span>
              </div>
            </div>
            <p class="subtitle">__M1_SUBTITLE__</p>
          </div>
          <div class="section-label">Clusters &middot; sorted by token count</div>
          <div class="cluster-list" id="list-m1"></div>
          <div class="cluster-footer" id="footer-m1"></div>
        </div>
        <div class="model-card">
          <div class="model-card-header">
            <div class="stripe" style="background: var(--m2-accent);"></div>
            <div class="name-row">
              <div>
                <span class="name">__M2_NAME__</span>
                <span class="sublabel">__M2_SUBLABEL__</span>
              </div>
            </div>
            <p class="subtitle">__M2_SUBTITLE__</p>
          </div>
          <div class="section-label">Clusters &middot; sorted by token count</div>
          <div class="cluster-list" id="list-m2"></div>
          <div class="cluster-footer" id="footer-m2"></div>
        </div>
        <div class="doc-panel" id="doc-panel">
          <div class="doc-panel-header" id="doc-panel-header">
            <div class="doc-panel-placeholder">
              Click a cluster on the left to see documents with that cluster's tokens highlighted.
            </div>
          </div>
          <div class="doc-panel-body" id="doc-panel-body"></div>
        </div>
      </div>
    </section>

    <section id="tab-documents" class="tab-panel" hidden>
      <div class="docs-controls">
        <label>Doc</label>
        <input id="doc-input" type="number" min="0" />
        <button class="doc-nav-btn" id="doc-prev">&lsaquo; Prev</button>
        <button class="doc-nav-btn" id="doc-next">Next &rsaquo;</button>
        <label style="margin-left:20px">Source</label>
        <select id="src-filter"></select>
        <label style="margin-left:8px">Jump to</label>
        <select id="doc-select"></select>
        <span class="doc-meta" id="doc-meta"></span>
      </div>
      <div class="docs-compare-grid">
        <div class="doc-view">
          <div class="doc-view-header" style="background: var(--m1-accent);">
            __M1_NAME__<span class="inline-sublabel">__M1_SUBLABEL__</span>
          </div>
          <div class="doc-view-text" id="doc-text-m1"></div>
        </div>
        <div class="doc-view">
          <div class="doc-view-header" style="background: var(--m2-accent);">
            __M2_NAME__<span class="inline-sublabel">__M2_SUBLABEL__</span>
          </div>
          <div class="doc-view-text" id="doc-text-m2"></div>
        </div>
      </div>
      <div class="legends">
        <div class="legend-card">
          <div class="legend-card-header">
            __M1_NAME__ &middot; clusters in this doc
            <button class="legend-clear" id="legend-clear-m1">clear filter</button>
          </div>
          <div class="legend-chips" id="legend-m1"></div>
        </div>
        <div class="legend-card">
          <div class="legend-card-header">
            __M2_NAME__ &middot; clusters in this doc
            <button class="legend-clear" id="legend-clear-m2">clear filter</button>
          </div>
          <div class="legend-chips" id="legend-m2"></div>
        </div>
      </div>
    </section>
  </main>

</div>

<div class="token-tooltip" id="tok-tooltip"></div>

<script>
const M1 = __M1_JSON__;
const M2 = __M2_JSON__;
const DOCS = __DOCS_JSON__;

// Lookup by di (docs are index-aligned)
const DOC_BY_ID = {};
for (const d of DOCS) DOC_BY_ID[d.di] = d;

// Cache of per-cluster doc lists (computed on demand). Keys: `${model}:${cid}`.
const CLUSTER_DOCS_CACHE = {};

function docsForCluster(model, cid) {
  const key = `${model}:${cid}`;
  if (CLUSTER_DOCS_CACHE[key]) return CLUSTER_DOCS_CACHE[key];
  const assignKey = model === 1 ? "c1" : "c2";
  const out = [];
  for (const d of DOCS) {
    let count = 0;
    const cArr = d[assignKey];
    for (let i = 0; i < cArr.length; i++) {
      if (cArr[i] === cid) count++;
    }
    if (count > 0) {
      out.push({di: d.di, c: count, n: d.n, p: count / d.n});
    }
  }
  out.sort((a, b) => (b.p - a.p) || (b.c - a.c) || (a.di - b.di));
  CLUSTER_DOCS_CACHE[key] = out;
  return out;
}

// Cluster lookup
const M1_BY_ID = {}; for (const c of M1.clusters) M1_BY_ID[c.id] = c;
const M2_BY_ID = {}; for (const c of M2.clusters) M2_BY_ID[c.id] = c;

const DOC_PAGE_SIZE = 30;
const STATE = {
  tab: "clusters",
  selectedModel: null,
  selectedClusterId: null,
  docIdx: DOCS.length ? DOCS[0].di : 0,
  legendFilter: {1: null, 2: null},
  docsShown: 0,  // how many docs currently rendered in the cluster doc-panel
};

// -------- Utilities --------
function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
// Newlines render as zero-width in pre-wrap, making their colored highlight
// invisible. Prepend non-breaking spaces so the colored span has visible
// width (like space tokens naturally do); the real newline still breaks the
// line after.
function tokenDisplay(t) {
  return /\n/.test(t) ? "  " + t : t;
}
function pad2(n) { return n.toString().padStart(2, "0"); }
function rgbaFromHex(hex, alpha) {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0,2), 16);
  const g = parseInt(h.slice(2,4), 16);
  const b = parseInt(h.slice(4,6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}
function sourceClass(s) {
  return "source-" + String(s).replace(/[^a-z0-9]/gi, "-").toLowerCase();
}

// -------- Cluster lists --------
function renderClusterLists() {
  renderClusterList(1, M1, "list-m1", "footer-m1");
  renderClusterList(2, M2, "list-m2", "footer-m2");
}

function renderClusterList(model, spec, listId, footerId) {
  const total = spec.clusters.reduce((s, c) => s + c.size, 0);
  const sorted = [...spec.clusters].sort((a, b) => b.size - a.size);
  const accent = model === 1 ? spec.accent : spec.accent;
  const parts = sorted.map(c => {
    const pct = (c.size / total * 100);
    return `<div class="cluster-item" data-model="${model}" data-cid="${c.id}">
      <span class="ci-dot" style="background:${c.color}"></span>
      <span class="ci-id">C${pad2(c.id)}</span>
      <span class="ci-label">${esc(c.label)}</span>
      <span class="ci-pct">${pct.toFixed(1)}%</span>
    </div>`;
  });
  const list = document.getElementById(listId);
  list.innerHTML = parts.join("");
  list.querySelectorAll(".cluster-item").forEach(el => {
    el.addEventListener("click", () => {
      selectCluster(parseInt(el.dataset.model), parseInt(el.dataset.cid));
    });
  });
  document.getElementById(footerId).textContent =
    `${spec.clusters.length} clusters · ${total.toLocaleString()} tokens`;
}

function selectCluster(model, cid) {
  STATE.selectedModel = model;
  STATE.selectedClusterId = cid;
  const spec = model === 1 ? M1 : M2;
  const cluster = (model === 1 ? M1_BY_ID : M2_BY_ID)[cid];
  const panel = document.getElementById("doc-panel");
  panel.style.setProperty("--hl-accent", spec.accent);
  panel.style.setProperty("--hl-soft", spec.accentSoft);
  panel.style.setProperty("--hl-dark", spec.accentDark);

  // Mark selection on both cluster lists
  document.querySelectorAll(".cluster-item").forEach(el => {
    const m = parseInt(el.dataset.model);
    const c = parseInt(el.dataset.cid);
    const isSel = m === model && c === cid;
    el.classList.toggle("selected", isSel);
    if (isSel) {
      el.style.setProperty("--hl-accent", spec.accent);
      el.style.setProperty("--hl-soft", spec.accentSoft);
      el.style.setProperty("--hl-dark", spec.accentDark);
    } else {
      el.style.removeProperty("--hl-accent");
      el.style.removeProperty("--hl-soft");
      el.style.removeProperty("--hl-dark");
    }
  });

  // Compute all matching docs on the fly (cached after first call)
  const matches = docsForCluster(model, cid);
  STATE.docsShown = 0;
  const header = document.getElementById("doc-panel-header");
  header.innerHTML = `
    <div class="doc-panel-title">
      <span class="dp-dot" style="background:${cluster.color}"></span>
      <span>${esc(spec.name)} &middot; C${pad2(cid)} — ${esc(cluster.label)}</span>
    </div>
    <div class="doc-panel-sub" id="doc-panel-sub">
      ${cluster.size.toLocaleString()} tokens &middot; ${matches.length.toLocaleString()} documents contain this cluster
    </div>`;
  document.getElementById("doc-panel-body").innerHTML = "";
  renderMoreClusterDocs(model, cid, matches, spec);
}

function renderMoreClusterDocs(model, cid, matches, spec) {
  const body = document.getElementById("doc-panel-body");
  const start = STATE.docsShown;
  const end = Math.min(matches.length, start + DOC_PAGE_SIZE);
  const slice = matches.slice(start, end);

  // Remove any existing load-more button
  const oldBtn = body.querySelector(".load-more");
  if (oldBtn) oldBtn.remove();

  const newHtml = slice.map(entry => renderDocCard(entry, model, cid, spec)).join("");
  body.insertAdjacentHTML("beforeend", newHtml);
  // Wire up the newly-added doc cards
  body.querySelectorAll(".doc-card:not([data-wired])").forEach(el => {
    el.setAttribute("data-wired", "1");
    el.addEventListener("click", () => {
      openDocInDocsTab(parseInt(el.dataset.di));
    });
  });

  STATE.docsShown = end;

  // Update sub-header
  const sub = document.getElementById("doc-panel-sub");
  if (sub) {
    const cluster = (model === 1 ? M1_BY_ID : M2_BY_ID)[cid];
    sub.innerHTML = `${cluster.size.toLocaleString()} tokens &middot; `
      + `showing ${end.toLocaleString()} of ${matches.length.toLocaleString()} documents &middot; `
      + `ranked by cluster density`;
  }

  // Add load-more button if more remain
  if (end < matches.length) {
    const btn = document.createElement("button");
    btn.className = "load-more";
    btn.textContent = `Load ${Math.min(DOC_PAGE_SIZE, matches.length - end)} more · ${(matches.length - end).toLocaleString()} remaining`;
    btn.addEventListener("click", () => renderMoreClusterDocs(model, cid, matches, spec));
    body.appendChild(btn);
  } else if (matches.length > DOC_PAGE_SIZE) {
    const done = document.createElement("div");
    done.className = "load-more-done";
    done.textContent = `All ${matches.length.toLocaleString()} documents shown.`;
    body.appendChild(done);
  }
}

function renderDocCard(entry, model, cid, spec) {
  const d = DOC_BY_ID[entry.di];
  if (!d) return "";
  const cArr = model === 1 ? d.c1 : d.c2;
  let body = "";
  for (let i = 0; i < d.t.length; i++) {
    if (cArr[i] === cid) {
      body += `<span class="hl">${esc(tokenDisplay(d.t[i]))}</span>`;
    } else {
      body += esc(d.t[i]);
    }
  }
  return `<div class="doc-card" data-di="${d.di}">
    <div class="doc-card-header">
      <span class="doc-id">Doc #${d.di}</span>
      <span class="source-badge ${sourceClass(d.s)}">${esc(d.s)}</span>
      <span class="doc-pct">${(entry.p * 100).toFixed(0)}% · ${entry.c}/${entry.n} tokens</span>
    </div>
    <div class="doc-body">${body}</div>
  </div>`;
}

// -------- Documents tab --------
function populateDocSelect() {
  const sel = document.getElementById("doc-select");
  // Full dropdown — every doc is accessible. DOCS is typically <15k, which
  // browsers handle without issue.
  const opts = DOCS.map(d => `<option value="${d.di}">#${d.di} (${esc(d.s)})</option>`);
  sel.innerHTML = opts.join("");
  sel.addEventListener("change", () => showDoc(parseInt(sel.value)));

  // Source filter: narrow the dropdown to a single source
  const srcSel = document.getElementById("src-filter");
  if (srcSel) {
    const sources = Array.from(new Set(DOCS.map(d => d.s))).sort();
    srcSel.innerHTML = `<option value="">all sources</option>`
      + sources.map(s => `<option value="${esc(s)}">${esc(s)}</option>`).join("");
    srcSel.addEventListener("change", () => {
      const v = srcSel.value;
      const filtered = v ? DOCS.filter(d => d.s === v) : DOCS;
      sel.innerHTML = filtered
        .map(d => `<option value="${d.di}">#${d.di} (${esc(d.s)})</option>`)
        .join("");
      // Keep the currently-selected doc in the dropdown if still there
      if (filtered.some(d => d.di === STATE.docIdx)) {
        sel.value = String(STATE.docIdx);
      } else if (filtered.length) {
        showDoc(filtered[0].di);
      }
    });
  }

  document.getElementById("doc-input").addEventListener("change", e => {
    const v = parseInt(e.target.value);
    if (DOC_BY_ID[v]) showDoc(v);
  });
  document.getElementById("doc-prev").addEventListener("click", () => stepDoc(-1));
  document.getElementById("doc-next").addEventListener("click", () => stepDoc(+1));
  document.getElementById("legend-clear-m1").addEventListener("click", () => {
    STATE.legendFilter[1] = null; renderDocViews();
  });
  document.getElementById("legend-clear-m2").addEventListener("click", () => {
    STATE.legendFilter[2] = null; renderDocViews();
  });
}
function stepDoc(dir) {
  const keys = Object.keys(DOC_BY_ID).map(x => parseInt(x)).sort((a, b) => a - b);
  const idx = keys.indexOf(STATE.docIdx);
  let next = idx + dir;
  if (next < 0) next = 0;
  if (next >= keys.length) next = keys.length - 1;
  showDoc(keys[next]);
}
function showDoc(docIdx) {
  if (!DOC_BY_ID[docIdx]) return;
  STATE.docIdx = docIdx;
  STATE.legendFilter = {1: null, 2: null};
  document.getElementById("doc-input").value = docIdx;
  const sel = document.getElementById("doc-select");
  if (sel.value !== String(docIdx)) sel.value = String(docIdx);
  const d = DOC_BY_ID[docIdx];
  document.getElementById("doc-meta").textContent =
    `source: ${d.s} · ${d.n} tokens`;
  renderDocViews();
}
function renderDocViews() {
  const d = DOC_BY_ID[STATE.docIdx];
  if (!d) return;
  renderDocView("doc-text-m1", d, 1, M1, M1_BY_ID);
  renderDocView("doc-text-m2", d, 2, M2, M2_BY_ID);
  renderLegend("legend-m1", d, 1, M1, M1_BY_ID);
  renderLegend("legend-m2", d, 2, M2, M2_BY_ID);
}
function renderDocView(containerId, d, model, spec, byId) {
  const cArr = model === 1 ? d.c1 : d.c2;
  const filter = STATE.legendFilter[model];
  const parts = [];
  for (let i = 0; i < d.t.length; i++) {
    const cid = cArr[i];
    const c = byId[cid];
    const color = c ? c.color : "#cbd5e0";
    const bg = rgbaFromHex(color, 0.32);
    const dimmed = filter !== null && cid !== filter;
    const cls = "tok" + (dimmed ? " dimmed" : "");
    const style = dimmed
      ? ""
      : `background:${bg};box-shadow:inset 0 -2px 0 ${color};`;
    const label = c ? `C${pad2(cid)} · ${c.label}` : `Cluster ${cid}`;
    parts.push(
      `<span class="${cls}" data-cid="${cid}" data-label="${esc(label)}" style="${style}">${esc(tokenDisplay(d.t[i]))}</span>`
    );
  }
  const el = document.getElementById(containerId);
  el.innerHTML = parts.join("");
  el.querySelectorAll(".tok").forEach(tok => {
    tok.addEventListener("mouseenter", e => showTooltip(e, tok.dataset.label));
    tok.addEventListener("mousemove", moveTooltip);
    tok.addEventListener("mouseleave", hideTooltip);
    tok.addEventListener("click", e => {
      e.stopPropagation();
      const cid = parseInt(tok.dataset.cid);
      const cur = STATE.legendFilter[model];
      STATE.legendFilter[model] = cur === cid ? null : cid;
      renderDocViews();
    });
  });
}
function renderLegend(containerId, d, model, spec, byId) {
  const cArr = model === 1 ? d.c1 : d.c2;
  const counts = {};
  for (const cid of cArr) counts[cid] = (counts[cid] || 0) + 1;
  const entries = Object.entries(counts)
    .map(([cid, cnt]) => [parseInt(cid), cnt])
    .sort((a, b) => b[1] - a[1]);
  const filter = STATE.legendFilter[model];
  const html = entries.map(([cid, cnt]) => {
    const c = byId[cid];
    const color = c ? c.color : "#cbd5e0";
    const label = c ? c.label : `Cluster ${cid}`;
    const pct = (cnt / d.n * 100).toFixed(0);
    const active = filter === cid;
    return `<span class="legend-chip${active ? " active" : ""}"
        data-model="${model}" data-cid="${cid}"
        style="background:${rgbaFromHex(color, 0.18)};color:${c ? c.color : "#555"};border-color:${rgbaFromHex(color, 0.55)}">
      <span class="lc-id">C${pad2(cid)}</span>${esc(label)}
      <span class="lc-count">${cnt} · ${pct}%</span>
    </span>`;
  }).join("");
  const el = document.getElementById(containerId);
  el.innerHTML = html;
  el.querySelectorAll(".legend-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const cid = parseInt(chip.dataset.cid);
      const cur = STATE.legendFilter[model];
      STATE.legendFilter[model] = cur === cid ? null : cid;
      renderDocViews();
    });
  });
}

// -------- Tabs --------
function switchTab(name) {
  STATE.tab = name;
  document.querySelectorAll(".tab").forEach(t => {
    t.classList.toggle("active", t.dataset.tab === name);
  });
  document.querySelectorAll(".tab-panel").forEach(p => {
    p.hidden = (p.id !== `tab-${name}`);
  });
}
document.querySelectorAll(".tab").forEach(t => {
  t.addEventListener("click", () => switchTab(t.dataset.tab));
});
function openDocInDocsTab(di) {
  switchTab("documents");
  showDoc(di);
}

// -------- Tooltip --------
function showTooltip(e, label) {
  const tip = document.getElementById("tok-tooltip");
  tip.textContent = label;
  tip.style.display = "block";
  moveTooltip(e);
}
function moveTooltip(e) {
  const tip = document.getElementById("tok-tooltip");
  tip.style.left = (e.clientX + 14) + "px";
  tip.style.top = (e.clientY + 14) + "px";
}
function hideTooltip() {
  document.getElementById("tok-tooltip").style.display = "none";
}

// -------- Init --------
renderClusterLists();
populateDocSelect();
showDoc(STATE.docIdx);
</script>
</body>
</html>
"""


def build_html(
    m1,
    m2,
    docs,
    label_1,
    label_2,
    sublabel_1,
    sublabel_2,
    subtitle_1,
    subtitle_2,
    out_path,
):
    pal1 = palette(m1["k"])
    pal2 = palette(m2["k"])

    m1_js = {
        "name": label_1,
        "accent": M1_ACCENT,
        "accentSoft": M1_ACCENT_SOFT,
        "accentDark": M1_ACCENT_DARK,
        "clusters": [
            {
                "id": c["id"],
                "label": c["label"],
                "category": c["category"],
                "size": c["size"],
                "color": pal1[c["id"]],
            }
            for c in m1["clusters"]
        ],
        "k": m1["k"],
    }
    m2_js = {
        "name": label_2,
        "accent": M2_ACCENT,
        "accentSoft": M2_ACCENT_SOFT,
        "accentDark": M2_ACCENT_DARK,
        "clusters": [
            {
                "id": c["id"],
                "label": c["label"],
                "category": c["category"],
                "size": c["size"],
                "color": pal2[c["id"]],
            }
            for c in m2["clusters"]
        ],
        "k": m2["k"],
    }

    total_tokens = sum(d["n"] for d in docs)
    repls = {
        "__TITLE__": f"{label_1} vs {label_2} — Cluster Comparison",
        "__M1_NAME__": label_1,
        "__M2_NAME__": label_2,
        "__M1_SUBLABEL__": sublabel_1,
        "__M2_SUBLABEL__": sublabel_2,
        "__M1_SUBTITLE__": subtitle_1,
        "__M2_SUBTITLE__": subtitle_2,
        "__M1_ACCENT__": M1_ACCENT,
        "__M1_ACCENT_SOFT__": M1_ACCENT_SOFT,
        "__M1_ACCENT_DARK__": M1_ACCENT_DARK,
        "__M2_ACCENT__": M2_ACCENT,
        "__M2_ACCENT_SOFT__": M2_ACCENT_SOFT,
        "__M2_ACCENT_DARK__": M2_ACCENT_DARK,
        "__K1__": str(m1["k"]),
        "__K2__": str(m2["k"]),
        "__N_DOCS__": f"{len(docs):,}",
        "__N_TOKENS__": f"{total_tokens:,}",
        "__M1_JSON__": safe_json(m1_js),
        "__M2_JSON__": safe_json(m2_js),
        "__DOCS_JSON__": safe_json(docs),
    }
    html = HTML_TEMPLATE
    for k, v in repls.items():
        html = html.replace(k, v)

    with open(out_path, "w") as f:
        f.write(html)
    size_mb = os.path.getsize(out_path) / 1e6
    logger.info(f"Wrote {out_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Single-HTML side-by-side comparison of two models' token clusterings"
    )
    parser.add_argument("--cluster-dir-1", required=True)
    parser.add_argument("--cluster-dir-2", required=True)
    parser.add_argument("--data-dir-1", default=None)
    parser.add_argument("--data-dir-2", default=None)
    parser.add_argument("--label-1", default="Model 1")
    parser.add_argument("--label-2", default="Model 2")
    parser.add_argument("--sublabel-1", default="")
    parser.add_argument("--sublabel-2", default="")
    parser.add_argument(
        "--subtitle-1",
        default="learns topical / semantic clusters",
        help="Italic tagline under the model header",
    )
    parser.add_argument(
        "--subtitle-2",
        default="learns syntactic / function-word clusters",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Cap number of docs (for smaller output).",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    m1 = load_model(args.cluster_dir_1, args.data_dir_1, args.label_1)
    m2 = load_model(args.cluster_dir_2, args.data_dir_2, args.label_2)

    docs = build_shared_doc_data(m1, m2, max_docs=args.max_docs)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    build_html(
        m1,
        m2,
        docs,
        args.label_1,
        args.label_2,
        args.sublabel_1,
        args.sublabel_2,
        args.subtitle_1,
        args.subtitle_2,
        args.output,
    )


if __name__ == "__main__":
    main()
