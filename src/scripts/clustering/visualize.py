"""
Generate interactive HTML cluster explorer.

Auto-detects token-level vs document-level from run_info.json.
Token-level shows each token highlighted in its document context.
Document-level shows document previews.

Usage:
    python -m src.scripts.clustering.visualize \\
        --cluster-dir .../probs_mean_pca_l2_spherical_kmeans_k64

    # With explicit data dir
    python -m src.scripts.clustering.visualize \\
        --cluster-dir .../probs_mean_pca_l2_spherical_kmeans_k64 \\
        --data-dir .../<model>/
"""

import argparse
import gzip
import json
import logging
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

CATEGORY_COLORS = {
    "code": "#4A90E2",
    "science": "#27AE60",
    "news": "#E67E22",
    "personal": "#E91E8C",
    "business": "#9B59B6",
    "health": "#E74C3C",
    "arts": "#F39C12",
    "education": "#1ABC9C",
    "reference": "#7F8C8D",
    "spam": "#BDC3C7",
}


def compute_umap(emb, cache_path, max_points=50000):
    """Compute or load cached UMAP 2D projection."""
    if os.path.exists(cache_path):
        logger.info("Loading cached UMAP coords...")
        return np.load(cache_path)

    import umap as umap_lib

    n = emb.shape[0]
    if n > max_points:
        logger.info(f"Subsampling {max_points}/{n} for UMAP...")
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_points, replace=False)
        idx.sort()
        emb_sub = emb[idx]
    else:
        idx = None
        emb_sub = emb

    logger.info(f"PCA-50 on {len(emb_sub)} points...")
    n_components = min(50, emb_sub.shape[0], emb_sub.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    reduced = normalize(pca.fit_transform(emb_sub), norm="l2")

    logger.info("Running UMAP...")
    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
        verbose=False,
    )
    coords = reducer.fit_transform(reduced)
    np.save(cache_path, coords)
    logger.info(f"Saved UMAP coords -> {cache_path}")
    return coords


def safe_json(obj):
    """JSON serialize with </script> escaping."""
    return json.dumps(obj).replace("</", "<\\/")


def build_token_data(cluster_dir, data_dir, run_info, info):
    """Build JS-ready data for token-level visualization."""
    k = run_info["k"]
    emb_name = run_info["embedding"]
    emb_file = os.path.join(data_dir, f"embeddings_{emb_name}.npy")

    emb = np.load(emb_file).astype(np.float32)
    labels = np.load(os.path.join(cluster_dir, "assignments.npy"))

    logger.info("Loading token metadata...")
    meta = []
    with gzip.open(os.path.join(data_dir, "metadata_tokens.jsonl.gz"), "rt") as f:
        for line in f:
            meta.append(json.loads(line))

    documents = np.load(os.path.join(data_dir, "documents.npy"))
    boundaries = np.load(os.path.join(data_dir, "doc_boundaries.npy"))

    summary = []
    summary_path = os.path.join(cluster_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    # Load tokenizer
    model_path = info.get("model_path", "")
    logger.info(f"Loading tokenizer from {model_path}...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    ctx_win = 10

    def get_token_context(token_idx):
        m = meta[token_idx]
        doc_idx = m["doc_index"]
        pos = m["token_position"]
        doc_start = int(boundaries[doc_idx])
        doc_end = int(boundaries[doc_idx + 1])
        doc_tokens = documents[doc_start:doc_end]
        ctx_start = max(0, pos - ctx_win)
        ctx_end = min(len(doc_tokens), pos + ctx_win + 1)
        before = (
            tokenizer.decode(doc_tokens[ctx_start:pos].tolist(), skip_special_tokens=True)
            if pos > ctx_start
            else ""
        )
        target = tokenizer.decode(doc_tokens[pos : pos + 1].tolist(), skip_special_tokens=True)
        after = (
            tokenizer.decode(doc_tokens[pos + 1 : ctx_end].tolist(), skip_special_tokens=True)
            if pos + 1 < ctx_end
            else ""
        )
        prefix = "\u2026" if ctx_start > 0 else ""
        suffix = "\u2026" if ctx_end < len(doc_tokens) else ""
        return prefix + before, target, after + suffix

    # UMAP
    umap_path = os.path.join(cluster_dir, "umap_coords.npy")
    n_total = emb.shape[0]
    max_umap = 50000
    if n_total > max_umap:
        rng = np.random.RandomState(42)
        umap_idx = np.sort(rng.choice(n_total, max_umap, replace=False))
    else:
        umap_idx = np.arange(n_total)

    coords = compute_umap(emb[umap_idx], umap_path)

    # Build per-token data
    logger.info(f"Building data for {len(umap_idx)} tokens...")
    docs_js = []
    for vi, token_idx in enumerate(umap_idx):
        m = meta[token_idx]
        before, target, after = get_token_context(token_idx)
        docs_js.append(
            {
                "i": int(token_idx),
                "c": int(labels[token_idx]),
                "s": m["source"],
                "di": m["doc_index"],
                "before": before,
                "target": target,
                "after": after,
                "x": round(float(coords[vi][0]), 3),
                "y": round(float(coords[vi][1]), 3),
            }
        )

    # Cluster summaries
    external_labels = {}
    labels_path = os.path.join(cluster_dir, "cluster_labels.json")
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            external_labels = json.load(f)

    clusters_js = []
    for c in summary:
        cid = c["cluster"]
        cid_str = str(cid)
        if cid_str in external_labels:
            label = external_labels[cid_str]["label"]
            cat = external_labels[cid_str].get("category", "reference")
        else:
            label, cat = f"Cluster {cid}", "reference"
        if cat not in CATEGORY_COLORS:
            cat = "reference"

        rep_tokens = []
        for rd in c.get("representative_samples", c.get("representative_docs", [])):
            idx = rd.get("idx")
            if idx is not None and idx < len(meta):
                before, target, after = get_token_context(idx)
            else:
                before, target, after = "", "(unknown)", ""
            rep_tokens.append(
                {
                    "source": rd.get("source", ""),
                    "before": before,
                    "target": target,
                    "after": after,
                }
            )

        clusters_js.append(
            {
                "id": cid,
                "label": label,
                "category": cat,
                "color": CATEGORY_COLORS[cat],
                "size": c["size"],
                "source_counts": c.get("source_counts", {}),
                "top_experts": c.get("top10_experts_global", []),
                "rep_samples": rep_tokens,
            }
        )

    return clusters_js, docs_js, k, "token"


def build_doc_data(cluster_dir, data_dir, run_info, info):
    """Build JS-ready data for document-level visualization."""
    k = run_info["k"]
    emb_name = run_info["embedding"]
    emb_file = os.path.join(data_dir, f"embeddings_{emb_name}.npy")

    emb = np.load(emb_file).astype(np.float32)
    labels = np.load(os.path.join(cluster_dir, "assignments.npy"))

    # Try doc-level metadata first, fall back to generic
    meta_path = os.path.join(data_dir, "metadata_docs.jsonl.gz")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(data_dir, "metadata.jsonl.gz")

    meta = []
    with gzip.open(meta_path, "rt") as f:
        for line in f:
            meta.append(json.loads(line))

    summary = []
    summary_path = os.path.join(cluster_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    # UMAP
    umap_path = os.path.join(cluster_dir, "umap_coords.npy")
    coords = compute_umap(emb, umap_path)

    # Build per-doc data
    docs_js = []
    for i in range(len(emb)):
        m = meta[i]
        preview = m.get("preview", "")[:500]
        docs_js.append(
            {
                "i": i,
                "c": int(labels[i]),
                "s": m["source"],
                "preview": preview,
                "doc_len": m.get("doc_len", 0),
                "x": round(float(coords[i][0]), 3),
                "y": round(float(coords[i][1]), 3),
            }
        )

    # Cluster summaries
    external_labels = {}
    labels_path = os.path.join(cluster_dir, "cluster_labels.json")
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            external_labels = json.load(f)

    clusters_js = []
    for c in summary:
        cid = c["cluster"]
        cid_str = str(cid)
        if cid_str in external_labels:
            label = external_labels[cid_str]["label"]
            cat = external_labels[cid_str].get("category", "reference")
        else:
            label, cat = f"Cluster {cid}", "reference"
        if cat not in CATEGORY_COLORS:
            cat = "reference"

        rep_docs = []
        for rd in c.get("representative_samples", c.get("representative_docs", [])):
            rep_docs.append(
                {
                    "source": rd.get("source", ""),
                    "preview": rd.get("preview", "")[:500],
                    "doc_len": rd.get("doc_len", 0),
                }
            )

        clusters_js.append(
            {
                "id": cid,
                "label": label,
                "category": cat,
                "color": CATEGORY_COLORS[cat],
                "size": c["size"],
                "source_counts": c.get("source_counts", {}),
                "top_experts": c.get("top10_experts_global", []),
                "rep_samples": rep_docs,
            }
        )

    return clusters_js, docs_js, k, "document"


def write_html(clusters_js, docs_js, k, granularity, info, emb_label, path):
    """Write the interactive HTML cluster explorer."""
    clusters_json = safe_json(clusters_js)
    docs_json = safe_json(docs_js)
    cat_colors_json = safe_json(CATEGORY_COLORS)
    model_path = info.get("model_path", "unknown")
    n_samples = len(docs_js)
    is_token = granularity == "token"

    # Sample rendering function (JS) differs by granularity
    if is_token:
        render_sample_js = """
        function renderSample(s) {
          return '<div class="sample-card">' +
            '<div class="sample-header"><span class="source-badge">' + esc(s.source) + '</span></div>' +
            '<div class="sample-text"><span class="dim">' + esc(s.before) + '</span>' +
            '<span class="highlight">' + esc(s.target) + '</span>' +
            '<span class="dim">' + esc(s.after) + '</span></div></div>';
        }
        function renderPoint(d) {
          return '<span class="dim">' + esc(d.before) + '</span>' +
            '<span class="highlight">' + esc(d.target) + '</span>' +
            '<span class="dim">' + esc(d.after) + '</span>';
        }
        """
    else:
        render_sample_js = """
        function renderSample(s) {
          return '<div class="sample-card">' +
            '<div class="sample-header"><span class="source-badge">' + esc(s.source) +
            '</span><span class="doc-len">' + (s.doc_len||'') + ' tokens</span></div>' +
            '<div class="sample-text">' + esc(s.preview||'') + '</div></div>';
        }
        function renderPoint(d) {
          return '<div>' + esc((d.preview||'').substring(0,200)) + '</div>';
        }
        """

    title = f"Emo {'Token' if is_token else 'Document'} Cluster Explorer"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title} — k={k} · {emb_label}</title>
<style>
:root {{ --bg:#0f1117; --surface:#1a1d27; --surface2:#232635; --border:#2e3347;
  --text:#e2e8f0; --dim:#8892a4; --accent:#4A90E2; --highlight:#fbbf24; }}
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:13px; height:100vh; display:flex; flex-direction:column; overflow:hidden; }}

#header {{ background:var(--surface); border-bottom:1px solid var(--border); padding:10px 16px;
  display:flex; align-items:center; gap:16px; flex-shrink:0; }}
#header h1 {{ font-size:15px; font-weight:600; }}
#header .meta {{ color:var(--dim); font-size:11px; }}
.view-tab {{ background:var(--surface2); border:1px solid var(--border); color:var(--dim);
  padding:5px 14px; border-radius:6px; cursor:pointer; font-size:12px; }}
.view-tab.active {{ background:var(--accent); color:#fff; border-color:var(--accent); }}

#main {{ display:flex; flex:1; overflow:hidden; }}

#sidebar {{ width:280px; flex-shrink:0; background:var(--surface); border-right:1px solid var(--border);
  display:flex; flex-direction:column; overflow:hidden; }}
#cluster-list {{ overflow-y:auto; flex:1; }}
.cluster-item {{ padding:8px 12px; cursor:pointer; border-bottom:1px solid var(--border); }}
.cluster-item:hover {{ background:var(--surface2); }}
.cluster-item.selected {{ background:var(--surface2); border-left:3px solid var(--accent); }}
.ci-header {{ display:flex; align-items:center; gap:6px; margin-bottom:4px; }}
.ci-dot {{ width:9px; height:9px; border-radius:50%; flex-shrink:0; }}
.ci-id {{ color:var(--dim); font-size:10px; width:24px; }}
.ci-label {{ font-size:12px; font-weight:500; flex:1; }}
.ci-size {{ color:var(--dim); font-size:10px; }}

#content {{ flex:1; overflow:hidden; display:flex; flex-direction:column; }}

#detail-view {{ flex:1; overflow-y:auto; display:none; padding:16px 20px; }}
#detail-view.active {{ display:block; }}
#detail-title {{ font-size:18px; font-weight:700; margin-bottom:4px; }}
#detail-meta {{ color:var(--dim); font-size:12px; margin-bottom:16px; }}
.section-title {{ font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:.06em;
  color:var(--dim); margin:16px 0 8px; }}
.sample-card {{ background:var(--surface2); border:1px solid var(--border); border-radius:8px;
  padding:12px 16px; margin-bottom:8px; }}
.sample-header {{ display:flex; gap:8px; margin-bottom:6px; align-items:center; }}
.source-badge {{ font-size:10px; padding:2px 7px; border-radius:10px; background:var(--accent);
  color:#fff; font-weight:600; }}
.doc-len {{ font-size:10px; color:var(--dim); }}
.sample-text {{ font-size:13px; line-height:1.7; font-family:'SF Mono','Fira Code',monospace;
  white-space:pre-wrap; word-break:break-all; }}
.highlight {{ background:var(--highlight); color:#000; padding:1px 2px; border-radius:3px; font-weight:700; }}
.dim {{ color:var(--dim); }}
.expert-tag {{ background:var(--surface2); border:1px solid var(--border); border-radius:4px;
  padding:3px 8px; font-size:12px; font-family:monospace; display:inline-block; margin:2px; }}

#src-breakdown {{ display:flex; flex-direction:column; gap:6px; }}
.src-row {{ display:flex; align-items:center; gap:8px; }}
.src-name {{ width:160px; font-size:12px; color:var(--dim); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.src-bar-wrap {{ flex:1; height:8px; background:var(--border); border-radius:4px; overflow:hidden; }}
.src-bar-fill {{ height:100%; border-radius:4px; }}
.src-count {{ width:60px; text-align:right; font-size:11px; color:var(--dim); }}

#umap-view {{ flex:1; overflow:hidden; display:none; position:relative; }}
#umap-view.active {{ display:block; }}
#umap-canvas {{ width:100%; height:100%; cursor:crosshair; }}
#umap-tooltip {{ position:fixed; background:var(--surface2); border:1px solid var(--border);
  border-radius:6px; padding:8px 12px; font-size:12px; pointer-events:none; z-index:50;
  display:none; max-width:360px; }}
#umap-legend {{ position:absolute; top:12px; right:12px; background:rgba(15,17,23,.85);
  border:1px solid var(--border); border-radius:8px; padding:10px 14px;
  max-height:80vh; overflow-y:auto; font-size:11px; }}
.legend-item {{ display:flex; align-items:center; gap:6px; margin-bottom:4px; cursor:pointer; }}
.legend-dot {{ width:9px; height:9px; border-radius:50%; flex-shrink:0; }}
</style>
</head>
<body>

<div id="header">
  <h1>{title}</h1>
  <span class="meta">k={k} · {emb_label} · {n_samples:,} {'tokens' if is_token else 'documents'} · {model_path.split('/')[-2] if '/' in model_path else model_path}</span>
  <div style="margin-left:auto;display:flex;gap:4px">
    <div class="view-tab active" onclick="switchView('detail')">Clusters</div>
    <div class="view-tab" onclick="switchView('umap')">UMAP</div>
  </div>
</div>

<div id="main">
  <div id="sidebar">
    <div id="cluster-list"></div>
  </div>
  <div id="content">
    <div id="detail-view" class="active">
      <div id="detail-title">Select a cluster</div>
      <div id="detail-meta"></div>
      <div id="detail-body"></div>
    </div>
    <div id="umap-view">
      <canvas id="umap-canvas"></canvas>
      <div id="umap-tooltip"></div>
      <div id="umap-legend"></div>
    </div>
  </div>
</div>

<script>
const CLUSTERS = {clusters_json};
const POINTS = {docs_json};
const CAT_COLORS = {cat_colors_json};
const IS_TOKEN = {'true' if is_token else 'false'};

function esc(s) {{ return s ? String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;') : ''; }}

{render_sample_js}

// Build sidebar
const clList = document.getElementById('cluster-list');
CLUSTERS.forEach(c => {{
  const el = document.createElement('div');
  el.className = 'cluster-item';
  el.dataset.id = c.id;
  el.innerHTML = '<div class="ci-header">' +
    '<div class="ci-dot" style="background:' + c.color + '"></div>' +
    '<span class="ci-id">#' + c.id + '</span>' +
    '<span class="ci-label">' + esc(c.label) + '</span>' +
    '<span class="ci-size">' + c.size + '</span></div>';
  el.onclick = () => selectCluster(c.id);
  clList.appendChild(el);
}});

let selectedCluster = null;
function selectCluster(id) {{
  selectedCluster = id;
  document.querySelectorAll('.cluster-item').forEach(el => {{
    el.classList.toggle('selected', +el.dataset.id === id);
  }});
  const c = CLUSTERS.find(x => x.id === id);
  if (!c) return;
  document.getElementById('detail-title').textContent = c.label;
  document.getElementById('detail-meta').textContent = c.size + ' samples · category: ' + c.category;

  let html = '<div class="section-title">Source Distribution</div><div id="src-breakdown">';
  const sorted = Object.entries(c.source_counts).sort((a,b) => b[1]-a[1]);
  const maxCount = sorted.length ? sorted[0][1] : 1;
  sorted.forEach(([src, count]) => {{
    const pct = (count/c.size*100).toFixed(1);
    html += '<div class="src-row"><span class="src-name">' + esc(src) + '</span>' +
      '<div class="src-bar-wrap"><div class="src-bar-fill" style="width:' + (count/maxCount*100) + '%;background:' + c.color + '"></div></div>' +
      '<span class="src-count">' + count + ' (' + pct + '%)</span></div>';
  }});
  html += '</div>';

  if (c.top_experts && c.top_experts.length) {{
    html += '<div class="section-title">Top Experts</div><div>';
    c.top_experts.forEach(e => {{ html += '<span class="expert-tag">L' + Math.floor(e/{info.get("num_standard_experts", 128)}) + '/E' + (e%{info.get("num_standard_experts", 128)}) + '</span>'; }});
    html += '</div>';
  }}

  if (c.rep_samples && c.rep_samples.length) {{
    html += '<div class="section-title">Representative Samples</div>';
    c.rep_samples.forEach(s => {{ html += renderSample(s); }});
  }}

  document.getElementById('detail-body').innerHTML = html;
}}

// View switching
function switchView(view) {{
  document.querySelectorAll('.view-tab').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('detail-view').classList.toggle('active', view === 'detail');
  document.getElementById('umap-view').classList.toggle('active', view === 'umap');
  if (view === 'umap' && !umapDrawn) drawUMAP();
}}

// UMAP
let umapDrawn = false;
function drawUMAP() {{
  umapDrawn = true;
  const canvas = document.getElementById('umap-canvas');
  const wrap = canvas.parentElement;
  canvas.width = wrap.clientWidth * 2;
  canvas.height = wrap.clientHeight * 2;
  canvas.style.width = wrap.clientWidth + 'px';
  canvas.style.height = wrap.clientHeight + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(2, 2);

  const W = wrap.clientWidth, H = wrap.clientHeight;
  const xs = POINTS.map(p => p.x), ys = POINTS.map(p => p.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const pad = 30;

  function tx(x) {{ return pad + (x - xMin) / (xMax - xMin) * (W - 2*pad); }}
  function ty(y) {{ return pad + (y - yMin) / (yMax - yMin) * (H - 2*pad); }}

  // Draw
  ctx.fillStyle = '#0f1117';
  ctx.fillRect(0, 0, W, H);

  const clusterMap = {{}};
  CLUSTERS.forEach(c => {{ clusterMap[c.id] = c; }});

  POINTS.forEach(p => {{
    const c = clusterMap[p.c];
    ctx.fillStyle = c ? c.color + '88' : '#55555588';
    ctx.beginPath();
    ctx.arc(tx(p.x), ty(p.y), 2, 0, Math.PI * 2);
    ctx.fill();
  }});

  // Legend
  const legend = document.getElementById('umap-legend');
  legend.innerHTML = CLUSTERS.map(c =>
    '<div class="legend-item" onclick="selectCluster(' + c.id + ');switchView(\\'detail\\')">' +
    '<div class="legend-dot" style="background:' + c.color + '"></div>' +
    '<span>#' + c.id + ' ' + esc(c.label) + ' (' + c.size + ')</span></div>'
  ).join('');

  // Tooltip
  const tooltip = document.getElementById('umap-tooltip');
  canvas.onmousemove = function(e) {{
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    let closest = null, minDist = 100;
    POINTS.forEach(p => {{
      const dx = tx(p.x) - mx, dy = ty(p.y) - my;
      const d = Math.sqrt(dx*dx + dy*dy);
      if (d < minDist) {{ minDist = d; closest = p; }}
    }});
    if (closest && minDist < 10) {{
      tooltip.style.display = 'block';
      tooltip.style.left = (e.clientX + 12) + 'px';
      tooltip.style.top = (e.clientY + 12) + 'px';
      const c = clusterMap[closest.c];
      tooltip.innerHTML = '<b>#' + closest.c + ' ' + esc(c ? c.label : '') + '</b><br>' +
        '<span style="color:var(--dim)">' + esc(closest.s) + '</span><br>' +
        '<div style="margin-top:4px">' + renderPoint(closest) + '</div>';
    }} else {{
      tooltip.style.display = 'none';
    }}
  }};
}}

// Select first cluster
if (CLUSTERS.length) selectCluster(CLUSTERS[0].id);
</script>
</body>
</html>"""

    with open(path, "w") as f:
        f.write(html)
    logger.info(f"Saved HTML -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate interactive HTML cluster explorer")
    parser.add_argument(
        "--cluster-dir", required=True, help="Directory with assignments.npy, run_info.json"
    )
    parser.add_argument(
        "--data-dir", default=None, help="Data directory (default: parent of cluster-dir)"
    )
    args = parser.parse_args()

    cluster_dir = args.cluster_dir
    data_dir = args.data_dir or os.path.dirname(os.path.normpath(cluster_dir))

    with open(os.path.join(cluster_dir, "run_info.json")) as f:
        run_info = json.load(f)
    with open(os.path.join(data_dir, "info.json")) as f:
        info = json.load(f)

    emb_name = run_info["embedding"]
    preprocess = run_info.get("preprocess", run_info.get("transform", "unknown"))
    method = run_info.get("method", run_info.get("cluster", "unknown"))
    emb_label = f"{emb_name}_{preprocess}_{method}"

    # Auto-detect granularity: doc-level embeddings start with "doc_"
    is_doc_level = emb_name.startswith("doc_")

    if is_doc_level:
        logger.info("Detected document-level clustering")
        clusters_js, docs_js, k, granularity = build_doc_data(cluster_dir, data_dir, run_info, info)
    else:
        logger.info("Detected token-level clustering")
        clusters_js, docs_js, k, granularity = build_token_data(
            cluster_dir, data_dir, run_info, info
        )

    html_path = os.path.join(cluster_dir, "cluster_explorer.html")
    write_html(clusters_js, docs_js, k, granularity, info, emb_label, html_path)


if __name__ == "__main__":
    main()
