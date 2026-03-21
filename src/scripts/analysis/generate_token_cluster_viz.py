"""
Generate interactive HTML visualizer for token-level cluster analysis.

Similar to generate_cluster_viz.py but adapted for token-level data:
- Each data point is a token, not a document
- Displays each token highlighted within its ±N surrounding tokens as context
- Loads documents.npy + doc_boundaries.npy for context recovery

Usage:
    python -m src.scripts.analysis.generate_token_cluster_viz \
        --cluster-dir claude_outputs/analysis/.../token_probs_mean_pca_l2_spherical_kmeans_k64 \
        --data-dir claude_outputs/analysis/.../<model_name> \
        --context-window 10
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

# Category colors (same as document-level viz)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-dir", required=True,
                        help="Directory containing assignments.npy, summary.json, run_info.json")
    parser.add_argument("--data-dir", default=None,
                        help="Dir with token data files. Defaults to parent of --cluster-dir.")
    parser.add_argument("--emb-file", default=None,
                        help="Path to embedding .npy file. Auto-detected from run_info.json.")
    parser.add_argument("--context-window", type=int, default=10,
                        help="Number of tokens to show before/after the target token (default: 10)")
    parser.add_argument("--model-path", default=None,
                        help="Path to HF model for tokenizer. Auto-detected from info.json.")
    args = parser.parse_args()

    cluster_dir = args.cluster_dir
    data_dir = args.data_dir or os.path.dirname(os.path.normpath(cluster_dir))
    ctx_win = args.context_window

    # Load run_info
    with open(os.path.join(cluster_dir, "run_info.json")) as f:
        run_info = json.load(f)
    k = run_info["k"]

    # Auto-detect embedding file
    if args.emb_file:
        emb_file = args.emb_file
    else:
        emb_name = run_info["embedding"]
        emb_file = os.path.join(data_dir, f"embeddings_{emb_name}.npy")

    # Load info
    with open(os.path.join(data_dir, "info.json")) as f:
        info = json.load(f)

    # Load tokenizer for decoding
    model_path = args.model_path or info.get("model_path", "")
    logger.info(f"Loading tokenizer from {model_path}...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load embeddings + assignments
    logger.info(f"Loading embeddings from {emb_file}...")
    emb = np.load(emb_file).astype(np.float32)
    labels = np.load(os.path.join(cluster_dir, "assignments.npy"))

    # Load token metadata
    logger.info("Loading token metadata...")
    meta = []
    meta_path = os.path.join(data_dir, "metadata_tokens.jsonl.gz")
    with gzip.open(meta_path, "rt") as f:
        for line in f:
            meta.append(json.loads(line))

    # Load documents for context
    logger.info("Loading documents for context recovery...")
    documents = np.load(os.path.join(data_dir, "documents.npy"))
    boundaries = np.load(os.path.join(data_dir, "doc_boundaries.npy"))

    # Load summary
    with open(os.path.join(cluster_dir, "summary.json")) as f:
        summary = json.load(f)

    # UMAP — subsample if too many tokens for performance
    umap_path = os.path.join(cluster_dir, "umap_coords.npy")
    n_total = emb.shape[0]
    max_umap = 50000  # subsample for UMAP if needed
    if n_total > max_umap:
        logger.info(f"Subsampling {max_umap}/{n_total} tokens for UMAP...")
        rng = np.random.RandomState(42)
        umap_idx = rng.choice(n_total, max_umap, replace=False)
        umap_idx.sort()
    else:
        umap_idx = np.arange(n_total)

    if os.path.exists(umap_path):
        logger.info("Loading cached UMAP coords...")
        coords_2d = np.load(umap_path)
    else:
        logger.info(f"PCA-50 on {len(umap_idx)} tokens...")
        pca = PCA(n_components=50, random_state=42)
        reduced = pca.fit_transform(emb[umap_idx])
        reduced_normed = normalize(reduced, norm="l2")

        logger.info("Running UMAP...")
        reducer = umap_lib.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                                metric="euclidean", random_state=42, verbose=False)
        coords_2d = reducer.fit_transform(reduced_normed)
        np.save(umap_path, coords_2d)
        logger.info(f"Saved UMAP coords → {umap_path}")

    # Build context strings for tokens
    logger.info("Building token context strings...")

    def get_token_context(token_idx):
        """Get context string with target token highlighted."""
        m = meta[token_idx]
        doc_idx = m["doc_index"]
        pos = m["token_position"]
        doc_start = int(boundaries[doc_idx])
        doc_end = int(boundaries[doc_idx + 1])
        doc_tokens = documents[doc_start:doc_end]

        # Context window
        ctx_start = max(0, pos - ctx_win)
        ctx_end = min(len(doc_tokens), pos + ctx_win + 1)

        # Decode each token individually for accurate highlighting
        before_ids = doc_tokens[ctx_start:pos].tolist()
        target_id = doc_tokens[pos:pos + 1].tolist()
        after_ids = doc_tokens[pos + 1:ctx_end].tolist()

        before_text = tokenizer.decode(before_ids, skip_special_tokens=True) if before_ids else ""
        target_text = tokenizer.decode(target_id, skip_special_tokens=True)
        after_text = tokenizer.decode(after_ids, skip_special_tokens=True) if after_ids else ""

        prefix = "…" if ctx_start > 0 else ""
        suffix = "…" if ctx_end < len(doc_tokens) else ""

        return prefix + before_text, target_text, after_text + suffix

    # Build per-token data for JS (only for UMAP-subsampled tokens)
    logger.info(f"Building data for {len(umap_idx)} tokens...")
    docs_js = []
    for vi, token_idx in enumerate(umap_idx):
        m = meta[token_idx]
        before, target, after = get_token_context(token_idx)
        xy = coords_2d[vi]
        docs_js.append({
            "i": int(token_idx),
            "c": int(labels[token_idx]),
            "s": m["source"],
            "di": m["doc_index"],
            "tp": m["token_position"],
            "tid": m["token_id"],
            "before": before,
            "target": target,
            "after": after,
            "x": round(float(xy[0]), 3),
            "y": round(float(xy[1]), 3),
        })

    # Build representative token contexts for each cluster summary
    logger.info("Building representative token contexts for clusters...")
    for c in summary:
        if "representative_docs" in c:
            for rd in c["representative_docs"]:
                idx = rd.get("idx")
                if idx is not None and idx < len(meta):
                    before, target, after = get_token_context(idx)
                    rd["before"] = before
                    rd["target"] = target
                    rd["after"] = after
                else:
                    rd["before"] = ""
                    rd["target"] = "(unknown)"
                    rd["after"] = ""

    # Load cluster labels
    external_labels = {}
    labels_path = os.path.join(cluster_dir, "cluster_labels.json")
    if os.path.exists(labels_path):
        logger.info(f"Loading cluster labels from {labels_path}")
        with open(labels_path) as f:
            external_labels = json.load(f)

    clusters_js = []
    for c in summary:
        cid = c["cluster"]
        cid_str = str(cid)
        if cid_str in external_labels:
            label = external_labels[cid_str]["label"]
            cat = external_labels[cid_str].get("category", "reference")
            if cat not in CATEGORY_COLORS:
                cat = "reference"
        else:
            label, cat = f"Cluster {cid}", "reference"

        rep_tokens = []
        for rd in c.get("representative_docs", []):
            rep_tokens.append({
                "source": rd.get("source", ""),
                "doc_index": rd.get("doc_index", rd.get("idx", -1)),
                "token_position": rd.get("token_position", -1),
                "before": rd.get("before", ""),
                "target": rd.get("target", ""),
                "after": rd.get("after", ""),
            })

        clusters_js.append({
            "id": cid,
            "label": label,
            "category": cat,
            "color": CATEGORY_COLORS[cat],
            "size": c["size"],
            "source_counts": c["source_counts"],
            "top_experts": c["top10_experts_global"],
            "rep_tokens": rep_tokens,
        })

    # Precompute per-document cluster stats (using ALL tokens, not just subsampled)
    logger.info("Computing per-document cluster statistics...")
    doc_cluster_map = {}  # doc_index -> {cluster_id: count}
    for i, (m, lbl) in enumerate(zip(meta, labels)):
        di = m["doc_index"]
        if di not in doc_cluster_map:
            doc_cluster_map[di] = {"source": m["source"], "clusters": {}, "total": 0}
        cl = int(lbl)
        doc_cluster_map[di]["clusters"][cl] = doc_cluster_map[di]["clusters"].get(cl, 0) + 1
        doc_cluster_map[di]["total"] += 1

    # Build doc stats list for JS
    doc_stats_js = []
    spread_counts = []  # number of clusters per doc
    for di in sorted(doc_cluster_map.keys()):
        ds = doc_cluster_map[di]
        n_clusters = len(ds["clusters"])
        spread_counts.append(n_clusters)
        doc_stats_js.append({
            "di": di,
            "s": ds["source"],
            "total": ds["total"],
            "nClusters": n_clusters,
            "clusters": ds["clusters"],  # {cluster_id: count}
        })

    # Aggregate stats
    spread_counts_np = np.array(spread_counts)
    agg_stats = {
        "num_docs": len(doc_cluster_map),
        "mean_clusters": round(float(spread_counts_np.mean()), 2),
        "median_clusters": int(np.median(spread_counts_np)),
        "max_clusters": int(spread_counts_np.max()),
        "min_clusters": int(spread_counts_np.min()),
        "std_clusters": round(float(spread_counts_np.std()), 2),
        "histogram": {},  # n_clusters -> count of docs
    }
    for nc in sorted(set(spread_counts)):
        agg_stats["histogram"][nc] = int((spread_counts_np == nc).sum())

    logger.info(f"  {agg_stats['num_docs']} docs, "
                f"mean {agg_stats['mean_clusters']} clusters/doc, "
                f"median {agg_stats['median_clusters']}, "
                f"max {agg_stats['max_clusters']}")

    # Build per-document full text with per-token cluster assignments
    logger.info("Building per-document full text with cluster assignments...")
    # Build position-to-cluster map: (doc_index, token_position) -> cluster_id
    pos_to_cluster = {}
    for i, m in enumerate(meta):
        pos_to_cluster[(m["doc_index"], m["token_position"])] = int(labels[i])

    doc_texts_js = []  # list of {di, source, tokens: [{t: decoded_text, c: cluster_or_-1}]}
    for di in sorted(doc_cluster_map.keys()):
        doc_start = int(boundaries[di])
        doc_end = int(boundaries[di + 1])
        doc_token_ids = documents[doc_start:doc_end]
        tokens_list = []
        for pos, tid in enumerate(doc_token_ids):
            decoded = tokenizer.decode([int(tid)], skip_special_tokens=False)
            cluster_id = pos_to_cluster.get((di, pos), -1)
            tokens_list.append({"t": decoded, "c": cluster_id})
        doc_texts_js.append({
            "di": di,
            "s": doc_cluster_map[di]["source"],
            "tokens": tokens_list,
        })
    logger.info(f"  Decoded {sum(len(d['tokens']) for d in doc_texts_js)} tokens across {len(doc_texts_js)} docs")

    # Build per-unique-token cluster distribution
    logger.info("Computing per-unique-token cluster distributions...")
    from collections import defaultdict
    token_cluster_counts = defaultdict(lambda: defaultdict(int))  # token_id -> {cluster: count}
    token_total_counts = defaultdict(int)
    for i, m in enumerate(meta):
        tid = m["token_id"]
        cl = int(labels[i])
        token_cluster_counts[tid][cl] += 1
        token_total_counts[tid] += 1

    # Build list for JS (only tokens appearing >= 3 times)
    min_count = 3
    unique_tokens_js = []
    for tid, total in sorted(token_total_counts.items(), key=lambda x: -x[1]):
        if total < min_count:
            continue
        decoded = tokenizer.decode([tid], skip_special_tokens=False)
        clusters = dict(token_cluster_counts[tid])  # {cluster_id: count}
        n_clusters = len(clusters)
        # Entropy of cluster distribution
        probs_arr = np.array(list(clusters.values()), dtype=np.float64)
        probs_arr = probs_arr / probs_arr.sum()
        entropy = float(-np.sum(probs_arr * np.log2(probs_arr)))
        unique_tokens_js.append({
            "id": tid,
            "t": decoded,
            "n": total,
            "nc": n_clusters,
            "ent": round(entropy, 3),
            "clusters": clusters,
        })
    logger.info(f"  {len(unique_tokens_js)} unique tokens (>= {min_count} occurrences) "
                f"out of {len(token_total_counts)} total unique tokens")

    emb_label = f"{run_info['embedding']}_{run_info['transform']}_{run_info['cluster']}"

    html_path = os.path.join(cluster_dir, "cluster_explorer.html")
    write_html(clusters_js, docs_js, doc_stats_js, agg_stats, doc_texts_js, unique_tokens_js,
               info, k, emb_label, html_path, ctx_win)
    logger.info(f"Saved HTML visualizer → {html_path}")


def write_html(clusters_js, docs_js, doc_stats_js, agg_stats, doc_texts_js, unique_tokens_js,
               info, k, emb_label, path, ctx_win):
    def safe_json(obj):
        return json.dumps(obj).replace("</", "<\\/")

    clusters_json = safe_json(clusters_js)
    docs_json = safe_json(docs_js)
    doc_stats_json = safe_json(doc_stats_js)
    agg_stats_json = safe_json(agg_stats)
    doc_texts_json = safe_json(doc_texts_js)
    unique_tokens_json = safe_json(unique_tokens_js)
    cat_colors_json = safe_json(CATEGORY_COLORS)
    model_path = info.get("model_path", "unknown")
    num_tokens = info.get("num_tokens", len(docs_js))
    num_docs = info.get("num_docs", 0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FlexMoE Token Cluster Explorer — k={k} · {emb_label}</title>
<style>
  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #232635;
    --border: #2e3347;
    --text: #e2e8f0;
    --text-dim: #8892a4;
    --accent: #4A90E2;
    --highlight: #fbbf24;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 13px; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}

  #header {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 10px 16px; display: flex; align-items: center; gap: 16px; flex-shrink: 0; }}
  #header h1 {{ font-size: 15px; font-weight: 600; color: var(--text); }}
  #header .meta {{ color: var(--text-dim); font-size: 11px; }}
  #view-tabs {{ margin-left: auto; display: flex; gap: 4px; }}
  .view-tab {{ background: var(--surface2); border: 1px solid var(--border); color: var(--text-dim); padding: 5px 14px; border-radius: 6px; cursor: pointer; font-size: 12px; }}
  .view-tab.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}

  #filterbar {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 8px 16px; display: flex; gap: 6px; flex-wrap: wrap; align-items: center; flex-shrink: 0; }}
  #search {{ background: var(--surface2); border: 1px solid var(--border); color: var(--text); padding: 5px 10px; border-radius: 6px; font-size: 12px; width: 200px; outline: none; }}
  #search:focus {{ border-color: var(--accent); }}
  .cat-btn {{ border: 1px solid var(--border); background: var(--surface2); color: var(--text-dim); padding: 4px 10px; border-radius: 12px; cursor: pointer; font-size: 11px; transition: all .15s; }}
  .cat-btn:hover {{ border-color: #555; color: var(--text); }}
  .cat-btn.active {{ color: #fff; }}
  #sort-select {{ background: var(--surface2); border: 1px solid var(--border); color: var(--text-dim); padding: 4px 8px; border-radius: 6px; font-size: 11px; outline: none; margin-left: auto; }}

  #main {{ display: flex; flex: 1; overflow: hidden; }}

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

  #content {{ flex: 1; overflow: hidden; display: flex; flex-direction: column; }}

  #detail-view {{ flex: 1; overflow-y: auto; display: none; flex-direction: column; }}
  #detail-view.active {{ display: flex; }}
  #detail-header {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 14px 20px; flex-shrink: 0; }}
  #detail-title {{ font-size: 18px; font-weight: 700; margin-bottom: 2px; }}
  #detail-meta {{ color: var(--text-dim); font-size: 12px; }}
  #detail-body {{ padding: 16px 20px; display: flex; flex-direction: column; gap: 16px; overflow-y: auto; flex: 1; }}
  .section-title {{ font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; color: var(--text-dim); margin-bottom: 8px; }}

  #src-breakdown {{ display: flex; flex-direction: column; gap: 6px; }}
  .src-row {{ display: flex; align-items: center; gap: 8px; }}
  .src-name {{ width: 160px; font-size: 12px; color: var(--text-dim); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .src-bar-wrap {{ flex: 1; height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; }}
  .src-bar-fill {{ height: 100%; border-radius: 4px; }}
  .src-count {{ width: 80px; text-align: right; font-size: 11px; color: var(--text-dim); }}

  #top-docs-list {{ display: flex; flex-direction: column; gap: 6px; }}
  .top-doc-row {{ display: flex; align-items: center; gap: 8px; background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; padding: 6px 10px; cursor: pointer; transition: border-color .15s; }}
  .top-doc-row:hover {{ border-color: var(--accent); }}
  .top-doc-pct {{ font-size: 14px; font-weight: 700; min-width: 48px; text-align: right; }}
  .top-doc-info {{ flex: 1; font-size: 11px; color: var(--text-dim); }}
  .top-doc-bar {{ width: 80px; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }}
  .top-doc-bar-fill {{ height: 100%; border-radius: 3px; }}

  #experts-list {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .expert-tag {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 4px; padding: 3px 8px; font-size: 12px; font-family: monospace; }}

  .token-context {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; }}
  .token-context-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }}
  .source-badge {{ font-size: 10px; padding: 2px 7px; border-radius: 10px; font-weight: 600; letter-spacing: .03em; }}
  .token-context-text {{ font-size: 13px; line-height: 1.8; font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; white-space: pre-wrap; word-break: break-all; }}
  .token-highlight {{ background: var(--highlight); color: #000; padding: 1px 2px; border-radius: 3px; font-weight: 700; }}
  .token-dim {{ color: var(--text-dim); }}

  /* UMAP */
  #umap-view {{ flex: 1; overflow: hidden; display: none; flex-direction: column; }}
  #umap-view.active {{ display: flex; }}
  #umap-controls {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 8px 16px; display: flex; gap: 12px; align-items: center; flex-shrink: 0; font-size: 12px; }}
  .umap-radio label {{ cursor: pointer; color: var(--text-dim); }}
  .umap-radio input {{ margin-right: 4px; }}
  #umap-tooltip {{ position: fixed; background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; font-size: 12px; pointer-events: none; z-index: 50; display: none; max-width: 360px; }}
  #umap-canvas-wrap {{ flex: 1; position: relative; overflow: hidden; }}
  #umap-canvas {{ display: block; width: 100%; height: 100%; cursor: crosshair; }}
  #umap-legend {{ position: absolute; top: 12px; right: 12px; background: rgba(15,17,23,.85); border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; max-height: 80vh; overflow-y: auto; font-size: 11px; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; margin-bottom: 5px; cursor: pointer; }}
  .legend-dot {{ width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }}
  .legend-label {{ color: var(--text-dim); }}
  .legend-item:hover .legend-label {{ color: var(--text); }}

  ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
  ::-webkit-scrollbar-track {{ background: transparent; }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}

  /* Document spread annotation on token cards */
  .doc-spread-bar {{ display: flex; height: 6px; border-radius: 3px; overflow: hidden; margin-top: 6px; background: var(--border); }}
  .doc-spread-bar-seg {{ height: 100%; }}
  .doc-spread-info {{ color: var(--text-dim); font-size: 10px; margin-top: 4px; }}

  /* Documents view */
  #docs-view {{ flex: 1; overflow: hidden; display: none; flex-direction: column; }}
  #docs-view.active {{ display: flex; }}
  #docs-view-header {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 12px 20px; flex-shrink: 0; }}
  #docs-view-stats {{ display: flex; gap: 24px; margin-bottom: 10px; }}
  .stat-card {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 10px 16px; min-width: 100px; }}
  .stat-value {{ font-size: 22px; font-weight: 700; color: var(--accent); }}
  .stat-label {{ font-size: 10px; color: var(--text-dim); text-transform: uppercase; letter-spacing: .05em; }}
  #docs-histogram {{ display: flex; align-items: flex-end; gap: 2px; height: 60px; margin: 8px 0; }}
  .hist-bar {{ background: var(--accent); border-radius: 2px 2px 0 0; min-width: 8px; position: relative; cursor: pointer; }}
  .hist-bar:hover {{ background: #6ab0ff; }}
  .hist-label {{ position: absolute; bottom: -14px; left: 50%; transform: translateX(-50%); font-size: 9px; color: var(--text-dim); white-space: nowrap; }}
  #docs-view-controls {{ display: flex; gap: 8px; align-items: center; }}
  #doc-filter {{ background: var(--surface2); border: 1px solid var(--border); color: var(--text); padding: 5px 10px; border-radius: 6px; font-size: 12px; width: 200px; outline: none; }}
  #doc-sort-select {{ background: var(--surface2); border: 1px solid var(--border); color: var(--text-dim); padding: 4px 8px; border-radius: 6px; font-size: 11px; outline: none; }}
  #docs-list-wrap {{ flex: 1; overflow-y: auto; padding: 12px 20px; }}
  .doc-card {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; cursor: pointer; transition: border-color .15s; }}
  .doc-card:hover {{ border-color: var(--accent); }}
  .doc-card-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }}
  .doc-card-id {{ color: var(--text-dim); font-size: 11px; font-family: monospace; }}
  .doc-card-stats {{ font-size: 11px; color: var(--text-dim); margin-left: auto; }}
  .doc-cluster-bar {{ display: flex; height: 10px; border-radius: 5px; overflow: hidden; background: var(--border); }}
  .doc-cluster-bar-seg {{ height: 100%; cursor: pointer; position: relative; }}
  .doc-cluster-bar-seg:hover {{ opacity: 0.8; }}
  .doc-card-clusters {{ display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }}
  .doc-cluster-tag {{ font-size: 10px; padding: 2px 6px; border-radius: 8px; border: 1px solid var(--border); }}

  /* Tokens view */
  #tokens-view {{ flex: 1; overflow: hidden; display: none; flex-direction: column; }}
  #tokens-view.active {{ display: flex; }}
  #tokens-view-header {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 12px 20px; flex-shrink: 0; }}
  #tokens-view-controls {{ display: flex; gap: 8px; align-items: center; margin-top: 8px; }}
  #token-filter {{ background: var(--surface2); border: 1px solid var(--border); color: var(--text); padding: 5px 10px; border-radius: 6px; font-size: 12px; width: 200px; outline: none; }}
  #token-sort-select {{ background: var(--surface2); border: 1px solid var(--border); color: var(--text-dim); padding: 4px 8px; border-radius: 6px; font-size: 11px; outline: none; }}
  #tokens-list-wrap {{ flex: 1; overflow-y: auto; padding: 12px 20px; }}
  .utok-card {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; margin-bottom: 6px; }}
  .utok-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }}
  .utok-text {{ font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; font-size: 15px; font-weight: 700; background: var(--highlight); color: #000; padding: 2px 6px; border-radius: 4px; max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: pre; }}
  .utok-stats {{ font-size: 11px; color: var(--text-dim); }}
  .utok-entropy {{ font-size: 11px; padding: 2px 6px; border-radius: 8px; }}
  .utok-cluster-bar {{ display: flex; height: 10px; border-radius: 5px; overflow: hidden; background: var(--border); margin-bottom: 4px; }}
  .utok-cluster-bar-seg {{ height: 100%; cursor: pointer; }}
  .utok-cluster-bar-seg:hover {{ opacity: 0.8; }}
  .utok-clusters {{ display: flex; flex-wrap: wrap; gap: 3px; }}
  .utok-cluster-tag {{ font-size: 9px; padding: 1px 5px; border-radius: 6px; border: 1px solid var(--border); }}

  /* Token detail panel (replaces tokens list when a token is selected) */
  #token-detail {{ display: none; flex: 1; overflow: hidden; flex-direction: column; }}
  #token-detail.active {{ display: flex; }}
  #token-detail-header {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 14px 20px; flex-shrink: 0; }}
  #token-detail-back {{ background: var(--surface2); border: 1px solid var(--border); color: var(--text-dim); padding: 4px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; margin-right: 12px; }}
  #token-detail-back:hover {{ color: var(--text); border-color: #555; }}
  #token-detail-title {{ font-size: 18px; font-weight: 700; display: inline; }}
  #token-detail-meta {{ color: var(--text-dim); font-size: 12px; margin-top: 4px; }}
  #token-detail-body {{ flex: 1; overflow-y: auto; padding: 16px 20px; }}
  .td-cluster-section {{ margin-bottom: 16px; }}
  .td-cluster-header {{ display: flex; align-items: center; gap: 8px; padding: 8px 12px; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; cursor: pointer; transition: border-color .15s; }}
  .td-cluster-header:hover {{ border-color: var(--accent); }}
  .td-cluster-header.expanded {{ border-radius: 8px 8px 0 0; border-bottom-color: transparent; }}
  .td-cluster-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
  .td-cluster-name {{ font-size: 13px; font-weight: 600; }}
  .td-cluster-count {{ font-size: 11px; color: var(--text-dim); margin-left: auto; }}
  .td-cluster-pct-bar {{ width: 60px; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }}
  .td-cluster-pct-fill {{ height: 100%; border-radius: 3px; }}
  .td-cluster-usages {{ display: none; border: 1px solid var(--border); border-top: none; border-radius: 0 0 8px 8px; padding: 8px; background: var(--bg); }}
  .td-cluster-usages.expanded {{ display: block; }}

  /* Document reader modal */
  #doc-reader-modal {{ position: fixed; inset: 0; background: rgba(0,0,0,.7); z-index: 100; display: none; align-items: center; justify-content: center; }}
  #doc-reader-modal.open {{ display: flex; }}
  #doc-reader-inner {{ background: var(--bg); border: 1px solid var(--border); border-radius: 12px; width: 90vw; max-width: 1000px; height: 85vh; display: flex; flex-direction: column; overflow: hidden; }}
  #doc-reader-header {{ background: var(--surface); padding: 14px 20px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 12px; flex-shrink: 0; }}
  #doc-reader-title {{ font-size: 16px; font-weight: 700; }}
  #doc-reader-meta {{ color: var(--text-dim); font-size: 12px; }}
  #doc-reader-close {{ margin-left: auto; background: var(--surface2); border: 1px solid var(--border); color: var(--text-dim); padding: 4px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; }}
  #doc-reader-close:hover {{ color: var(--text); border-color: #555; }}
  #doc-reader-legend {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 8px 20px; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; flex-shrink: 0; font-size: 11px; }}
  .legend-cluster-tag {{ padding: 2px 8px; border-radius: 10px; cursor: pointer; font-size: 10px; font-weight: 600; border: 1px solid transparent; }}
  .legend-cluster-tag:hover {{ opacity: 0.8; }}
  .legend-cluster-tag.dim {{ opacity: 0.3; }}
  #doc-reader-body {{ flex: 1; overflow-y: auto; padding: 20px; font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; font-size: 13px; line-height: 2.0; white-space: pre-wrap; word-break: break-all; }}
  .doc-token {{ padding: 1px 0; border-radius: 2px; cursor: default; position: relative; }}
  .doc-token.sampled {{ padding: 1px 2px; border-radius: 3px; cursor: pointer; }}
  .doc-token.sampled:hover {{ outline: 2px solid #fff; outline-offset: 1px; }}
  .doc-token.unsampled {{ color: var(--text-dim); }}
  #doc-reader-tooltip {{ position: fixed; background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; padding: 6px 10px; font-size: 11px; pointer-events: none; z-index: 110; display: none; white-space: nowrap; }}

  /* Document preview cards */
  .doc-preview-card {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 10px; overflow: hidden; }}
  .doc-preview-header {{ display: flex; align-items: center; gap: 8px; padding: 8px 12px; border-bottom: 1px solid var(--border); cursor: pointer; }}
  .doc-preview-header:hover {{ background: var(--bg); }}
  .doc-preview-pct {{ font-size: 14px; font-weight: 700; min-width: 48px; text-align: right; }}
  .doc-preview-info {{ font-size: 11px; color: var(--text-dim); flex: 1; }}
  .doc-preview-body {{ padding: 10px 14px; font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; font-size: 12px; line-height: 1.9; white-space: pre-wrap; word-break: break-all; max-height: 200px; overflow-y: auto; }}
  .doc-preview-body .tok-hl {{ background: var(--highlight); color: #000; padding: 1px 2px; border-radius: 3px; font-weight: 700; }}
  .doc-preview-body .tok-dim {{ color: var(--text-dim); opacity: 0.5; }}

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
  <h1>FlexMoE Token Cluster Explorer — {emb_label}</h1>
  <span class="meta">k={k} · {num_tokens:,} tokens · {num_docs:,} docs · context ±{ctx_win} · {model_path.split('/')[-2]}</span>
  <div id="view-tabs">
    <button class="view-tab active" onclick="setView('detail')">Clusters</button>
    <button class="view-tab" onclick="setView('docs')">Documents</button>
    <button class="view-tab" onclick="setView('tokens')">Tokens</button>
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
    <div id="detail-view" class="active">
      <div id="detail-header">
        <div id="detail-title">Select a cluster</div>
        <div id="detail-meta"></div>
      </div>
      <div id="detail-body">
        <div id="detail-placeholder" style="color:var(--text-dim);padding:20px 0;">
          Click a cluster in the sidebar to explore its tokens and expert routing patterns.
        </div>
        <div id="detail-content" style="display:none">
          <div>
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
              <div class="section-title" style="margin-bottom:0">Document Previews (tokens in cluster highlighted)</div>
              <select id="doc-preview-count" onchange="renderDocPreviews()" style="background:var(--surface2);border:1px solid var(--border);color:var(--text-dim);padding:4px 8px;border-radius:6px;font-size:11px;outline:none;">
                <option value="10">Top 10</option>
                <option value="20" selected>Top 20</option>
                <option value="50">Top 50</option>
                <option value="100">Top 100</option>
                <option value="200">Top 200</option>
              </select>
            </div>
            <div id="doc-preview-list"></div>
            <div id="doc-preview-load-more-wrap" style="text-align:center;padding:8px 0;"></div>
          </div>
          <div>
            <div class="section-title">Source Breakdown</div>
            <div id="src-breakdown"></div>
          </div>
        </div>
      </div>
    </div>

    <div id="docs-view">
      <div id="docs-view-header">
        <div class="section-title">Document-Cluster Spread Analysis</div>
        <div id="docs-view-stats"></div>
        <div id="docs-histogram-wrap">
          <div style="font-size:10px;color:var(--text-dim);margin-bottom:2px;">Distribution: # clusters per document</div>
          <div id="docs-histogram"></div>
        </div>
        <div id="docs-view-controls" style="margin-top:10px;">
          <input id="doc-filter" type="text" placeholder="Filter by source…" oninput="filterDocList()">
          <select id="doc-sort-select" onchange="sortDocList()">
            <option value="spread-desc">Sort: Most spread</option>
            <option value="spread-asc">Sort: Least spread</option>
            <option value="size-desc">Sort: Most tokens</option>
            <option value="id-asc">Sort: Doc ID</option>
          </select>
          <span id="docs-count" style="color:var(--text-dim);font-size:11px;margin-left:auto;"></span>
        </div>
      </div>
      <div id="docs-list-wrap"></div>
    </div>

    <div id="tokens-view">
      <div id="tokens-view-header">
        <div class="section-title">Unique Token Cluster Distribution</div>
        <div style="color:var(--text-dim);font-size:12px;margin-bottom:6px;">
          For each unique token, shows which clusters it tends to be routed to.
          Low entropy = token is consistently routed to the same cluster(s). High entropy = spread across many.
        </div>
        <div id="tokens-view-controls">
          <input id="token-filter" type="text" placeholder="Search tokens…" oninput="filterUniqueTokens()">
          <select id="token-sort-select" onchange="sortUniqueTokens()">
            <option value="count-desc">Sort: Most frequent</option>
            <option value="count-asc">Sort: Least frequent</option>
            <option value="entropy-desc">Sort: Highest entropy (most spread)</option>
            <option value="entropy-asc">Sort: Lowest entropy (most focused)</option>
            <option value="clusters-desc">Sort: Most clusters</option>
            <option value="clusters-asc">Sort: Fewest clusters</option>
          </select>
          <span id="tokens-count" style="color:var(--text-dim);font-size:11px;margin-left:auto;"></span>
        </div>
      </div>
      <div id="tokens-list-wrap"></div>
      <div id="token-detail">
        <div id="token-detail-header">
          <button id="token-detail-back" onclick="closeTokenDetail()">&larr; Back to tokens</button>
          <div id="token-detail-title"></div>
          <div id="token-detail-meta"></div>
        </div>
        <div id="token-detail-body"></div>
      </div>
    </div>

    <div id="umap-view">
      <div id="umap-controls">
        <span style="color:var(--text-dim)">Color by:</span>
        <label class="umap-radio"><input type="radio" name="umap-color" value="cluster" checked onchange="redrawUmap()"> Cluster</label>
        <label class="umap-radio"><input type="radio" name="umap-color" value="source" onchange="redrawUmap()"> Data Source</label>
        <label class="umap-radio"><input type="radio" name="umap-color" value="docspread" onchange="redrawUmap()"> Doc Spread</label>
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

<div id="doc-reader-modal">
  <div id="doc-reader-inner">
    <div id="doc-reader-header">
      <div id="doc-reader-title"></div>
      <div id="doc-reader-meta"></div>
      <button id="doc-reader-close" onclick="closeDocReader()">Close</button>
    </div>
    <div id="doc-reader-legend">
      <span style="color:var(--text-dim)">Clusters:</span>
      <span id="doc-reader-legend-tags"></span>
      <span style="color:var(--text-dim);margin-left:8px;">Unsampled tokens shown in gray</span>
    </div>
    <div id="doc-reader-body"></div>
    <div id="doc-reader-tooltip"></div>
  </div>
</div>

<script>
const CLUSTERS = {clusters_json};
const DOCS = {docs_json};
const DOC_STATS = {doc_stats_json};
const AGG_STATS = {agg_stats_json};
const DOC_TEXTS = {doc_texts_json};
const UNIQUE_TOKENS = {unique_tokens_json};
const CAT_COLORS = {cat_colors_json};
const SOURCE_COLORS = {{
  'dclm': '#64b0f4', 'starcoder': '#4caf78', 'pes2o': '#f4a04a',
  'proofpile-2-arxiv': '#c084f5', 'proofpile-2-open-web-math': '#f5c842',
  'proofpile-2-stack': '#42c8f5', 'wikipedia': '#aaaaaa',
}};

let selectedCluster = null;
let currentView = 'detail';
let activeCat = 'all';
let umapInitialized = false;

// Build doc lookup: doc_index -> DOC_STATS entry
const docLookup = {{}};
DOC_STATS.forEach(ds => {{ docLookup[ds.di] = ds; }});
// Build doc text lookup: doc_index -> DOC_TEXTS entry
const docTextLookup = {{}};
DOC_TEXTS.forEach(dt => {{ docTextLookup[dt.di] = dt; }});

// Precompute per-token: how many clusters its parent doc spans
const docSpreadByToken = {{}};
DOCS.forEach(d => {{
  if (!(d.di in docSpreadByToken)) {{
    const ds = docLookup[d.di];
    docSpreadByToken[d.di] = ds ? ds.nClusters : 1;
  }}
}});

function init() {{
  buildCatButtons();
  buildSidebar();
  filterClusters();
  buildDocStats();
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
  let vis = 0;
  document.querySelectorAll('.cluster-item').forEach(el => {{
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
  document.querySelectorAll('.view-tab').forEach((t,i) => t.classList.toggle('active', ['detail','docs','tokens','umap'][i] === view));
  document.getElementById('detail-view').classList.toggle('active', view === 'detail');
  document.getElementById('docs-view').classList.toggle('active', view === 'docs');
  document.getElementById('tokens-view').classList.toggle('active', view === 'tokens');
  document.getElementById('umap-view').classList.toggle('active', view === 'umap');
  document.getElementById('sidebar').style.display = view === 'detail' ? '' : 'none';
  document.getElementById('filterbar').style.display = view === 'detail' ? '' : 'none';
  if (view === 'umap' && !umapInitialized) {{ initUmap(); umapInitialized = true; }}
  if (view === 'umap') redrawUmap();
  if (view === 'docs') renderDocList();
  if (view === 'tokens') renderUniqueTokens();
}}

function selectCluster(id) {{
  selectedCluster = id;
  const c = CLUSTERS.find(x => x.id === id);
  if (!c) return;

  document.querySelectorAll('.cluster-item').forEach(el => {{
    el.classList.toggle('selected', parseInt(el.dataset.id) === id);
  }});

  document.getElementById('detail-title').textContent = `Cluster ${{c.id}} — ${{c.label}}`;
  document.getElementById('detail-meta').innerHTML =
    `<span style="background:${{c.color}}22;color:${{c.color}};padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600">${{c.category}}</span>  ` +
    `${{c.size.toLocaleString()}} tokens`;

  document.getElementById('detail-placeholder').style.display = 'none';
  document.getElementById('detail-content').style.display = '';

  // Sources
  const cid = c.id;
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

  // Render document previews
  currentClusterId = id;
  docPreviewsShown = 0;
  document.getElementById('doc-preview-list').innerHTML = '';
  renderDocPreviews();
}}

let clusterTokens = [];
let filteredClusterTokens = [];
let tokenBrowserOpen = false;

function toggleTokenBrowser() {{
  tokenBrowserOpen = !tokenBrowserOpen;
  document.getElementById('token-browser-wrap').style.display = tokenBrowserOpen ? '' : 'none';
  document.getElementById('token-browser-toggle').textContent = tokenBrowserOpen ? 'Hide' : 'Show';
  if (tokenBrowserOpen) renderTokenBrowser();
}}

function filterTokenBrowser() {{
  const q = document.getElementById('token-search').value.toLowerCase();
  filteredClusterTokens = q ? clusterTokens.filter(d =>
    d.s.includes(q) || d.target.toLowerCase().includes(q) ||
    d.before.toLowerCase().includes(q) || d.after.toLowerCase().includes(q)
  ) : [...clusterTokens];
  renderTokenBrowser();
}}

function renderTokenBrowser() {{
  const wrap = document.getElementById('token-browser-list');
  wrap.innerHTML = '';
  const limit = 200;
  filteredClusterTokens.slice(0, limit).forEach((tok, i) => {{
    const badgeClass = 'badge-' + tok.s.replace(/[^a-z0-9]/g, '-');
    const div = document.createElement('div');
    div.className = 'token-context';
    div.innerHTML = `
      <div class="token-context-header">
        <span class="source-badge ${{badgeClass}}">${{tok.s}}</span>
        <span style="color:var(--text-dim);font-size:11px">doc #${{tok.di}} · pos ${{tok.tp}}</span>
        <span style="margin-left:auto;color:var(--text-dim);font-size:11px">#${{i+1}}</span>
      </div>
      <div class="token-context-text"><span class="token-dim">${{escHtml(tok.before)}}</span><span class="token-highlight">${{escHtml(tok.target)}}</span><span class="token-dim">${{escHtml(tok.after)}}</span></div>
      ${{renderDocSpreadBar(tok.di)}}`;
    wrap.appendChild(div);
  }});
  if (filteredClusterTokens.length > limit) {{
    const more = document.createElement('div');
    more.style.cssText = 'color:var(--text-dim);text-align:center;padding:12px;font-size:12px;';
    more.textContent = `… ${{(filteredClusterTokens.length - limit).toLocaleString()}} more tokens (use filter to narrow)`;
    wrap.appendChild(more);
  }}
}}

// ── Document previews for cluster detail ──
let currentClusterId = null;
let docPreviewsShown = 0;
let cachedDocsWithCluster = [];

function renderDocPreviews() {{
  if (currentClusterId === null) return;

  const batchSize = parseInt(document.getElementById('doc-preview-count').value) || 20;
  const cid = currentClusterId;
  const wrap = document.getElementById('doc-preview-list');
  const loadMoreWrap = document.getElementById('doc-preview-load-more-wrap');

  // Rebuild sorted doc list if starting fresh
  if (docPreviewsShown === 0) {{
    cachedDocsWithCluster = DOC_STATS
      .filter(ds => ds.clusters[cid] !== undefined)
      .map(ds => ({{ ...ds, clusterCount: ds.clusters[cid], pct: ds.clusters[cid] / ds.total }}))
      .sort((a, b) => b.pct - a.pct);
  }}

  const c = CLUSTERS.find(x => x.id === cid);
  const hlColor = c ? c.color : '#fbbf24';
  const end = Math.min(docPreviewsShown + batchSize, cachedDocsWithCluster.length);

  for (let idx = docPreviewsShown; idx < end; idx++) {{
    const ds = cachedDocsWithCluster[idx];
    const dt = docTextLookup[ds.di];
    if (!dt) continue;

    const pctStr = (ds.pct * 100).toFixed(1);
    const badgeClass = 'badge-' + ds.s.replace(/[^a-z0-9]/g, '-');

    const card = document.createElement('div');
    card.className = 'doc-preview-card';

    const header = document.createElement('div');
    header.className = 'doc-preview-header';
    header.innerHTML = `
      <div class="doc-preview-pct" style="color:${{hlColor}}">${{pctStr}}%</div>
      <span class="source-badge ${{badgeClass}}">${{ds.s}}</span>
      <div class="doc-preview-info">Doc #${{ds.di}} · ${{ds.clusterCount}}/${{ds.total}} tokens in cluster · ${{ds.nClusters}} clusters total</div>
      <span style="color:var(--text-dim);font-size:11px;cursor:pointer" title="Open in full reader">⤢</span>`;
    header.onclick = () => openDocReader(ds.di);
    card.appendChild(header);

    const body = document.createElement('div');
    body.className = 'doc-preview-body';
    dt.tokens.forEach(tok => {{
      const span = document.createElement('span');
      span.textContent = tok.t;
      if (tok.c === cid) {{
        span.className = 'tok-hl';
      }} else {{
        span.className = 'tok-dim';
      }}
      body.appendChild(span);
    }});
    card.appendChild(body);
    wrap.appendChild(card);
  }}

  docPreviewsShown = end;

  // Update load more button
  const remaining = cachedDocsWithCluster.length - docPreviewsShown;
  if (remaining > 0) {{
    loadMoreWrap.innerHTML = `<button onclick="renderDocPreviews()" style="background:var(--surface2);border:1px solid var(--border);color:var(--text);padding:8px 24px;border-radius:6px;cursor:pointer;font-size:12px;">Load ${{Math.min(batchSize, remaining)}} more (${{remaining}} remaining)</button>`;
  }} else {{
    loadMoreWrap.innerHTML = docPreviewsShown > 0
      ? `<span style="color:var(--text-dim);font-size:11px;">All ${{cachedDocsWithCluster.length}} documents shown</span>`
      : '<div style="color:var(--text-dim);font-size:12px;">No documents found</div>';
  }}
}}

// ── Doc spread bar for a token ──
function renderDocSpreadBar(docIndex) {{
  const ds = docLookup[docIndex];
  if (!ds) return '';
  const entries = Object.entries(ds.clusters).sort((a,b) => b[1] - a[1]);
  const total = ds.total;
  const segs = entries.map(([cid, cnt]) => {{
    const c = CLUSTERS.find(x => x.id === parseInt(cid));
    const color = c ? c.color : '#555';
    const pct = (cnt / total * 100).toFixed(1);
    return `<div class="doc-spread-bar-seg" style="width:${{pct}}%;background:${{color}}" title="Cluster ${{cid}}: ${{cnt}} tokens (${{pct}}%)"></div>`;
  }}).join('');
  return `<div class="doc-spread-info">Doc #${{docIndex}} spans ${{ds.nClusters}} cluster${{ds.nClusters > 1 ? 's' : ''}} (${{total}} tokens sampled)</div>
    <div class="doc-spread-bar">${{segs}}</div>`;
}}

// ── Aggregate doc stats ──
function buildDocStats() {{
  const el = document.getElementById('docs-view-stats');
  el.innerHTML = `
    <div class="stat-card"><div class="stat-value">${{AGG_STATS.num_docs}}</div><div class="stat-label">Documents</div></div>
    <div class="stat-card"><div class="stat-value">${{AGG_STATS.mean_clusters}}</div><div class="stat-label">Mean clusters/doc</div></div>
    <div class="stat-card"><div class="stat-value">${{AGG_STATS.median_clusters}}</div><div class="stat-label">Median clusters/doc</div></div>
    <div class="stat-card"><div class="stat-value">${{AGG_STATS.max_clusters}}</div><div class="stat-label">Max clusters/doc</div></div>
    <div class="stat-card"><div class="stat-value">${{AGG_STATS.std_clusters}}</div><div class="stat-label">Std dev</div></div>`;

  // Histogram
  const hist = document.getElementById('docs-histogram');
  hist.innerHTML = '';
  const entries = Object.entries(AGG_STATS.histogram).map(([k,v]) => [parseInt(k), v]);
  entries.sort((a,b) => a[0] - b[0]);
  const maxCount = Math.max(...entries.map(e => e[1]));
  entries.forEach(([nClusters, count]) => {{
    const h = Math.max(4, (count / maxCount) * 55);
    const bar = document.createElement('div');
    bar.className = 'hist-bar';
    bar.style.height = h + 'px';
    bar.style.width = Math.max(8, Math.floor(400 / entries.length) - 2) + 'px';
    bar.title = `${{nClusters}} clusters: ${{count}} docs`;
    bar.innerHTML = `<div class="hist-label">${{nClusters}}</div>`;
    hist.appendChild(bar);
  }});
}}

// ── Documents browser ──
let sortedDocStats = [...DOC_STATS];
let filteredDocStats = [...DOC_STATS];

function filterDocList() {{
  const q = document.getElementById('doc-filter').value.toLowerCase();
  filteredDocStats = q ? sortedDocStats.filter(ds => ds.s.includes(q)) : [...sortedDocStats];
  renderDocList();
}}

function sortDocList() {{
  const mode = document.getElementById('doc-sort-select').value;
  const sorted = [...DOC_STATS];
  if (mode === 'spread-desc') sorted.sort((a,b) => b.nClusters - a.nClusters || b.total - a.total);
  else if (mode === 'spread-asc') sorted.sort((a,b) => a.nClusters - b.nClusters || b.total - a.total);
  else if (mode === 'size-desc') sorted.sort((a,b) => b.total - a.total);
  else sorted.sort((a,b) => a.di - b.di);
  sortedDocStats = sorted;
  filterDocList();
}}

function renderDocList() {{
  const wrap = document.getElementById('docs-list-wrap');
  wrap.innerHTML = '';
  document.getElementById('docs-count').textContent = `${{filteredDocStats.length}} documents`;
  const limit = 200;
  filteredDocStats.slice(0, limit).forEach(ds => {{
    const card = document.createElement('div');
    card.className = 'doc-card';
    const badgeClass = 'badge-' + ds.s.replace(/[^a-z0-9]/g, '-');
    const entries = Object.entries(ds.clusters).sort((a,b) => b[1] - a[1]);
    const barSegs = entries.map(([cid, cnt]) => {{
      const c = CLUSTERS.find(x => x.id === parseInt(cid));
      const color = c ? c.color : '#555';
      const pct = (cnt / ds.total * 100).toFixed(1);
      return `<div class="doc-cluster-bar-seg" style="width:${{pct}}%;background:${{color}}" title="Cluster ${{cid}}${{c ? ' ('+c.label+')' : ''}}: ${{cnt}} tokens (${{pct}}%)" onclick="event.stopPropagation();selectCluster(${{cid}});setView('detail');"></div>`;
    }}).join('');
    const tags = entries.slice(0, 8).map(([cid, cnt]) => {{
      const c = CLUSTERS.find(x => x.id === parseInt(cid));
      const color = c ? c.color : '#555';
      const pct = (cnt / ds.total * 100).toFixed(0);
      return `<span class="doc-cluster-tag" style="border-color:${{color}}40;color:${{color}}">C${{cid}} ${{pct}}%</span>`;
    }}).join('');
    const moreTags = entries.length > 8 ? `<span class="doc-cluster-tag" style="color:var(--text-dim)">+${{entries.length - 8}} more</span>` : '';
    card.innerHTML = `
      <div class="doc-card-header">
        <span class="doc-card-id">Doc #${{ds.di}}</span>
        <span class="source-badge ${{badgeClass}}">${{ds.s}}</span>
        <span class="doc-card-stats">${{ds.total}} tokens · ${{ds.nClusters}} cluster${{ds.nClusters > 1 ? 's' : ''}}</span>
      </div>
      <div class="doc-cluster-bar">${{barSegs}}</div>
      <div class="doc-card-clusters">${{tags}}${{moreTags}}</div>`;
    card.onclick = () => openDocReader(ds.di);
    wrap.appendChild(card);
  }});
  if (filteredDocStats.length > limit) {{
    const more = document.createElement('div');
    more.style.cssText = 'color:var(--text-dim);text-align:center;padding:12px;font-size:12px;';
    more.textContent = `… ${{(filteredDocStats.length - limit).toLocaleString()}} more documents (use filter to narrow)`;
    wrap.appendChild(more);
  }}
}}

// ── Unique tokens view ──
let sortedUniqueTokens = [...UNIQUE_TOKENS];
let filteredUniqueTokens = [...UNIQUE_TOKENS];

function filterUniqueTokens() {{
  const q = document.getElementById('token-filter').value.toLowerCase();
  filteredUniqueTokens = q ? sortedUniqueTokens.filter(ut =>
    ut.t.toLowerCase().includes(q)) : [...sortedUniqueTokens];
  renderUniqueTokens();
}}

function sortUniqueTokens() {{
  const mode = document.getElementById('token-sort-select').value;
  const sorted = [...UNIQUE_TOKENS];
  if (mode === 'count-desc') sorted.sort((a,b) => b.n - a.n);
  else if (mode === 'count-asc') sorted.sort((a,b) => a.n - b.n);
  else if (mode === 'entropy-desc') sorted.sort((a,b) => b.ent - a.ent || b.n - a.n);
  else if (mode === 'entropy-asc') sorted.sort((a,b) => a.ent - b.ent || b.n - a.n);
  else if (mode === 'clusters-desc') sorted.sort((a,b) => b.nc - a.nc || b.n - a.n);
  else if (mode === 'clusters-asc') sorted.sort((a,b) => a.nc - b.nc || b.n - a.n);
  sortedUniqueTokens = sorted;
  filterUniqueTokens();
}}

function renderUniqueTokens() {{
  const wrap = document.getElementById('tokens-list-wrap');
  wrap.innerHTML = '';
  document.getElementById('tokens-count').textContent =
    `${{filteredUniqueTokens.length}} of ${{UNIQUE_TOKENS.length}} unique tokens`;
  const limit = 200;
  const maxEnt = Math.log2(CLUSTERS.length);  // max possible entropy
  filteredUniqueTokens.slice(0, limit).forEach(ut => {{
    const card = document.createElement('div');
    card.className = 'utok-card';
    const entries = Object.entries(ut.clusters).map(([cid, cnt]) => [parseInt(cid), cnt]).sort((a,b) => b[1] - a[1]);
    const barSegs = entries.map(([cid, cnt]) => {{
      const c = CLUSTERS.find(x => x.id === cid);
      const color = c ? c.color : '#555';
      const pct = (cnt / ut.n * 100).toFixed(1);
      const label = c ? c.label : `Cluster ${{cid}}`;
      return `<div class="utok-cluster-bar-seg" style="width:${{pct}}%;background:${{color}}" title="Cluster ${{cid}} (${{label}}): ${{cnt}} (${{pct}}%)" onclick="event.stopPropagation();selectCluster(${{cid}});setView('detail');"></div>`;
    }}).join('');
    const tags = entries.slice(0, 6).map(([cid, cnt]) => {{
      const c = CLUSTERS.find(x => x.id === cid);
      const color = c ? c.color : '#555';
      const pct = (cnt / ut.n * 100).toFixed(0);
      return `<span class="utok-cluster-tag" style="border-color:${{color}}40;color:${{color}}">C${{cid}} ${{pct}}%</span>`;
    }}).join('');
    const moreTags = entries.length > 6 ? `<span class="utok-cluster-tag" style="color:var(--text-dim)">+${{entries.length - 6}} more</span>` : '';
    // Color entropy badge: green (low/focused) to red (high/spread)
    const entRatio = ut.ent / maxEnt;
    const entColor = entRatio < 0.3 ? '#27AE60' : entRatio < 0.6 ? '#F39C12' : '#E74C3C';
    card.style.cursor = 'pointer';
    card.innerHTML = `
      <div class="utok-header">
        <span class="utok-text">${{escHtml(ut.t)}}</span>
        <span class="utok-stats">${{ut.n.toLocaleString()}} occurrences · ${{ut.nc}} cluster${{ut.nc > 1 ? 's' : ''}}</span>
        <span class="utok-entropy" style="background:${{entColor}}22;color:${{entColor}}">entropy: ${{ut.ent.toFixed(2)}} / ${{maxEnt.toFixed(2)}}</span>
      </div>
      <div class="utok-cluster-bar">${{barSegs}}</div>
      <div class="utok-clusters">${{tags}}${{moreTags}}</div>`;
    card.onclick = (e) => {{ if (!e.target.closest('.utok-cluster-bar-seg')) openTokenDetail(ut); }};
    wrap.appendChild(card);
  }});
  if (filteredUniqueTokens.length > limit) {{
    const more = document.createElement('div');
    more.style.cssText = 'color:var(--text-dim);text-align:center;padding:12px;font-size:12px;';
    more.textContent = `… ${{(filteredUniqueTokens.length - limit).toLocaleString()}} more tokens (use search to filter)`;
    wrap.appendChild(more);
  }}
}}

// ── Token detail view ──
function openTokenDetail(ut) {{
  // Hide list, show detail
  document.getElementById('tokens-list-wrap').style.display = 'none';
  document.getElementById('tokens-view-header').style.display = 'none';
  const detail = document.getElementById('token-detail');
  detail.classList.add('active');

  const maxEnt = Math.log2(CLUSTERS.length);
  const entRatio = ut.ent / maxEnt;
  const entColor = entRatio < 0.3 ? '#27AE60' : entRatio < 0.6 ? '#F39C12' : '#E74C3C';

  document.getElementById('token-detail-title').innerHTML =
    `<span style="background:var(--highlight);color:#000;padding:2px 8px;border-radius:4px;font-family:monospace;font-size:18px;">${{escHtml(ut.t)}}</span>`;
  document.getElementById('token-detail-meta').innerHTML =
    `${{ut.n.toLocaleString()}} occurrences · ${{ut.nc}} cluster${{ut.nc > 1 ? 's' : ''}} · ` +
    `<span style="color:${{entColor}}">entropy ${{ut.ent.toFixed(2)}} / ${{maxEnt.toFixed(2)}}</span>` +
    `  ·  token ID: ${{ut.id}}`;

  // Build cluster sections sorted by count desc
  const body = document.getElementById('token-detail-body');
  body.innerHTML = '';
  const entries = Object.entries(ut.clusters).map(([cid, cnt]) => [parseInt(cid), cnt]).sort((a,b) => b[1] - a[1]);

  entries.forEach(([cid, cnt]) => {{
    const c = CLUSTERS.find(x => x.id === cid);
    const color = c ? c.color : '#555';
    const label = c ? c.label : `Cluster ${{cid}}`;
    const pct = (cnt / ut.n * 100).toFixed(1);
    const sectionId = `td-section-${{cid}}`;

    const section = document.createElement('div');
    section.className = 'td-cluster-section';
    section.innerHTML = `
      <div class="td-cluster-header" onclick="toggleTokenClusterUsages('${{sectionId}}', ${{ut.id}}, ${{cid}}, this)">
        <div class="td-cluster-dot" style="background:${{color}}"></div>
        <div class="td-cluster-name" style="color:${{color}}">Cluster ${{cid}} — ${{label}}</div>
        <div class="td-cluster-pct-bar"><div class="td-cluster-pct-fill" style="width:${{pct}}%;background:${{color}}"></div></div>
        <div class="td-cluster-count">${{cnt}} (${{pct}}%)</div>
      </div>
      <div class="td-cluster-usages" id="${{sectionId}}"></div>`;
    body.appendChild(section);
  }});

  // Auto-expand the first cluster
  if (entries.length > 0) {{
    const firstId = `td-section-${{entries[0][0]}}`;
    const firstHeader = body.querySelector('.td-cluster-header');
    toggleTokenClusterUsages(firstId, ut.id, entries[0][0], firstHeader);
  }}
}}

function toggleTokenClusterUsages(sectionId, tokenId, clusterId, headerEl) {{
  const usagesEl = document.getElementById(sectionId);
  const isExpanded = usagesEl.classList.contains('expanded');

  if (isExpanded) {{
    usagesEl.classList.remove('expanded');
    headerEl.classList.remove('expanded');
    return;
  }}

  // Collapse all others
  document.querySelectorAll('.td-cluster-usages.expanded').forEach(el => el.classList.remove('expanded'));
  document.querySelectorAll('.td-cluster-header.expanded').forEach(el => el.classList.remove('expanded'));

  usagesEl.classList.add('expanded');
  headerEl.classList.add('expanded');

  // Find usages from DOCS matching this token_id and cluster
  if (usagesEl.dataset.loaded) return;  // already loaded

  const matches = DOCS.filter(d => d.tid === tokenId && d.c === clusterId);
  usagesEl.innerHTML = '';

  if (matches.length === 0) {{
    usagesEl.innerHTML = '<div style="color:var(--text-dim);padding:8px;font-size:12px;">No usages found in subsampled set for this cluster.</div>';
  }} else {{
    const limit = 100;
    matches.slice(0, limit).forEach((tok, i) => {{
      const badgeClass = 'badge-' + tok.s.replace(/[^a-z0-9]/g, '-');
      const div = document.createElement('div');
      div.className = 'token-context';
      div.innerHTML = `
        <div class="token-context-header">
          <span class="source-badge ${{badgeClass}}">${{tok.s}}</span>
          <span style="color:var(--text-dim);font-size:11px">doc #${{tok.di}} · pos ${{tok.tp}}</span>
          <span style="margin-left:auto;color:var(--text-dim);font-size:11px">#${{i+1}}</span>
        </div>
        <div class="token-context-text"><span class="token-dim">${{escHtml(tok.before)}}</span><span class="token-highlight">${{escHtml(tok.target)}}</span><span class="token-dim">${{escHtml(tok.after)}}</span></div>`;
      usagesEl.appendChild(div);
    }});
    if (matches.length > limit) {{
      const more = document.createElement('div');
      more.style.cssText = 'color:var(--text-dim);text-align:center;padding:8px;font-size:11px;';
      more.textContent = `… ${{matches.length - limit}} more usages`;
      usagesEl.appendChild(more);
    }}
  }}
  usagesEl.dataset.loaded = '1';
}}

function closeTokenDetail() {{
  document.getElementById('token-detail').classList.remove('active');
  document.getElementById('tokens-list-wrap').style.display = '';
  document.getElementById('tokens-view-header').style.display = '';
}}

// ── UMAP ──
let umapCanvas, umapCtx, umapScale = {{}};

function initUmap() {{
  umapCanvas = document.getElementById('umap-canvas');
  umapCtx = umapCanvas.getContext('2d');
  buildUmapLegend();
  new ResizeObserver(() => redrawUmap()).observe(document.getElementById('umap-canvas-wrap'));
  umapCanvas.addEventListener('mousemove', onUmapMouseMove);
  umapCanvas.addEventListener('click', onUmapClick);
  umapCanvas.addEventListener('mouseleave', () => document.getElementById('umap-tooltip').style.display = 'none');
}}

function spreadToColor(n) {{
  // Blue (1 cluster) -> Yellow (mid) -> Red (max)
  const maxSpread = AGG_STATS.max_clusters;
  const t = Math.min(1, (n - 1) / Math.max(1, maxSpread - 1));
  const r = Math.round(t < 0.5 ? t * 2 * 255 : 255);
  const g = Math.round(t < 0.5 ? 100 + t * 310 : 255 * (1 - (t - 0.5) * 2));
  const b = Math.round(t < 0.5 ? 255 * (1 - t * 2) : 0);
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function getUmapColor(d) {{
  const mode = document.querySelector('input[name="umap-color"]:checked').value;
  if (mode === 'cluster') {{
    const c = CLUSTERS.find(x => x.id === d.c);
    return c ? c.color : '#555';
  }} else if (mode === 'docspread') {{
    const spread = docSpreadByToken[d.di] || 1;
    return spreadToColor(spread);
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
  // First pass: draw non-highlighted tokens
  DOCS.forEach(d => {{
    if (highlightedDoc !== null && d.di === highlightedDoc) return;
    const [px, py] = toCanvas(d.x, d.y);
    const isSelected = d.c === selectedCluster;
    const dimmed = highlightedDoc !== null;
    umapCtx.beginPath();
    umapCtx.arc(px, py, isSelected ? 3 : 1.5, 0, Math.PI*2);
    umapCtx.fillStyle = getUmapColor(d) + (dimmed ? '33' : isSelected ? 'ff' : '88');
    umapCtx.fill();
  }});
  // Second pass: draw highlighted document tokens on top
  if (highlightedDoc !== null) {{
    DOCS.forEach(d => {{
      if (d.di !== highlightedDoc) return;
      const [px, py] = toCanvas(d.x, d.y);
      const color = getUmapColor(d);
      // Outer glow
      umapCtx.beginPath();
      umapCtx.arc(px, py, 6, 0, Math.PI*2);
      umapCtx.fillStyle = color + '44';
      umapCtx.fill();
      // Inner dot
      umapCtx.beginPath();
      umapCtx.arc(px, py, 4, 0, Math.PI*2);
      umapCtx.fillStyle = color;
      umapCtx.fill();
      // White border
      umapCtx.beginPath();
      umapCtx.arc(px, py, 4, 0, Math.PI*2);
      umapCtx.strokeStyle = '#fff';
      umapCtx.lineWidth = 1;
      umapCtx.stroke();
    }});
  }}
  buildUmapLegend();
}}

function toCanvas(x, y) {{
  const {{xMin,xMax,yMin,yMax,W,H,pad}} = umapScale;
  return [pad + (x - xMin) / (xMax - xMin) * (W - 2*pad), pad + (y - yMin) / (yMax - yMin) * (H - 2*pad)];
}}
function fromCanvas(px, py) {{
  const {{xMin,xMax,yMin,yMax,W,H,pad}} = umapScale;
  return [xMin + (px - pad) / (W - 2*pad) * (xMax - xMin), yMin + (py - pad) / (H - 2*pad) * (yMax - yMin)];
}}

function onUmapMouseMove(e) {{
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
  const tooltip = document.getElementById('umap-tooltip');
  if (best) {{
    const c = CLUSTERS.find(x => x.id === best.c);
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX + 14) + 'px';
    tooltip.style.top = (e.clientY - 10) + 'px';
    tooltip.innerHTML = `<b>Cluster ${{best.c}}</b>: ${{c ? c.label : ''}}<br>
      <span style="color:var(--text-dim)">${{best.s}} · doc #${{best.di}} · pos ${{best.tp}}</span><br>
      <span style="color:var(--text-dim)">${{escHtml(best.before)}}</span><span style="background:#fbbf24;color:#000;padding:0 2px;border-radius:2px">${{escHtml(best.target)}}</span><span style="color:var(--text-dim)">${{escHtml(best.after)}}</span>`;
  }} else {{
    tooltip.style.display = 'none';
  }}
}}

let highlightedDoc = null;

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
    // Toggle document highlight on UMAP
    if (highlightedDoc === best.di) {{
      highlightedDoc = null;
    }} else {{
      highlightedDoc = best.di;
    }}
    redrawUmap();
    const ds = docLookup[best.di];
    const status = document.getElementById('umap-status');
    if (highlightedDoc !== null && ds) {{
      status.innerHTML = `Highlighting doc #${{best.di}} (${{ds.s}}) — ${{ds.total}} tokens, ${{ds.nClusters}} clusters. Click again to clear.`;
    }} else {{
      status.textContent = '';
    }}
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
  }} else if (mode === 'docspread') {{
    const maxS = AGG_STATS.max_clusters;
    const steps = Math.min(maxS, 8);
    for (let i = 0; i <= steps; i++) {{
      const n = Math.round(1 + i * (maxS - 1) / steps);
      const item = document.createElement('div');
      item.className = 'legend-item';
      item.innerHTML = `<div class="legend-dot" style="background:${{spreadToColor(n)}}"></div><div class="legend-label">${{n}} cluster${{n>1?'s':''}}</div>`;
      leg.appendChild(item);
    }}
    leg.innerHTML += `<div style="margin-top:6px;color:var(--text-dim);font-size:10px">Mean: ${{AGG_STATS.mean_clusters}}<br>Median: ${{AGG_STATS.median_clusters}}</div>`;
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

// ── Document reader ──
let docReaderHighlight = null;  // cluster id to highlight, or null for all

function openDocReader(docIndex) {{
  const dt = docTextLookup[docIndex];
  if (!dt) return;
  const ds = docLookup[docIndex];

  document.getElementById('doc-reader-title').textContent = `Document #${{docIndex}}`;
  const badgeClass = 'badge-' + dt.s.replace(/[^a-z0-9]/g, '-');
  document.getElementById('doc-reader-meta').innerHTML =
    `<span class="source-badge ${{badgeClass}}">${{dt.s}}</span>  ` +
    `${{dt.tokens.length}} tokens · ${{ds ? ds.nClusters + ' clusters' : ''}}`;

  // Build legend of clusters present in this doc
  const clustersInDoc = {{}};
  dt.tokens.forEach(tok => {{
    if (tok.c >= 0) clustersInDoc[tok.c] = (clustersInDoc[tok.c] || 0) + 1;
  }});
  const legendEl = document.getElementById('doc-reader-legend-tags');
  legendEl.innerHTML = '';
  const allBtn = document.createElement('span');
  allBtn.className = 'legend-cluster-tag';
  allBtn.style.cssText = 'background:var(--surface2);color:var(--text);border-color:var(--border);';
  allBtn.textContent = 'All';
  allBtn.onclick = () => {{ docReaderHighlight = null; renderDocReaderBody(docIndex); updateDocReaderLegend(docIndex); }};
  legendEl.appendChild(allBtn);

  Object.entries(clustersInDoc).sort((a,b) => b[1] - a[1]).forEach(([cid, cnt]) => {{
    const c = CLUSTERS.find(x => x.id === parseInt(cid));
    const color = c ? c.color : '#555';
    const label = c ? c.label : `Cluster ${{cid}}`;
    const tag = document.createElement('span');
    tag.className = 'legend-cluster-tag';
    tag.style.cssText = `background:${{color}}33;color:${{color}};`;
    tag.textContent = `${{cid}}: ${{label}} (${{cnt}})`;
    tag.title = `${{cnt}} tokens in cluster ${{cid}}`;
    tag.dataset.cid = cid;
    tag.onclick = () => {{
      docReaderHighlight = docReaderHighlight === parseInt(cid) ? null : parseInt(cid);
      renderDocReaderBody(docIndex);
      updateDocReaderLegend(docIndex);
    }};
    legendEl.appendChild(tag);
  }});

  docReaderHighlight = null;
  renderDocReaderBody(docIndex);
  document.getElementById('doc-reader-modal').classList.add('open');
}}

function updateDocReaderLegend(docIndex) {{
  document.querySelectorAll('#doc-reader-legend-tags .legend-cluster-tag').forEach(tag => {{
    const cid = tag.dataset.cid;
    if (cid === undefined) {{
      // "All" button
      tag.classList.toggle('dim', docReaderHighlight !== null);
    }} else {{
      tag.classList.toggle('dim', docReaderHighlight !== null && docReaderHighlight !== parseInt(cid));
    }}
  }});
}}

function renderDocReaderBody(docIndex) {{
  const dt = docTextLookup[docIndex];
  if (!dt) return;
  const body = document.getElementById('doc-reader-body');
  body.innerHTML = '';

  dt.tokens.forEach((tok, pos) => {{
    const span = document.createElement('span');
    const text = tok.t;

    if (tok.c >= 0) {{
      // Sampled token — color by cluster
      const c = CLUSTERS.find(x => x.id === tok.c);
      const color = c ? c.color : '#555';
      span.className = 'doc-token sampled';

      if (docReaderHighlight === null) {{
        // Show all clusters
        span.style.cssText = `background:${{color}}44;color:${{color}};`;
      }} else if (docReaderHighlight === tok.c) {{
        // This token's cluster is highlighted
        span.style.cssText = `background:${{color}}88;color:#fff;font-weight:700;`;
      }} else {{
        // Different cluster — dim it
        span.style.cssText = `color:var(--text-dim);opacity:0.4;`;
      }}

      span.dataset.cid = tok.c;
      span.dataset.pos = pos;
      span.onmouseenter = (e) => {{
        const tip = document.getElementById('doc-reader-tooltip');
        const cName = c ? c.label : `Cluster ${{tok.c}}`;
        tip.innerHTML = `<b style="color:${{color}}">Cluster ${{tok.c}}</b>: ${{cName}}<br>Position ${{pos}}`;
        tip.style.display = 'block';
        tip.style.left = (e.clientX + 12) + 'px';
        tip.style.top = (e.clientY - 8) + 'px';
      }};
      span.onmouseleave = () => {{
        document.getElementById('doc-reader-tooltip').style.display = 'none';
      }};
      span.onclick = (e) => {{
        e.stopPropagation();
        docReaderHighlight = docReaderHighlight === tok.c ? null : tok.c;
        renderDocReaderBody(docIndex);
        updateDocReaderLegend(docIndex);
      }};
    }} else {{
      // Unsampled token
      span.className = 'doc-token unsampled';
      if (docReaderHighlight !== null) {{
        span.style.opacity = '0.3';
      }}
    }}

    span.textContent = text;
    body.appendChild(span);
  }});
}}

function closeDocReader() {{
  document.getElementById('doc-reader-modal').classList.remove('open');
}}
document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeDocReader(); }});

init();
</script>
</body>
</html>"""

    with open(path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
