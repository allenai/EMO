"""
k-selection experiment: is there a principled k for the oracle partition, and does a
better k reduce k-means seed variance?

Every ``fit`` arm clusters rows of the FROZEN preprocessed cache (the subsample-stability
experiment showed the transform is stable, so only k-means varies here), assigns all rows,
and records selection criteria:

  objective   mean cosine of every doc to its own centroid (the spherical k-means
              objective, computed over ALL rows during assignment) -> elbow curve
  silhouette / davies_bouldin   cosine metric, on a fixed 50K-row evaluation sample

Arms:
  --fit-rows all         fit on all rows (global fit); vary --seed for stability pairs
  --fit-rows 1000000     fit on a 1M-row draw; the draw AND k-means use --seed

``aggregate`` computes, per k: seed-pair stability of global fits, cross-draw stability
of subsample fits, subsample->global recoverability (all ARI + matched accuracy where
k matches), criterion curves per fit set, and each criterion's chosen k.

Usage:
    PYTHONPATH=.:src python -m src.scripts.clustering.k_selection fit \\
        --data-dir modular_extension/cluster/emo100b_step23842_100B-130B \\
        --k 32 --seed 1 --fit-rows all

    PYTHONPATH=.:src python -m src.scripts.clustering.k_selection aggregate \\
        --data-dir modular_extension/cluster/emo100b_step23842_100B-130B
"""

import argparse
import itertools
import json
import logging
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("OMP_NUM_THREADS", "16")

import numpy as np

from src.scripts.clustering.cluster import fit_spherical_kmeans
from src.scripts.clustering.subsample_stability import (
    ari_from_contingency,
    contingency,
    hungarian_accuracy,
    nmi_from_contingency,
)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

EVAL_SAMPLE = 50_000
EVAL_SEED = 12345  # fixed so every arm scores criteria on the same rows


def arm_name(k: int, fit_rows: str, seed: int) -> str:
    if fit_rows == "all":
        tag = "global"
    else:
        n = int(fit_rows)
        tag = f"sub{n // 1_000_000}M" if n >= 1_000_000 else f"sub{n // 1_000}K"
    return f"{tag}_k{k}_seed{seed}"


def assign_all_with_objective(km, pre, chunk: int):
    """Assign every row to its nearest L2 centroid; also return the mean cosine
    of each row to its own centroid (the spherical k-means objective)."""
    from sklearn.preprocessing import normalize

    cent = normalize(km.cluster_centers_, norm="l2").astype(np.float32)
    n = pre.shape[0]
    labels = np.empty(n, dtype=np.int16)
    obj = 0.0
    t0 = time.time()
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        sims = np.asarray(pre[lo:hi], dtype=np.float32) @ cent.T
        lab = sims.argmax(axis=1)
        labels[lo:hi] = lab
        obj += float(sims[np.arange(hi - lo), lab].sum())
        logger.info(f"  assigned {hi:,}/{n:,} rows ({time.time() - t0:.0f}s)")
    return labels, obj / n


def fit(args):
    out_dir = os.path.join(args.out_root, arm_name(args.k, args.fit_rows, args.seed))
    labels_path = os.path.join(out_dir, "labels.npy")
    if os.path.exists(labels_path):
        logger.info(f"SKIP: {labels_path} already exists")
        return
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    cache_path = os.path.join(args.data_dir, args.preprocess_cache)
    pre = np.load(cache_path, mmap_mode="r")
    n_total = pre.shape[0]

    if args.fit_rows == "all":
        fit_x = np.asarray(pre, dtype=np.float32)
    else:
        n_fit = int(args.fit_rows)
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(n_total, n_fit, replace=False))
        fit_x = np.asarray(pre[idx], dtype=np.float32)
    km = fit_spherical_kmeans(fit_x, args.k, random_state=args.seed)
    del fit_x

    labels, objective = assign_all_with_objective(km, pre, args.chunk_size)

    # criteria on a fixed evaluation sample, identical across arms
    from sklearn.metrics import davies_bouldin_score, silhouette_score

    ev = np.sort(np.random.default_rng(EVAL_SEED).choice(n_total, EVAL_SAMPLE, replace=False))
    ev_x = np.asarray(pre[ev], dtype=np.float32)
    ev_lab = labels[ev]
    sil = (
        float(silhouette_score(ev_x, ev_lab, metric="cosine"))
        if len(set(ev_lab.tolist())) > 1
        else float("nan")
    )
    db = (
        float(davies_bouldin_score(ev_x, ev_lab)) if len(set(ev_lab.tolist())) > 1 else float("nan")
    )

    sizes = np.bincount(labels, minlength=args.k)
    run_info = {
        "k": args.k,
        "fit_rows": args.fit_rows,
        "seed": args.seed,
        "objective": round(objective, 6),
        "silhouette_cosine": round(sil, 6),
        "davies_bouldin": round(db, 6),
        "eval_sample": EVAL_SAMPLE,
        "cluster_size_min": int(sizes.min()),
        "cluster_size_median": int(np.median(sizes)),
        "cluster_size_max": int(sizes.max()),
        "n_empty_clusters": int((sizes == 0).sum()),
        "runtime_s": round(time.time() - t_start, 1),
    }
    np.save(labels_path, labels)
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(run_info, f, indent=2)
    logger.info(f"DONE {arm_name(args.k, args.fit_rows, args.seed)}: {run_info}")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def pair_metrics(lab_a, lab_b):
    k = int(max(lab_a.max(), lab_b.max())) + 1
    c = contingency(lab_a, lab_b, k)
    out = {"ari": round(ari_from_contingency(c), 4), "nmi": round(nmi_from_contingency(c), 4)}
    if lab_a.max() == lab_b.max():  # same k: matched accuracy is meaningful
        out["acc"] = round(hungarian_accuracy(c), 4)
    return out


def elbow_k(ks, objectives):
    """Knee of the objective curve: max vertical distance to the chord between the
    endpoints, in (log2 k, normalized objective) space."""
    x = np.log2(np.asarray(ks, dtype=float))
    y = np.asarray(objectives, dtype=float)
    y = (y - y[0]) / (y[-1] - y[0])
    x = (x - x[0]) / (x[-1] - x[0])
    dist = y - x  # chord is the identity line after normalization
    return int(ks[int(np.argmax(dist))])


def aggregate(args):
    arms = {}
    for name in sorted(os.listdir(args.out_root)):
        rj = os.path.join(args.out_root, name, "run.json")
        lp = os.path.join(args.out_root, name, "labels.npy")
        if os.path.exists(rj) and os.path.exists(lp):
            arms[name] = json.load(open(rj))
    logger.info(f"{len(arms)} arms present")

    def labels(name):
        return np.load(os.path.join(args.out_root, name, "labels.npy"))

    by_fitset: dict = {}
    for name, info in arms.items():
        fitset = (arm_name(0, info["fit_rows"], 0).split("_")[0], info["seed"])
        by_fitset.setdefault(fitset, {})[info["k"]] = name

    # criterion curves + chosen k per fit set
    chosen = {}
    curves = {}
    for fitset, by_k in sorted(by_fitset.items()):
        ks = sorted(by_k)
        cur = {
            "k": ks,
            "objective": [arms[by_k[k]]["objective"] for k in ks],
            "silhouette_cosine": [arms[by_k[k]]["silhouette_cosine"] for k in ks],
            "davies_bouldin": [arms[by_k[k]]["davies_bouldin"] for k in ks],
        }
        key = f"{fitset[0]}_seed{fitset[1]}"
        curves[key] = cur
        chosen[key] = {
            "elbow_objective": elbow_k(ks, cur["objective"]),
            "silhouette_argmax": int(ks[int(np.nanargmax(cur["silhouette_cosine"]))]),
            "davies_bouldin_argmin": int(ks[int(np.nanargmin(cur["davies_bouldin"]))]),
        }

    all_ks = sorted({info["k"] for info in arms.values()})
    stability = {"global": {}, "sub1M": {}, "recover": {}}
    for k in all_ks:
        g = [n for n, i in arms.items() if i["k"] == k and i["fit_rows"] == "all"]
        s = [n for n, i in arms.items() if i["k"] == k and i["fit_rows"] != "all"]
        if len(g) >= 2:
            stability["global"][k] = pair_metrics(labels(g[0]), labels(g[1]))
        if len(s) >= 2:
            pairs = [pair_metrics(labels(a), labels(b)) for a, b in itertools.combinations(s, 2)]
            stability["sub1M"][k] = {
                m: round(float(np.mean([p[m] for p in pairs])), 4) for m in pairs[0]
            }
        if g and s:
            pairs = [pair_metrics(labels(a), labels(b)) for a in s for b in g]
            stability["recover"][k] = {
                m: round(float(np.mean([p[m] for p in pairs])), 4) for m in pairs[0]
            }
        logger.info(
            f"k={k}: global={stability['global'].get(k)} "
            f"sub={stability['sub1M'].get(k)} recover={stability['recover'].get(k)}"
        )

    out = {"arms": arms, "curves": curves, "chosen_k": chosen, "stability": stability}
    out_path = os.path.join(args.out_root, "k_selection_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fit", help="fit one arm and assign all rows")
    pf.add_argument("--data-dir", required=True)
    pf.add_argument("--k", type=int, required=True)
    pf.add_argument("--seed", type=int, required=True)
    pf.add_argument(
        "--fit-rows",
        default="all",
        help="'all' for a global fit, or a row count (e.g. 1000000) for a draw",
    )
    pf.add_argument("--preprocess-cache", default="preprocessed_doc_probs_mean_pca_l2.npy")
    pf.add_argument("--out-root", default=None)
    pf.add_argument("--chunk-size", type=int, default=1_000_000)

    pa = sub.add_parser("aggregate", help="criterion curves, chosen k, stability tables")
    pa.add_argument("--data-dir", required=True)
    pa.add_argument("--out-root", default=None)

    args = parser.parse_args()
    if args.out_root is None:
        args.out_root = os.path.join(args.data_dir, "k_selection")
    fit(args) if args.cmd == "fit" else aggregate(args)


if __name__ == "__main__":
    main()
