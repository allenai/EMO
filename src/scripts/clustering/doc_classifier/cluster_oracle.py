"""Part A — the leakage-free oracle: fit clustering on TRAIN only, validate it reproduces
the GLOBAL (all-data) partition on held-out val, and emit the classifier's ground-truth labels.

Uses one shared clustering implementation (`DocClustering`) for both fits, differing only in
which rows are fed in:

  * train model  = DocClustering.fit(emb[train])   -> its predictions on all rows are the
                   classifier's ground-truth labels (val/test labels don't depend on val/test
                   being in any fit -> leakage-free).
  * global model = DocClustering.fit(emb[all])      -> the reference partition (== existing
                   assignments.npy up to ARI~1, since it is the same code).

**The one meaningful check**: align the train-model cluster ids to the global ids with the
Hungarian algorithm on the TRAIN confusion matrix, apply that fixed map to VAL, and report
top-1 accuracy + macro-F1 of train-model-val vs global-val (plus an alignment-free ARI). High
=> the train clustering recovers the global partition out-of-sample, so the labels are
trustworthy. Gate before Part B.

    PYTHONPATH=.:src python -m src.scripts.clustering.doc_classifier.cluster_oracle \
        --data-dir modular_extension/cluster/emo100b_step23842
"""

from __future__ import annotations

import argparse
import json
import logging
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "64")
os.environ.setdefault("OMP_NUM_THREADS", "64")

import numpy as np

from src.scripts.clustering.doc_classifier.clustering import DocClustering

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def align_labels(pred: np.ndarray, ref: np.ndarray, k: int) -> np.ndarray:
    """Map `pred` cluster ids onto `ref` ids maximizing agreement (Hungarian)."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(ref, pred, labels=list(range(k)))
    # maximize matches -> minimize negative
    row_ind, col_ind = linear_sum_assignment(-cm.T)  # pred id (row) -> ref id (col)
    mapping = np.arange(k)
    for pred_id, ref_id in zip(row_ind, col_ind):
        mapping[pred_id] = ref_id
    return mapping


def agreement(
    train_labels: np.ndarray, global_labels: np.ndarray, idx: np.ndarray, k: int, mapping
):
    from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score

    aligned = mapping[train_labels[idx]]
    ref = global_labels[idx]
    return {
        "accuracy": float(accuracy_score(ref, aligned)),
        "macro_f1": float(
            f1_score(ref, aligned, average="macro", labels=list(range(k)), zero_division=0)
        ),
        "ari": float(adjusted_rand_score(ref, train_labels[idx])),
        "n": int(len(idx)),
    }


def plot_confusion(train_labels, global_labels, idx, mapping, k, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    aligned = mapping[train_labels[idx]]
    cm = confusion_matrix(global_labels[idx], aligned, labels=list(range(k)))
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cmn, cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("train-model cluster (aligned to global)")
    ax.set_ylabel("global cluster")
    ax.set_title(
        "Val agreement: train-fit vs global partition\n(row-normalized; diagonal = agreement)"
    )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--embedding", default="embeddings_doc_probs.npy")
    ap.add_argument("--split", default=None, help="default: <data-dir>/doc_classifier/split.npz")
    ap.add_argument("--out", default=None, help="default: <data-dir>/doc_classifier/oracle")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument(
        "--existing-assignments",
        default="doc_probs_mean_pca_l2_spherical_kmeans_k64/assignments.npy",
        help="relative to data-dir; refactor-faithfulness sanity (ARI vs shared-code global)",
    )
    args = ap.parse_args()

    split_path = args.split or os.path.join(args.data_dir, "doc_classifier", "split.npz")
    out = args.out or os.path.join(args.data_dir, "doc_classifier", "oracle")
    os.makedirs(out, exist_ok=True)

    emb = np.load(os.path.join(args.data_dir, args.embedding))  # (N, D)
    sp = np.load(split_path)
    train_idx, val_idx, test_idx = sp["train_idx"], sp["val_idx"], sp["test_idx"]
    logger.info(
        f"emb {emb.shape}; train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}"
    )

    # --- fit train model (leakage-free) -> ground-truth labels for everyone ------------
    logger.info("Fitting TRAIN clustering ...")
    train_model = DocClustering.fit(emb[train_idx], k=args.k)
    train_model.save(os.path.join(out, "train"))
    train_labels = train_model.predict(emb)  # (N,) train-model applied to all rows
    np.save(os.path.join(out, "train_model_labels.npy"), train_labels)

    # --- fit global model (reference partition) ----------------------------------------
    logger.info("Fitting GLOBAL clustering ...")
    global_model = DocClustering.fit(emb, k=args.k)
    global_model.save(os.path.join(out, "global"))
    global_labels = global_model.predict(emb)
    np.save(os.path.join(out, "global_labels.npy"), global_labels)

    # --- refactor-faithfulness sanity: shared-code global vs existing assignments ------
    from sklearn.metrics import adjusted_rand_score

    existing_ari = None
    existing_path = os.path.join(args.data_dir, args.existing_assignments)
    if os.path.exists(existing_path):
        existing = np.load(existing_path)
        existing_ari = float(adjusted_rand_score(existing, global_labels))
        logger.info(
            f"refactor sanity: ARI(shared-code global, existing assignments) = {existing_ari:.4f}"
        )

    # --- the one meaningful check: train-fit vs global agreement on val (and test) -----
    mapping = align_labels(train_labels[train_idx], global_labels[train_idx], args.k)
    report = {
        "k": args.k,
        "n_train": int(len(train_idx)),
        "refactor_sanity_ari_vs_existing": existing_ari,
        "train": agreement(train_labels, global_labels, train_idx, args.k, mapping),
        "val": agreement(train_labels, global_labels, val_idx, args.k, mapping),
        "test": agreement(train_labels, global_labels, test_idx, args.k, mapping),
    }
    with open(os.path.join(out, "cluster_agreement.json"), "w") as f:
        json.dump(report, f, indent=2)
    plot_confusion(
        train_labels,
        global_labels,
        val_idx,
        mapping,
        args.k,
        os.path.join(out, "agreement_confusion.png"),
    )

    v = report["val"]
    logger.info("=== Part-A gate: train-fit vs global partition ===")
    logger.info(f"  refactor sanity ARI vs existing: {existing_ari}")
    logger.info(
        f"  VAL  accuracy={v['accuracy']:.4f}  macro_f1={v['macro_f1']:.4f}  ari={v['ari']:.4f}"
    )
    logger.info(
        f"  TEST accuracy={report['test']['accuracy']:.4f}  macro_f1={report['test']['macro_f1']:.4f}"
    )
    logger.info(f"  wrote {out}/cluster_agreement.json + agreement_confusion.png")


if __name__ == "__main__":
    main()
