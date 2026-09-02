"""Part B — train a cheap classifier to predict a document's cluster, eval on val/test.

Loads a cached feature matrix (dense .npy or sparse .npz), the shared split, and the
train-fit clustering labels (the leakage-free ground truth). Fits a supervised pipeline on
train (tf-idf/standardize is fit on train only), selects nothing beyond defaults, and reports
accuracy / macro-F1 / top-k on val and test against the majority-class baseline.

    PYTHONPATH=.:src python -m src.scripts.clustering.doc_classifier.classify \
        --features <feats.npz|.npy> --split <split.npz> \
        --labels <oracle/train_model_labels.npy> --out <run_dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "64")
os.environ.setdefault("OMP_NUM_THREADS", "64")

import numpy as np

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def load_features(path: str):
    if path.endswith(".npz"):
        import scipy.sparse as sp

        return sp.load_npz(path), True
    return np.load(path, mmap_mode="r"), False


def build_pipeline(is_sparse: bool, classifier: str, C: float, max_iter: int):
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.pipeline import Pipeline

    steps = []
    if is_sparse:
        from sklearn.feature_extraction.text import TfidfTransformer

        steps.append(("tfidf", TfidfTransformer()))
    else:
        from sklearn.preprocessing import StandardScaler

        steps.append(("scale", StandardScaler()))

    if classifier == "logreg":
        solver = "saga" if is_sparse else "lbfgs"
        clf = LogisticRegression(C=C, max_iter=max_iter, solver=solver, n_jobs=-1)
    elif classifier == "sgd":
        clf = SGDClassifier(loss="log_loss", alpha=1.0 / (C * 1e5), max_iter=max_iter, n_jobs=-1)
    else:
        raise ValueError(classifier)
    steps.append(("clf", clf))
    return Pipeline(steps)


def evaluate(pipe, X, y, idx, classes):
    from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

    Xi = X[idx]
    proba = pipe.predict_proba(Xi)
    pred = classes[proba.argmax(1)]
    ref = y[idx]
    out = {
        "n": int(len(idx)),
        "accuracy": float(accuracy_score(ref, pred)),
        "macro_f1": float(f1_score(ref, pred, average="macro", labels=classes, zero_division=0)),
    }
    for k in (3, 5):
        out[f"top{k}_accuracy"] = float(top_k_accuracy_score(ref, proba, k=k, labels=classes))
    return out, pred


def plot_confusion(ref, pred, classes, out_path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(ref, pred, labels=classes)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cmn, cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("predicted cluster")
    ax.set_ylabel("true cluster (oracle)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--labels", required=True, help="train-fit clustering labels (N,)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--classifier", default="logreg", choices=["logreg", "sgd"])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    X, is_sparse = load_features(args.features)
    y = np.load(args.labels)
    sp = np.load(args.split)
    train_idx, val_idx, test_idx = sp["train_idx"], sp["val_idx"], sp["test_idx"]
    classes = np.unique(y)
    logger.info(f"X {X.shape} sparse={is_sparse}; {len(classes)} classes; train={len(train_idx):,}")

    # majority-class baseline (train prior applied to test)
    vals, cnts = np.unique(y[train_idx], return_counts=True)
    majority = int(vals[cnts.argmax()])
    baseline_acc = float((y[test_idx] == majority).mean())

    pipe = build_pipeline(is_sparse, args.classifier, args.C, args.max_iter)
    logger.info("fitting ...")
    pipe.fit(X[train_idx], y[train_idx])

    val_m, _ = evaluate(pipe, X, y, val_idx, classes)
    test_m, test_pred = evaluate(pipe, X, y, test_idx, classes)

    from sklearn.metrics import classification_report

    per_cluster = classification_report(
        y[test_idx], test_pred, labels=classes, output_dict=True, zero_division=0
    )
    report = {
        "features": args.features,
        "classifier": args.classifier,
        "n_features": int(X.shape[1]),
        "majority_baseline_test_acc": baseline_acc,
        "val": val_m,
        "test": test_m,
        "per_cluster_test": per_cluster,
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    plot_confusion(
        y[test_idx],
        test_pred,
        classes,
        os.path.join(args.out, "confusion.png"),
        f"{os.path.basename(args.features)} — test (acc={test_m['accuracy']:.3f})",
    )

    logger.info("=== results ===")
    logger.info(f"  baseline (majority) test acc: {baseline_acc:.4f}")
    logger.info(
        f"  VAL  acc={val_m['accuracy']:.4f} macroF1={val_m['macro_f1']:.4f} top5={val_m['top5_accuracy']:.4f}"
    )
    logger.info(
        f"  TEST acc={test_m['accuracy']:.4f} macroF1={test_m['macro_f1']:.4f} top5={test_m['top5_accuracy']:.4f}"
    )
    logger.info(f"  wrote {args.out}/metrics.json + confusion.png")


if __name__ == "__main__":
    main()
