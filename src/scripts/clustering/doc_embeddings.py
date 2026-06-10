"""
Shared loaders for per-document expert-usage extractions produced by
extract_document.py (the weborganizer pipeline).

An extraction dir contains:
    embeddings_doc_probs.npy      (num_docs, num_layers * num_standard_experts)
    embeddings_doc_topk_freq.npy  (num_docs, num_layers * num_standard_experts)
    metadata_docs.jsonl.gz        per-doc {doc_index, source, doc_len, preview}
    info.json                     extraction config + model dims
"""

import gzip
import json
import os
from typing import Tuple

import numpy as np

EMB_TYPES = ("probs", "topk_freq")


def load_info(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "info.json")) as f:
        return json.load(f)


def load_doc_labels(data_dir: str) -> np.ndarray:
    """Per-doc topic label (the `source` metadata field), shape (num_docs,)."""
    labels = []
    with gzip.open(os.path.join(data_dir, "metadata_docs.jsonl.gz"), "rt") as f:
        for line in f:
            labels.append(json.loads(line)["source"])
    return np.array(labels)


def load_doc_embeddings(data_dir: str, emb_type: str) -> Tuple[np.ndarray, dict]:
    """
    Load one embedding file reshaped to (num_docs, num_layers, num_experts).

    Returns (emb_3d, info). num_experts here is the *standard* (routed,
    non-shared) expert count — the extractor excludes shared experts.
    """
    if emb_type not in EMB_TYPES:
        raise ValueError(f"emb_type must be one of {EMB_TYPES}, got {emb_type!r}")
    info = load_info(data_dir)
    num_layers = info["num_layers"]
    num_experts = info["num_standard_experts"]
    emb = np.load(os.path.join(data_dir, f"embeddings_doc_{emb_type}.npy"))
    assert emb.shape[1] == num_layers * num_experts, (
        f"{data_dir}: embedding dim {emb.shape[1]} != "
        f"num_layers ({num_layers}) * num_standard_experts ({num_experts})"
    )
    return emb.reshape(emb.shape[0], num_layers, num_experts).astype(np.float32), info


def layer_distributions(emb_3d: np.ndarray) -> np.ndarray:
    """Renormalize each (doc, layer) slice to a probability distribution."""
    sums = emb_3d.sum(axis=-1, keepdims=True)
    sums = np.where(sums > 0, sums, 1.0)
    return emb_3d / sums


def assert_same_doc_set(dir_a: str, dir_b: str) -> None:
    """
    Cross-model per-doc comparisons are only valid when both extractions saw
    the identical document list (same composition file + shuffle seed).
    """
    la, lb = load_doc_labels(dir_a), load_doc_labels(dir_b)
    if len(la) != len(lb) or not (la == lb).all():
        raise ValueError(
            f"Doc sets differ between {dir_a} ({len(la)} docs) and {dir_b} "
            f"({len(lb)} docs); re-extract with a shared mix_composition.json "
            "and shuffle seed."
        )
