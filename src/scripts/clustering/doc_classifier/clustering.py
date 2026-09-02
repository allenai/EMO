"""Persistable document clustering — one implementation, reused for global and train fits.

`DocClustering` wraps the published router-embedding recipe (mean-center + PCA to 95%
variance + L2, then spherical k-means, seed 42) behind a fit / predict / save / load
interface. The point is a *single* clustering code path: the global (all-data) fit and the
train-only fit differ only in which rows are fed to `.fit()`, so they cannot silently
diverge, and the fitted transform can be persisted and re-applied to held-out docs (val/test)
or a production stream without refitting.

It reuses `transform.fit_apply_mean_pca_l2` / `apply_mean_pca_l2` and
`cluster.fit_spherical_kmeans` — the same functions the `cluster.py` CLI uses — so a global
fit here reproduces the existing `assignments.npy`.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from src.scripts.clustering.cluster import fit_spherical_kmeans
from src.scripts.clustering.transform import apply_mean_pca_l2, fit_apply_mean_pca_l2

logger = logging.getLogger(__name__)


class DocClustering:
    """PCA(95%)+L2 preprocess followed by spherical k-means, persistable for reuse."""

    def __init__(self, state: tuple, km, k: int, variance: float):
        self.state = state  # (mean, pca) — the fitted preprocess transform
        self.km = km  # fitted MiniBatchKMeans with L2-normalized centroids
        self.k = k
        self.variance = variance

    # -- fit / predict ------------------------------------------------------

    @classmethod
    def fit(cls, emb: np.ndarray, k: int = 64, variance: float = 0.95) -> "DocClustering":
        """Fit preprocess + spherical k-means on ``emb`` (N, D)."""
        emb = np.asarray(emb, dtype=np.float32)
        logger.info(f"DocClustering.fit: {emb.shape[0]:,} rows x {emb.shape[1]} dims, k={k}")
        transformed, state = fit_apply_mean_pca_l2(emb, variance)
        km = fit_spherical_kmeans(transformed, k)
        return cls(state, km, k, variance)

    def _preprocess(self, emb: np.ndarray) -> np.ndarray:
        return apply_mean_pca_l2(np.asarray(emb, dtype=np.float32), self.state)

    def predict(self, emb: np.ndarray) -> np.ndarray:
        """Assign rows of ``emb`` to clusters via the saved transform + nearest centroid."""
        from sklearn.preprocessing import normalize

        transformed = self._preprocess(emb)
        return self.km.predict(normalize(transformed, norm="l2")).astype(np.int32)

    # -- persistence --------------------------------------------------------

    def save(self, out_dir: str) -> None:
        import joblib

        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(
            {"state": self.state, "km": self.km, "k": self.k, "variance": self.variance},
            os.path.join(out_dir, "clustering.joblib"),
        )
        # Also drop L2 centroids as a plain .npy for quick inspection / non-python use.
        np.save(os.path.join(out_dir, "centroids.npy"), self.km.cluster_centers_)
        logger.info(f"  saved DocClustering to {out_dir}/clustering.joblib")

    @classmethod
    def load(cls, out_dir: str) -> "DocClustering":
        import joblib

        d = joblib.load(os.path.join(out_dir, "clustering.joblib"))
        return cls(d["state"], d["km"], d["k"], d["variance"])
