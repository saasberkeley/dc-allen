"""Gaussian-mixture motifs for continuous spectral band envelopes.

Each row of ``X`` is one timepoint and each column is a frequency-band
envelope. Unlike :mod:`src.get_motifs`, values are not reduced to signs before
clustering. This extracts the reusable part of Jahan's exploratory notebook.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture


@dataclass
class GMMMotifResult:
    """A fitted continuous-envelope motif model."""

    model: GaussianMixture
    labels: np.ndarray
    centers: np.ndarray
    confidence: np.ndarray
    order: np.ndarray
    bic: float
    seed: int

    @property
    def n_motifs(self) -> int:
        return self.centers.shape[0]

    def occupancy(self, groups: np.ndarray) -> dict:
        """Return the fraction of each named group assigned to every motif."""
        groups = np.asarray(groups)
        if groups.shape != self.labels.shape:
            raise ValueError("groups must have one value per timepoint")
        return {
            group: np.bincount(self.labels[groups == group], minlength=self.n_motifs)
            / np.sum(groups == group)
            for group in np.unique(groups)
        }


def _validate_data(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2:
        raise ValueError("X must be a (timepoints, bands) array with at least two of each")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values")
    return X


def shift_bands(X: np.ndarray, step: int = 20) -> np.ndarray:
    """Circularly offset each band to disrupt simultaneous cross-band structure."""
    X = _validate_data(X)
    if step < 1:
        raise ValueError("step must be positive")
    return np.column_stack([np.roll(X[:, band], step * band) for band in range(X.shape[1])])


def _slow_fast_order(centers: np.ndarray, low_band_count: int | None) -> np.ndarray:
    """Order centers from slow-dominant to fast-dominant for stable plotting."""
    n_bands = centers.shape[1]
    split = low_band_count if low_band_count is not None else max(1, n_bands // 2)
    if not 1 <= split < n_bands:
        raise ValueError("low_band_count must be between 1 and n_bands - 1")
    score = centers[:, :split].mean(axis=1) - centers[:, split:].mean(axis=1)
    return np.argsort(score)[::-1]


def fit_gmm_motifs(
    X: np.ndarray,
    k: int,
    *,
    seed: int = 0,
    n_init: int = 5,
    covariance_type: str = "full",
    low_band_count: int | None = None,
) -> GMMMotifResult:
    """Fit K continuous spectral motifs and return consistently ordered labels."""
    X = _validate_data(X)
    if not 1 <= k <= X.shape[0]:
        raise ValueError("k must be between 1 and the number of timepoints")
    model = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        n_init=n_init,
        random_state=seed,
    ).fit(X)

    order = _slow_fast_order(model.means_, low_band_count)
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    labels = remap[model.predict(X)]
    probabilities = model.predict_proba(X)[:, order]
    return GMMMotifResult(
        model=model,
        labels=labels,
        centers=model.means_[order],
        confidence=probabilities.max(axis=1),
        order=order,
        bic=float(model.bic(X)),
        seed=seed,
    )


def bic_sweep(
    X: np.ndarray,
    ks=range(1, 13),
    *,
    seed: int = 0,
    n_init: int = 5,
    covariance_type: str = "full",
) -> tuple[np.ndarray, np.ndarray]:
    """Return tested K values and their BIC scores (lower is preferred)."""
    X = _validate_data(X)
    ks = np.asarray(list(ks), dtype=int)
    if ks.size == 0 or np.any(ks < 1) or np.any(ks > X.shape[0]):
        raise ValueError("ks must contain valid positive cluster counts")
    bics = [
        GaussianMixture(
            n_components=int(k),
            covariance_type=covariance_type,
            n_init=n_init,
            random_state=seed,
        ).fit(X).bic(X)
        for k in ks
    ]
    return ks, np.asarray(bics)


def seed_stability(
    X: np.ndarray,
    k: int,
    *,
    seeds=range(5),
    n_init: int = 1,
    covariance_type: str = "full",
) -> tuple[np.ndarray, float]:
    """Return pairwise ARIs and their mean across independently seeded fits."""
    seeds = list(seeds)
    if len(seeds) < 2:
        raise ValueError("at least two seeds are required")
    labels = [
        fit_gmm_motifs(
            X,
            k,
            seed=int(seed),
            n_init=n_init,
            covariance_type=covariance_type,
        ).labels
        for seed in seeds
    ]
    scores = np.asarray(
        [adjusted_rand_score(labels[a], labels[b]) for a, b in combinations(range(len(seeds)), 2)]
    )
    return scores, float(scores.mean())
