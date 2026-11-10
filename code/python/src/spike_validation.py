"""How closely each motif pipeline tracks neuron population spiking.

Per state (awake, isoflurane), on spontaneous seconds (stimulus blocks excluded):
  - neuron ensembles from getClust (Mishler) on the region's well-isolated units, preprocessed as Mishler
    describes (smoothed 1-s rate, squared, z-scored);
  - targets: the population rate, and each ensemble's activity (getClust's group support);
  - features from each pipeline: motif expression (and, for the tracker, motif baselines);
  - score: held-out R^2 of a ridge regression (blocked 5-fold cross-validation, 10 s trimmed at test-fold
    edges), with a null from circularly shifting the target in time (keeps both signals' autocorrelation,
    breaks their alignment);
  - Mishler-style pole contrast: target in + pole seconds minus - pole seconds, with the same null.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from . import session_data as sd
from .get_clust import get_clust, group_support, preprocess


def unit_rates(nwb_path, region, t, sigma=1.0):
    """Spike counts in 1-s bins centred on t, Gaussian-smoothed (sigma in s): (len(t), units)."""
    io, nwb = sd.open_nwb(nwb_path)
    trains, ids, _ = sd.good_units(nwb, region)
    io.close()
    edges_lo, edges_hi = t - 0.5, t + 0.5
    counts = np.stack([np.searchsorted(s, edges_hi) - np.searchsorted(s, edges_lo) for s in trains], axis=1).astype(float)
    # smooth within contiguous stretches only
    out = np.empty_like(counts)
    breaks = np.r_[0, np.flatnonzero(np.diff(t) > 1.5) + 1, t.size]
    for a, b in zip(breaks[:-1], breaks[1:]):
        out[a:b] = gaussian_filter1d(counts[a:b], sigma, axis=0, mode="nearest")
    return out, ids


def spontaneous_mask(t, info, margin=5.0):
    m = np.ones(t.size, bool)
    for _, a, b in list(info.stim_blocks) + list(info.visual_blocks):
        m &= ~((t >= a - margin) & (t < b + margin))
    return m


def state_masks(t, info, spontaneous=True):
    w = sd.analysis_windows(info)
    base = spontaneous_mask(t, info) if spontaneous else np.ones(t.size, bool)
    return {s: base & (t >= w[s][0]) & (t < w[s][1]) for s in ("awake", "isoflurane")}


def ensembles(rate, sel, seeds=range(5)):
    """getClust on one state's seconds for several seeds; returns (clusters for seed 0, stability)."""
    X = preprocess(rate[sel])
    runs = [get_clust(X, 1.0, seed=s) for s in seeds]
    ref = runs[0]
    stab = []
    for c in ref:
        c = set(c.tolist())
        stab.append(np.mean([max((len(c & set(d.tolist())) / len(c | set(d.tolist())) for d in r), default=0)
                             for r in runs[1:]]))
    return ref, np.array(stab), X


def zscore(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-12)


def _ridge(F, y, alpha):
    mu, sdv = F.mean(0), F.std(0) + 1e-12
    A = (F - mu) / sdv
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (y - y.mean()))
    return lambda G: ((G - mu) / sdv) @ w + y.mean()


def cv_r2(F, y, n_folds=5, block=60, gap=10, alpha=1.0):
    """Held-out R^2 (pooled over folds). Folds interleave 60-s blocks (block i goes to fold i mod 5), so
    every fold samples the whole state; test seconds within `gap` of a block edge are dropped so the 5-s
    smoothing does not leak between training and test. F with no columns gives 0."""
    if F is None or F.shape[1] == 0:
        return 0.0
    i = np.arange(y.size)
    fid = (i // block) % n_folds
    inner = ((i % block) >= gap) & ((i % block) < block - gap)
    pred = np.full(y.size, np.nan)
    for f in range(n_folds):
        model = _ridge(F[fid != f], y[fid != f], alpha)
        test = (fid == f) & inner
        pred[test] = model(F[test])
    ok = np.isfinite(pred)
    return float(1 - np.sum((y[ok] - pred[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2))


def cv_predict(F, y, n_folds=5, block=60, gap=10, alpha=1.0):
    """Held-out predictions with the same folds as cv_r2 (NaN where not tested)."""
    i = np.arange(y.size)
    fid = (i // block) % n_folds
    inner = ((i % block) >= gap) & ((i % block) < block - gap)
    pred = np.full(y.size, np.nan)
    if F is None or F.shape[1] == 0:
        return pred
    for f in range(n_folds):
        model = _ridge(F[fid != f], y[fid != f], alpha)
        test = (fid == f) & inner
        pred[test] = model(F[test])
    return pred


def detrend(v, t, window=300):
    """Subtract a local-linear baseline (window s) within each contiguous stretch; works on 1-D or 2-D."""
    from .motif_tracking import local_linear
    v = np.asarray(v, float)
    two_d = v.ndim == 2
    V = v if two_d else v[:, None]
    out = np.empty_like(V)
    breaks = np.r_[0, np.flatnonzero(np.diff(t) > 1.5) + 1, t.size]
    for a, b in zip(breaks[:-1], breaks[1:]):
        half = int(min(window // 2, max(15, (b - a) // 2)))
        out[a:b] = V[a:b] - local_linear(V[a:b], half)
    return out if two_d else out[:, 0]


def shift_null(F, y, rng, n_shift=100, min_shift=60):
    if F is None or F.shape[1] == 0:
        return np.zeros(n_shift)
    ks = rng.integers(min_shift, max(min_shift + 1, y.size - min_shift), n_shift)
    return np.array([cv_r2(F, np.roll(y, k)) for k in ks])


def pole_contrast(x, y, rng, n_shift=1000, min_shift=60):
    """Mean z-scored y when x > 0 minus when x < 0, and a two-sided shift p-value."""
    s = np.sign(x)
    z = zscore(y)
    if (s > 0).sum() < 10 or (s < 0).sum() < 10:
        return np.nan, np.nan
    d = z[s > 0].mean() - z[s < 0].mean()
    ks = rng.integers(min_shift, max(min_shift + 1, z.size - min_shift), n_shift)
    null = np.array([np.roll(z, k)[s > 0].mean() - np.roll(z, k)[s < 0].mean() for k in ks])
    return float(d), float((1 + np.sum(np.abs(null) >= abs(d))) / (n_shift + 1))


def evaluate(features, targets, seed=0, n_shift=100):
    """features: {pipeline: (n, k) array or None}; targets: {name: (n,) array}. One row per pair."""
    rng = np.random.default_rng(seed)
    rows = []
    for tname, y in targets.items():
        y = zscore(y)
        for pname, F in features.items():
            r2 = cv_r2(F, y)
            null = shift_null(F, y, rng, n_shift)
            rows.append({"target": tname, "pipeline": pname, "n_features": 0 if F is None else F.shape[1],
                         "r2": r2, "null95": float(np.percentile(null, 95)),
                         "p": float((1 + np.sum(null >= r2)) / (null.size + 1))})
    return pd.DataFrame(rows)


def ensemble_targets(X, clusters, rate):
    """Population rate plus each ensemble's group support (on Mishler-preprocessed activity)."""
    out = {"population rate": rate.sum(axis=1)}
    for i, c in enumerate(clusters):
        out[f"ensemble {i + 1} ({len(c)} units)"] = group_support(X[:, c])
    return out
