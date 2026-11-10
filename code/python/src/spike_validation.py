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


# ---------- opponency: do opposite motif poles recruit opposite neuron groups? ----------

def remove_global(R):
    """z-score each neuron, then regress the population mean out of it (removes shared gain changes)."""
    Z = (R - R.mean(0)) / (R.std(0) + 1e-12)
    pop = Z.mean(1)
    pop = pop - pop.mean()
    beta = (Z * pop[:, None]).sum(0) / (pop @ pop + 1e-12)
    return Z - np.outer(pop, beta)


def _shifts(n, rng, n_shift, min_shift=60):
    return rng.integers(min_shift, max(min_shift + 1, n - min_shift), n_shift)


def pole_contrasts(x, R, rng, n_shift=1000, alpha=0.05):
    """Per neuron: mean activity in + pole seconds (x > 0) minus - pole seconds. The null circularly shifts
    the pole sequence, the same shift for all neurons. A neuron counts as +preferring (-preferring) when its
    difference is above the (1 - alpha/2) quantile (below the alpha/2 quantile) of its own null. The same
    rule applied to each null shift gives the null distribution of the counts."""
    s = np.sign(x)
    def diff(sig):
        return R[sig > 0].mean(0) - R[sig < 0].mean(0)
    d = diff(s)
    ks = _shifts(s.size, rng, n_shift)
    null = np.array([diff(np.roll(s, k)) for k in ks])
    hi, lo = np.quantile(null, 1 - alpha / 2, axis=0), np.quantile(null, alpha / 2, axis=0)
    plus, minus = d > hi, d < lo
    null_plus, null_minus = (null > hi).sum(1), (null < lo).sum(1)
    both_null = np.minimum(null_plus, null_minus)
    return {
        "d": d, "plus": plus, "minus": minus,
        "n_plus": int(plus.sum()), "n_minus": int(minus.sum()),
        "p_plus": float((1 + (null_plus >= plus.sum()).sum()) / (n_shift + 1)),
        "p_minus": float((1 + (null_minus >= minus.sum()).sum()) / (n_shift + 1)),
        "p_both": float((1 + (both_null >= min(plus.sum(), minus.sum())).sum()) / (n_shift + 1)),
        "null_plus_95": float(np.percentile(null_plus, 95)), "null_minus_95": float(np.percentile(null_minus, 95)),
    }


def heldout_opponency(x, R, rng, n_shift=200, n_folds=5, block=60, gap=10, q=0.75):
    """Groups A (+preferring) and B (-preferring) are defined on training blocks: neurons whose training
    pole difference has the right sign and is in the top quarter by size. On held-out blocks, correlate the
    motif expression with A's mean activity, B's mean activity and A - B. Null: the same after circularly
    shifting x relative to the neurons."""
    i = np.arange(x.size)
    fid = (i // block) % n_folds
    inner = ((i % block) >= gap) & ((i % block) < block - gap)

    def run(xx):
        A = np.full(xx.size, np.nan); B = np.full(xx.size, np.nan)
        for f in range(n_folds):
            tr, te = fid != f, (fid == f) & inner
            s = np.sign(xx[tr])
            d = R[tr][s > 0].mean(0) - R[tr][s < 0].mean(0)
            cut = np.quantile(np.abs(d), q)
            a, b = (d > 0) & (np.abs(d) >= cut), (d < 0) & (np.abs(d) >= cut)
            if a.any():
                A[te] = R[te][:, a].mean(1)
            if b.any():
                B[te] = R[te][:, b].mean(1)
        ok = np.isfinite(A) & np.isfinite(B)
        c = lambda u: float(np.corrcoef(xx[ok], u[ok])[0, 1])
        return c(A), c(B), c(A - B)

    real = run(x)
    null = np.array([run(np.roll(x, k)) for k in _shifts(x.size, rng, n_shift)])
    return {"r_A": real[0], "r_B": real[1], "r_AminusB": real[2],
            "p_A": float((1 + (null[:, 0] >= real[0]).sum()) / (n_shift + 1)),
            "p_B": float((1 + (null[:, 1] <= real[1]).sum()) / (n_shift + 1)),
            "p_AminusB": float((1 + (null[:, 2] >= real[2]).sum()) / (n_shift + 1)),
            "null95_AminusB": float(np.percentile(null[:, 2], 95))}


def ensemble_alternation(X, clusters):
    """Correlation between every pair of getClust ensembles' activity (negative = they alternate)."""
    sup = [group_support(X[:, c]) for c in clusters]
    k = len(sup)
    return np.array([[np.corrcoef(sup[a], sup[b])[0, 1] for b in range(k)] for a in range(k)]) if k else np.zeros((0, 0))


# ---------- does a motif's baseline shift go with a shift in its neuron populations? ----------

def _corr_cols(R, x):
    Rc, xc = R - R.mean(0), x - x.mean()
    return (Rc * xc[:, None]).sum(0) / (np.sqrt((Rc ** 2).sum(0) * (xc @ xc)) + 1e-12)


def slow_opponency(x, base, R, t, rng, n_shift=500, min_shift=300):
    """(a) Groups A (+preferring) and B (-preferring) are defined on the fast parts (both detrended).
    Then: does the slow part of A - B (its 5-min local-linear baseline) follow the motif's baseline?
    Null: circular shifts of at least min_shift seconds."""
    Rf, xf = detrend(R, t), detrend(x, t)
    pc = pole_contrasts(xf, Rf, rng, n_shift=200)
    a, b = pc["plus"], pc["minus"]
    if not a.any() or not b.any():
        return {"n_A": int(a.sum()), "n_B": int(b.sum()), "r": np.nan, "p": np.nan}
    ab = R[:, a].mean(1) - R[:, b].mean(1)
    slow = ab - detrend(ab, t)
    r = float(np.corrcoef(base, slow)[0, 1])
    null = np.array([np.corrcoef(np.roll(base, k), slow)[0, 1] for k in _shifts(base.size, rng, n_shift, min_shift)])
    return {"n_A": int(a.sum()), "n_B": int(b.sum()), "r": r,
            "p": float((1 + (np.abs(null) >= abs(r)).sum()) / (n_shift + 1)), "null95": float(np.percentile(np.abs(null), 95))}


def membership_shift(x, base, R, t, rng, n_shift=200, min_shift=300, block=60):
    """(b) Split seconds into thirds by the motif's baseline. In each third, each neuron's coupling is its
    correlation with the motif's fast expression (both detrended). Each third is split into even and odd
    60-s blocks. within = r(coupling, same third, other half); between = r(coupling, low third, high third),
    also across halves so both use the same amount of data. reorganisation = within - between: about 0 if
    the same neurons go on and off at low and high baseline. Null: circularly shifted baseline."""
    Rf, xf = detrend(R, t), detrend(x, t)
    half = (np.arange(t.size) // block) % 2 == 0

    def index(bb):
        lo, hi = np.quantile(bb, [1 / 3, 2 / 3])
        thirds = {"low": bb <= lo, "high": bb >= hi}
        c = {(k, h): _corr_cols(Rf[m & (half if h else ~half)], xf[m & (half if h else ~half)])
             for k, m in thirds.items() for h in (True, False)}
        r = lambda u, v: np.corrcoef(u, v)[0, 1]
        within = (r(c["low", True], c["low", False]) + r(c["high", True], c["high", False])) / 2
        between = (r(c["low", True], c["high", False]) + r(c["low", False], c["high", True])) / 2
        return within, between

    w, b = index(base)
    null = np.array([np.subtract(*index(np.roll(base, k))) for k in _shifts(base.size, rng, n_shift, min_shift)])
    return {"within": float(w), "between": float(b), "reorganisation": float(w - b),
            "p": float((1 + (null >= w - b).sum()) / (n_shift + 1)), "null95": float(np.percentile(null, 95))}
