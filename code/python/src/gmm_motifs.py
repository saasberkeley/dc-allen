"""Motif candidates from a Gaussian mixture fitted to the full band space (no motif axis given in advance).

Each second is a point in band space (one z-scored envelope value per band). A K-component Gaussian
mixture with full covariances is fitted to all seconds of a time bin; each component's mean is a
candidate motif (a band pattern), and its responsibility per second is that candidate's expression.

Choosing K. Seconds are strongly autocorrelated (envelopes are smoothed over 5 s), so BIC, which
assumes independent samples, keeps adding components. Instead:
  - cross-validated log-likelihood on held-out contiguous blocks of time, and
  - the same on a phase-randomized surrogate that keeps every band's autocorrelation and every
    pair of bands' cross-correlation (same random phases for all bands), so it is a Gaussian
    process with the same linear structure but no clusters.
The gain of the real data over the surrogate at each K measures cluster structure beyond
correlated Gaussian noise.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture

from .state_motifs import minority_sign_bands


def phase_surrogate(pieces, rng):
    """Same random Fourier phases for every band within each piece."""
    out = []
    for z in pieces:
        n = z.shape[0]
        F = np.fft.rfft(z - z.mean(axis=0), axis=0)
        ph = np.exp(2j * np.pi * rng.random(F.shape[0]))
        ph[0] = 1
        if n % 2 == 0:
            ph[-1] = 1
        s = np.fft.irfft(F * ph[:, None], n=n, axis=0) + z.mean(axis=0)
        out.append(s)
    # keep the per-band scale of the real data
    allz, alls = np.concatenate(pieces), np.concatenate(out)
    return [(s - alls.mean(0)) / alls.std(0) * allz.std(0) + allz.mean(0) for s in out]


def fit(Z, k, seed=0, n_init=3):
    return GaussianMixture(k, covariance_type="full", n_init=n_init, random_state=seed, reg_covar=1e-4).fit(Z)


def folds(pieces, n_folds=5):
    """Fold id per second: contiguous blocks of time, about equal size, across the concatenated pieces."""
    n = sum(z.shape[0] for z in pieces)
    return np.minimum(np.arange(n) * n_folds // n, n_folds - 1)


def cv_loglik(pieces, ks, n_folds=5, seed=0, gap=10):
    """Mean held-out log-likelihood per second for each K. `gap` seconds at each fold edge are dropped
    from the test block so smoothing does not leak between training and test."""
    Z = np.concatenate(pieces)
    fid = folds(pieces, n_folds)
    out = np.zeros((len(ks), n_folds))
    for f in range(n_folds):
        idx = np.flatnonzero(fid == f)
        test = idx[gap:-gap] if idx.size > 3 * gap else idx
        train = fid != f
        for i, k in enumerate(ks):
            out[i, f] = fit(Z[train], k, seed, n_init=2).score(Z[test])
    return out.mean(axis=1), out


def match(a, b):
    """Pair rows of a with rows of b (Hungarian on correlation); returns pairs and their r."""
    A = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-12)
    B = (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-12)
    R = A @ B.T / a.shape[1]
    r, c = linear_sum_assignment(-R)
    return list(zip(r, c)), R[r, c]


def split_half(pieces, k, seed=0, block=300):
    """Fit on alternating 5-min blocks (even vs odd) so both halves sample every state, and match the means.
    Also returns the adjusted Rand index between the two models' hard labels on all seconds."""
    from sklearn.metrics import adjusted_rand_score
    Z = np.concatenate(pieces)
    half = (np.arange(Z.shape[0]) // block) % 2 == 0
    g1, g2 = fit(Z[half], k, seed), fit(Z[~half], k, seed)
    pairs, r = match(g1.means_, g2.means_)
    ari = adjusted_rand_score(g1.predict(Z), g2.predict(Z))
    return g1, g2, pairs, r, ari


def describe(gm, Z):
    """One row per component: weight, size of the mean pattern, broadband flag, dwell time."""
    resp = gm.predict_proba(Z)
    lab = resp.argmax(axis=1)
    rows = []
    for k in range(gm.n_components):
        mu = gm.means_[k]
        runs = _runs(lab == k)
        rows.append({
            "weight": gm.weights_[k],
            "rms": float(np.sqrt(np.mean(mu ** 2))),
            "broadband": minority_sign_bands(mu) <= 1,
            "median_dwell_s": float(np.median(runs)) if runs.size else 0.0,
        })
    return rows, resp, lab


def _runs(mask):
    """Lengths of consecutive True stretches."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def opponents(means, r_max=-0.7):
    """Pairs of components whose mean patterns are strongly anti-correlated."""
    C = np.corrcoef(means)
    k = len(means)
    return [(i, j, C[i, j]) for i in range(k) for j in range(i + 1, k) if C[i, j] <= r_max]
