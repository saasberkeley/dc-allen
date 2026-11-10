"""Python port of getClust.m (Mishler et al.): groups of neurons with coordinated activity epochs.

Line-by-line port of code/matlab/getClust.m. Differences from MATLAB:
  - indices are 0-based;
  - the random coalitions use a seedable numpy Generator (MATLAB's randperm there is unseeded);
  - bwconncomp on a 1-D mask is replaced by an equivalent run-length search; graph/conncomp by
    scipy's connected_components.

Input X is (time samples x neurons), already preprocessed upstream (Mishler: square and z-score each
neuron's continuous rate trace). fs is the sampling rate of X's rows; it sets the 3-s minimum event.
Returns a list of integer arrays, one per cluster (column indices of X).

Behaviour worth knowing (tested in tests/test_get_clust.py): a single ensemble, or two ensembles that
alternate (opponent ensembles), are recovered. Two independent ensembles whose episodes overlap in time
tend to be proposed as one candidate group; random 3-neuron coalitions then mix them, every member's
reliability sits near the threshold of 1, and the group can be rejected. That follows from the MATLAB
algorithm, not from the port.
"""

import numpy as np
from scipy.sparse.csgraph import connected_components

N_SUB = 500
SUB_SIZE = 3
SUPPORT_Z = 1.0
SCORE_THRESH = 1.0
MERGE_SCORE_THRESH = 0.6
MIN_CLUST_SIZE = 5
MAX_ITER = 20


def _mean(x):
    return x.mean() if x.size else np.nan        # MATLAB mean([]) is NaN


def group_support(Xg):
    """(fraction of members with X > 0) * (mean of members), per time sample."""
    return (Xg > 0).mean(axis=1) * Xg.mean(axis=1)


def event_mask(support, min_dur, z=SUPPORT_Z):
    """Samples >= mean + z*SD (MATLAB std, N-1), kept only in runs of at least min_dur samples."""
    thresh = support.mean() + z * support.std(ddof=1)
    m = support >= thresh
    d = np.diff(np.r_[0, m.astype(int), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    out = np.zeros_like(m)
    for a, b in zip(starts, ends):
        if b - a >= min_dur:
            out[a:b] = True
    return out


def recruitment(X, cells, mask):
    """Each cell's mean during events minus mean outside them."""
    return np.array([_mean(X[mask, c]) - _mean(X[~mask, c]) for c in cells])


def get_clust(X, fs, seed=None):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    min_dur = int(round(3 * fs))
    remaining = np.arange(X.shape[1])
    all_clust = []

    while remaining.size:
        remaining_prev_outer = remaining.copy()
        master = []
        seed_cells = remaining.copy()

        while seed_cells.size:
            total_keep = np.array([], int)
            groups = []
            group_keep = seed_cells
            while total_keep.size != group_keep.size:
                g = np.setdiff1d(group_keep, total_keep)
                for it in range(1, MAX_ITER + 1):
                    old = g.copy()
                    mask = event_mask(group_support(X[:, g]), min_dur)
                    score = recruitment(X, g, mask)
                    with np.errstate(invalid="ignore"):
                        g = g[score > min(1.0, 0.3 * it)]
                    if np.array_equal(g, old):
                        break
                if g.size == 0:
                    break
                groups.append(g)
                total_keep = np.concatenate([total_keep, g])
            if not groups:
                break
            master.append(groups)
            used = np.unique(np.concatenate(groups))
            prev = seed_cells
            seed_cells = np.setdiff1d(seed_cells, used)
            if np.array_equal(seed_cells, prev):
                break

        raw = [g for groups in master for g in groups]
        if not raw:
            break
        raw = [g for g in raw if g.size >= MIN_CLUST_SIZE]
        if not raw:
            break

        clean = []
        for cells in raw:
            n = cells.size
            sub_size = min(SUB_SIZE, n - 1)
            gx = X[:, cells]
            s_sum, s_sq, count = np.zeros(n), np.zeros(n), np.zeros(n)
            for _ in range(N_SUB):
                sub = rng.permutation(n)[:sub_size]
                mask = event_mask(group_support(gx[:, sub]), min_dur)
                if not mask.any():
                    continue
                test = np.setdiff1d(np.arange(n), sub)
                sc = recruitment(gx, test, mask)
                s_sum[test] += sc
                s_sq[test] += sc ** 2
                count[test] += 1
            mean = s_sum / np.maximum(count, 1)
            std = np.sqrt(np.maximum(s_sq / np.maximum(count, 1) - mean ** 2, 0))
            rel = mean / (std + np.finfo(float).eps)
            order = np.argsort(-rel, kind="stable")
            kept = cells[order][rel[order] > SCORE_THRESH]
            if kept.size >= MIN_CLUST_SIZE:
                clean.append(kept)
        if not clean:
            break
        accepted = np.unique(np.concatenate(clean))
        all_clust += clean
        remaining = np.setdiff1d(remaining, accepted)
        if np.array_equal(remaining, remaining_prev_outer):
            break

    # merge clusters that recruit each other (either direction), transitively
    while len(all_clust) >= 2:
        k = len(all_clust)
        masks = [event_mask(group_support(X[:, c]), min_dur) for c in all_clust]
        cross = np.full((k, k), np.nan)
        for a in range(k):
            if not masks[a].any():
                continue
            for b in range(k):
                if a != b:
                    cross[a, b] = np.mean(recruitment(X, all_clust[b], masks[a]))
        merge = np.nan_to_num(cross, nan=-np.inf) > MERGE_SCORE_THRESH
        np.fill_diagonal(merge, False)
        if not merge.any():
            break
        n_comp, comp = connected_components(merge | merge.T, directed=False)
        new = [np.unique(np.concatenate([all_clust[i] for i in np.flatnonzero(comp == c)])) for c in range(n_comp)]
        if len(new) < len(all_clust):
            all_clust = new
        else:
            break
    return all_clust


def preprocess(rate):
    """Mishler's preprocessing: square each neuron's rate trace, then z-score it (time x neurons)."""
    sq = np.asarray(rate, float) ** 2
    sd = sq.std(axis=0, ddof=1)
    return (sq - sq.mean(axis=0)) / np.where(sd > 0, sd, 1)
