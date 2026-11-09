"""Python port of getMotifs.m (Mishler spectral-motif discovery).

Input: a list with one (bands, time) array per session, z-scored and free of NaN
(trim SmoothEnv's NaN edges first). Output: a (bands, 2 * n_motifs) array; the
second half holds the sign-inverted opponent of each motif in the first half.

The branch-consolidation loop in getMotifs.m walks per-leaf ancestor chains,
sweep after sweep. Every live chain is just "the path from some tree node to the
root", and whether two siblings merge depends only on the timepoints under them,
so that loop's fixpoint can be computed in one bottom-up pass over the tree per
threshold. tests/test_get_motifs.py checks that this gives the same candidates, in
the same order, as a line-by-line translation of the MATLAB loop.

Known differences from MATLAB (all on paths where MATLAB would error or is
undefined): a branch that consolidates all the way to the root is kept as a
candidate (MATLAB would index past the end of its chain); consolidation stops once a
single template remains (MATLAB would call linkage on one point); NaN correlation
distances from constant templates are treated as 1 (uncorrelated). Tied distances
can be broken differently by scipy's and MATLAB's linkage, and the null shifts
come from numpy's RNG, so results match MATLAB in distribution, not bit for bit.
"""

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from .smooth_env import mround

CORR_THRESHOLDS = (0.4, 0.5, 0.6, 0.7, 0.8)


@dataclass
class MotifResult:
    motifs: np.ndarray
    """(bands, 2 * n_motifs). Column i + n_motifs equals -column i."""
    thresholds: tuple
    null_max_size: np.ndarray
    """Per threshold: size of the largest motif in any shifted null dataset."""
    candidate_sizes: list
    """Per threshold: sizes of the motif candidates found in the real data."""
    n_kept: np.ndarray
    """Per threshold: candidates that beat the null and appear in every session."""
    n_timepoints: int
    """Non-trivial ~1 Hz timepoints in the real data."""
    shifts: np.ndarray = field(repr=False)
    """(bands, 1 + n_null) circular shifts in samples; column 0 is the real data."""
    null_largest: np.ndarray = field(default=None, repr=False)
    """(n_null, n_thresholds) size of the largest candidate in each null dataset."""
    kept_candidates: list = field(default_factory=list)
    """Candidates that passed the null and session filters: threshold, size, p, sessions_present."""
    motif_support: list = field(default_factory=list)
    """Per final motif: the kept candidates whose strongest PCA loading is on its axis."""
    representation: str = "binary"
    """Input representation clustered at ~1 Hz: ``binary`` or ``continuous``."""

    @property
    def n_motifs(self) -> int:
        return self.motifs.shape[1] // 2


@dataclass
class _Tree:
    a: np.ndarray       # left child of internal node n + k
    b: np.ndarray       # right child of internal node n + k
    parent: np.ndarray  # parent node id, -1 for the root
    size: np.ndarray    # timepoints under each node
    rep: np.ndarray     # smallest leaf position under each node (MATLAB list order)
    r: np.ndarray       # correlation between the two children's expression time courses
    sums: np.ndarray    # sum of member sign patterns under each node


def _expression_corr(t1, t2, second_moment, mean):
    """Row-wise corr(allCorr' * t1, allCorr' * t2) over all timepoints.

    Pearson correlation of projections only needs the data's mean and second
    moment, so this costs O(bands^2) per pair instead of O(timepoints).
    """
    m1, m2 = t1 @ mean, t2 @ mean
    c1, c2 = t1 @ second_moment, t2 @ second_moment
    e11, e22 = (c1 * t1).sum(1), (c2 * t2).sum(1)
    cov = (c1 * t2).sum(1) - m1 * m2
    v1, v2 = e11 - m1 ** 2, e22 - m2 ** 2
    flat = (v1 <= 1e-12 * np.maximum(e11, 1e-300)) | (v2 <= 1e-12 * np.maximum(e22, 1e-300))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt(v1 * v2)
    r[flat] = np.nan  # MATLAB corr() of a constant vector is NaN, which never passes r >= thresh
    return r


def _build_tree(Z, leaf_sums, leaf_sizes, second_moment, mean) -> _Tree:
    n = leaf_sums.shape[0]
    a = Z[:, 0].astype(np.int64)
    b = Z[:, 1].astype(np.int64)
    parent = np.full(2 * n - 1, -1, dtype=np.int64)
    parent[a] = np.arange(n, 2 * n - 1)
    parent[b] = np.arange(n, 2 * n - 1)

    sums = np.empty((2 * n - 1, leaf_sums.shape[1]))
    sums[:n] = leaf_sums
    size = np.empty(2 * n - 1)
    size[:n] = leaf_sizes
    rep = np.empty(2 * n - 1, dtype=np.int64)
    rep[:n] = np.arange(n)
    for k in range(n - 1):
        sums[n + k] = sums[a[k]] + sums[b[k]]
        size[n + k] = size[a[k]] + size[b[k]]
        rep[n + k] = min(rep[a[k]], rep[b[k]])

    templates = sums / size[:, None]
    r = _expression_corr(templates[a], templates[b], second_moment, mean)
    return _Tree(a=a, b=b, parent=parent, size=size, rep=rep, r=r, sums=sums)


def _consolidate(tree: _Tree, thresh: float) -> np.ndarray:
    """Tree nodes that end the MATLAB inner consolidation loop as motif candidates.

    Returned in MATLAB's order: branches still live at the end (by list position),
    then branches frozen for failing the threshold (by the sweep they froze in,
    then list position).
    """
    n = tree.a.size + 1
    live = np.zeros(2 * n - 1, dtype=bool)
    live[:n] = True
    ready = np.zeros(2 * n - 1, dtype=np.int64)  # sweep at whose start a node is live
    ready[:n] = 1
    frozen_in = np.zeros(2 * n - 1, dtype=np.int64)
    merge = (tree.r >= thresh).tolist()
    a, b = tree.a.tolist(), tree.b.tolist()
    live_l, ready_l, frozen_l = live.tolist(), ready.tolist(), frozen_in.tolist()
    for k in range(n - 1):
        ak, bk = a[k], b[k]
        if live_l[ak] and live_l[bk]:
            sweep = max(ready_l[ak], ready_l[bk])
            if merge[k]:
                live_l[n + k] = True
                ready_l[n + k] = sweep + 1
            else:
                frozen_l[ak] = frozen_l[bk] = sweep
    live = np.array(live_l)
    frozen_in = np.array(frozen_l)

    parent_live = np.zeros_like(live)
    has_parent = tree.parent >= 0
    parent_live[has_parent] = live[tree.parent[has_parent]]
    still_live = np.flatnonzero(live & ~parent_live & (frozen_in == 0))
    frozen = np.flatnonzero(frozen_in > 0)
    still_live = still_live[np.argsort(tree.rep[still_live], kind="stable")]
    frozen = frozen[np.lexsort((tree.rep[frozen], frozen_in[frozen]))]
    return np.concatenate([still_live, frozen])


def _leaf_owner(tree: _Tree, nodes: np.ndarray) -> np.ndarray:
    """For each leaf, the index (into `nodes`) of the candidate subtree containing it."""
    owner = np.full(tree.parent.size, -1, dtype=np.int64)
    owner[nodes] = np.arange(nodes.size)
    parent = tree.parent.tolist()
    owner_l = owner.tolist()
    for v in range(len(owner_l) - 1, -1, -1):
        if owner_l[v] < 0:
            owner_l[v] = owner_l[parent[v]]
    n = tree.a.size + 1
    return np.array(owner_l[:n], dtype=np.int64)


def _correlation_linkage(points: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = pdist(points, "correlation")
    return linkage(np.nan_to_num(d, nan=1.0), "average")


def _candidates(X, tree0, thresh, second_moment, mean, need_labels):
    """One threshold of getMotifs' consolidation, outer re-clustering loop included.

    Returns (sizes, labels): candidate sizes in MATLAB order, and for each
    timepoint the index of its candidate (None unless need_labels).
    """
    nodes = _consolidate(tree0, thresh)
    labels = _leaf_owner(tree0, nodes) if need_labels else None
    sums, sizes = tree0.sums[nodes], tree0.size[nodes]
    history = [nodes.size]
    while not (len(history) > 1 and history[-1] == history[-2]) and sizes.size > 1:
        tree = _build_tree(_correlation_linkage(sums / sizes[:, None]), sums, sizes, second_moment, mean)
        nodes = _consolidate(tree, thresh)
        if need_labels:
            labels = _leaf_owner(tree, nodes)[labels]
        sums, sizes = tree.sums[nodes], tree.size[nodes]
        history.append(nodes.size)
    return sizes, labels


def candidate_partition(X: np.ndarray, thresh: float):
    """Motif candidates for (timepoints, bands) sign data at one threshold, before any null test."""
    X = np.asarray(X, dtype=float)
    second_moment, mean = X.T @ X / X.shape[0], X.mean(axis=0)
    tree0 = _build_tree(_correlation_linkage(X), X, np.ones(X.shape[0]), second_moment, mean)
    return _candidates(X, tree0, thresh, second_moment, mean, need_labels=True)


def _pca_uncentered(F):
    """MATLAB pca(F, 'Centered', 'off'), including its coefficient sign convention."""
    _, _, vt = np.linalg.svd(F, full_matrices=False)
    coeff = vt.T
    biggest = np.argmax(np.abs(coeff), axis=0)
    sign = np.sign(coeff[biggest, np.arange(coeff.shape[1])])
    sign[sign == 0] = 1
    coeff = coeff * sign
    return coeff, F @ coeff


def _null_shifts(n_bands, fs, n_null, rng):
    shift_times = np.array([mround(fs * 20 * k) for k in range(n_bands)])
    perms = np.stack([shift_times[rng.permutation(n_bands)] for _ in range(100)])
    perms = np.unique(perms, axis=0)  # sorted rows, as MATLAB's unique(..., 'rows')
    return np.column_stack([np.zeros(n_bands, dtype=np.int64), perms[:n_null].T])


def _represent(sessions, shift, hop, representation):
    shifted = [np.stack([np.roll(s[f], shift[f]) for f in range(s.shape[0])]) for s in sessions]
    sampled = np.concatenate(shifted, axis=1)[:, ::hop]
    if representation == "continuous":
        return sampled.T, np.ones(sampled.shape[1], dtype=bool)
    signs = np.sign(sampled)
    nontrivial = np.abs(signs.sum(axis=0)) != signs.shape[0]
    return signs[:, nontrivial].T, nontrivial


def get_motifs(
    lfp_in,
    fs: float,
    corr_thresholds=CORR_THRESHOLDS,
    n_null: int = 20,
    seed=None,
    verbose: bool = False,
    null_mode: str = "max",
    alpha: float = 0.05,
    min_session_frac: float = 1.0,
    min_pca_support: int = 2,
    normalize_size: bool = False,
    representation: str = "binary",
) -> MotifResult:
    """Port of getMotifs(LFPIn, LFPfs).

    lfp_in: list of (bands, time) arrays, one per session, same bands in the same
    order. Must not contain NaN. fs: sampling rate of the time axis in Hz.

    The defaults reproduce getMotifs.m. The other settings loosen its three filters:
      null_mode="pvalue": keep a candidate if its permutation p-value is <= alpha, where
          p = (1 + #nulls whose largest candidate is at least as big) / (n_null + 1).
          "max" keeps only candidates bigger than the largest candidate of every null.
      min_session_frac: fraction of sessions a candidate must have timepoints in (1.0 = all).
      min_pca_support: candidates that must load on a PCA axis for it to become a motif
          (getMotifs.m requires 2).
      normalize_size: compare candidate size as a fraction of each dataset's non-trivial
          timepoints. Shuffling breaks broadband co-fluctuation, so null datasets keep more
          timepoints than the real data and grow bigger clusters for that reason alone.

    ``representation="binary"`` reproduces the MATLAB method by clustering the
    sign of each z-scored band. ``representation="continuous"`` is an
    experimental ablation that clusters the actual z-scored band-envelope values
    sampled at the same ~1 Hz cadence. The null shifts and all downstream
    significance tests are otherwise identical.
    """
    if null_mode not in ("max", "pvalue"):
        raise ValueError("null_mode must be 'max' or 'pvalue'")
    sessions = [np.asarray(s, dtype=float) for s in lfp_in]
    n_bands = sessions[0].shape[0]
    if any(s.ndim != 2 or s.shape[0] != n_bands for s in sessions):
        raise ValueError("every session must be a (bands, time) array with the same number of bands")
    if not all(np.isfinite(s).all() for s in sessions):
        raise ValueError("input contains NaN/inf; trim SmoothEnv's NaN edges before calling get_motifs")
    if representation not in {"binary", "continuous"}:
        raise ValueError("representation must be 'binary' or 'continuous'")

    hop = mround(fs)
    session_of = np.concatenate([np.full(s.shape[1], i) for i, s in enumerate(sessions)])[::hop]
    shifts = _null_shifts(n_bands, fs, n_null, np.random.default_rng(seed))
    thresholds = tuple(corr_thresholds)

    null_largest = np.zeros((shifts.shape[1] - 1, len(thresholds)))
    null_n = np.ones(shifts.shape[1] - 1)
    real = None
    for s in range(shifts.shape[1]):
        t0 = time.time()
        X, nontrivial = _represent(sessions, shifts[:, s], hop, representation)
        second_moment, mean = X.T @ X / X.shape[0], X.mean(axis=0)
        tree0 = _build_tree(_correlation_linkage(X), X, np.ones(X.shape[0]), second_moment, mean)
        per_thresh = [_candidates(X, tree0, t, second_moment, mean, need_labels=(s == 0)) for t in thresholds]
        if s == 0:
            real = (X, session_of[nontrivial], per_thresh)
        else:
            null_largest[s - 1] = [sizes.max() for sizes, _ in per_thresh]
            null_n[s - 1] = X.shape[0]
        if verbose:
            kind = "real data" if s == 0 else f"null {s}/{shifts.shape[1] - 1}"
            print(f"  [{kind}] {X.shape[0]} timepoints, candidates per threshold: "
                  f"{[sizes.size for sizes, _ in per_thresh]} ({time.time() - t0:.1f}s)", flush=True)

    X, session_ids, per_thresh = real
    if normalize_size:
        null_largest = null_largest / null_n[:, None] * X.shape[0]  # null sizes rescaled to the real data's length
    null_max = null_largest.max(axis=0) if null_largest.size else np.zeros(len(thresholds))
    templates, n_kept, kept_info = [], [], []
    for t, ((sizes, labels), limit) in enumerate(zip(per_thresh, null_max)):
        p = (1 + (null_largest[:, t][None, :] >= sizes[:, None]).sum(axis=1)) / (null_largest.shape[0] + 1)
        passes = sizes > limit if null_mode == "max" else p <= alpha
        kept = 0
        for k in np.flatnonzero(passes):
            members = labels == k
            n_present = np.unique(session_ids[members]).size
            if n_present >= min_session_frac * len(sessions) - 1e-9:
                templates.append(X[members].mean(axis=0))
                kept_info.append({"threshold": thresholds[t], "size": int(sizes[k]), "p": float(p[k]),
                                  "sessions_present": int(n_present)})
                kept += 1
        n_kept.append(kept)

    result = MotifResult(
        motifs=np.zeros((n_bands, 0)),
        thresholds=thresholds,
        null_max_size=null_max,
        candidate_sizes=[sizes for sizes, _ in per_thresh],
        n_kept=np.array(n_kept),
        n_timepoints=X.shape[0],
        shifts=shifts,
        null_largest=null_largest,
        kept_candidates=kept_info,
        representation=representation,
    )
    if not templates:
        return result

    coeff, score = _pca_uncentered(np.array(templates))
    magnitude = np.abs(score)
    which, best_axis = np.nonzero(magnitude == magnitude.max(axis=1, keepdims=True))
    axes = np.flatnonzero(np.bincount(best_axis, minlength=coeff.shape[1]) >= min_pca_support)
    if axes.size == 0:
        return result
    result.motif_support = [[kept_info[c] for c in which[best_axis == a]] for a in axes]

    proj = X @ coeff[:, axes]
    proj = np.hstack([proj, -proj])
    best = proj.max(axis=1)
    motifs = np.full((n_bands, axes.size), np.nan)
    for i in range(axes.size):
        pooled = np.vstack([X[proj[:, i] == best], -X[proj[:, i + axes.size] == best]])
        if pooled.size:
            motifs[:, i] = pooled.mean(axis=0)
    result.motifs = np.hstack([motifs, -motifs])
    return result
