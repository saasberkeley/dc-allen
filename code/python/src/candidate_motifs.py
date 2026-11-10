"""Candidate-motif extraction with switchable design choices, for maximizing candidates.

Each choice is a switch, so variants can be compared and combined:
  time bins     "standard" (awake and isoflurane, stimulus-free) or "fine" (spontaneous awake,
                awake + electrical, awake + visual, early isoflurane, isoflurane + electrical,
                isoflurane + visual, late isoflurane)
  binarize      cluster signs (Mishler) or the z-scored values themselves
  significance  "strict" (getMotifs.m), "relaxed" (p-value, half the chunks), "any" (p-value,
                present in at least one chunk), or "none" (every candidate kept, p-value as a score)
  poles         "mirror" (motif and its exact negative), "gmm" (2-component Gaussian mixture
                along each motif axis; the poles can differ in strength but are always opposite
                in shape), or "gmm3" (3 components: low pole, middle, high pole; poles can differ
                in shape)
  baseline      add a "baseline" label for quiet seconds (all bands near average)
  reference     EEG as recorded, or common-average referenced (CAR) across valid EEG channels

Bands are the 10 overlapping bands of the 2026-11-01 report (state_motifs.BANDS), so results
compare directly with it. Templates are compared by signed correlation; |r| >= MERGE_R with the same sign counts as the
same candidate when building unions.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

from . import band_eda as be
from . import session_data as sd
from .get_motifs import get_motifs
from .smooth_env import smooth_env
from .state_motifs import BANDS, FS, minority_sign_bands

EDGE = 10.0
MERGE_R = 0.9
QUIET_RMS = 0.5
STANDARD_BINS = ("awake", "isoflurane")
FINE_BINS = ("awake", "awake_electrical", "awake_visual", "iso_early", "iso_electrical", "iso_visual", "iso_late")
SIGNIFICANCE = {
    "strict": {"n_null": 20},
    "relaxed": {"n_null": 100, "null_mode": "pvalue", "alpha": 0.05, "min_session_frac": 0.5, "normalize_size": True},
    "any": {"n_null": 100, "null_mode": "pvalue", "alpha": 0.05, "min_session_frac": 1e-9, "normalize_size": True},
    "none": {"n_null": 100, "null_mode": "pvalue", "alpha": 1.0, "min_session_frac": 1e-9, "normalize_size": True},
}


@dataclass
class Variant:
    name: str
    bins: str = "standard"
    binarize: bool = True
    significance: str = "relaxed"
    poles: str = "mirror"
    baseline: bool = False


@dataclass
class Candidate:
    signal: str
    bin: str
    variant: str
    template: np.ndarray
    kind: str                      # "pole", "baseline"
    p: float = np.nan              # best permutation p of the supporting candidates
    expression: np.ndarray = field(default=None, repr=False)
    seconds: np.ndarray = field(default=None, repr=False)

    @property
    def broadband(self):
        return self.kind == "pole" and minority_sign_bands(self.template) <= 1


def time_bins(info):
    """Named lists of (start, stop) intervals, shrunk by EDGE so smoothing does not leak across edges."""
    spont = be.segments(info, "excluded")
    windows = sd.analysis_windows(info)
    elec = {lab: (a, b) for lab, a, b in info.stim_blocks}
    vis = {lab: (a, b) for lab, a, b in info.visual_blocks}

    def clip(a, b):
        return [(max(s0, a), min(s1, b)) for s0, s1 in spont if min(s1, b) > max(s0, a)]

    raw = {
        "awake": clip(*windows["awake"]),
        "isoflurane": clip(*windows["isoflurane"]),
        "awake_electrical": [elec["awake"]],
        "awake_visual": [vis["awake"]],
        "iso_early": clip(windows["isoflurane"][0], elec["isoflurane"][0]),
        "iso_electrical": [elec["isoflurane"]],
        "iso_visual": [vis["isoflurane"]],
        "iso_late": clip(elec["isoflurane"][1], windows["isoflurane"][1]),
    }
    return {k: [(a + EDGE, b - EDGE) for a, b in v if b - a - 2 * EDGE >= 60] for k, v in raw.items()}


def car_reference(nwb, cache):
    """Mean of every valid EEG channel at FS, cached to disk."""
    cache = Path(cache)
    if cache.exists():
        f = np.load(cache)
        return f["t"], f["mean"]
    valid = np.flatnonzero(nwb.acquisition["ElectricalSeriesEEG"].electrodes.to_dataframe().is_data_valid.values)
    total = None
    for ch in valid:
        t, x, _ = sd.decimate(*sd.read_raw(nwb, ("eeg", int(ch))), FS)
        total = x if total is None else total + x
    mean = total / valid.size
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, t=t, mean=mean)
    return t, mean


def signal_envelope(nwb_path, source, car_cache=None):
    """1 Hz envelopes over the whole recording (up to isoflurane off), not z-scored."""
    io, nwb = sd.open_nwb(nwb_path)
    info = sd.read_info(nwb, Path(nwb_path).name)
    src = ("lfp", source[1], sd.lfp_channel(nwb, source[1], source[2])) if source[0] == "lfp" else ("eeg", source[1])
    t, x, fs = sd.decimate(*sd.read_raw(nwb, src), FS)
    if car_cache is not None:
        tc, mean = car_reference(nwb, car_cache)
        x = x - np.interp(t, tc, mean)
    io.close()
    k = (t >= info.t_start) & (t < info.iso_off)
    hop = int(round(fs))
    env = np.concatenate([smooth_env(x[k], fs, BANDS[i:i + 5], zscore=False)[::hop] for i in range(0, len(BANDS), 5)], axis=1)
    tt = t[k][::hop]
    ok = np.isfinite(env).all(axis=1)
    return info, tt[ok], env[ok]


def bin_pieces(t, env, intervals):
    """Per-interval (times, z) with z-scoring done across the whole bin."""
    pieces = [(t[(t >= a) & (t < b)], env[(t >= a) & (t < b)]) for a, b in intervals]
    pieces = [p for p in pieces if p[0].size >= 60]
    if not pieces:
        return []
    allv = np.concatenate([p[1] for p in pieces])
    mu, sdv = allv.mean(axis=0), allv.std(axis=0, ddof=1)
    return [(tt, (e - mu) / sdv) for tt, e in pieces]


def detect(pieces, significance, binarize, seed=0, chunk_sec=300):
    """Motif axes (bands, k) and the best p of each, chunking where possible."""
    sessions = []
    for _, z in pieces:
        n = z.shape[0] // chunk_sec or (1 if z.shape[0] >= chunk_sec / 2 else 0)
        sessions += [part.T for part in (np.array_split(z, n) if n else [])]
    kw = dict(SIGNIFICANCE[significance])
    if len(sessions) < 2:
        sessions = [np.concatenate([z for _, z in pieces]).T]
        kw["min_session_frac"] = 1e-9
    res = get_motifs(sessions, 1.0, seed=seed, binarize=binarize, **kw)
    ps = [min(c["p"] for c in s) for s in res.motif_support]
    axes = res.motifs[:, : res.n_motifs]
    # get_motifs leaves a motif as NaN when no timepoint is assigned to either of its poles
    ok = np.isfinite(axes).all(axis=0)
    return axes[:, ok], [p for p, k in zip(ps, ok) if k]


def poles_for(axes, ps, Z, mode):
    """[(template, expression per second, p)] for every pole of every axis."""
    out = []
    for i in range(axes.shape[1]):
        u = axes[:, i] / np.linalg.norm(axes[:, i])
        score = Z @ u
        if mode == "mirror":
            out += [(axes[:, i], score, ps[i]), (-axes[:, i], -score, ps[i])]
            continue
        # With 2 components on z-scored data the two weighted means are always exact negatives
        # (responsibilities sum to 1 and Z averages to 0). A middle component frees the poles.
        k_comp = 3 if mode == "gmm3" else 2
        gm = GaussianMixture(k_comp, random_state=0).fit(score[:, None])
        resp = gm.predict_proba(score[:, None])
        order = np.argsort(gm.means_.ravel())
        for k in (order[0], order[-1]):
            w = resp[:, k]
            out.append(((w @ Z) / w.sum(), w, ps[i]))
    return out


def run_signal(nwb_path, key, source, variants, car_cache=None, keep_expression=False):
    """Every variant's candidates for one signal."""
    info, t, env = signal_envelope(nwb_path, source, car_cache)
    bins = time_bins(info)
    cands = []
    cache = {}
    for v in variants:
        for b in (STANDARD_BINS if v.bins == "standard" else FINE_BINS):
            pieces = bin_pieces(t, env, bins[b])
            if not pieces:
                continue
            Z = np.concatenate([z for _, z in pieces])
            secs = np.concatenate([tt for tt, _ in pieces])
            ck = (b, v.significance, v.binarize)
            if ck not in cache:
                cache[ck] = detect(pieces, v.significance, v.binarize)
            axes, ps = cache[ck]
            for template, expr, p in poles_for(axes, ps, Z, v.poles):
                cands.append(Candidate(key, b, v.name, np.asarray(template, float), "pole", p,
                                       expr if keep_expression else None, secs if keep_expression else None))
            if v.baseline:
                quiet = (np.linalg.norm(Z, axis=1) / np.sqrt(Z.shape[1]) < QUIET_RMS).astype(float)
                cands.append(Candidate(key, b, v.name, np.zeros(Z.shape[1]), "baseline", np.nan,
                                       quiet if keep_expression else None, secs if keep_expression else None))
    return cands


def run_many(nwb_path, signals, variants, car_cache=None, workers=None):
    """run_signal over many (key, source, use_car) in parallel."""
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    out = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(run_signal, str(nwb_path), key, src, variants, car_cache if car else None)
                for key, src, car in signals]
        for job in as_completed(jobs):
            out += job.result()
    return out


def _standardized(cands):
    """Rows scaled so a dot product between two rows is their Pearson correlation."""
    T = np.array([c.template for c in cands], dtype=float)
    T = T - T.mean(axis=1, keepdims=True)
    return T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)


def unique(cands, r=MERGE_R, sort=True):
    """Greedy union: keep a pole candidate unless it matches (signed r >= r) one already kept.
    Candidates are taken in order of p-value (sort=True) or as given."""
    poles = [c for c in cands if c.kind == "pole"]
    if sort:
        poles = sorted(poles, key=lambda c: np.nan_to_num(c.p, nan=1.0))
    if not poles:
        return []
    T = _standardized(poles)
    keep = []
    for i in range(len(poles)):
        if not keep or (T[keep] @ T[i]).max() < r:
            keep.append(i)
    return [poles[i] for i in keep]


def matched_share(a, b, r=MERGE_R):
    """Share of pole candidates in a that have a match (signed r >= r) in b."""
    a = [c for c in a if c.kind == "pole"]
    b = [c for c in b if c.kind == "pole"]
    if not a or not b:
        return np.nan
    return float(np.mean((_standardized(a) @ _standardized(b).T).max(axis=1) >= r))
