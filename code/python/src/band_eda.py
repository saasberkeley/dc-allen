"""Exploratory analysis of band-envelope structure for one LFP channel.

Band sets: n log-spaced bands from 1 to 80 Hz, each overlapping half of its neighbours
(band i spans edges i to i+2 of n+2 log-spaced points). Stimulus handling: "included"
keeps the whole recording; "excluded" drops electrical and visual stimulus blocks
(plus a margin). Envelopes are computed at 250 Hz with smooth_env and kept at 1 Hz.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from diptest import diptest

from . import session_data as sd
from .get_motifs import get_motifs
from .smooth_env import smooth_env
from .state_motifs import FS, VERSIONS

BAND_COUNTS = (5, 6, 8, 10, 20)
STIM = ("included", "excluded")
WINDOWS = ("awake", "isoflurane", "full")
MARGIN = 5.0
BAND_BATCH = 5


def band_set(n, lo=1.0, hi=80.0):
    edges = np.geomspace(lo, hi, n + 2)
    return np.column_stack([edges[:-2], edges[2:]])


def band_names(bands):
    return [f"{a:.1f}-{b:.1f}" for a, b in bands]


def segments(info, stim):
    """Stretches analysed: the whole recording, or everything outside stimulus blocks."""
    if stim == "included":
        return [(info.t_start, info.iso_off)]
    blocks = sorted((a - MARGIN, b + MARGIN) for _, a, b in info.stim_blocks + info.visual_blocks)
    out, cursor = [], info.t_start
    for a, b in blocks:
        if a > cursor:
            out.append((cursor, min(a, info.iso_off)))
        cursor = max(cursor, b)
    if cursor < info.iso_off:
        out.append((cursor, info.iso_off))
    return [(a, b) for a, b in out if b - a > 60]


def envelopes(t, x, fs, segs, bands):
    """1 Hz envelopes (not z-scored) per stretch: list of (times, (seconds, bands))."""
    hop = int(round(fs))
    pieces = []
    for a, b in segs:
        k = (t >= a) & (t < b)
        seg = x[k]
        env = np.concatenate([smooth_env(seg, fs, bands[i:i + BAND_BATCH], zscore=False)[::hop]
                              for i in range(0, len(bands), BAND_BATCH)], axis=1)
        tt = t[k][::hop]
        ok = np.isfinite(env).all(axis=1)
        pieces.append((tt[ok], env[ok]))
    return pieces


def build_cache(nwb_path, cache_dir, probe="probeF", region="PL5", force=False):
    """Compute (or load) envelopes for every band count and stimulus option."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    io, nwb = sd.open_nwb(nwb_path)
    info = sd.read_info(nwb, Path(nwb_path).name)
    channel = sd.lfp_channel(nwb, probe, region)
    need = [(n, s) for n in BAND_COUNTS for s in STIM if force or not (cache_dir / f"env_{n}_{s}.npz").exists()]
    if need:
        t, x, fs = sd.decimate(*sd.read_raw(nwb, ("lfp", probe, channel)), FS)
    io.close()
    for n, s in need:
        pieces = envelopes(t, x, fs, segments(info, s), band_set(n))
        np.savez(cache_dir / f"env_{n}_{s}.npz", **{f"t{i}": p[0] for i, p in enumerate(pieces)},
                 **{f"e{i}": p[1] for i, p in enumerate(pieces)})
        print(f"cached {n} bands, stimulus {s}: {sum(p[0].size for p in pieces)} s", flush=True)
    return info, channel


def load(cache_dir, n, stim):
    f = np.load(Path(cache_dir) / f"env_{n}_{stim}.npz")
    k = len([key for key in f.files if key.startswith("t")])
    return [(f[f"t{i}"], f[f"e{i}"]) for i in range(k)]


@dataclass
class WindowData:
    t: np.ndarray        # seconds
    raw: np.ndarray      # (seconds, bands) envelope amplitude
    z: np.ndarray        # z-scored per band within the window
    scaled: np.ndarray   # divided by the window SD, not mean-subtracted
    piece: np.ndarray    # which stretch each second belongs to
    state: np.ndarray


def window_data(pieces, info, win):
    a, b = sd.analysis_windows(info)[win]
    ts, xs, ids = [], [], []
    for i, (tt, env) in enumerate(pieces):
        k = (tt >= a) & (tt < b)
        if k.sum() >= 60:
            ts.append(tt[k]); xs.append(env[k]); ids.append(np.full(k.sum(), i))
    t, raw, piece = np.concatenate(ts), np.concatenate(xs), np.concatenate(ids)
    sd_ = raw.std(axis=0, ddof=1)
    return WindowData(t, raw, (raw - raw.mean(axis=0)) / sd_, raw / sd_, piece, sd.state_of_time(t, info))


def pca(X, center=True):
    """Components (rows), scores, variance share. Signs: largest |loading| positive."""
    Xc = X - X.mean(axis=0) if center else X
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    vt = vt * np.sign(vt[np.arange(vt.shape[0]), np.abs(vt).argmax(axis=1)])[:, None]
    return vt, Xc @ vt.T, s ** 2 / (s ** 2).sum()


def chunks(wd, n_bands, chunk_sec=300):
    """~5-min chunks (longer when the null's band shifts need it) as (bands, seconds) arrays."""
    length = max(chunk_sec, 20 * (n_bands - 1) + 60)
    out = []
    for p in np.unique(wd.piece):
        idx = np.flatnonzero(wd.piece == p)
        n = idx.size // length or (1 if idx.size >= length / 2 else 0)
        out += [wd.z[part].T for part in (np.array_split(idx, n) if n else [])]
    return out, length


def motif_axes(wd, n_bands, seed=0):
    """Motif directions from the relaxed version, falling back to the original if it finds none."""
    sessions, length = chunks(wd, n_bands)
    if len(sessions) >= 2:
        res = get_motifs(sessions, 1.0, seed=seed, **VERSIONS["relaxed"]["kw"])
        if res.n_motifs:
            return res.motifs[:, : res.n_motifs], "relaxed", length
    res = get_motifs([wd.z.T], 1.0, seed=seed, **VERSIONS["original"]["kw"])
    return res.motifs[:, : res.n_motifs], "original", length


def band_shift_null(wd, rng):
    """Each band circularly shifted by a random amount within each stretch."""
    Z = wd.z.copy()
    for p in np.unique(wd.piece):
        idx = np.flatnonzero(wd.piece == p)
        for b in range(Z.shape[1]):
            Z[idx, b] = np.roll(Z[idx, b], rng.integers(idx.size))
    return Z


def pole_split(Z, axis, side, min_seconds=60):
    """Dip statistic of the main residual direction within one pole (seconds beyond 0.5 SD)."""
    score = Z @ axis
    cut = 0.5 * score.std()
    members = score > cut if side > 0 else score < -cut
    if members.sum() < min_seconds:
        return np.nan, members, None
    R = Z[members] - np.outer(score[members], axis)
    vt, sc, _ = pca(R)
    return diptest(sc[:, 0])[0], members, (vt[0], sc[:, 0])


def quiet_share(Z, radius=0.5):
    """Share of seconds whose bands are all near average (RMS z below radius)."""
    return float(np.mean(np.linalg.norm(Z, axis=1) / np.sqrt(Z.shape[1]) < radius))


def baseline_test(wd, n_null=200, seed=0):
    """Are quiet seconds more common than in a Gaussian with the same band correlations?"""
    rng = np.random.default_rng(seed)
    cov = np.cov(wd.z.T)
    real = quiet_share(wd.z)
    null = np.array([quiet_share(rng.multivariate_normal(np.zeros(len(cov)), cov, size=wd.z.shape[0]))
                     for _ in range(n_null)])
    return {"quiet": real, "quiet_gaussian": float(np.median(null)), "ratio": real / max(np.median(null), 1e-9),
            "p_quiet": float((1 + (null >= real).sum()) / (n_null + 1))}


def axis_tests(wd, axis, n_null=100, seed=0):
    """Bimodality along the axis, and a split within each pole (perpendicular to the axis)."""
    rng = np.random.default_rng(seed)
    axis = axis / np.linalg.norm(axis)
    score = wd.z @ axis
    real = {"dip_axis": diptest(score)[0], "dip_plus": pole_split(wd.z, axis, 1)[0],
            "dip_minus": pole_split(wd.z, axis, -1)[0]}
    nulls = {k: [] for k in real}
    for _ in range(n_null):
        Z = band_shift_null(wd, rng)
        nulls["dip_axis"].append(diptest(Z @ axis)[0])
        nulls["dip_plus"].append(pole_split(Z, axis, 1)[0])
        nulls["dip_minus"].append(pole_split(Z, axis, -1)[0])
    out = {}
    for k, v in real.items():
        v_null = np.array(nulls[k], dtype=float)
        v_null = v_null[np.isfinite(v_null)]
        out[k] = v
        out[f"{k}_null"] = float(np.median(v_null)) if v_null.size else np.nan
        out[f"p_{k}"] = (float((1 + (v_null >= v).sum()) / (v_null.size + 1))
                         if np.isfinite(v) and v_null.size else np.nan)
    out["score"] = score
    out["null_scores"] = np.array([band_shift_null(wd, rng) @ axis for _ in range(20)])
    return out
