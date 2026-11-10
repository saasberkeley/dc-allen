"""Tracking motifs over time: direction, baseline, spread, polarity and presence, without fixed epochs.

For one signal (any LFP region or EEG channel, any session):

1. Band envelopes at 1 Hz (report bands), z-scored once over the recording (global units).
2. Sliding baseline and spread per band: centered rolling mean and SD (`baseline_window` s), computed
   inside segments cut at known event boundaries (isoflurane on, end of induction, stimulus block
   edges) so no window mixes two conditions. Residual r = (z - baseline) / spread: the fast part.
3. Motif detection (Mishler, signs) in sliding windows (`window` s, every `step` s) on the residual;
   candidates with permutation p <= alpha are kept.
4. Tracks: candidates in consecutive windows are linked when |cos| >= link_cos (Hungarian matching,
   gaps of up to max_gap windows allowed). Tracks living >= min_life windows form the library,
   merged when their mean directions match (|cos| >= merge_cos).
5. Per library motif, over time:
   per second  expression (r . u), baseline position (baseline . u), rolling spread of expression;
   per window  presence and best p, pole occupancy (+ / 0 / -), bimodality against phase surrogates,
               local direction and its angle to u with a split-half noise floor, and a held-out test of
               whether a locally re-estimated direction explains more variance than the fixed one.

Everything is returned as plain arrays and DataFrames, so results for many signals and mice can be saved
and compared.
"""

import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from diptest import diptest
from scipy.optimize import linear_sum_assignment

from . import candidate_motifs as cm
from . import gmm_motifs as gx
from . import session_data as sd
from .state_motifs import minority_sign_bands


@dataclass
class TrackConfig:
    baseline_window: int = 300
    window: int = 300
    step: int = 60
    alpha: float = 0.05
    link_cos: float = 0.8
    max_gap: int = 1
    min_life: int = 3
    merge_cos: float = 0.9
    pole_z: float = 0.5
    half_block: int = 30
    n_sur: int = 19          # bimodal flag at p < 0.05
    cut_at_events: bool = True
    seed: int = 0


@dataclass
class Signal:
    key: str
    info: object
    t: np.ndarray
    Z: np.ndarray          # global z
    base: np.ndarray       # sliding baseline (global z)
    spread: np.ndarray     # sliding SD (global z)
    resid: np.ndarray      # (Z - base) / spread
    segment: np.ndarray


@dataclass
class LibMotif:
    template: np.ndarray
    tracks: list
    windows: list

    @property
    def life(self):
        return len(set(self.windows))

    @property
    def broadband(self):
        return minority_sign_bands(self.template) <= 1


@dataclass
class TrackResult:
    key: str
    config: TrackConfig
    t: np.ndarray
    centres: np.ndarray                 # window centres (s)
    candidates: pd.DataFrame            # window, centre, p, axis (array), track
    tracks: pd.DataFrame                # track, first, last, n_windows, mean axis, library motif
    library: list
    second: pd.DataFrame                # t, segment, expr_M*, base_M*, spread_M*
    window_stats: pd.DataFrame          # one row per (window, motif)
    n_candidates: np.ndarray            # per window
    base_bands: np.ndarray = field(repr=False, default=None)


# ---------- 1-2: envelopes, baseline, residual ----------

def event_cuts(info):
    cuts = [info.iso_on, info.induction_end]
    for _, a, b in list(info.stim_blocks) + list(info.visual_blocks):
        cuts += [a, b]
    return np.sort(np.array(cuts))


def _window_sums(v, half):
    """Centered window sums of v (n, ...) over [j - half, j + half], truncated at the ends."""
    c = np.concatenate([np.zeros((1,) + v.shape[1:]), np.cumsum(v, axis=0)])
    j = np.arange(v.shape[0])
    lo, hi = np.maximum(j - half, 0), np.minimum(j + half, v.shape[0] - 1) + 1
    return c[hi] - c[lo]


def local_linear(y, half):
    """Baseline from a straight line fitted in a centered window (truncated at the segment ends) and
    evaluated at the centre. Unlike a rolling mean, it does not lag at segment edges."""
    n = y.shape[0]
    x = np.arange(n, dtype=float)[:, None]
    s0 = _window_sums(np.ones((n, 1)), half)
    sx, sxx = _window_sums(x, half), _window_sums(x ** 2, half)
    sy, sxy = _window_sums(y, half), _window_sums(x * y, half)
    den = s0 * sxx - sx ** 2
    ok = den > 1e-9
    slope = np.where(ok, (s0 * sxy - sx * sy) / np.where(ok, den, 1.0), 0.0)
    return (sy - slope * sx) / s0 + slope * x


def rolling_sd(r, half):
    s0 = _window_sums(np.ones((r.shape[0], 1)), half)
    m = _window_sums(r, half) / s0
    var = np.maximum(_window_sums(r ** 2, half) / s0 - m ** 2, 1e-12)
    return np.sqrt(var * s0 / np.maximum(s0 - 1, 1))


def split_baseline(Z, segment, window):
    """Per segment: local-linear baseline and rolling SD of the residual around it. Segments shorter
    than the window use half their own length as the half-window."""
    base, spread = np.full_like(Z, np.nan), np.full_like(Z, np.nan)
    for s in np.unique(segment):
        i = np.flatnonzero(segment == s)
        if i.size < 30:
            continue
        half = int(min(window // 2, max(15, i.size // 2)))
        base[i] = local_linear(Z[i], half)
        spread[i] = rolling_sd(Z[i] - base[i], half)
    return base, spread


def segments_of(t, cuts):
    seg = np.searchsorted(np.asarray(cuts, float), t, side="right") * 100000 + np.r_[0, np.cumsum(np.diff(t) > 2)]
    return np.unique(seg, return_inverse=True)[1]


def prepare_arrays(key, info, t, env, cfg=TrackConfig(), cuts=None):
    """Signal from envelopes already in memory (seconds x bands); cuts default to the session's events."""
    Z = (env - env.mean(axis=0)) / env.std(axis=0, ddof=1)
    if cuts is None:
        cuts = event_cuts(info) if (cfg.cut_at_events and info is not None) else []
    seg = segments_of(t, cuts)
    base, spread = split_baseline(Z, seg, cfg.baseline_window)
    return Signal(key, info, t, Z, base, spread, (Z - base) / spread, seg)


def prepare(nwb_path, key, source, cfg=TrackConfig(), car_cache=None):
    info, t, env = cm.signal_envelope(nwb_path, source, car_cache)
    return prepare_arrays(key, info, t, env, cfg)


# ---------- 3-4: window detection and tracks ----------

def window_starts(sig, cfg):
    return np.arange(sig.t[0], sig.t[-1] - cfg.window + 1, cfg.step)


def window_candidates(sig, cfg):
    rows = []
    ok = np.isfinite(sig.resid).all(axis=1)
    for w, a in enumerate(window_starts(sig, cfg)):
        sel = (sig.t >= a) & (sig.t < a + cfg.window) & ok
        if sel.sum() < 0.8 * cfg.window:
            continue
        axes, ps = cm.detect([(sig.t[sel], sig.resid[sel])], "none", True)
        for i in range(axes.shape[1]):
            if ps[i] <= cfg.alpha:
                rows.append({"window": w, "centre": a + cfg.window / 2, "p": ps[i],
                             "axis": axes[:, i] / np.linalg.norm(axes[:, i])})
    return pd.DataFrame(rows, columns=["window", "centre", "p", "axis"])


def link(cands, cfg):
    """Assign a track id to every candidate."""
    cands = cands.copy()
    cands["track"] = -1
    tracks = {}           # id -> (last window, last axis)
    nxt = 0
    for w in sorted(cands.window.unique()):
        idx = cands.index[cands.window == w]
        A = np.stack(cands.loc[idx, "axis"].to_numpy())
        live = [k for k, (lw, _) in tracks.items() if w - lw <= cfg.max_gap + 1]
        assigned = set()
        if live:
            B = np.stack([tracks[k][1] for k in live])
            C = np.abs(A @ B.T)
            r, c = linear_sum_assignment(-C)
            for i, j in zip(r, c):
                if C[i, j] >= cfg.link_cos:
                    k = live[j]
                    ax = A[i] * np.sign(A[i] @ tracks[k][1])
                    cands.at[idx[i], "axis"] = ax
                    cands.at[idx[i], "track"] = k
                    tracks[k] = (w, ax)
                    assigned.add(i)
        for i in range(len(idx)):
            if i not in assigned:
                cands.at[idx[i], "track"] = nxt
                tracks[nxt] = (w, A[i])
                nxt += 1
    return cands


def build_library(cands, cfg):
    rows = []
    for k, g in cands.groupby("track"):
        ax = np.stack(g["axis"].to_numpy()).mean(axis=0)
        rows.append({"track": k, "first": g.window.min(), "last": g.window.max(), "n_windows": g.window.nunique(),
                     "best_p": g.p.min(), "axis": ax / np.linalg.norm(ax)})
    tracks = pd.DataFrame(rows, columns=["track", "first", "last", "n_windows", "best_p", "axis"])
    tracks["motif"] = -1
    lib = []
    for i, r in tracks.sort_values("n_windows", ascending=False).iterrows():
        if r.n_windows < cfg.min_life:
            continue
        for mi, m in enumerate(lib):
            c = m.template @ r.axis
            if abs(c) >= cfg.merge_cos:
                ws = cands.loc[cands.track == r.track, "window"].tolist()
                n_old = len(m.windows)
                v = m.template * n_old + np.sign(c) * r.axis * len(ws)
                m.template = v / np.linalg.norm(v)
                m.tracks.append(r.track)
                m.windows += ws
                tracks.at[i, "motif"] = mi
                break
        else:
            lib.append(LibMotif(r.axis.copy(), [r.track], cands.loc[cands.track == r.track, "window"].tolist()))
            tracks.at[i, "motif"] = len(lib) - 1
    return tracks, lib


# ---------- 5: per-motif measures over time ----------

def _closest_eigvec(C, u):
    w, V = np.linalg.eigh(C)
    j = int(np.argmax(np.abs(V.T @ u)))
    v = V[:, j]
    return v * np.sign(v @ u), w[j]


def _angle(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)), 0, 1))))


def measure(sig, cands, lib, cfg):
    rng = np.random.default_rng(cfg.seed)
    U = np.array([m.template for m in lib]) if lib else np.zeros((0, sig.Z.shape[1]))
    E = sig.resid @ U.T
    Bm = sig.base @ U.T
    sec = pd.DataFrame({"t": sig.t, "segment": sig.segment})
    for i in range(len(lib)):
        sec[f"expr_M{i + 1}"] = E[:, i]
        sec[f"base_M{i + 1}"] = Bm[:, i]
        sec[f"spread_M{i + 1}"] = pd.Series(E[:, i]).rolling(cfg.baseline_window, center=True,
                                                             min_periods=cfg.baseline_window // 4).std().to_numpy()
    starts = window_starts(sig, cfg)
    ok = np.isfinite(sig.resid).all(axis=1)
    rows, n_c = [], np.zeros(starts.size, int)
    for w, a in enumerate(starts):
        sel = np.flatnonzero((sig.t >= a) & (sig.t < a + cfg.window) & ok)
        cw = cands[cands.window == w]
        n_c[w] = len(cw)
        if sel.size < 0.8 * cfg.window:
            continue
        R = sig.resid[sel]
        half = ((sig.t[sel] - a) // cfg.half_block) % 2 == 0
        Ca, Cb, Cw = np.corrcoef(R[half].T), np.corrcoef(R[~half].T), np.corrcoef(R.T)
        sur = [gx.phase_surrogate([R], rng)[0] for _ in range(cfg.n_sur)]
        for mi, u in enumerate(U):
            e = R @ u
            d = diptest(e)[0]
            d_sur = [diptest(s @ u)[0] for s in sur]
            va, _ = _closest_eigvec(Ca, u)
            vb, _ = _closest_eigvec(Cb, u)
            vw, _ = _closest_eigvec(Cw, u)
            match = [(abs(ax @ u), p) for ax, p in zip(cw.axis, cw.p) if abs(ax @ u) >= cfg.link_cos]
            closest = max((abs(ax @ u) for ax in cw.axis), default=np.nan)
            rows.append({
                "window": w, "centre": a + cfg.window / 2, "motif": mi,
                "present": bool(match), "best_p": min((p for _, p in match), default=np.nan),
                "mean_expr": e.mean(), "sd_expr": e.std(),
                "p_plus": np.mean(e > cfg.pole_z), "p_zero": np.mean(np.abs(e) <= cfg.pole_z), "p_minus": np.mean(e < -cfg.pole_z),
                "dip": d, "bimodal": d > max(d_sur),
                "var_along": float(u @ Cw @ u),
                "angle_detected": float(np.degrees(np.arccos(min(closest, 1.0)))) if np.isfinite(closest) else np.nan,
                "angle_local": _angle(vw, u), "angle_noise": _angle(va, vb),
                # held-out: does the half-A local direction explain half B better than the fixed one?
                "heldout_gain": float((va @ Cb @ va + vb @ Ca @ vb) / 2 - (u @ Cb @ u + u @ Ca @ u) / 2),
                "heldout_gain_rel": float(((va @ Cb @ va + vb @ Ca @ vb) - (u @ Cb @ u + u @ Ca @ u)) / (u @ Cb @ u + u @ Ca @ u)),
                "base_pos": float(np.nanmean(sig.base[sel] @ u)),
            })
    return sec, pd.DataFrame(rows), n_c


def track_prepared(sig, cfg=TrackConfig()):
    cands = link(window_candidates(sig, cfg), cfg)
    tracks, lib = build_library(cands, cfg)
    cands = cands.merge(tracks[["track", "motif"]], on="track", how="left")
    sec, ws, n_c = measure(sig, cands, lib, cfg)
    centres = window_starts(sig, cfg) + cfg.window / 2
    return TrackResult(sig.key, cfg, sig.t, centres, cands, tracks, lib, sec, ws, n_c, sig.base)


def track_signal(nwb_path, key, source, cfg=TrackConfig(), car_cache=None):
    sig = prepare(nwb_path, key, source, cfg, car_cache)
    return track_prepared(sig, cfg), sig


def _job(args):
    nwb_path, key, source, cfg, car_cache, out_dir = args
    res, _ = track_signal(nwb_path, key, source, cfg, car_cache)
    if out_dir:
        save(res, Path(out_dir) / f"{key}.pkl")
    return res


def track_many(nwb_path, signals, cfg=TrackConfig(), out_dir=None, workers=None, car_cache=None):
    """signals: [(key, source)]. Runs in parallel; saves one file per signal when out_dir is given."""
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    jobs = [(str(nwb_path), k, tuple(s), cfg, car_cache, out_dir) for k, s in signals]
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(_job, j) for j in jobs]):
            r = f.result()
            out[r.key] = r
    return out


def save(res, path):
    with open(path, "wb") as f:
        pickle.dump(res, f)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def behaviour(nwb_path):
    """Running speed and pupil width as (t, values) pairs, plus the session info."""
    io, nwb = sd.open_nwb(nwb_path)
    info = sd.read_info(nwb, Path(nwb_path).name)
    beh = nwb.processing["behavior"]
    rs = beh["BehavioralTimeSeries"]["running_speed"]
    run = rs.data[:]
    run_t = rs.timestamps[:] if rs.timestamps is not None else rs.starting_time + np.arange(run.size) / rs.rate
    pw = beh["PupilTracking"]["pupil_width"]
    pupil, pupil_t = pw.data[:], pw.timestamps[:]
    io.close()
    return info, (run_t, run), (pupil_t, pupil)
