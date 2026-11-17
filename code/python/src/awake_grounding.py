"""Do LFP motif poles track neuron groups in spontaneous awake activity? (Mishler et al. 2026, Fig. 3)

For every recording with Neuropixels units, every region with enough well-isolated units and an LFP channel
in that same region is one "signal". Only awake seconds with no stimulation are used:
  - isoflurane sessions: before isoflurane is switched on (minus 30 s);
  - the psilocybin session: before the psilocybin injection, without the saline injection (+60 s after);
  - every electrical / visual stimulation block is cut out with a MARGIN-second margin.

Three motif pipelines run on exactly these seconds:
  original  getMotifs.m rules on the joined awake data (20 nulls, larger than every null, one session);
            expression = sign-binarized z-scored envelopes projected on the template (Mishler's LFP projection)
  relaxed   5-min chunks as sessions, permutation p <= 0.05, present in at least half the chunks; same projection
  tracker   sliding baseline (5 min) and spread removed, motifs found in sliding 5-min windows and linked into a
            library; expression = residual projected on each library direction

Mishler's claims, each tested held-out (neuron groups chosen on training blocks, scored on held-out blocks):
  C1  ensembles: each neuron is assigned to the motif pole (motif k, sign +/-) whose expression correlates most
      with its rate; the + ensemble's rate rises with expression, the - ensemble's falls
  C2  the + and - ensembles of one motif are anticorrelated
  C3  a motif tracks its ensembles better than a single band does (each band gets its own +/- ensembles, chosen
      the same way, so motif and band are compared on equal terms)
  C4  the opponent split is recoverable from spiking alone (getClust; PCA opponent axis)
Significance: the whole procedure (assignment included) is repeated after circularly shifting the LFP relative to
the spikes by at least 300 s (a third of the data if shorter, same for every method); this keeps both signals'
autocorrelation and breaks their alignment.

The tracker is tested on every per-second output, each as if it were the motif signal: expression (fast part),
baseline (slow part), spread, presence, and the unsplit projection. In C3 each output is compared with the same
quantity computed for single bands. Part B asks which method's outputs, all together, explain spiking best
(held-out ridge R^2 for every neuron and the population rate, minus the same on shifted features).
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from . import motif_tracking as mt
from . import session_data as sd
from .get_clust import get_clust, group_support, preprocess
from .get_motifs import get_motifs
from .smooth_env import smooth_env
from .state_motifs import BANDS, FS, VERSIONS, chunk, minority_sign_bands, Piece

MARGIN = 10.0          # s cut around each stimulation block
MIN_UNITS = 20
MIN_PIECE = 60.0       # s; envelopes lose 10 s at each end
BLOCK, GAP, N_FOLDS = 60, 10, 5
TRACK_CFG = mt.TrackConfig(step=30, min_life=4)   # awake data are short: finer step, same 5-min window


# ---------------------------------------------------------------- which seconds, which signals

def awake_intervals(nwb):
    """Spontaneous awake (start, stop) intervals and a short description of how they were cut."""
    lfp_t = sd.first_timestamps(nwb)
    t0 = float(lfp_t[0])
    cuts = []
    if "isoflurane_level" in nwb.acquisition:
        iso = nwb.acquisition["isoflurane_level"]
        level = iso.data[:]
        iso_t = iso.starting_time + np.arange(level.size) / iso.rate
        end = float(iso_t[np.argmax(level > 0.5)]) - 30.0
        how = "before isoflurane on (-30 s)"
    else:
        ep = nwb.epochs.to_dataframe()
        tags = ep.tags.map(lambda v: " ".join(v).lower())
        end = float(ep.start_time[tags.str.contains("psilocybin")].min())
        for _, r in ep[tags.str.contains("saline")].iterrows():
            cuts.append((r.start_time, r.stop_time + 60.0))
        how = "before psilocybin injection; saline injection (+60 s) cut"
    tr = nwb.trials.to_dataframe()
    for _, g in tr.groupby(["stimulus_type", "behavioral_epoch"]):
        cuts.append((g.start_time.min() - MARGIN, g.stop_time.max() + MARGIN))
    out, start = [], t0
    for a, b in sorted(cuts):
        if a > start:
            out.append((start, min(a, end)))
        start = max(start, b)
    if start < end:
        out.append((start, end))
    return [(a, b) for a, b in out if b - a >= MIN_PIECE], how


def is_grey(region):
    return region[:1].isupper() and region not in ("STR",)


def signals_in(nwb, min_units=MIN_UNITS):
    """[(probe, region, n_good_units)] for grey-matter regions with units and LFP on the same probe."""
    units = nwb.units.to_dataframe()
    good = ((units.isi_violations < sd.GOOD_UNIT["isi_violations"])
            & (units.amplitude_cutoff < sd.GOOD_UNIT["amplitude_cutoff"])
            & (units.presence_ratio > sd.GOOD_UNIT["presence_ratio"]))
    counts = units[good].location.value_counts()
    lfp = {}
    for k, s in nwb.acquisition["LFP"].electrical_series.items():
        el = s.electrodes.to_dataframe()
        for reg, n in el.location[el.is_data_valid].value_counts().items():
            if n > lfp.get(reg, ("", 0))[1]:
                lfp[reg] = (k.replace("ElectricalSeries", ""), n)
    return [(lfp[r][0], r, int(n)) for r, n in counts.items() if n >= min_units and r in lfp and is_grey(r)]


# ---------------------------------------------------------------- data for one signal

@dataclass
class SignalData:
    session: str
    probe: str
    region: str
    how: str
    channel: int
    t: np.ndarray                 # 1 Hz seconds kept
    env: np.ndarray               # (seconds, bands) smoothed envelopes, not z-scored
    piece: np.ndarray             # contiguous piece id per second
    rate: np.ndarray              # (seconds, units) spike counts in 1-s bins, Gaussian-smoothed (sigma 1 s)
    unit_ids: np.ndarray
    unit_depth: np.ndarray = field(default=None)


def load_signal(path, probe, region):
    io, nwb = sd.open_nwb(path)
    intervals, how = awake_intervals(nwb)
    series = nwb.acquisition["LFP"][f"ElectricalSeries{probe}"]
    ch = sd.lfp_channel(nwb, probe, region)
    ts = series.timestamps
    i_end = int(np.searchsorted(ts, intervals[-1][1] + 1))
    t_raw = ts[:i_end]
    x_raw = series.data[:i_end, ch] * series.conversion * 1e6
    fs_raw = 1 / np.median(np.diff(t_raw[:100_000]))
    t, x, fs = sd.decimate(t_raw, x_raw, fs_raw, FS)
    del t_raw, x_raw
    hop = int(round(fs))
    T, E, P = [], [], []
    for i, (a, b) in enumerate(intervals):
        k = (t >= a) & (t < b)
        env = smooth_env(x[k], fs, BANDS, zscore=False)[::hop]
        tt = t[k][::hop]
        ok = np.isfinite(env).all(1)
        T.append(tt[ok]); E.append(env[ok]); P.append(np.full(ok.sum(), i))
    t1, env, piece = np.concatenate(T), np.concatenate(E), np.concatenate(P)

    trains, ids, _ = sd.good_units(nwb, region)
    units = nwb.units.to_dataframe().loc[ids]
    io.close()
    counts = np.stack([np.searchsorted(s, t1 + 0.5) - np.searchsorted(s, t1 - 0.5) for s in trains], 1).astype(float)
    rate = np.empty_like(counts)
    for p in np.unique(piece):
        m = piece == p
        rate[m] = gaussian_filter1d(counts[m], 1.0, axis=0, mode="nearest")
    return SignalData(Path(path).name[:15], probe, region, how, ch, t1, env, piece, rate, ids, units.y.to_numpy())


# ---------------------------------------------------------------- three pipelines

@dataclass
class Pipeline:
    name: str
    templates: np.ndarray          # (k, bands), unit length
    expr: np.ndarray               # (seconds, k)
    broadband: list
    note: str = ""


def _zscore_cols(a):
    return (a - a.mean(0)) / a.std(0, ddof=1)


def static_pipeline(d, version, seed=0):
    Z = _zscore_cols(d.env)
    pieces = [Piece(d.t[d.piece == p], Z[d.piece == p]) for p in np.unique(d.piece)]
    if VERSIONS[version]["chunked"]:
        sessions = chunk(pieces, 1.0)
        note = f"{len(sessions)} 5-min chunks"
    else:
        sessions = [Piece(np.concatenate([p.t for p in pieces]), np.concatenate([p.env for p in pieces]))]
        note = "joined"
    kw = dict(VERSIONS[version]["kw"])
    if len(sessions) == 0:
        return Pipeline(version, np.zeros((0, Z.shape[1])), np.zeros((Z.shape[0], 0)), [], "no data")
    res = get_motifs([s.env.T for s in sessions], 1.0, seed=seed, **kw)
    M = res.motifs[:, : res.n_motifs]
    M = M[:, np.isfinite(M).all(0)]
    U = (M / np.linalg.norm(M, axis=0)).T if M.size else np.zeros((0, Z.shape[1]))
    expr = np.sign(Z) @ U.T
    return Pipeline(version, U, expr, [minority_sign_bands(u) <= 1 for u in U], note)


def tracker_pipeline(d, cfg=TRACK_CFG):
    """Tracker on the awake seconds only. Baselines are computed inside each contiguous piece; detection windows
    slide along the concatenated awake seconds (stimulation blocks removed), so short pieces still contribute."""
    # every second here is awake and stimulation-free, so all of them are the z-score reference
    sig = mt.prepare_arrays("x", None, d.t, d.env, cfg, cuts=[], reference=np.ones(d.t.size, bool))
    sig_idx = mt.Signal(sig.key, None, np.arange(d.t.size, dtype=float), sig.Z, sig.base, sig.spread, sig.resid,
                        sig.segment)
    res = mt.track_prepared(sig_idx, cfg)
    U = np.array([m.template for m in res.library]) if res.library else np.zeros((0, d.env.shape[1]))
    expr = sig.resid @ U.T
    p = Pipeline("tracker", U, expr, [m.broadband for m in res.library],
                 f"{len(res.centres)} windows, {len(res.candidates)} candidates")
    p.resid = sig.resid
    p.track = res
    p.outputs, p.band_outputs = tracker_outputs(sig, res, U, cfg)
    return p


def _segment_sd(V, segment, window):
    """Rolling SD of V (n, k) inside each contiguous segment, as the tracker computes band spread."""
    out = np.full(V.shape, np.nan)
    for s in np.unique(segment):
        i = np.flatnonzero(segment == s)
        if i.size < 30 or not np.isfinite(V[i]).all():
            continue
        half = int(min(window // 2, max(15, i.size // 2)))
        out[i] = mt.rolling_sd(V[i] - V[i].mean(0), half)
    return out


def tracker_outputs(sig, res, U, cfg):
    """Every per-second output of the tracker for each library motif, and the same quantity for each single band
    (the band version is what a motif output is compared with in C3):
      expression  residual (fast part) projected on the motif           | band residual
      baseline    sliding baseline projected on the motif (slow part)   | band baseline
      spread      rolling SD of the expression (how strongly it swings) | band spread
      presence    share of the 5-min windows covering this second in which the motif was detected | (none)
      unsplit     z-scored envelopes projected on the motif, no baseline split | band z
    """
    k = U.shape[0]
    n = sig.Z.shape[0]
    E = sig.resid @ U.T
    pres = np.full((n, k), np.nan)
    if k:
        ws = res.window_stats
        cover = np.zeros(n)
        hits = np.zeros((n, k))
        for c in np.unique(ws.centre):
            a = int(c - cfg.window / 2)
            sl = slice(max(a, 0), min(a + cfg.window, n))
            cover[sl] += 1
            row = ws[ws.centre == c].set_index("motif").present
            hits[sl] += np.array([float(row.get(m, False)) for m in range(k)])
        with np.errstate(invalid="ignore"):
            pres = hits / cover[:, None]
        pres = pd.DataFrame(pres).ffill().bfill().to_numpy()     # seconds after the last window start
    outs = {"expression": E, "baseline": sig.base @ U.T, "spread": _segment_sd(E, sig.segment, cfg.baseline_window),
            "presence": pres, "unsplit": sig.Z @ U.T}
    bands = {"expression": sig.resid, "baseline": sig.base, "spread": sig.spread, "presence": None, "unsplit": sig.Z}
    return outs, bands


# ---------------------------------------------------------------- held-out Mishler tests

def folds(n):
    i = np.arange(n)
    fid = (i // BLOCK) % N_FOLDS
    inner = ((i % BLOCK) >= GAP) & ((i % BLOCK) < BLOCK - GAP)
    return fid, inner


def _corr_matrix(R, X):
    """Pearson r between every column of R (n, N) and every column of X (n, k): (N, k)."""
    Rc = (R - R.mean(0)) / (R.std(0) + 1e-12)
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-12)
    return Rc.T @ Xc / R.shape[0]


def _r(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 30 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def heldout_ensembles(X, R, fid, inner):
    """Mishler assignment on training folds, ensemble rates on held-out seconds.
    Returns A, B: (n, k) held-out + / - ensemble rates (NaN where not tested or empty), and the per-fold sizes."""
    n, k = X.shape
    A = np.full((n, k), np.nan)
    B = np.full((n, k), np.nan)
    sizes = np.zeros((N_FOLDS, k, 2), int)
    for f in range(N_FOLDS):
        tr, te = fid != f, (fid == f) & inner
        C = _corr_matrix(R[tr], X[tr])
        best = np.argmax(np.abs(C), axis=1)
        sign = np.sign(C[np.arange(C.shape[0]), best])
        for j in range(k):
            a, b = (best == j) & (sign > 0), (best == j) & (sign < 0)
            sizes[f, j] = a.sum(), b.sum()
            if a.any():
                A[te, j] = R[te][:, a].mean(1)
            if b.any():
                B[te, j] = R[te][:, b].mean(1)
    return A, B, sizes


def claim_scores(X, R, fid, inner):
    """Per motif: C1 r(expr, +ens), r(expr, -ens), r(expr, + minus -); C2 r(+ens, -ens)."""
    A, B, sizes = heldout_ensembles(X, R, fid, inner)
    out = []
    for j in range(X.shape[1]):
        x = np.where(inner, X[:, j], np.nan)
        out.append({"r_plus": _r(x, A[:, j]), "r_minus": _r(x, B[:, j]), "r_diff": _r(x, A[:, j] - B[:, j]),
                    "r_ens": _r(A[:, j], B[:, j]),
                    "n_plus": float(sizes[:, j, 0].mean()), "n_minus": float(sizes[:, j, 1].mean())})
    return out


def single_feature_diff(x, R, fid, inner):
    """Held-out r(x, + minus - ensemble) when x alone defines the ensembles (used for motifs and bands alike)."""
    A, B, _ = heldout_ensembles(x[:, None], R, fid, inner)
    return _r(np.where(inner, x, np.nan), A[:, 0] - B[:, 0])


def shift_values(n, rng, n_shift, min_shift=60):
    return rng.integers(min_shift, max(min_shift + 1, n - min_shift), n_shift)


def test_pipeline(X, broadband, R, bands, rng, n_shift=200, min_shift=60):
    """All held-out claim scores for one motif signal X (seconds, k), with circular-shift nulls (the LFP side is
    shifted). bands (seconds, n_bands): the single-band version of the same quantity for C3, or None."""
    n = R.shape[0]
    fid, inner = folds(n)
    if X.shape[1] == 0:
        return pd.DataFrame()
    real = claim_scores(X, R, fid, inner)
    band_real = [single_feature_diff(bands[:, b], R, fid, inner) for b in range(bands.shape[1])] \
        if bands is not None else [np.nan]
    null = []
    for s in shift_values(n, rng, n_shift, min_shift):
        sc = claim_scores(np.roll(X, s, axis=0), R, fid, inner)
        null.append([[r["r_plus"], r["r_minus"], r["r_diff"], r["r_ens"]] for r in sc])
    null = np.array(null)                      # (shifts, motifs, 4)
    rows = []
    has_bands = bands is not None and np.isfinite(band_real).any()
    for j, r in enumerate(real):
        nj = null[:, j]
        p_of = lambda v, col, side: float((1 + np.sum(side * nj[:, col] >= side * v)) / (n_shift + 1)) \
            if np.isfinite(v) else np.nan
        rows.append({
            "motif": f"M{j + 1}", "broadband": broadband[j], **r,
            "p_plus": p_of(r["r_plus"], 0, 1), "p_minus": p_of(r["r_minus"], 1, -1),
            "p_diff": p_of(r["r_diff"], 2, 1), "p_ens": p_of(r["r_ens"], 3, -1),
            "null95_diff": float(np.nanpercentile(nj[:, 2], 95)), "null_mean_ens": float(np.nanmean(nj[:, 3])),
            "best_band_r_diff": float(np.nanmax(band_real)) if has_bands else np.nan,
            "best_band": int(np.nanargmax(band_real)) if has_bands else -1,
            "beats_best_band": bool(r["r_diff"] > np.nanmax(band_real)) if has_bands else np.nan,
            "frac_bands_beaten": float(np.nanmean(r["r_diff"] > np.array(band_real)))
            if has_bands and np.isfinite(r["r_diff"]) else np.nan,
            "band_r_diff": list(band_real), "min_shift": int(min_shift),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- B: which method's outputs explain spiking best

def _usable(F):
    if F is None or F.size == 0:
        return None
    keep = np.isfinite(F).all(0) & (np.nanstd(F, 0) > 1e-9)
    return F[:, keep] if keep.any() else None


def heldout_r2(F, Y, fid, inner, alpha=10.0):
    """Held-out R^2 of a ridge regression from F to every column of Y (pooled over folds): (n_targets,)."""
    pred = np.full(Y.shape, np.nan)
    for f in range(N_FOLDS):
        tr, te = fid != f, (fid == f) & inner
        mu, sdv = F[tr].mean(0), F[tr].std(0) + 1e-12
        A = (F[tr] - mu) / sdv
        ym = Y[tr].mean(0)
        W = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (Y[tr] - ym))
        pred[te] = ((F[te] - mu) / sdv) @ W + ym
    ok = np.isfinite(pred[:, 0])
    res = ((Y[ok] - pred[ok]) ** 2).sum(0)
    tot = ((Y[ok] - Y[ok].mean(0)) ** 2).sum(0)
    return 1 - res / tot


def info_compare(featuresets, R, rng, n_shift=100, min_shift=300):
    """For each feature set: held-out R^2 for every neuron and for the population rate, against a null in which the
    features are circularly shifted (same shifts for every set). A set with more or slower features fits shifted data
    better too, so the comparison is real minus null."""
    n = R.shape[0]
    fid, inner = folds(n)
    Z = (R - R.mean(0)) / (R.std(0) + 1e-12)
    Y = np.column_stack([Z, Z.mean(1)])
    shifts = shift_values(n, rng, n_shift, min_shift)
    rows = []
    for name, F in featuresets.items():
        F = _usable(F)
        if F is None:
            continue
        real = heldout_r2(F, Y, fid, inner)
        null = np.array([heldout_r2(np.roll(F, s, axis=0), Y, fid, inner) for s in shifts])
        neu_p = (1 + (null[:, :-1] >= real[:-1]).sum(0)) / (n_shift + 1)
        rows.append({"features": name, "n_features": F.shape[1],
                     "neuron_r2": float(real[:-1].mean()), "neuron_null": float(null[:, :-1].mean()),
                     "neuron_gain": float(real[:-1].mean() - null[:, :-1].mean()),
                     "neurons_beating_null": float((neu_p < 0.05).mean()),
                     "pop_r2": float(real[-1]), "pop_null95": float(np.percentile(null[:, -1], 95)),
                     "pop_gain": float(real[-1] - null[:, -1].mean()),
                     "pop_p": float((1 + (null[:, -1] >= real[-1]).sum()) / (n_shift + 1))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- C4: from spiking alone

def lfp_assignment(X, R):
    """Assignment on all seconds: (motif index, sign) per neuron."""
    C = _corr_matrix(R, X)
    best = np.argmax(np.abs(C), axis=1)
    return best, np.sign(C[np.arange(C.shape[0]), best])


def pca_agreement(X, R, rng, n_perm=1000):
    """Per motif, neurons assigned to it by the LFP: PCA on their z-scored rates; split by the sign of their PC1
    loadings; agreement = share of neurons whose split matches their +/- assignment (best of the two labelings).
    Null: the same with the +/- labels permuted among those neurons."""
    best, sign = lfp_assignment(X, R)
    Z = (R - R.mean(0)) / (R.std(0) + 1e-12)
    out = []
    for j in range(X.shape[1]):
        m = best == j
        if m.sum() < 6 or (sign[m] > 0).sum() < 2 or (sign[m] < 0).sum() < 2:
            out.append({"n": int(m.sum()), "agreement": np.nan, "p": np.nan, "null_mean": np.nan})
            continue
        _, _, Vt = np.linalg.svd(Z[:, m] - Z[:, m].mean(0), full_matrices=False)
        split = np.sign(Vt[0])
        lab = sign[m]
        agree = lambda L: max(np.mean(split == L), np.mean(split == -L))
        a = agree(lab)
        null = np.array([agree(rng.permutation(lab)) for _ in range(n_perm)])
        out.append({"n": int(m.sum()), "agreement": float(a), "p": float((1 + (null >= a).sum()) / (n_perm + 1)),
                    "null_mean": float(null.mean())})
    return out


def getclust_vs_lfp(R, X, rng, seeds=range(5), n_shift=200, min_shift=60):
    """getClust on spiking alone (Mishler preprocessing). For each cluster: run-to-run stability, and how its
    activity (group support) correlates with each motif's expression, with a circular-shift null."""
    Xp = preprocess(R)
    runs = [get_clust(Xp, 1.0, seed=s) for s in seeds]
    ref = runs[0]
    rows = []
    for ci, c in enumerate(ref):
        cs = set(c.tolist())
        stab = np.mean([max((len(cs & set(d.tolist())) / len(cs | set(d.tolist())) for d in r), default=0)
                        for r in runs[1:]])
        sup = group_support(Xp[:, c])
        for j in range(X.shape[1]):
            r = _r(sup, X[:, j])
            null = np.array([_r(sup, np.roll(X[:, j], s)) for s in shift_values(sup.size, rng, n_shift, min_shift)])
            rows.append({"cluster": ci + 1, "size": len(c), "stability": float(stab), "motif": f"M{j + 1}",
                         "r": r, "p": float((1 + np.sum(np.abs(null) >= abs(r))) / (n_shift + 1))})
    return ref, pd.DataFrame(rows)


# ---------------------------------------------------------------- one signal end to end

TRACKER_OUTPUTS = ("expression", "baseline", "spread", "presence", "unsplit")


def run_signal(path, probe, region, n_shift=200, seed=0):
    """Claims C1-C4 for the original and relaxed motifs and for every tracker output, plus the information
    comparison (B). One min_shift for every method of a signal: 300 s, or a third of the data if shorter."""
    rng = np.random.default_rng(seed)
    d = load_signal(path, probe, region)
    static = {"original": static_pipeline(d, "original"), "relaxed": static_pipeline(d, "relaxed")}
    tr = tracker_pipeline(d)
    ok = np.isfinite(tr.resid).all(1)          # common seconds for every method
    n = int(ok.sum())
    min_shift = int(min(300, n // 3))
    R = d.rate[ok]
    R_glob = remove_global(R)
    bandsZ = _zscore_cols(d.env)[ok]
    methods = {name: (p.expr[ok], p.broadband, bandsZ) for name, p in static.items()}
    for o in TRACKER_OUTPUTS:
        bo = tr.band_outputs[o]
        methods[f"tracker {o}"] = (tr.outputs[o][ok], tr.broadband, None if bo is None else bo[ok])
    tables, c4 = [], []
    for name, (X, bb, bands) in methods.items():
        if X.shape[1] == 0:
            continue
        for rate_kind, RR in (("raw", R), ("population removed", R_glob)):
            tb = test_pipeline(X, bb, RR, bands, rng, n_shift, min_shift)
            if len(tb):
                tables.append(tb.assign(method=name, rates=rate_kind))
        if np.isfinite(X).all() and (X.std(0) > 1e-9).all():
            for j, row in enumerate(pca_agreement(X, R, rng)):
                c4.append({"method": name, "motif": f"M{j + 1}", **row})
    cols = [(name, j) for name, (X, _, _) in methods.items() for j in range(X.shape[1])]
    Xall = np.column_stack([methods[nm][0][:, j] for nm, j in cols]) if cols else np.zeros((n, 0))
    clusters, gc = getclust_vs_lfp(R, Xall, rng, min_shift=min_shift)
    if len(gc):
        idx = gc.motif.str[1:].astype(int) - 1
        gc["method"] = [cols[i][0] for i in idx]
        gc["motif"] = [f"M{cols[i][1] + 1}" for i in idx]

    E = tr.outputs
    fs = {"original": methods["original"][0], "relaxed": methods["relaxed"][0],
          "tracker expression": E["expression"][ok],
          "tracker all outputs": np.column_stack([E[o][ok] for o in ("expression", "baseline", "spread", "presence")])
          if tr.expr.shape[1] else None,
          "10 bands": bandsZ,
          "10 bands, split (fast + slow)": np.column_stack([tr.band_outputs["expression"][ok],
                                                            tr.band_outputs["baseline"][ok]])}
    info = info_compare(fs, R, rng, n_shift=100, min_shift=min_shift)

    meta = {"session": d.session, "probe": d.probe, "region": d.region, "how": d.how, "channel": d.channel,
            "n_units": R.shape[1], "n_seconds": n, "n_pieces": int(np.unique(d.piece[ok]).size), "min_shift": min_shift,
            "n_motifs_original": static["original"].expr.shape[1], "n_motifs_relaxed": static["relaxed"].expr.shape[1],
            "n_motifs_tracker": tr.expr.shape[1], **{f"note_{k}": p.note for k, p in static.items()},
            "note_tracker": tr.note, "n_getclust": len(clusters)}
    key = {"session": d.session, "region": d.region}
    return {"meta": meta,
            "claims": pd.concat(tables, ignore_index=True).assign(**key) if tables else pd.DataFrame(),
            "pca": pd.DataFrame(c4).assign(**key), "getclust": gc.assign(**key) if len(gc) else gc,
            "info": info.assign(**key),
            "templates": {"original": static["original"].templates, "relaxed": static["relaxed"].templates,
                          "tracker": tr.templates},
            "outputs": {name: X for name, (X, _, _) in methods.items()}, "t": d.t[ok], "rate": R,
            "unit_ids": d.unit_ids, "clusters": clusters, "bandsZ": bandsZ}


def remove_global(R):
    """z-score each neuron, then regress the population mean out of it (removes shared gain changes)."""
    Z = (R - R.mean(0)) / (R.std(0) + 1e-12)
    pop = Z.mean(1) - Z.mean(1).mean()
    beta = (Z * pop[:, None]).sum(0) / (pop @ pop + 1e-12)
    return Z - np.outer(pop, beta)


def save(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)
