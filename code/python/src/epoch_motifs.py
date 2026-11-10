"""Epoch-wise motif union with Gaussian-mixture expression levels along each motif (one channel).

Steps
  1. Envelopes over the whole recording, z-scored once globally (so positions along an axis are
     comparable between epochs). Detection itself uses each epoch's own z-scoring, as before.
  2. Run the (relaxed) Mishler detection in every epoch (spontaneous awake, awake + electrical,
     awake + visual, early isoflurane, isoflurane + electrical, isoflurane + visual, late isoflurane).
  3. Union of motif directions: an axis joins an existing motif if |cosine| >= cos_merge (sign does
     not matter, an axis and its negative are the same direction); templates are sign-aligned means.
  4. For every epoch and motif: project the epoch's seconds onto the motif axis and fit a 1-D GMM with
     3 components started at -init_scale, 0, +init_scale.
  5. Per motif, pool the epoch Gaussians (ignoring those with weight < min_weight) and merge those
     closer than merge_d (separation d' = |mu1 - mu2| / sqrt((sd1^2 + sd2^2) / 2), complete linkage).
     The number of groups is k; their weighted mean positions start the final fit.
  6. Per motif, a k-component 1-D GMM over the whole recording. Each second gets a responsibility
     for every component of every motif.
  7. A "zero" (baseline) motif with no poles: either RMS of the z-scored bands below quiet_rms
     ("rms"), or the product over motifs of the responsibility of each motif's component nearest
     zero ("nearzero").
"""

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score
from sklearn.mixture import GaussianMixture

from . import candidate_motifs as cm
from . import gmm_motifs as gx
from .state_motifs import minority_sign_bands

EPOCHS = cm.FINE_BINS


@dataclass
class Params:
    significance: str = "relaxed"
    cos_merge: float = 0.9
    init_scale: float = 1.0
    merge_d: float = 1.0
    min_weight: float = 0.05
    baseline: str = "rms"
    quiet_rms: float = 0.5
    epochs: tuple = EPOCHS


@dataclass
class Data:
    info: object
    t: np.ndarray            # seconds of the whole recording (1 Hz)
    env: np.ndarray          # envelopes, not z-scored
    Z: np.ndarray            # globally z-scored envelopes
    epoch: np.ndarray        # epoch label per second ("" outside every epoch)
    pieces: dict             # epoch -> [(times, epoch-z-scored envelopes)] for detection


@dataclass
class Motif:
    template: np.ndarray     # unit vector
    members: list            # [(epoch, p)]

    @property
    def epochs(self):
        return sorted({e for e, _ in self.members}, key=EPOCHS.index)

    @property
    def p(self):
        return min(p for _, p in self.members)

    @property
    def broadband(self):
        return minority_sign_bands(self.template) <= 1


@dataclass
class Result:
    params: Params
    motifs: list
    epoch_gauss: pd.DataFrame      # one row per (motif, epoch, component)
    final: list                    # per motif: fitted GaussianMixture with components sorted by mean
    comps: pd.DataFrame            # one row per final component
    R: np.ndarray                  # seconds x components (all motifs), responsibilities
    baseline: np.ndarray           # seconds, baseline expression (0..1)
    scores: np.ndarray             # seconds x motifs, projection onto each motif axis


def load(nwb_path, source=("lfp", "probeF", "PL5"), epochs=EPOCHS):
    info, t, env = cm.signal_envelope(nwb_path, source)
    Z = (env - env.mean(axis=0)) / env.std(axis=0, ddof=1)
    bins = cm.time_bins(info)
    lab = np.full(t.size, "", dtype=object)
    pieces = {}
    for e in epochs:
        for a, b in bins[e]:
            lab[(t >= a) & (t < b)] = e
        pieces[e] = cm.bin_pieces(t, env, bins[e])
    return Data(info, t, env, Z, lab, pieces)


def detect_epochs(data, significance="relaxed", epochs=EPOCHS):
    """Every motif axis found in every epoch: [(epoch, unit axis, p)]."""
    out = []
    for e in epochs:
        if not data.pieces.get(e):
            continue
        axes, ps = cm.detect(data.pieces[e], significance, True)
        for i in range(axes.shape[1]):
            out.append((e, axes[:, i] / np.linalg.norm(axes[:, i]), ps[i]))
    return out


def union_axes(dets, cos_merge):
    """Greedy union in order of p: join the first kept motif with |cos| >= cos_merge."""
    motifs, sums = [], []
    for e, u, p in sorted(dets, key=lambda d: d[2]):
        if motifs:
            cos = np.array([s @ u / np.linalg.norm(s) for s in sums])
            j = int(np.argmax(np.abs(cos)))
            if abs(cos[j]) >= cos_merge:
                sums[j] = sums[j] + np.sign(cos[j]) * u
                motifs[j].members.append((e, p))
                continue
        sums.append(u.copy())
        motifs.append(Motif(u.copy(), [(e, p)]))
    for m, s in zip(motifs, sums):
        m.template = s / np.linalg.norm(s)
    return motifs


def fit1d(x, means_init=None, k=None, seed=0):
    k = k or len(means_init)
    kw = {"means_init": np.asarray(means_init, float)[:, None]} if means_init is not None else {"n_init": 3}
    return GaussianMixture(k, random_state=seed, **kw).fit(x[:, None])


def _sorted_params(gm):
    o = np.argsort(gm.means_.ravel())
    return gm.means_.ravel()[o], np.sqrt(gm.covariances_.ravel()[o]), gm.weights_[o]


def epoch_gaussians(data, motifs, params, mask=None):
    rows = []
    s0 = params.init_scale
    for mi, m in enumerate(motifs):
        score = data.Z @ m.template
        for e in params.epochs:
            sel = data.epoch == e
            if mask is not None:
                sel &= mask
            if sel.sum() < 60:
                continue
            mu, sd, w = _sorted_params(fit1d(score[sel], [-s0, 0.0, s0]))
            rows += [{"motif": mi, "epoch": e, "comp": j, "mean": mu[j], "sd": sd[j], "weight": w[j],
                      "n_seconds": int(sel.sum())} for j in range(3)]
    return pd.DataFrame(rows)


def dprime(a, b):
    return abs(a[0] - b[0]) / np.sqrt((a[1] ** 2 + b[1] ** 2) / 2)


def merge_gaussians(g, merge_d, min_weight):
    """Group one motif's epoch Gaussians. Returns (group label per row or -1 if ignored, group means)."""
    keep = (g["weight"] >= min_weight).to_numpy()
    lab = np.full(len(g), -1)
    P = g.loc[keep, ["mean", "sd"]].to_numpy()
    if len(P) == 0:
        return lab, np.array([0.0])
    if len(P) == 1:
        lab[keep] = 0
        return lab, P[:, 0]
    from scipy.spatial.distance import pdist
    Zl = linkage(pdist(P, dprime), "complete")
    groups = fcluster(Zl, merge_d, "distance") - 1
    # weight each Gaussian by its share of seconds overall
    w = (g.loc[keep, "weight"] * g.loc[keep, "n_seconds"]).to_numpy()
    means = np.array([np.average(P[groups == k, 0], weights=w[groups == k]) for k in range(groups.max() + 1)])
    order = np.argsort(means)
    remap = np.empty_like(order); remap[order] = np.arange(order.size)
    lab[keep] = remap[groups]
    return lab, means[order]


def run(data, dets, params):
    motifs = union_axes(dets, params.cos_merge)
    eg = epoch_gaussians(data, motifs, params)
    eg["group"] = -1
    final, comps, Rs, scores = [], [], [], []
    for mi, m in enumerate(motifs):
        idx = np.flatnonzero(eg["motif"] == mi)
        lab, init = merge_gaussians(eg.iloc[idx], params.merge_d, params.min_weight)
        eg.loc[eg.index[idx], "group"] = lab
        score = data.Z @ m.template
        scores.append(score)
        gm = fit1d(score, init)
        mu, sd, w = _sorted_params(gm)
        o = np.argsort(gm.means_.ravel())
        R = gm.predict_proba(score[:, None])[:, o]
        final.append(gm)
        Rs.append(R)
        for j in range(len(mu)):
            comps.append({"motif": mi, "comp": j, "label": f"M{mi + 1}:{mu[j]:+.1f}", "mean": mu[j], "sd": sd[j],
                          "weight": w[j], "broadband": m.broadband, "found_in": ",".join(m.epochs)})
    R = np.hstack(Rs) if Rs else np.zeros((data.t.size, 0))
    comps = pd.DataFrame(comps)
    if params.baseline == "rms" or not len(motifs):
        base = (np.linalg.norm(data.Z, axis=1) / np.sqrt(data.Z.shape[1]) < params.quiet_rms).astype(float)
    else:
        base = np.ones(data.t.size)
        for mi in range(len(motifs)):
            cols = np.flatnonzero(comps["motif"] == mi)
            near = cols[np.argmin(np.abs(comps.loc[cols, "mean"].to_numpy()))]
            base *= R[:, near]
    return Result(params, motifs, eg, final, comps, R, base, np.array(scores).T if scores else np.zeros((data.t.size, 0)))


# ---------- evaluation ----------

def epoch_nmi(res, data):
    """Per motif: normalized mutual information between hard component labels and epoch labels."""
    sel = data.epoch != ""
    out = []
    for mi in range(len(res.motifs)):
        cols = np.flatnonzero(res.comps["motif"] == mi)
        if len(cols) < 2:
            out.append(np.nan)
            continue
        hard = res.R[:, cols].argmax(axis=1)
        out.append(normalized_mutual_info_score(data.epoch[sel], hard[sel]))
    return np.array(out)


def occupancy(res, data):
    """Mean responsibility of every component (rows) in every epoch (columns), plus baseline."""
    cols = {e: res.R[data.epoch == e].mean(axis=0) for e in res.params.epochs if (data.epoch == e).any()}
    df = pd.DataFrame(cols, index=res.comps["label"])
    df.loc["baseline"] = [res.baseline[data.epoch == e].mean() for e in df.columns]
    return df


def split_half(data, motifs, params, block=300):
    """Re-run steps 4-5 on alternating 5-min blocks; per motif, k in each half and matched |mean| gap."""
    t0 = data.t - data.t[0]
    even = (t0 // block) % 2 == 0
    out = []
    halves = [epoch_gaussians(data, motifs, params, mask) for mask in (even, ~even)]
    for mi in range(len(motifs)):
        ms = [merge_gaussians(h[h["motif"] == mi], params.merge_d, params.min_weight)[1] for h in halves]
        k1, k2 = len(ms[0]), len(ms[1])
        if k1 == k2:
            gap = float(np.max(np.abs(ms[0] - ms[1])))
        else:
            C = np.abs(ms[0][:, None] - ms[1][None, :])
            r, c = linear_sum_assignment(C)
            gap = float(C[r, c].max())
        out.append({"motif": mi, "k_half1": k1, "k_half2": k2, "same_k": k1 == k2, "max_mean_gap": gap})
    return pd.DataFrame(out)


def cv_gain(score, k, init=None, n_folds=5, gap=10):
    """Held-out log-likelihood per second of a k-component fit minus a 1-component fit."""
    fid = np.minimum(np.arange(score.size) * n_folds // score.size, n_folds - 1)
    out = 0.0
    for f in range(n_folds):
        idx = np.flatnonzero(fid == f)
        test, train = idx[gap:-gap], fid != f
        gk = fit1d(score[train], init, k) if init is not None else fit1d(score[train], k=k)
        g1 = fit1d(score[train], k=1)
        out += (gk.score(score[test][:, None]) - g1.score(score[test][:, None])) / n_folds
    return out


def sweep(data, dets_by_sig, grid, with_stability=False):
    """One row of summary metrics per parameter combination."""
    rows = []
    for p in grid:
        res = run(data, dets_by_sig[p.significance], p)
        ks = res.comps.groupby("motif").size() if len(res.comps) else pd.Series(dtype=int)
        nmi = epoch_nmi(res, data)
        row = {**p.__dict__, "n_motifs": len(res.motifs),
               "n_not_broadband": sum(not m.broadband for m in res.motifs),
               "total_components": int(ks.sum()), "mean_k": float(ks.mean()) if len(ks) else np.nan,
               "max_k": int(ks.max()) if len(ks) else 0,
               "motifs_k_ge_2": int((ks >= 2).sum()),
               "mean_nmi_with_epochs": float(np.nanmean(nmi)) if np.isfinite(nmi).any() else np.nan,
               "baseline_share": float(res.baseline.mean())}
        if with_stability and res.motifs:
            st = split_half(data, res.motifs, p)
            row["share_same_k_split_half"] = float(st["same_k"].mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.drop(columns=["epochs"])


# ---------- measurement: state and expression kept apart ----------
#
# Expression is measured in each epoch's own units, (x - mu_e) / sd_e, which is Mishler's frame: the
# same sign template means the same motif in every epoch. The state is described separately: each
# epoch's baseline shift from the global baseline (global z units), split into the part along each
# motif, inside all motifs together, and perpendicular to all of them.

KS = (1, 2, 3)


@dataclass
class Measurement:
    motifs: list
    epochs: list
    mu: dict                 # epoch -> band means (envelope units)
    sd: dict                 # epoch -> band SDs
    shift: pd.DataFrame      # per epoch: baseline shift size and its split
    kfit: pd.DataFrame       # per (motif, epoch): chosen k, peaks, gains, Gaussian parameters
    second: pd.DataFrame     # per second: t, epoch, rms deviation, zero flag, suppression flag, expression per motif
    resp: dict               # (motif index, epoch) -> (seconds in epoch, k) responsibilities, sorted by mean


def _cv_gains(pieces, n_folds=5, gap=10, ks=KS):
    """Held-out log-likelihood per second for each k, relative to k = 1 (contiguous folds)."""
    s = np.concatenate(pieces)[:, None]
    fid = gx.folds(pieces, n_folds)
    out = np.zeros(len(ks))
    for f in range(n_folds):
        idx = np.flatnonzero(fid == f)
        test = idx[gap:-gap] if idx.size > 3 * gap else idx
        for i, k in enumerate(ks):
            gm = GaussianMixture(k, n_init=3, random_state=0).fit(s[fid != f])
            out[i] += gm.score(s[test]) / n_folds
    return out - out[0]


def n_peaks(gm, x):
    xs = np.linspace(np.percentile(x, 0.5), np.percentile(x, 99.5), 800)
    p = np.exp(gm.score_samples(xs[:, None]))
    return int(np.sum((p[1:-1] > p[:-2]) & (p[1:-1] > p[2:])))


def choose_k(real_pieces, surrogate_pieces, min_step=0.005, ks=KS):
    """Smallest-first: accept k if its held-out gain beats every surrogate's gain at k and adds at least
    min_step over k - 1. Returns (k, gains, max surrogate gains)."""
    g = _cv_gains(real_pieces, ks=ks)
    gs = np.array([_cv_gains(sp, ks=ks) for sp in surrogate_pieces])
    k = ks[0]
    for i in range(1, len(ks)):
        if g[i] > gs[:, i].max() and g[i] - g[i - 1] > min_step:
            k = ks[i]
        else:
            break
    return k, g, gs.max(axis=0)


def baseline_shift(data, motifs, epochs=EPOCHS):
    """Each epoch's mean minus the pooled mean of all epoch seconds (global z), and its split."""
    sel = data.epoch != ""
    g = data.Z[sel].mean(axis=0)
    U = np.array([m.template for m in motifs])
    Q, _ = np.linalg.qr(U.T)
    rows = []
    for e in epochs:
        if not (data.epoch == e).any():
            continue
        d = data.Z[data.epoch == e].mean(axis=0) - g
        n2 = float(d @ d)
        span = float(np.sum((Q.T @ d) ** 2) / n2)
        rows.append({"epoch": e, "shift size": np.sqrt(n2),
                     **{f"M{i + 1} offset": float(u @ d) for i, u in enumerate(U)},
                     **{f"M{i + 1} share": float((u @ d) ** 2 / n2) for i, u in enumerate(U)},
                     "share in motif span": span, "share perpendicular": 1 - span})
    return pd.DataFrame(rows).set_index("epoch")


def suppression(data, z_by_second, epoch_mask_iso, level=-1.0, max_up=1):
    """Heuristic: isoflurane seconds where the bands average below `level` (own-epoch z) and at most
    `max_up` bands are above the epoch mean: every band low at once."""
    mean = np.nanmean(z_by_second, axis=1)
    up = np.sum(z_by_second > 0, axis=1)
    return epoch_mask_iso & (mean < level) & (up <= max_up)


def detrend(z, window):
    """Subtract a centered rolling mean (per band) of `window` seconds from one stretch."""
    if not window:
        return z
    m = pd.DataFrame(z).rolling(window, center=True, min_periods=max(10, window // 4)).mean().to_numpy()
    return z - m


def measure(data, motifs, epochs=EPOCHS, n_sur=10, quiet_rms=0.5, seed=0, baseline_window=None):
    """Expression, k per (motif, epoch), responsibilities, flags. With baseline_window (seconds), each
    stretch also has a sliding baseline of that length removed, so slow drift within an epoch counts as
    state rather than expression."""
    rng = np.random.default_rng(seed)
    Zown = np.full_like(data.env, np.nan)
    mu, sd, own_pieces = {}, {}, {}
    for e in epochs:
        sel = data.epoch == e
        if sel.sum() < 60 or not data.pieces.get(e):
            continue
        mu[e], sd[e] = data.env[sel].mean(axis=0), data.env[sel].std(axis=0, ddof=1)
        own_pieces[e] = []
        for tt, z in data.pieces[e]:            # same own-epoch z-scoring, by contiguous stretch
            z = detrend(z, baseline_window)
            Zown[np.searchsorted(data.t, tt)] = z
            own_pieces[e].append(z)
    eps = list(mu)
    U = np.array([m.template for m in motifs])
    E = Zown @ U.T                                   # expression, seconds x motifs (NaN outside epochs)
    rms = np.sqrt(np.nanmean(Zown ** 2, axis=1))
    sec = pd.DataFrame({"t": data.t, "epoch": data.epoch, "rms_deviation": rms, "zero": rms < quiet_rms})
    iso = np.isin(data.epoch, [e for e in eps if e.startswith("iso")])
    sec["suppression"] = suppression(data, Zown, iso)
    for i in range(len(motifs)):
        sec[f"M{i + 1}"] = E[:, i]

    rows, resp = [], {}
    for e in eps:
        zp = own_pieces[e]
        surs = [gx.phase_surrogate(zp, rng) for _ in range(n_sur)]
        sel = np.flatnonzero((data.epoch == e) & np.isfinite(E[:, 0]))
        for mi, m in enumerate(motifs):
            k, g, gmax = choose_k([z @ m.template for z in zp], [[z @ m.template for z in s] for s in surs])
            x = E[sel, mi]
            gm = GaussianMixture(k, n_init=3, random_state=0).fit(x[:, None])
            o = np.argsort(gm.means_.ravel())
            resp[(mi, e)] = gm.predict_proba(x[:, None])[:, o]
            rows.append({"motif": mi, "epoch": e, "k": k, "peaks": n_peaks(gm, x),
                         "gain k2": g[1], "gain k3": g[2], "surrogate max k2": gmax[1], "surrogate max k3": gmax[2],
                         "means": gm.means_.ravel()[o].round(3).tolist(),
                         "sds": np.sqrt(gm.covariances_.ravel()[o]).round(3).tolist(),
                         "weights": gm.weights_[o].round(3).tolist(), "n_seconds": sel.size})
    return Measurement(motifs, eps, mu, sd, baseline_shift(data, motifs, eps), pd.DataFrame(rows), sec, resp)


# ---------- tracking over time ----------

def rolling(x, t, window):
    """Centered rolling mean and SD (seconds) on the 1 Hz grid, ignoring NaN."""
    s = pd.Series(x, index=np.round(t).astype(int))
    s = s[~s.index.duplicated()].reindex(np.arange(s.index.min(), s.index.max() + 1))
    r = s.rolling(window, center=True, min_periods=window // 2)
    return s.index.to_numpy(), r.mean().to_numpy(), r.std().to_numpy()


def direction_over_time(data, motifs, window=300, step=30):
    """In sliding windows, z-score every band within the window and compute, for each motif, the
    variance along it (u' C u; 1 = what a random direction gets, top eigenvalue = the most any
    direction gets). Returns window centres, enrichment (windows x motifs) and the top eigenvalue."""
    t = data.t
    centres, enr, top = [], [], []
    U = np.array([m.template for m in motifs])
    for c in np.arange(t[0] + window / 2, t[-1] - window / 2, step):
        sel = np.abs(t - c) <= window / 2
        if sel.sum() < window * 0.8:
            continue
        X = data.env[sel]
        C = np.corrcoef(X.T)
        centres.append(c)
        enr.append(np.einsum("ij,jk,ik->i", U, C, U))
        top.append(np.linalg.eigvalsh(C)[-1])
    return np.array(centres), np.array(enr), np.array(top)


# ---------- final pipeline: sliding baseline ----------
#
# Each second is split into a slow baseline (centered rolling mean, `baseline_window` seconds, computed
# within each contiguous stretch of an epoch so it never mixes epochs) and a fast residual around it.
#   - Detection (Mishler, per epoch) runs on the residual, so the library holds fast motifs, not drift.
#   - Expression = residual projected on each motif, in residual-SD units of that epoch (Mishler's frame).
#   - Baseline outputs: the rolling baseline in global z units, per band and projected on every motif,
#     which carries the slow state (dose, stimulation, arousal).
#   - k per (motif, epoch) from 1-3 Gaussians on the expression, surrogate rule; peaks counted.

@dataclass
class Config:
    significance: str = "relaxed"
    cos_merge: float = 0.9
    baseline_window: int = 300
    n_sur: int = 10
    quiet_rms: float = 0.5
    epochs: tuple = EPOCHS
    seed: int = 0


@dataclass
class PipelineResult:
    config: Config
    library: list                # Motif objects
    second: pd.DataFrame         # t, epoch, rms_residual, zero, suppression, M{i} expression, base_M{i}
    baseline_bands: np.ndarray   # seconds x bands, rolling baseline in global z (NaN outside epochs)
    kfit: pd.DataFrame           # per (motif, epoch): k, peaks, gains, Gaussian parameters
    resp: dict                   # (motif index, epoch) -> responsibilities, sorted by Gaussian mean
    shift: pd.DataFrame          # per epoch: mean baseline shift from the pooled baseline and its split
    residual_sd: dict            # epoch -> residual SD per band (own-epoch z units)


def _rolling_mean(z, window):
    return pd.DataFrame(z).rolling(window, center=True, min_periods=max(10, window // 4)).mean().to_numpy()


def pipeline(data, cfg=Config()):
    rng = np.random.default_rng(cfg.seed)
    n, nb = data.env.shape
    own = np.full((n, nb), np.nan)        # own-epoch z (for the suppression label)
    resid = np.full((n, nb), np.nan)      # residual around the sliding baseline, residual-SD units
    base_g = np.full((n, nb), np.nan)     # sliding baseline, global z units
    res_pieces, rsd = {}, {}
    for e in cfg.epochs:
        pieces = data.pieces.get(e) or []
        if not pieces:
            continue
        raw = []
        for tt, z in pieces:
            idx = np.searchsorted(data.t, tt)
            r = z - _rolling_mean(z, cfg.baseline_window)
            own[idx] = z
            base_g[idx] = _rolling_mean(data.Z[idx], cfg.baseline_window)
            raw.append((idx, r))
        rsd[e] = np.concatenate([r for _, r in raw]).std(axis=0, ddof=1)
        res_pieces[e] = []
        for (idx, r), (tt, _) in zip(raw, pieces):
            r = r / rsd[e]
            resid[idx] = r
            res_pieces[e].append((tt, r))

    dets = []
    for e, pieces in res_pieces.items():
        axes, ps = cm.detect(pieces, cfg.significance, True)
        for i in range(axes.shape[1]):
            dets.append((e, axes[:, i] / np.linalg.norm(axes[:, i]), ps[i]))
    library = union_axes(dets, cfg.cos_merge)
    U = np.array([m.template for m in library]) if library else np.zeros((0, nb))

    E = resid @ U.T
    rms = np.sqrt(np.nanmean(resid ** 2, axis=1))
    sec = pd.DataFrame({"t": data.t, "epoch": data.epoch, "rms_residual": rms, "zero": rms < cfg.quiet_rms})
    iso = np.isin(data.epoch, [e for e in res_pieces if e.startswith("iso")])
    sec["suppression"] = suppression(data, own, iso)
    for i in range(len(library)):
        sec[f"M{i + 1}"] = E[:, i]
        sec[f"base_M{i + 1}"] = base_g @ U[i]

    rows, resp = [], {}
    for e, pieces in res_pieces.items():
        zp = [r for _, r in pieces]
        surs = [gx.phase_surrogate(zp, rng) for _ in range(cfg.n_sur)]
        sel = np.flatnonzero((data.epoch == e) & np.isfinite(rms))
        for mi, m in enumerate(library):
            k, g, gmax = choose_k([z @ m.template for z in zp], [[z @ m.template for z in s] for s in surs])
            x = E[sel, mi]
            gm = GaussianMixture(k, n_init=3, random_state=0).fit(x[:, None])
            o = np.argsort(gm.means_.ravel())
            resp[(mi, e)] = gm.predict_proba(x[:, None])[:, o]
            rows.append({"motif": mi, "epoch": e, "k": k, "peaks": n_peaks(gm, x),
                         "gain k2": g[1], "gain k3": g[2], "surrogate max k2": gmax[1], "surrogate max k3": gmax[2],
                         "means": gm.means_.ravel()[o].round(3).tolist(),
                         "sds": np.sqrt(gm.covariances_.ravel()[o]).round(3).tolist(),
                         "weights": gm.weights_[o].round(3).tolist(), "n_seconds": sel.size})
    shift = baseline_shift(data, library, list(res_pieces)) if library else pd.DataFrame()
    return PipelineResult(cfg, library, sec, base_g, pd.DataFrame(rows), resp, shift, rsd)
