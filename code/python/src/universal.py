"""Universal motifs: do motifs learned in one mouse / region / modality translate to others, and are they grounded in
spiking? (U1-U5; awake references; stimulation excluded; locally re-referenced channels: CSD on probes, common average
on EEG.)

Inputs
  data/processed/multichannel_local/*.npz            per-channel band envelopes (every ~40 um on probes, every EEG electrode)
  data/processed/channel_tracker_local/_coarse/*.pkl per-channel awake motif libraries (every 4th probe channel, every
                                                      EEG electrode)
Per channel: awake-referenced z, sliding baseline, residual (channel_tracker.signal_of); the residual covariance in
awake and in isoflurane seconds summarises everything U1, U2 and U4 need (variance a direction u captures = u'Cu).

U1  dictionary      sign-free k-lines clustering of the pooled awake templates (each mouse weighted equally);
                    k chosen on the discovery mice by leave-one-mouse-out reconstruction of held-out channels' residual
                    variance, against (a) random smooth directions matched to the templates' smoothness, (b) the pooled
                    band PCA of the training mice; universality = the motif reappears in every fold and in an
                    independent fit on the confirmation mice
U2  transfer        for every source channel s (with a library) and target channel c: share of c's top direction
                    variance captured by s's best template, u'C_c u / lambda1(C_c); decomposed by same mouse, same
                    region class / area, modality (LFP / EEG), atlas distance, state
U3  grounding       per unit, nearest local channel on its own probe: coupling of each universal motif's expression with
                    the unit's rate (population mean removed), against circular shifts; unique coupling after removing
                    spectral tilt, the array's band PC1 and named bands (partial r); signed relationships (motif ~ raw
                    local firing; narrow- vs broad-spiking)
U4  states          awake dictionary on isoflurane residual covariances; isoflurane principal directions clustered like
                    U1 and matched to the awake dictionary; U3 signs under isoflurane
U5  EEG -> beneath  coupling of EEG motif expression with units within 1 mm horizontally vs >= 3 mm
"""
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment

from . import awake_grounding as ag
from . import channel_tracker as ct
from . import locality as lc
from .state_motifs import BANDS

REPO = Path(__file__).resolve().parents[3]
MC = REPO / "data/processed/multichannel_local"
LIBS = REPO / "data/processed/channel_tracker_local/_coarse"
CENTRE = np.sqrt(BANDS.prod(1))


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


TILT = unit(np.log(CENTRE) - np.log(CENTRE).mean())            # spectral tilt (1/f slope change) in band space
NAMED = {"delta 1-4 Hz": 0, "theta 5-9 Hz": 2, "beta 13-20 Hz": 5, "low gamma 35-55 Hz": 8, "high gamma 50-80 Hz": 9}
NARROW_MS = 0.4

CLASSES = [("hippocampus", ("CA", "DG", "POST", "ProS", "SUB")),
           ("thalamus", ("AD", "AM", "AV", "CL", "IAD", "LD", "LGd", "LGv", "LP", "MD", "MG", "PO", "RT", "VAL", "VL",
                         "VPL", "VPM", "TH", "PCN", "SGN", "IGL")),
           ("striatum / septum / pallidum", ("ACB", "CP", "STR", "GP", "LS", "MS", "OT")),
           ("midbrain", ("APN", "MB", "MRN", "NOT", "PPT", "RPF", "SC")),
           ("olfactory", ("AON", "TT", "DP")),
           ("cortex", ("ACA", "ILA", "MO", "ORB", "PL", "RSP", "SS", "VIS", "AUD", "FRP", "AI", "GU", "TE"))]


def region_class(r):
    for name, prefixes in CLASSES:
        if r.startswith(prefixes):
            return name
    return "other"


def area_of(r):
    return re.sub(r"(1|2/3|4|5|6a|6b)$", "", r) if region_class(r) == "cortex" else r


# ---------------------------------------------------------------- per-channel summaries

def array_channels(npz):
    """Per live channel: awake and isoflurane residual covariance, metadata, own awake library (if any)."""
    key = Path(npz).stem
    session, array = key.split("_")
    d = np.load(npz, allow_pickle=True)
    t, st, piece, env = d["t"], d["state"], d["piece"], d["env"].astype(float)
    libs = ag.load(LIBS / f"{key}.pkl") if (LIBS / f"{key}.pkl").exists() else {}
    rows = []
    for j in range(env.shape[1]):
        if np.isnan(env[:, j]).all():
            continue
        region = str(d["region"][j])
        if array != "EEG" and not ag.is_grey(region):
            continue
        Z, base, resid = ct.signal_of(t, env[:, j], piece, st == 0)
        ok = np.isfinite(resid).all(1)
        cov = {}
        for s, name in ((0, "C_awake"), (1, "C_iso")):
            m = ok & (st == s)
            cov[name] = np.cov(resid[m].T) if m.sum() >= 120 else None
        if cov["C_awake"] is None:
            continue
        U = np.asarray(libs[j]["U"], float) if j in libs else None
        rows.append({"session": session, "mouse": session[:6], "array": array, "is_eeg": array == "EEG", "j": j,
                     "depth_um": float(d["depth_um"][j]), "region": region, "class": "EEG" if array == "EEG" else region_class(region),
                     "area": area_of(region), "ccf": np.asarray(d["ccf"][j], float), "has_library": j in libs,
                     "U": unit(U) if U is not None and len(U) else np.zeros((0, 10)), **cov})
    return rows


# ---------------------------------------------------------------- nulls and dictionary

def lag1(T):
    T = unit(T)
    return float(np.mean((T[:, :-1] * T[:, 1:]).sum(1)))


def smooth_directions(n, ell, rng):
    i = np.arange(10)
    K = np.exp(-(i[:, None] - i[None]) ** 2 / (2 * ell ** 2)) + 1e-6 * np.eye(10)
    return unit(rng.standard_normal((n, 10)) @ np.linalg.cholesky(K).T)


def fit_ell(T, rng):
    """Length scale of the random smooth null so that its adjacent-band similarity matches the real templates'."""
    target = lag1(T)
    grid = np.linspace(0.3, 8, 78)
    return float(grid[np.argmin([abs(lag1(smooth_directions(4000, g, rng)) - target) for g in grid])])


def orient(D):
    D = unit(D)
    return D * np.sign(D[np.arange(len(D)), np.abs(D).argmax(1)])[:, None]


def fit_lines(T, w, k, rng, n_init=20, iters=100):
    """Sign-free k-lines clustering (each centre = top eigenvector of its members' weighted outer products)."""
    T = unit(T)
    best = None
    for _ in range(n_init):
        D = T[rng.choice(len(T), k, replace=False, p=w / w.sum())]
        for _ in range(iters):
            a = np.abs(T @ D.T).argmax(1)
            new = []
            for i in range(k):
                m = a == i
                new.append(np.linalg.eigh((T[m] * w[m, None]).T @ T[m])[1][:, -1] if m.any() else T[rng.integers(len(T))])
            new = np.array(new)
            done = np.allclose(np.abs((new * D).sum(1)), 1, atol=1e-9)
            D = new
            if done:
                break
        a = np.abs(T @ D.T).argmax(1)
        score = float((w * np.abs(T @ D.T).max(1) ** 2).sum())
        if best is None or score > best[0]:
            best = (score, D, a)
    _, D, a = best
    share = np.array([w[a == i].sum() / w.sum() for i in range(k)])
    o = np.argsort(-share)
    return orient(D[o]), share[o]


def template_pool(Ch, mice):
    P = Ch[Ch.mouse.isin(mice) & Ch.has_library]
    T, w, meta = [], [], []
    per_mouse = P.groupby("mouse").U.apply(lambda s: sum(len(u) for u in s))
    for _, r in P.iterrows():
        for u in r.U:
            T.append(u); w.append(1.0 / per_mouse[r.mouse]); meta.append((r.mouse, r["class"], r.is_eeg))
    return np.array(T), np.array(w), pd.DataFrame(meta, columns=["mouse", "class", "is_eeg"])


def captured(D, C):
    """Share of the residual variance captured by span(D), and the same for the channel's own top-k PCs."""
    Q = np.linalg.qr(np.atleast_2d(D).T)[0]
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    return float(np.trace(Q.T @ C @ Q) / ev.sum()), float(ev[:Q.shape[1]].sum() / ev.sum())


def pooled_pca(Ch, mice, k):
    M = sum(c / np.trace(c) for c in Ch[Ch.mouse.isin(mice)].C_awake)
    return np.linalg.eigh(M)[1][:, ::-1][:, :k].T


def match(A, B):
    """Hungarian match of two motif sets by |cos|; returns |cos| per row of A (NaN if unmatched)."""
    S = np.abs(unit(A) @ unit(B).T)
    r, c = linear_sum_assignment(-S)
    out = np.full(len(A), np.nan)
    out[r] = S[r, c]
    return out, dict(zip(r, c))


def heldout_reconstruction(Ch, train, test, D, rng, ell, n_random=100, cov="C_awake"):
    """Per held-out channel: share captured by D, by the training mice's pooled PCA, by random smooth directions,
    relative to the channel's own top-k PCs (the ceiling)."""
    k = len(D)
    Pp = pooled_pca(Ch, train, k)
    R = [smooth_directions(k, ell, rng) for _ in range(n_random)]
    rows = []
    for _, c in Ch[Ch.mouse.isin(test)].iterrows():
        C = c[cov]
        if C is None:
            continue
        d, ceil = captured(D, C)
        rows.append({"mouse": c.mouse, "is_eeg": c.is_eeg, "class": c["class"], "dictionary": d / ceil,
                     "pooled PCA": captured(Pp, C)[0] / ceil,
                     "random smooth": float(np.mean([captured(r, C)[0] for r in R])) / ceil,
                     "tilt": captured(TILT[None], C)[0] / captured(TILT[None], C)[1] if k == 1 else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- U2 transfer

def transfer_pairs(Ch, targets_every=4):
    """Every source channel with a library x every target channel (probes: every 4th extracted channel)."""
    src = Ch[Ch.has_library & (Ch.U.map(len) > 0)].reset_index(drop=True)
    tgt = Ch[Ch.is_eeg | (Ch.j % targets_every == 0)].reset_index(drop=True)
    out = []
    for state in ("C_awake", "C_iso"):
        ok = tgt[state].map(lambda c: c is not None).to_numpy()
        Cs = np.stack([c if c is not None else np.eye(10) for c in tgt[state]])
        lam = np.linalg.eigvalsh(Cs)[:, -1]
        for i, s in src.iterrows():
            q = np.einsum("ui,cij,uj->uc", s.U, Cs, s.U).max(0) / lam
            out.append(pd.DataFrame({"src": i, "tgt": np.arange(len(tgt)), "state": state[2:], "score": np.where(ok, q, np.nan)}))
    P = pd.concat(out, ignore_index=True)
    skey = (src.session + "_" + src.array + "_" + src.j.astype(str)).to_numpy()
    tkey = (tgt.session + "_" + tgt.array + "_" + tgt.j.astype(str)).to_numpy()
    P = P[skey[P.src] != tkey[P.tgt]]
    S, T = src.loc[P.src].reset_index(drop=True), tgt.loc[P.tgt].reset_index(drop=True)
    P = P.reset_index(drop=True)
    P["src_mouse"], P["tgt_mouse"] = S.mouse, T.mouse
    P["same_mouse"] = S.mouse.to_numpy() == T.mouse.to_numpy()
    P["same_array"] = P.same_mouse & (S.array.to_numpy() == T.array.to_numpy()) & (S.session.to_numpy() == T.session.to_numpy())
    P["src_kind"] = np.where(S.is_eeg, "EEG", "LFP")
    P["tgt_kind"] = np.where(T.is_eeg, "EEG", "LFP")
    P["same_class"] = (S["class"].to_numpy() == T["class"].to_numpy()) & ~S.is_eeg.to_numpy() & ~T.is_eeg.to_numpy()
    P["same_area"] = S.area.to_numpy() == T.area.to_numpy()
    P["atlas_mm"] = np.linalg.norm(np.stack(S.ccf) - np.stack(T.ccf), axis=1) * 0.025
    P["depth_gap_um"] = np.where(P.same_array, np.abs(S.depth_um.to_numpy() - T.depth_um.to_numpy()), np.nan)
    P["src_class"], P["tgt_class"] = S["class"], T["class"]
    return P, src, tgt


def own_scores(tgt, rng, ell, n_random=200):
    """Per target: own library (if any), spectral tilt, random smooth direction (mean), awake and isoflurane."""
    rows = []
    R = smooth_directions(n_random, ell, rng)
    for _, c in tgt.iterrows():
        for state in ("C_awake", "C_iso"):
            C = c[state]
            if C is None:
                continue
            lam = np.linalg.eigvalsh(C)[-1]
            rows.append({"session": c.session, "array": c.array, "j": c.j, "mouse": c.mouse, "state": state[2:],
                         "kind": "EEG" if c.is_eeg else "LFP",
                         "own library": float(np.max(np.einsum("ui,ij,uj->u", c.U, C, c.U)) / lam) if len(c.U) else np.nan,
                         "tilt": float(TILT @ C @ TILT / lam),
                         "random smooth": float(np.mean(np.einsum("ui,ij,uj->u", R, C, R)) / lam)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- U3 / U5 session signals

def _rates(trains, sec):
    piece = np.r_[0, np.cumsum(np.diff(sec) > 1)]
    counts = np.stack([np.searchsorted(s, sec + 0.5) - np.searchsorted(s, sec - 0.5) for s in trains], 1).astype(float)
    rate = np.empty_like(counts)
    for p in np.unique(piece):
        k = piece == p
        rate[k] = gaussian_filter1d(counts[k], 1.0, axis=0, mode="nearest")
    return counts, rate


def _zc(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-12)


def _resid_on(X, Cv):
    """Residual of the columns of X after least squares on the covariate columns Cv (both time x k)."""
    A = np.c_[np.ones(len(Cv)), Cv]
    return X - A @ np.linalg.lstsq(A, X, rcond=None)[0]


def array_resid(npz):
    d = np.load(npz, allow_pickle=True)
    t, st, piece, env = d["t"], d["state"], d["piece"], d["env"].astype(float)
    live = np.flatnonzero(~np.isnan(env).all(axis=(0, 2)))
    R = np.full((len(t), len(live), 10), np.nan)
    for i, j in enumerate(live):
        R[:, i] = ct.signal_of(t, env[:, j], piece, st == 0)[2]
    aw = (st == 0) & np.isfinite(R).all((1, 2))
    pc1 = np.linalg.eigh(np.mean([np.cov(R[aw, i].T) for i in range(len(live))], 0))[1][:, -1]
    return {"sec": np.round(t).astype(int), "state": st, "resid": R, "depth_um": d["depth_um"][live],
            "ccf": d["ccf"][live], "region": d["region"][live], "pc1": pc1}


def signal_block(r, D, pc1):
    """Per second: universal motifs, then the controls (tilt, array band PC1, named bands)."""
    ctrl = np.c_[r @ TILT, r @ pc1, r[:, list(NAMED.values())]]
    return r @ D.T, ctrl


def grounding_probe(U, A, D, state, rng, unit_ids, n_shift=100):
    """Units of one probe; their nearest live channel; per unit and motif: r with population-removed rate,
    partial r beyond controls, signed r with the raw rate. Shift null shared by all units (same shifts)."""
    m = (A["state"] == state) & np.isfinite(A["resid"]).all((1, 2))
    sec = A["sec"][m]
    U = U[U.unit_id.isin(unit_ids)].reset_index(drop=True)
    if len(sec) < 300 or len(U) < 10:
        return None
    counts, rate = _rates(U.train.to_list(), sec)
    near = np.array([int(np.argmin(np.abs(A["depth_um"] - z))) for z in U.depth_um])
    Rp, Rr = _zc(ag.remove_global(rate)), _zc(rate)
    k = len(D)
    n = len(sec)
    ms = min(300, n // 3)
    shifts = rng.integers(ms, max(ms + 1, n - ms), n_shift)
    res = {f: np.zeros((len(U), k)) for f in ("r", "partial", "signed_raw")}
    null = {f: np.zeros((n_shift, len(U), k)) for f in ("r", "partial")}
    for ch in np.unique(near):
        idx = np.flatnonzero(near == ch)
        Mo, Ct = signal_block(A["resid"][m, ch], D, A["pc1"])
        Mz, Mpz = _zc(Mo), _zc(_resid_on(Mo, Ct))
        Rpart = _zc(_resid_on(Rp[:, idx], Ct))
        res["r"][idx] = Rp[:, idx].T @ Mz / n
        res["partial"][idx] = Rpart.T @ Mpz / n
        res["signed_raw"][idx] = Rr[:, idx].T @ Mz / n
        for si, s in enumerate(shifts):
            Ms, Cs = np.roll(Mo, s, 0), np.roll(Ct, s, 0)
            null["r"][si, idx] = Rp[:, idx].T @ _zc(Ms) / n
            null["partial"][si, idx] = _zc(_resid_on(Rp[:, idx], Cs)).T @ _zc(_resid_on(Ms, Cs)) / n
    return {"units": U.drop(columns="train").assign(narrow=U.waveform_ms.to_numpy() < NARROW_MS), **res,
            "null": null, "n_sec": n}


def session_units(session):
    path = glob.glob(str(REPO / f"data/raw/{session}*.nwb"))[0]
    from . import session_data as sd
    io, nwb = sd.open_nwb(path)
    U = lc.units_of(nwb)
    io.close()
    return U


def grounding_session(session, D, rng, n_shift=100):
    """U3 for every probe of a session, awake and isoflurane (units kept by the awake >= 0.5 Hz rule)."""
    U = session_units(session)
    out = []
    for f in sorted(glob.glob(str(MC / f"{session}_probe*.npz"))):
        probe = Path(f).stem.split("_")[1]
        Up = U[U.probe == probe]
        if len(Up) < 10:
            continue
        A = array_resid(f)
        aw = (A["state"] == 0) & np.isfinite(A["resid"]).all((1, 2))
        c, _ = _rates(Up.train.to_list(), A["sec"][aw])
        keep = Up.unit_id.to_numpy()[c.mean(0) >= 0.5]
        for state, name in ((0, "awake"), (1, "isoflurane")):
            g = grounding_probe(Up, A, D, state, rng, keep, n_shift)
            if g is not None:
                out.append({"session": session, "probe": probe, "state": name, **g})
    return out


def eeg_beneath_session(session, D, rng, n_shift=100, near_mm=1.0, far_mm=3.0):
    """U5: |r| of each EEG electrode's motif expression with each unit's population-removed rate; units within
    near_mm horizontally (CCF x, z) vs beyond far_mm; shift null shared across pairs."""
    f = MC / f"{session}_EEG.npz"
    if not f.exists():
        return []
    U = session_units(session)
    if len(U) < 20:
        return []
    E = array_resid(f)
    out = []
    for state, name in ((0, "awake"), (1, "isoflurane")):
        m = (E["state"] == state) & np.isfinite(E["resid"]).all((1, 2))
        sec = E["sec"][m]
        if len(sec) < 300:
            continue
        c, rate = _rates(U.train.to_list(), sec)
        if state == 0:
            keep = c.mean(0) >= 0.5
        Rp = _zc(ag.remove_global(rate[:, keep]))
        Uk = U[keep].reset_index(drop=True)
        h = np.linalg.norm(np.stack(Uk.ccf)[:, None, [0, 2]] - E["ccf"][None, :, [0, 2]], axis=2) * 0.025
        n = len(sec)
        ms = min(300, n // 3)
        shifts = rng.integers(ms, max(ms + 1, n - ms), n_shift)
        for i in range(len(D)):
            X = _zc(E["resid"][m] @ D[i])                  # seconds x electrodes
            r = np.abs(Rp.T @ X / n)
            nl = np.array([np.abs(Rp.T @ np.roll(X, s, 0) / n) for s in shifts])
            for lab, mask in (("beneath", h <= near_mm), ("far", h >= far_mm)):
                if mask.sum() < 10:
                    continue
                out.append({"session": session, "state": name, "motif": i, "where": lab, "pairs": int(mask.sum()),
                            "abs_r": float(r[mask].mean()), "null": nl[:, mask].mean(1)})
    return out
