"""Phase 2 of the state <-> ensembles program: a fixed catalog of LFP-tracker <-> spiking relationships.

Written for the awake reference first; every function takes the seconds and references as arguments so the same
code runs on isoflurane later with the awake-defined units, ensembles and z-score reference.

Inputs per signal: the awake_grounding result (seconds, band z-scores, motif templates) and the Phase 1 ensembles
result (kept units, rates, ensemble labels). The tracker motif used is the signal's longest-lived tracker motif (M1);
+ / - ensembles are the neurons significantly coupled to its expression (Phase 1, raw rates).

Pupil width and running speed are arousal covariates in every test (partialled out). They are confounded with each
other and, under isoflurane, with dose (experimenters adjusted isoflurane from live pupil readouts), so they are
controls, never state markers.

Per-second relationships (per signal: effect + null from circular shifts of the LFP side; shifts are shared by all
signals of a session so simultaneous signals are not treated as independent):
  R1  level       partial r(baseline, population rate), and with + and - ensemble rates
  R2  membership  per neuron: rate ~ E + B + E*B + pupil + running; share of neurons whose coupling to the motif
                  changes with the baseline (|t| of E*B > 2), in excess of the null (interaction term shifted only)
  R4  intensity   ensemble contrast (+ minus -) ~ E + B + E*B + covariates: slope on E, and gain change with B
  R11 timing      lag of the peak cross-correlation between E and the ensemble contrast (descriptive)
  R12 beyond arousal  held-out R^2 gain for the population rate and the contrast when tracker outputs are added to
                  pupil + running
Window relationships (WIN-s windows every STEP s; Spearman across windows, partial on pupil and running):
  R3  baseline velocity       <-> membership turnover
  R5  pole separation         <-> + / - opponency and coupling strength
  R6  pole asymmetry/midpoint <-> + / - balance
  R7  spread                  <-> synchrony, dimensionality, Fano factor
  R8  baseline rotation       <-> membership turnover
  R9  presence                <-> share of neurons coupled
  R10 umbrella: are windows that look alike on the LFP side also alike in spiking (Mantel r)?
"""

import numpy as np
import pandas as pd

from . import awake_grounding as ag
from . import motif_tracking as mt
from . import session_data as sd

WIN, STEP = 120, 60
N_SHIFT = 200


# ---------------------------------------------------------------- inputs

def behaviour_at(path, t):
    """Running speed (mean in each 1-s bin) and pupil width (interpolated, 5-s smoothed) at seconds t."""
    io, nwb = sd.open_nwb(path)
    beh = nwb.processing["behavior"]
    rs = beh["BehavioralTimeSeries"]["running_speed"]
    run = np.asarray(rs.data[:], float)
    run_t = rs.timestamps[:] if rs.timestamps is not None else rs.starting_time + np.arange(run.size) / rs.rate
    pw = beh["PupilTracking"]["pupil_width"]
    pupil, pupil_t = np.asarray(pw.data[:], float), pw.timestamps[:]
    io.close()
    c = np.r_[0, np.cumsum(np.nan_to_num(run))]
    lo, hi = np.searchsorted(run_t, t - 0.5), np.searchsorted(run_t, t + 0.5)
    speed = (c[hi] - c[lo]) / np.maximum(hi - lo, 1)
    ok = np.isfinite(pupil)
    pup = np.interp(t, pupil_t[ok], pupil[ok]) if ok.sum() > 10 else np.zeros_like(t)
    pup = pd.Series(pup).rolling(5, center=True, min_periods=1).mean().to_numpy()
    return speed, pup


def assemble(path, r, e):
    """Everything the catalog needs for one signal, on the common seconds."""
    t = r["t"]
    sig = mt.prepare_arrays("x", None, t, r["bandsZ"], mt.TrackConfig(), cuts=[], reference=np.ones(len(t), bool))
    T = r["templates"]["tracker"]
    if T.shape[0] == 0:
        return None
    u = T[0] / np.linalg.norm(T[0])
    out = r["outputs"]
    run, pup = behaviour_at(path, t)
    lab = e["labels"]["lfp tracker expression M1 (raw)"].to_numpy() if "lfp tracker expression M1 (raw)" in e["labels"] \
        else np.zeros(e["rate"].shape[1], int)
    return {"meta": r["meta"], "t": t, "E": out["tracker expression"][:, 0], "B": out["tracker baseline"][:, 0],
            "S": out["tracker spread"][:, 0], "P": out["tracker presence"][:, 0], "U": sig.Z @ u,
            "Bvec": sig.base, "rate": e["rate"], "label": lab, "run": run, "pupil": pup, "u": u,
            "cov": np.column_stack([run, pup]), "cov_names": ["running", "pupil"], "Z": sig.Z, "resid": sig.resid}


# ---------------------------------------------------------------- helpers

def zs(v):
    v = np.asarray(v, float)
    return (v - np.nanmean(v, 0)) / (np.nanstd(v, 0) + 1e-12)


def residualize(Y, C):
    """Y (n,) or (n, k) with the covariates C (n, c) and an intercept regressed out."""
    X = np.column_stack([np.ones(len(C)), C])
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    return Y - X @ beta


def ens_rates(d):
    Z = zs(d["rate"])
    plus, minus = d["label"] > 0, d["label"] < 0
    p = Z[:, plus].mean(1) if plus.any() else np.full(len(Z), np.nan)
    m = Z[:, minus].mean(1) if minus.any() else np.full(len(Z), np.nan)
    return Z.mean(1), p, m


def session_shifts(session, n, k=N_SHIFT, min_shift=60):
    rng = np.random.default_rng(int(session.replace('-', '')))   # deterministic per session
    return rng.integers(min_shift, max(min_shift + 1, n - min_shift), k)


def fz(r):
    return np.arctanh(np.clip(r, -0.999, 0.999))


# ---------------------------------------------------------------- per-second relationships

def r1_level(d, shifts):
    C = zs(d["cov"])
    pop, p, m = ens_rates(d)
    B = residualize(zs(d["B"]), C)
    out = {}
    for name, y in (("pop", pop), ("plus", p), ("minus", m)):
        if not np.isfinite(y).all():
            continue
        y = residualize(y, C)
        out[f"R1 baseline~{name} rate"] = (ag._r(B, y), np.array([ag._r(np.roll(B, s), y) for s in shifts]))
    return out


def _ols_t(X, Y, col):
    """t statistic of column col for every column of Y."""
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ beta
    dof = X.shape[0] - X.shape[1]
    s2 = (res ** 2).sum(0) / dof
    cov = np.linalg.inv(X.T @ X)
    return beta[col] / np.sqrt(s2 * cov[col, col])


def r2_membership(d, shifts):
    E, B = zs(d["E"]), zs(d["B"])
    Y = zs(d["rate"])
    base = [np.ones(len(E)), E, B] + list(zs(d["cov"]).T)
    col = len(base)
    real = np.mean(np.abs(_ols_t(np.column_stack(base + [E * B]), Y, col)) > 2)
    null = np.array([np.mean(np.abs(_ols_t(np.column_stack(base + [E * np.roll(B, s)]), Y, col)) > 2) for s in shifts])
    return {"R2 neurons whose motif coupling changes with baseline (share)": (real, null)}


def r4_intensity(d, shifts):
    pop, p, m = ens_rates(d)
    if not (np.isfinite(p).all() and np.isfinite(m).all()):
        return {}
    y = zs(p - m)
    E, B = zs(d["E"]), zs(d["B"])
    cov = list(zs(d["cov"]).T)
    X = lambda e, b: np.column_stack([np.ones(len(E)), e, b, e * b] + cov)
    beta = lambda X_: np.linalg.lstsq(X_, y, rcond=None)[0]
    real = beta(X(E, B))
    nulls = np.array([beta(X(np.roll(E, s), np.roll(B, s))) for s in shifts])
    nullg = np.array([beta(np.column_stack([np.ones(len(E)), E, B, E * np.roll(B, s)] + cov))[3] for s in shifts])
    return {"R4 contrast slope on expression": (real[1], nulls[:, 1]),
            "R4 gain change of that slope with baseline": (real[3], nullg)}


def r11_timing(d, max_lag=10):
    pop, p, m = ens_rates(d)
    if not (np.isfinite(p).all() and np.isfinite(m).all()):
        return {}
    x, y = zs(d["E"]), zs(p - m)
    lags = np.arange(-max_lag, max_lag + 1)
    cc = [ag._r(x[max(0, -k): len(x) - max(0, k)], y[max(0, k): len(y) - max(0, -k)]) for k in lags]
    return {"R11 lag of peak (s; + = spiking follows LFP)": float(lags[int(np.nanargmax(cc))])}


def r12_beyond_arousal(d, shifts, alpha=10.0):
    pop, p, m = ens_rates(d)
    fid, inner = ag.folds(len(pop))
    beh = zs(d["cov"])
    trk = np.column_stack([zs(d[k]) for k in ("E", "B", "S", "P") if np.nanstd(d[k]) > 1e-9])

    def r2(F, y):
        pred = np.full(len(y), np.nan)
        for f in range(ag.N_FOLDS):
            tr, te = fid != f, (fid == f) & inner
            A = np.column_stack([np.ones(tr.sum()), F[tr]])
            w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ y[tr])
            pred[te] = np.column_stack([np.ones(te.sum()), F[te]]) @ w
        ok = np.isfinite(pred)
        return 1 - np.sum((y[ok] - pred[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2)

    out = {}
    targets = {"pop": pop}
    if np.isfinite(p).all() and np.isfinite(m).all():
        targets["contrast"] = p - m
    for name, y in targets.items():
        y = zs(y)
        b = r2(beh, y)
        out[f"R12 R2 gain over pupil+running ({name})"] = (
            r2(np.column_stack([beh, trk]), y) - b,
            np.array([r2(np.column_stack([beh, np.roll(trk, s, axis=0)]), y) - b for s in shifts[:100]]))
    return out


# ---------------------------------------------------------------- windows

def windows(n):
    return [np.arange(a, a + WIN) for a in range(0, n - WIN + 1, STEP)]


def lfp_windows(d):
    rows, prev = [], None
    for w in windows(len(d["t"])):
        U = d["U"][w]
        med = np.median(U)
        hi, lo = U[U > med].mean(), U[U <= med].mean()
        bv = d["Bvec"][w].mean(0)
        rows.append({"plus_pole": hi, "minus_pole": lo, "midpoint": (hi + lo) / 2, "separation": hi - lo,
                     "asymmetry": (hi - med) - (med - lo), "baseline": d["B"][w].mean(), "spread": d["S"][w].mean(),
                     "presence": d["P"][w].mean(), "bvec": bv, "cov": d["cov"][w].mean(0)})
    W = pd.DataFrame(rows)
    bv = np.stack(W.bvec)
    W["baseline_velocity"] = np.r_[np.nan, np.linalg.norm(np.diff(bv, axis=0), axis=1)]
    unitv = bv / (np.linalg.norm(bv, axis=1, keepdims=True) + 1e-12)
    W["baseline_rotation_deg"] = np.r_[np.nan, np.degrees(np.arccos(np.clip(np.sum(unitv[1:] * unitv[:-1], 1), -1, 1)))]
    return W


def coupling_in(E, Z):
    Ez = zs(E)
    return (zs(Z).T @ Ez) / len(Ez)


def neural_windows(d, rng, n_null=50):
    Z = zs(d["rate"])
    plus, minus = d["label"] > 0, d["label"] < 0
    rows, members = [], []
    for w in windows(len(d["t"])):
        Zw = Z[w]
        C = np.corrcoef(Zw.T)
        iu = np.triu_indices(C.shape[0], 1)
        ev = np.clip(np.linalg.eigvalsh(np.cov(Zw.T)), 0, None)
        cpl = coupling_in(d["E"][w], Zw)
        null = np.concatenate([coupling_in(np.roll(d["E"][w], rng.integers(20, WIN - 20)), Zw) for _ in range(n_null)])
        thr = np.nanpercentile(np.abs(null), 95)
        mem = np.abs(cpl) > thr
        members.append(np.sign(cpl) * mem)
        p = Zw[:, plus].mean(1) if plus.any() else np.full(WIN, np.nan)
        m = Zw[:, minus].mean(1) if minus.any() else np.full(WIN, np.nan)
        rate = d["rate"][w]
        rows.append({"pop_rate": d["rate"][w].mean(), "plus_rate": np.nanmean(p), "minus_rate": np.nanmean(m),
                     "balance": np.nanmean(p) - np.nanmean(m), "opponency": ag._r(p, m),
                     "synchrony": float(np.nanmean(C[iu])), "dimensionality": float(ev.sum() ** 2 / (ev ** 2).sum()),
                     "fano": float(np.nanmean(rate.var(0) / np.maximum(rate.mean(0), 1e-9))),
                     "coupled_share": float(mem.mean()), "coupling_strength": float(np.nanmean(np.abs(cpl[mem]))) if mem.any() else np.nan})
    N = pd.DataFrame(rows)
    M = np.array(members)
    # turnover between windows that do not overlap (two steps apart), and the same within a window (noise reference)
    lag = WIN // STEP
    N["turnover"] = [np.nan] * lag + [float(np.mean(M[i] != M[i - lag])) for i in range(lag, len(M))]
    return N, M


def window_relationships(W, N):
    """Partial Spearman r (pupil and running partialled out) for each window pair in the catalog."""
    pairs = {"R3 baseline velocity ~ turnover": ("baseline_velocity", "turnover"),
             "R5 separation ~ opponency": ("separation", "opponency"),
             "R5 separation ~ coupling strength": ("separation", "coupling_strength"),
             "R6 asymmetry ~ balance": ("asymmetry", "balance"),
             "R6 midpoint ~ balance": ("midpoint", "balance"),
             "R7 spread ~ synchrony": ("spread", "synchrony"),
             "R7 spread ~ dimensionality": ("spread", "dimensionality"),
             "R7 spread ~ fano": ("spread", "fano"),
             "R8 baseline rotation ~ turnover": ("baseline_rotation_deg", "turnover"),
             "R9 presence ~ coupled share": ("presence", "coupled_share"),
             "R1w baseline ~ population rate": ("baseline", "pop_rate")}
    cov = np.column_stack([pd.Series(c).rank().to_numpy() for c in np.stack(W["cov"]).T])
    out = {}
    for name, (a, b) in pairs.items():
        x, y = W[a].to_numpy(float), N[b].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 6 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
            continue
        xr = residualize(pd.Series(x[ok]).rank().to_numpy(), cov[ok])
        yr = residualize(pd.Series(y[ok]).rank().to_numpy(), cov[ok])
        out[name] = _rsmall(xr, yr)
    return out


def _rsmall(a, b):
    """Pearson r for short series (window level), NaN if either is constant."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def mantel(W, N, M, min_gap=2):
    """r between window-by-window distances on the LFP side (baseline vector + pole measures) and the spiking side
    (all neurons' rates + coupling pattern), over window pairs at least min_gap steps apart."""
    lf = np.column_stack([np.stack(W.bvec), W[["plus_pole", "minus_pole", "separation", "spread"]].to_numpy()])
    lf = zs(lf)
    sp = zs(np.column_stack([N[["pop_rate", "synchrony", "dimensionality", "fano"]].to_numpy(), M]))
    lf, sp = np.nan_to_num(lf), np.nan_to_num(sp)
    Dl = np.linalg.norm(lf[:, None] - lf[None], axis=2)
    Ds = np.linalg.norm(sp[:, None] - sp[None], axis=2)
    i, j = np.triu_indices(len(lf), min_gap)
    return _rsmall(Dl[i, j], Ds[i, j]) if len(i) >= 6 else np.nan


# ---------------------------------------------------------------- one signal

def run_signal(d):
    """All relationships for one assembled signal. Returns {'second': {name: (real, null)}, 'window': {...}}."""
    rng = np.random.default_rng(0)
    m = d["meta"]
    shifts = session_shifts(m["session"], len(d["t"]), min_shift=m["min_shift"])
    sec = {}
    for f in (r1_level, r2_membership, r4_intensity, r12_beyond_arousal):
        sec.update(f(d, shifts))
    timing = r11_timing(d)
    W = lfp_windows(d)
    N, M = neural_windows(d, rng)
    real_w = window_relationships(W, N)
    real_w["R10 umbrella (Mantel r)"] = mantel(W, N, M)
    # window null: circularly shift the LFP-side windows by >= 2 steps (all possible shifts)
    nulls = {k: [] for k in real_w}
    for s in range(2, len(W) - 1):
        Ws = W.copy()
        for c in W.columns:
            if c == "bvec":
                Ws[c] = list(np.roll(np.stack(W.bvec), s, axis=0))
            elif c != "cov":                       # covariates stay aligned with the spikes
                Ws[c] = np.roll(W[c].to_numpy(), s)
        rw = window_relationships(Ws, N)
        rw["R10 umbrella (Mantel r)"] = mantel(Ws, N, M)
        for k in nulls:
            nulls[k].append(rw.get(k, np.nan))
    win = {k: (v, np.array(nulls[k])) for k, v in real_w.items()}
    return {"meta": m, "second": sec, "window": win, "timing": timing, "W": W, "N": N}


# ---------------------------------------------------------------- pooling over signals

def pool(results, n_draw=5000, seed=0):
    """One row per relationship: mean effect over signals, a null for that mean, per-mouse consistency, BH.
    Null draws pick one shift per session (the same for every signal of that session), so simultaneous signals
    stay dependent in the null as they are in the data."""
    from .ensembles import bh
    rng = np.random.default_rng(seed)
    rows = []
    names = sorted({(lvl, n) for o in results.values() for lvl in ("second", "window") for n in o[lvl]})
    for lvl, name in names:
        items = [(k, o[lvl][name]) for k, o in results.items() if name in o[lvl] and np.isfinite(o[lvl][name][0])]
        if len(items) < 3:
            continue
        real = np.array([v for _, (v, _) in items])
        nulls = [np.asarray(nl, float) for _, (_, nl) in items]
        sess = [k[:15] for k, _ in items]
        us = {s: rng.random(n_draw) for s in set(sess)}
        draws = np.array([[nl[int(u * len(nl))] for u in us[s]] for nl, s in zip(nulls, sess)])   # (signals, draws)
        null_mean = np.nanmean(draws, 0)
        centre = np.nanmean(null_mean)
        obs = real.mean()
        p = float((1 + np.sum(np.abs(null_mean - centre) >= abs(obs - centre))) / (n_draw + 1))
        excess = real - np.array([np.nanmean(nl) for nl in nulls])
        mice = sorted({k[:6] for k, _ in items})
        per_mouse = {m: np.mean([e for (k, _), e in zip(items, excess) if k[:6] == m]) for m in mice}
        rows.append({"level": lvl, "relationship": name, "signals": len(items), "mean effect": obs, "null mean": centre,
                     "excess over null": obs - centre, "p": p,
                     "mice same sign": f"{sum(np.sign(v) == np.sign(obs - centre) for v in per_mouse.values())}/{len(mice)}",
                     "per mouse excess": {m: round(float(v), 3) for m, v in per_mouse.items()}})
    df = pd.DataFrame(rows)
    df["BH significant"] = bh(df.p.to_numpy(), 0.05)
    return df
