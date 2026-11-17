"""E2: do LFP-tracker -> spiking mappings persist across states? Fit in one state, apply unchanged in the other.

Per signal, for each direction of transfer (awake -> isoflurane, isoflurane -> awake):
  model     ridge regression from the tracker outputs of the awake motif (expression, baseline, spread, presence) to
            a spiking target; targets: population rate, + minus - ensemble contrast (awake labels), every neuron.
  features  in two frames: "awake units" (as the tracker gives them: the isoflurane baseline sits far below the awake
            one) and "within-state z" (each state standardized on its own: tests the shape of the mapping, not its
            offset).
  score     Pearson r between prediction and target in the target state (insensitive to the ~4x rate drop).
  ceiling   the same model fitted inside the target state (5-fold blocked cross-validation).
  null      transfer with the target state's features circularly shifted (>= 5 min).
  retained  = transfer r / ceiling r: the share of the within-state relationship that survives the change of state.
"""
import numpy as np
import pandas as pd

from . import awake_grounding as ag
from . import relationships as rl


def features(d, frame, ref=None):
    F = np.column_stack([d[k] for k in ("E", "B", "S", "P")])
    if frame == "within-state z":
        F = (F - F.mean(0)) / (F.std(0) + 1e-12)
    return F


def targets(d):
    Z = rl.zs(d["rate"])
    plus, minus = d["label"] > 0, d["label"] < 0
    out = {"population rate": Z.mean(1)}
    if plus.any() and minus.any():
        out["+ minus - contrast"] = Z[:, plus].mean(1) - Z[:, minus].mean(1)
    return out, Z


def ridge(F, y, alpha=10.0):
    mu, sd = F.mean(0), F.std(0) + 1e-12
    A = np.column_stack([np.ones(len(F)), (F - mu) / sd])
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ y)
    return lambda G: np.column_stack([np.ones(len(G)), (G - mu) / sd]) @ w


def cv_r(F, y):
    fid, inner = ag.folds(len(y))
    pred = np.full(len(y), np.nan)
    for f in range(ag.N_FOLDS):
        tr, te = fid != f, (fid == f) & inner
        pred[te] = ridge(F[tr], y[tr])(F[te])
    return ag._r(pred, y)


def run(da, di, n_shift=100, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    ok_f = lambda F: np.isfinite(F).all(1)
    for src, dst, name in ((da, di, "awake -> isoflurane"), (di, da, "isoflurane -> awake")):
        for frame in ("awake units", "within-state z"):
            Fs, Fd = features(src, frame), features(dst, frame)
            ts, Zs_ = targets(src)
            td, Zd = targets(dst)
            ms, md = ok_f(Fs), ok_f(Fd)
            shifts = rl.session_shifts(dst["meta"]["session"], int(md.sum()), n_shift, min(300, int(md.sum()) // 3))
            for tname in td:
                if tname not in ts:
                    continue
                model = ridge(Fs[ms], ts[tname][ms])
                y = td[tname][md]
                r_tr = ag._r(model(Fd[md]), y)
                null = np.array([ag._r(model(np.roll(Fd[md], s, axis=0)), y) for s in shifts])
                ceil = cv_r(Fd[md], y)
                rows.append({"direction": name, "frame": frame, "target": tname, "transfer_r": r_tr, "ceiling_r": ceil,
                             "null95": float(np.nanpercentile(null, 95)),
                             "p": float((1 + np.sum(null >= r_tr)) / (n_shift + 1))})
            # every neuron: median transfer r and share beating its ceiling's half
            W = [ridge(Fs[ms], Zs_[ms][:, j]) for j in range(Zs_.shape[1])]
            rt = np.array([ag._r(W[j](Fd[md]), Zd[md][:, j]) for j in range(Zd.shape[1])])
            rc = np.array([cv_r(Fd[md], Zd[md][:, j]) for j in range(Zd.shape[1])])
            rows.append({"direction": name, "frame": frame, "target": "single neurons (median)",
                         "transfer_r": float(np.nanmedian(rt)), "ceiling_r": float(np.nanmedian(rc)),
                         "null95": np.nan, "p": np.nan})
    m = da["meta"]
    return pd.DataFrame(rows).assign(session=m["session"], region=m["region"])
