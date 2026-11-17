"""Do motifs add anything over simpler readouts of the LFP? Three tests on the awake grounding results.

1. Same size, no spikes used. Each direction in band space gives one number per second; every direction is scored
   with the same held-out C1 test (neurons split into + / - ensembles by the direction, scored on held-out blocks).
   Directions: the motifs (original, relaxed, tracker), the top principal components of the bands, the broadband
   direction (all bands equal) and 200 random directions. All found without spikes. Each frame is scored on its
   own: Mishler's sign-binarized projection, the continuous projection, and the tracker's residual.
2. Transfer across mice. For each mouse, consensus motif templates are learned from the other mice's signals and
   applied unchanged. Competitor: a ridge regression from the 10 bands to the population rate, trained on the other
   mice and applied unchanged. Both are scored in the held-out mouse (C1 and correlation with its population rate).
3. EEG (see eeg_test). LFP-derived and EEG-own motifs applied to the EEG channel nearest the region, scored against
   that region's neurons.

Inputs: the per-signal pickles of awake_grounding (data/processed/awake_grounding/).
"""

import numpy as np
import pandas as pd

from . import awake_grounding as ag
from . import motif_tracking as mt

N_RANDOM = 200


def unit(v):
    return v / np.linalg.norm(v)


def frames(r):
    """Per-second band representations of one signal, all on the same seconds."""
    Z = r["bandsZ"]
    sig = mt.prepare_arrays("x", None, r["t"], Z, mt.TrackConfig(), cuts=[], reference=np.ones(len(Z), bool))
    return {"binarized": np.sign(Z), "continuous": Z, "tracker": sig.resid}


def pcs(F, k=3):
    C = np.cov(F[np.isfinite(F).all(1)].T)
    w, V = np.linalg.eigh(C)
    return V[:, ::-1][:, :k].T


def c1(x, R, fid, inner):
    return ag.single_feature_diff(x, R, fid, inner)


def c1_with_null(x, R, fid, inner, rng, n_shift, min_shift):
    real = c1(x, R, fid, inner)
    null = np.array([c1(np.roll(x, s), R, fid, inner) for s in ag.shift_values(len(x), rng, n_shift, min_shift)])
    return real, float((1 + np.sum(null >= real)) / (n_shift + 1)), float(np.nanpercentile(null, 95))


def directions_test(r, rng, n_shift=200, n_random=N_RANDOM):
    """Test 1 for one signal. One row per (frame, direction)."""
    m = r["meta"]
    F = frames(r)
    R = r["rate"]
    fid, inner = ag.folds(R.shape[0])
    nb = r["bandsZ"].shape[1]
    rand = np.array([unit(v) for v in rng.standard_normal((n_random, nb))])
    rows = []
    for fname, X in F.items():
        motif_sets = {"original": r["templates"]["original"], "relaxed": r["templates"]["relaxed"]} \
            if fname != "tracker" else {"tracker": r["templates"]["tracker"]}
        dirs = {}
        for ms, T in motif_sets.items():
            for j, u in enumerate(T):
                dirs[(f"{ms} motif", f"M{j + 1}")] = unit(u)
        for j, u in enumerate(pcs(X)):
            dirs[("band PC", f"PC{j + 1}")] = u
        dirs[("broadband", "")] = unit(np.ones(nb))
        rand_scores = np.array([c1(X @ u, R, fid, inner) for u in rand])
        for (kind, name), u in dirs.items():
            real, p, n95 = c1_with_null(X @ u, R, fid, inner, rng, n_shift, m["min_shift"])
            rows.append({"frame": fname, "kind": kind, "direction": name, "r_diff": real, "p": p, "null95": n95,
                         "random_percentile": float(np.mean(rand_scores < real)) if np.isfinite(real) else np.nan,
                         "random_median": float(np.nanmedian(rand_scores)),
                         "abs_cos_broadband": float(abs(u @ unit(np.ones(nb)))),
                         "abs_cos_pc1": float(abs(u @ dirs[("band PC", "PC1")]))})
    return pd.DataFrame(rows).assign(session=m["session"], region=m["region"])


# ---------------------------------------------------------------- 2. transfer across mice

def consensus_templates(templates, cos=0.9, max_k=2, min_share=0.15):
    """Greedy consensus of unit templates (sign-free): repeatedly take the template with most neighbours
    (|cos| >= cos), average its sign-aligned neighbours, remove them. Keeps clusters holding >= min_share."""
    T = np.array([unit(t) for t in templates])
    left = np.arange(len(T))
    out = []
    while left.size and len(out) < max_k:
        C = np.abs(T[left] @ T[left].T)
        nb = (C >= cos).sum(1)
        i = int(np.argmax(nb))
        members = left[C[i] >= cos]
        if members.size < min_share * len(T):
            break
        ref = T[left[i]]
        out.append((unit(sum(np.sign(T[j] @ ref) * T[j] for j in members)), int(members.size)))
        left = np.setdiff1d(left, members)
    return out


def _pop(R):
    Z = (R - R.mean(0)) / (R.std(0) + 1e-12)
    p = Z.mean(1)
    return (p - p.mean()) / p.std()


def pooled_regression(signals, alpha=10.0):
    """Ridge from the 10 z-scored bands to the z-scored population rate, pooled over signals."""
    X = np.concatenate([r["bandsZ"] for r in signals])
    y = np.concatenate([_pop(r["rate"]) for r in signals])
    w = np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)
    return w


def inmouse_regression_pred(r, alpha=10.0):
    """Held-out prediction of the population rate from the signal's own bands (same folds): the ceiling."""
    X, y = r["bandsZ"], _pop(r["rate"])
    fid, inner = ag.folds(len(y))
    pred = np.full(len(y), np.nan)
    for f in range(ag.N_FOLDS):
        tr, te = fid != f, (fid == f) & inner
        w = np.linalg.solve(X[tr].T @ X[tr] + alpha * np.eye(X.shape[1]), X[tr].T @ y[tr])
        pred[te] = X[te] @ w
    return pred


def transfer_test(R_all, rng, n_shift=200, template_key="original"):
    mouse = lambda r: r["meta"]["session"][:6]
    mice = sorted({mouse(r) for r in R_all})
    rows, cons_rows = [], []
    for mo in mice:
        train = [r for r in R_all if mouse(r) != mo]
        test = [r for r in R_all if mouse(r) == mo]
        temps = [u for r in train for u in r["templates"][template_key]]
        cons = consensus_templates(temps)
        w = pooled_regression(train)
        for k, (c, n) in enumerate(cons):
            cons_rows.append({"held_out_mouse": mo, "consensus": f"C{k + 1}", "n_templates": n, "of": len(temps),
                              "template": np.round(c, 3).tolist(), "abs_cos_broadband": float(abs(c @ unit(np.ones(len(c)))))})
        for r in test:
            m = r["meta"]
            Z, R = r["bandsZ"], r["rate"]
            fid, inner = ag.folds(len(Z))
            pop = _pop(R)
            reads = {}
            for k, (c, _) in enumerate(cons):
                reads[f"consensus motif C{k + 1} (binarized)"] = np.sign(Z) @ c
                reads[f"consensus motif C{k + 1} (continuous)"] = Z @ c
            for j, u in enumerate(r["templates"][template_key]):
                reads[f"own motif M{j + 1} (binarized)"] = np.sign(Z) @ unit(u)
            reads["transferred regression"] = Z @ w
            reads["broadband"] = Z @ unit(np.ones(Z.shape[1]))
            reads["in-mouse regression (ceiling)"] = inmouse_regression_pred(r)
            for name, x in reads.items():
                # the in-mouse regression only predicts held-out seconds: scored on the population rate only
                rc1, p1 = (np.nan, np.nan) if name.startswith("in-mouse") else \
                    c1_with_null(x, R, fid, inner, rng, n_shift, m["min_shift"])[:2]
                c1v = ag._r(x, pop)     # motif signs are arbitrary: compare |r| across readouts
                null_pop = np.array([ag._r(np.roll(x, s), pop) for s in ag.shift_values(len(x), rng, n_shift, m["min_shift"])])
                rows.append({"held_out_mouse": mo, "session": m["session"], "region": m["region"], "readout": name,
                             "c1_r_diff": rc1, "c1_p": p1, "r_pop": c1v,
                             "p_pop": float((1 + np.sum(np.abs(null_pop) >= abs(c1v))) / (n_shift + 1))})
    return pd.DataFrame(rows), pd.DataFrame(cons_rows)


# ---------------------------------------------------------------- 3. EEG

def eeg_channels(nwb, region):
    """Nearest and farthest valid EEG electrode to the region's units, by horizontal (AP, ML) CCF distance in mm."""
    el = nwb.acquisition["ElectricalSeriesEEG"].electrodes.to_dataframe()
    u = nwb.units.to_dataframe()
    u = u[u.location == region]
    v = el[el.is_data_valid]
    d = np.hypot(v.x - u.x.mean(), v.z - u.z.mean()) * 25 / 1000
    return {"near": (int(d.idxmin()), str(v.location[d.idxmin()]), float(d.min())),
            "far": (int(d.idxmax()), str(v.location[d.idxmax()]), float(d.max()))}


def eeg_envelope(path, channel, t_target):
    """1 Hz band envelopes of one EEG channel on the awake stimulation-free intervals, sampled at t_target
    (the LFP signal's seconds), z-scored over those seconds."""
    from . import session_data as sd
    from .smooth_env import smooth_env
    from .state_motifs import BANDS, FS
    io, nwb = sd.open_nwb(path)
    intervals, _ = ag.awake_intervals(nwb)
    s = nwb.acquisition["ElectricalSeriesEEG"]
    ts = s.timestamps
    i_end = int(np.searchsorted(ts, intervals[-1][1] + 1))
    t_raw = ts[:i_end]
    x_raw = s.data[:i_end, channel] * s.conversion * 1e6
    io.close()
    fs_raw = 1 / np.median(np.diff(t_raw[:100_000]))
    t, x, fs = sd.decimate(t_raw, x_raw, fs_raw, FS)
    hop = int(round(fs))
    T, E = [], []
    for a, b in intervals:
        k = (t >= a) & (t < b)
        env = smooth_env(x[k], fs, BANDS, zscore=False)[::hop]
        ok = np.isfinite(env).all(1)
        T.append(t[k][::hop][ok]); E.append(env[ok])
    T, E = np.concatenate(T), np.concatenate(E)
    i = np.clip(np.searchsorted(T, t_target), 1, len(T) - 1)
    i = np.where(np.abs(T[i - 1] - t_target) < np.abs(T[i] - t_target), i - 1, i)
    out = E[i]
    out[np.abs(T[i] - t_target) > 1.0] = np.nan
    return (out - np.nanmean(out, 0)) / np.nanstd(out, 0, ddof=1)


def eeg_own_motifs(Zeeg, t, seed=0):
    """Original-rule motifs found on the EEG channel itself (same rules as the LFP 'original' pipeline)."""
    from .get_motifs import get_motifs
    from .state_motifs import VERSIONS
    res = get_motifs([Zeeg.T], 1.0, seed=seed, **VERSIONS["original"]["kw"])
    M = res.motifs[:, : res.n_motifs]
    M = M[:, np.isfinite(M).all(0)]
    return [unit(M[:, j]) for j in range(M.shape[1])]


def eeg_test(r, eeg, consensus, reg_w, rng, n_shift=200):
    """Test 3 for one signal. eeg: {"near": Z, "far": Z} EEG band z-scores on the signal's seconds.
    consensus: LFP consensus templates learned without this mouse; reg_w: EEG-band -> population-rate ridge
    weights learned on the other mice's EEG. One row per (electrode, readout)."""
    m = r["meta"]
    R = r["rate"]
    pop = _pop(R)
    rows = []
    for where, Ze in eeg.items():
        ok = np.isfinite(Ze).all(1)
        Zf = np.where(ok[:, None], Ze, 0.0)
        reads = {}
        own = eeg_own_motifs(Ze[ok], r["t"][ok])
        for j, u in enumerate(own):
            reads[f"EEG own motif M{j + 1}"] = np.sign(Zf) @ u
        for j, u in enumerate(r["templates"]["original"]):
            reads[f"LFP motif M{j + 1} on EEG"] = np.sign(Zf) @ unit(u)
        for k, c in enumerate(consensus):
            reads[f"consensus motif C{k + 1} on EEG"] = np.sign(Zf) @ c
        reads["EEG PC1"] = Zf @ pcs(Zf, 1)[0]
        reads["EEG broadband"] = Zf @ unit(np.ones(Zf.shape[1]))
        if reg_w is not None:
            reads["EEG regression (other mice)"] = Zf @ reg_w
        Rk = R[ok]
        fid, inner = ag.folds(int(ok.sum()))
        for name, x in reads.items():
            x = x[ok]
            rc1, p1, n95 = c1_with_null(x, Rk, fid, inner, rng, n_shift, m["min_shift"])
            rp = ag._r(x, pop[ok])
            rows.append({"electrode": where, "readout": name, "c1_r_diff": rc1, "c1_p": p1, "c1_null95": n95,
                         "abs_r_pop": abs(rp), "n_seconds": int(ok.sum())})
    return pd.DataFrame(rows).assign(session=m["session"], region=m["region"])
