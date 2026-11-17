"""Isoflurane step of the state <-> ensembles program, built on the awake references.

Isoflurane seconds: stimulation-free maintenance (end of induction to isoflurane off, every electrical / visual block
cut with a 10-s margin). Everything that defines the analysis comes from the awake state and is reused unchanged:
  - envelopes are z-scored with the mean / SD of the awake, stimulation-free seconds;
  - the tracker baseline / residual are computed on awake + isoflurane seconds together, cut at isoflurane on and the
    end of induction, so baseline values are in awake units;
  - the motif is the signal's awake tracker motif (M1); presence under isoflurane = share of 5-min windows in which
    a motif candidate matches it (|cos| >= 0.8);
  - the units are the awake-kept units; the + / - ensembles are the awake-defined ones;
  - covariates: running, pupil and isoflurane dose (all confounded with each other).

assemble_iso() returns the same dictionary as relationships.assemble(), so the whole catalog runs unchanged.
cross_state() compares each neuron's coupling awake vs isoflurane, matched for data length.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from . import awake_grounding as ag
from . import ensembles as en
from . import motif_tracking as mt
from . import relationships as rl
from . import session_data as sd
from .smooth_env import smooth_env
from .state_motifs import BANDS, FS

MARGIN = 10.0


def iso_intervals(info):
    blocks = sorted((a - MARGIN, b + MARGIN) for _, a, b in list(info.stim_blocks) + list(info.visual_blocks))
    segs, s = [], info.induction_end
    for a, b in blocks:
        if b <= s:
            continue
        if a > s:
            segs.append((s, min(a, info.iso_off)))
        s = max(s, b)
    if s < info.iso_off:
        segs.append((s, info.iso_off))
    return [(a, b) for a, b in segs if b - a >= 60]


def envelopes(path, probe, channel, intervals):
    io, nwb = sd.open_nwb(path)
    series = nwb.acquisition["LFP"][f"ElectricalSeries{probe}"]
    ts = series.timestamps
    i_end = int(np.searchsorted(ts, intervals[-1][1] + 1))
    t_raw = ts[:i_end]
    x_raw = series.data[:i_end, channel] * series.conversion * 1e6
    io.close()
    t, x, fs = sd.decimate(t_raw, x_raw, 1 / np.median(np.diff(t_raw[:100_000])), FS)
    hop = int(round(fs))
    T, E = [], []
    for a, b in intervals:
        k = (t >= a) & (t < b)
        env = smooth_env(x[k], fs, BANDS, zscore=False)[::hop]
        ok = np.isfinite(env).all(1)
        T.append(t[k][::hop][ok]); E.append(env[ok])
    return np.concatenate(T), np.concatenate(E)


def presence_of(u, sig_iso, cfg=mt.TrackConfig(step=30, min_life=4), link_cos=0.8):
    """Share of the 5-min windows covering each second in which a motif candidate matches u."""
    n = len(sig_iso.t)
    idx = mt.Signal("x", None, np.arange(n, dtype=float), sig_iso.Z, sig_iso.base, sig_iso.spread, sig_iso.resid,
                    sig_iso.segment)
    cands = mt.window_candidates(idx, cfg)
    cover, hits = np.zeros(n), np.zeros(n)
    for w, a in enumerate(mt.window_starts(idx, cfg)):
        sl = slice(int(a), min(int(a) + cfg.window, n))
        cover[sl] += 1
        c = cands[cands.window == w]
        if len(c) and max(abs(ax @ u) for ax in c.axis) >= link_cos:
            hits[sl] += 1
    with np.errstate(invalid="ignore"):
        p = hits / cover
    return pd.Series(p).ffill().bfill().to_numpy()


def unit_trains(nwb, unit_ids):
    u = nwb.units.to_dataframe().loc[unit_ids]
    return [np.asarray(s) for s in u.spike_times]


def assemble_iso(path, r, e):
    """Isoflurane signal on the awake references. r, e: the signal's awake_grounding and ensembles results."""
    m = r["meta"]
    T = r["templates"]["tracker"]
    if T.shape[0] == 0:
        return None
    u = T[0] / np.linalg.norm(T[0])
    io, nwb = sd.open_nwb(path)
    if "isoflurane_level" not in nwb.acquisition:
        io.close()
        return None
    info = sd.read_info(nwb, path)
    awake_iv, _ = ag.awake_intervals(nwb)
    iso_iv = iso_intervals(info)
    trains = unit_trains(nwb, e["labels"].unit_id.to_numpy())
    io.close()
    if not iso_iv:
        return None
    t, env = envelopes(path, m["probe"], m["channel"], awake_iv + iso_iv)
    awake = t < info.iso_on
    sig = mt.prepare_arrays("x", None, t, env, mt.TrackConfig(), cuts=[info.iso_on, info.induction_end], reference=awake)
    iso = (t >= info.induction_end) & np.isfinite(sig.resid).all(1)
    sub = lambda a: a[iso]
    sig_iso = mt.Signal("x", None, sub(t), sub(sig.Z), sub(sig.base), sub(sig.spread), sub(sig.resid), sub(sig.segment))
    ti = sig_iso.t
    E = sig_iso.resid @ u
    seg = sig_iso.segment
    S = np.full(len(ti), np.nan)
    for sgi in np.unique(seg):
        i = np.flatnonzero(seg == sgi)
        if i.size >= 30:
            half = int(min(150, max(15, i.size // 2)))
            S[i] = mt.rolling_sd((E[i] - E[i].mean())[:, None], half)[:, 0]
    piece = np.r_[0, np.cumsum(np.diff(ti) > 1.5)]
    counts = np.stack([np.searchsorted(s, ti + 0.5) - np.searchsorted(s, ti - 0.5) for s in trains], 1).astype(float)
    rate = np.empty_like(counts)
    for p in np.unique(piece):
        k = piece == p
        rate[k] = gaussian_filter1d(counts[k], 1.0, axis=0, mode="nearest")
    run, pup = rl.behaviour_at(path, ti)
    dose = np.interp(ti, info.iso_t, info.iso_level)
    lab = e["labels"]["lfp tracker expression M1 (raw)"].to_numpy() if "lfp tracker expression M1 (raw)" in e["labels"] \
        else np.zeros(rate.shape[1], int)
    n = len(ti)
    meta = {**m, "state": "isoflurane", "n_seconds": n, "min_shift": int(min(300, n // 3)),
            "dose_median": float(np.median(dose)), "dose_range": [float(dose.min()), float(dose.max())]}
    return {"meta": meta, "t": ti, "E": E, "B": sig_iso.base @ u, "S": S, "P": presence_of(u, sig_iso), "U": sig_iso.Z @ u,
            "Bvec": sig_iso.base, "rate": rate, "counts": counts, "label": lab, "run": run, "pupil": pup, "dose": dose,
            "u": u, "cov": np.column_stack([run, pup, dose]), "cov_names": ["running", "pupil", "dose"],
            "Z": sig_iso.Z, "resid": sig_iso.resid,
            "awake_baseline_vec_mean": sig.base[awake].mean(0), "iso_baseline_vec_mean": sig_iso.base.mean(0)}


# ---------------------------------------------------------------- per-neuron awake vs isoflurane

def _membership(x, R, rng, min_shift):
    return en.lfp_membership(x, R, rng, min_shift, n_shift=300)


def cross_state(r, e, d, rng, n_blocks=4):
    """Per neuron: coupling to the motif awake vs under isoflurane, and in/out turnover against within-state
    turnover, all on stretches of equal length (half the awake seconds), so the longer isoflurane recording cannot
    make more neurons look coupled. Done on raw rates and with the population mean removed (under isoflurane, global
    on/off firing couples almost every neuron to any LFP signal).
      within awake : awake half A vs awake half B (even / odd 60-s blocks)
      within iso   : disjoint isoflurane stretches of that length
      across       : each awake half vs each isoflurane stretch
    Turnover = share of neurons whose label (+, -, none) differs."""
    Ea, Ei = r["outputs"]["tracker expression"][:, 0], d["E"]
    na = len(Ea)
    half = (np.arange(na) // 60) % 2 == 0
    L = int(min(half.sum(), (~half).sum()))
    ms = int(min(300, L // 3))
    starts = np.linspace(0, len(Ei) - L, n_blocks).astype(int) if len(Ei) >= 2 * L else np.array([0])
    turn = lambda a, b: float(np.mean(a != b))
    out = {"L_seconds": L, "n_iso_blocks": len(starts)}
    per = {"unit_id": e["labels"].unit_id.to_numpy(), "rate_awake": e["rate"].mean(0), "rate_iso": d["counts"].mean(0)}
    for kind, f in (("raw", lambda R: R), ("population removed", ag.remove_global)):
        aw = [_membership(Ea[h][:L], f(e["rate"][h][:L]), rng, ms)[0] for h in (half, ~half)]
        iso = [_membership(Ei[s:s + L], f(d["rate"][s:s + L]), rng, ms)[0] for s in starts]
        pairs = [(i, j) for i in range(len(iso)) for j in range(i + 1, len(iso)) if starts[j] - starts[i] >= L]
        la, ra, _ = _membership(Ea, f(e["rate"]), rng, min(300, na // 3))
        li, ri, _ = _membership(Ei, f(d["rate"]), rng, min(300, len(Ei) // 3))
        both = (la != 0) & (li != 0)
        out[kind] = {"within_awake": turn(*aw),
                     "within_iso": float(np.mean([turn(iso[i], iso[j]) for i, j in pairs])) if pairs else np.nan,
                     "across": float(np.mean([turn(a, b) for a in aw for b in iso])),
                     "share_coupled_awake_half": float(np.mean([np.mean(a != 0) for a in aw])),
                     "share_coupled_iso_block": float(np.mean([np.mean(b != 0) for b in iso])),
                     "sign_kept_share": float(np.mean(np.sign(ra[both]) == np.sign(ri[both]))) if both.any() else np.nan,
                     "n_coupled_both": int(both.sum()),
                     "coupling_r_across_neurons": float(np.corrcoef(ra, ri)[0, 1]) if len(ra) > 3 else np.nan}
        per.update({f"label_awake ({kind})": la, f"label_iso ({kind})": li, f"r_awake ({kind})": ra, f"r_iso ({kind})": ri})
    out["per_neuron"] = pd.DataFrame(per)
    return out
