"""D2: the tracker on every extracted channel (multichannel.py output), on the awake references.

Per channel:
  - envelopes z-scored with the mean / SD of the channel's awake seconds;
  - sliding baseline / spread / residual computed inside contiguous pieces (awake and isoflurane pieces are separate);
  - motif library from the AWAKE seconds only (5-min windows, 30-s step, tracks living >= 4 windows), so the
    motif identity is an awake reference; if a channel has too little awake data for a library (two recordings
    with ~6 min awake, the EEG-only mouse), the fallback is the consensus of the other mice's awake libraries of
    the same kind (LFP or EEG), which earlier transferred well across mice;
  - per second and motif: expression (residual . u), baseline (baseline . u), spread (rolling SD of expression,
    within pieces), presence (share of covering 5-min windows with a candidate matching u, |cos| >= 0.8; computed
    separately in awake and isoflurane seconds), unsplit projection (z . u);
  - the 10-band baseline vector per second (for spatial maps of baseline shifts).
Output: one pickle per probe / EEG array.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import motif_tracking as mt
from . import motif_value as mv

CFG = mt.TrackConfig(step=30, min_life=4)
LINK = 0.8


def signal_of(t, env, piece, ref):
    Z = (env - env[ref].mean(0)) / env[ref].std(0, ddof=1)
    base, spread = mt.split_baseline(Z, piece, CFG.baseline_window)
    return Z, base, (Z - base) / spread


def index_signal(Z, base, resid, piece):
    n = len(Z)
    return mt.Signal("x", None, np.arange(n, dtype=float), Z, base, np.ones_like(Z), resid, piece)


def awake_library(Z, base, resid, piece):
    sig = index_signal(Z, base, resid, piece)
    if len(Z) < CFG.window + (CFG.min_life - 1) * CFG.step:
        return np.zeros((0, Z.shape[1]))
    cands = mt.link(mt.window_candidates(sig, CFG), CFG)
    _, lib = mt.build_library(cands, CFG)
    return np.array([m.template for m in lib]) if lib else np.zeros((0, Z.shape[1]))


def presence(U, Z, base, resid, piece):
    n = len(Z)
    out = np.full((n, len(U)), np.nan)
    if n < CFG.window or len(U) == 0:
        return out
    sig = index_signal(Z, base, resid, piece)
    cands = mt.window_candidates(sig, CFG)
    cover, hits = np.zeros(n), np.zeros((n, len(U)))
    for w, a in enumerate(mt.window_starts(sig, CFG)):
        sl = slice(int(a), min(int(a) + CFG.window, n))
        cover[sl] += 1
        c = cands[cands.window == w]
        if len(c):
            A = np.stack(c.axis.to_numpy())
            hits[sl] += (np.abs(A @ U.T).max(0) >= LINK)
    with np.errstate(invalid="ignore"):
        out = hits / cover[:, None]
    return pd.DataFrame(out).ffill().bfill().to_numpy()


def rolling_spread(E, piece):
    S = np.full(E.shape, np.nan)
    for p in np.unique(piece):
        i = np.flatnonzero(piece == p)
        if i.size >= 30:
            half = int(min(150, max(15, i.size // 2)))
            S[i] = mt.rolling_sd(E[i] - E[i].mean(0), half)
    return S


def run_array(npz_path):
    """Per channel: awake library + per-second outputs. Returns dict with arrays and per-channel libraries."""
    d = np.load(npz_path, allow_pickle=True)
    t, state, piece, env = d["t"], d["state"], d["piece"], d["env"].astype(float)
    awake = state == 0
    chans = []
    for j in range(env.shape[1]):
        Z, base, resid = signal_of(t, env[:, j], piece, awake)
        ok = np.isfinite(resid).all(1)
        aw = awake & ok
        U = awake_library(Z[aw], base[aw], resid[aw], piece[aw])
        chans.append({"Z": Z, "base": base, "resid": resid, "ok": ok, "U": U})
    return {"t": t, "state": state, "piece": piece, "channels": chans,
            **{k: d[k] for k in ("channel", "depth_um", "region", "ccf") if k in d}}


def finish(arr, fallback):
    """Fill missing libraries with the fallback templates and compute the per-second outputs."""
    t, state, piece = arr["t"], arr["state"], arr["piece"]
    out = []
    for ch in arr["channels"]:
        U = ch["U"] if len(ch["U"]) else np.asarray(fallback)
        src = "own awake library" if len(ch["U"]) else "consensus of other mice"
        ok = ch["ok"]
        E = ch["resid"] @ U.T
        P = np.full(E.shape, np.nan)
        for st in (0, 1):
            m = (state == st) & ok
            if m.sum():
                P[m] = presence(U, ch["Z"][m], ch["base"][m], ch["resid"][m], piece[m])
        out.append({"U": U.astype(np.float32), "source": src, "ok": ok,
                    "E": E.astype(np.float32), "B": (ch["base"] @ U.T).astype(np.float32),
                    "S": rolling_spread(np.where(ok[:, None], E, 0.0), piece).astype(np.float32),
                    "P": P.astype(np.float32), "Uproj": (ch["Z"] @ U.T).astype(np.float32),
                    "base_vec": ch["base"].astype(np.float32)})
    return {**{k: arr[k] for k in arr if k != "channels"}, "channels": out}


def consensus_for(libs):
    """Consensus motifs (sign-free) from a list of template arrays."""
    T = [u for L in libs for u in L]
    return [c for c, _ in mv.consensus_templates(T)] if T else []
