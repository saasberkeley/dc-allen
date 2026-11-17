"""S-D (shape part): does the motif itself change with depth / layer / region along a probe?

Input: the per-channel awake motif libraries of run_channel_tracker.py (coarse pass: every 4th extracted channel,
~160 um apart on probes; every EEG electrode). Caveat: these were learned on the as-recorded LFP, which is
near-identical along a probe, so shape differences are understated (a lower bound on local structure).

Measures
  pair similarity   best sign-free match between two channels' libraries: max |cos| over template pairs
  canonicality      per channel, |cos| of its best template with the cross-mouse consensus motif (from the OTHER
                    mice's libraries, so a channel never defines its own reference)
  library size      number of motifs the channel's awake library holds
  band profile      per channel, the sign-aligned (to the consensus) best template, 10 bands
Tests
  similarity ~ distance   within probe: Spearman r of pair similarity vs depth distance; null permutes channel
                          depths within probe (keeps each probe's set of templates); pooled mean r over probes
  layer / region profile  canonicality, library size and band loadings per layer; reproducibility of layer and
                          region means between the spatial-split discovery and confirmation mice
                          (spatial_maps.reproducibility)
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from . import awake_grounding as ag
from . import motif_value as mv
from . import position_pole as pp

REPO = Path(__file__).resolve().parents[3]
COARSE = REPO / "data/processed/channel_tracker/_coarse"
MC = REPO / "data/processed/multichannel"


def channel_table():
    rows = []
    for f in sorted(glob.glob(str(COARSE / "*.pkl"))):
        key = Path(f).stem
        session, array = key.split("_")
        res = ag.load(f)
        d = np.load(MC / f"{key}.npz", allow_pickle=True)
        for j, v in res.items():
            rows.append({"session": session, "mouse": session[:6], "array": array, "is_eeg": array == "EEG",
                         "j": int(j), "depth_um": float(d["depth_um"][j]), "region": str(d["region"][j]),
                         "ccf_x": float(d["ccf"][j][0]), "ccf_y": float(d["ccf"][j][1]),
                         "U": np.asarray(v["U"], float)})
    T = pd.DataFrame(rows)
    T["n_motifs"] = T.U.map(len)
    T["layer"] = T.region.map(pp.layer_of)
    return T


def consensus_without(T, mouse, eeg):
    libs = T[(T.mouse != mouse) & (T.is_eeg == eeg)].U
    c = mv.consensus_templates([u for L in libs for u in L], max_k=1)
    return c[0][0] if c else None


def add_canonical(T):
    refs = {(m, e): consensus_without(T, m, e) for m, e in T[["mouse", "is_eeg"]].drop_duplicates().to_numpy()}
    can, prof = [], []
    for _, r in T.iterrows():
        ref = refs[(r.mouse, r.is_eeg)]
        if len(r.U) == 0 or ref is None:
            can.append(np.nan); prof.append(np.full(10, np.nan)); continue
        U = r.U / np.linalg.norm(r.U, axis=1, keepdims=True)
        c = U @ ref
        k = int(np.argmax(np.abs(c)))
        can.append(float(abs(c[k]))); prof.append(U[k] * np.sign(c[k]))
    T = T.copy()
    T["canonical"] = can
    for b in range(10):
        T[f"band{b}"] = [p[b] for p in prof]
    return T, refs


def pair_sim(A, B):
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return float(np.abs(A @ B.T).max())


def distance_test(T, rng, n_perm=1000):
    """Within each probe: Spearman r of pair similarity vs depth distance; null permutes depths within probe."""
    out = []
    for (s, a), P in T[(~T.is_eeg) & (T.n_motifs > 0)].groupby(["session", "array"]):
        if len(P) < 5:
            continue
        L, z = P.U.to_list(), P.depth_um.to_numpy()
        iu = np.triu_indices(len(P), 1)
        S = np.array([[pair_sim(L[i], L[j]) for j in range(len(P))] for i in range(len(P))])[iu]
        def r_of(zz):
            D = np.abs(zz[:, None] - zz[None])[iu]
            return pd.Series(S).rank().corr(pd.Series(D).rank())
        real = r_of(z)
        null = np.array([r_of(rng.permutation(z)) for _ in range(n_perm)])
        D = np.abs(z[:, None] - z[None])[iu]
        out.append({"session": s, "mouse": s[:6], "array": a, "n_channels": len(P), "r": real, "null": null,
                    "sim_near": float(S[D <= 400].mean()) if (D <= 400).any() else np.nan,
                    "sim_far": float(S[D >= 1600].mean()) if (D >= 1600).any() else np.nan})
    return pd.DataFrame(out)


def pooled(Dt, mice):
    P = Dt[Dt.mouse.isin(mice)]
    if P.empty:
        return {"probes": 0, "mean_r": np.nan, "p_lower": np.nan}
    real = P.r.mean()
    null = np.stack(P.null.to_numpy()).mean(0)
    return {"probes": len(P), "mean_r": float(real), "p_lower": float((1 + (null <= real).sum()) / (len(null) + 1)),
            "near_minus_far": float((P.sim_near - P.sim_far).mean())}
