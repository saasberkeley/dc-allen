"""S-D and S-E: where do the tracker's state changes happen, and do EEG changes track the neurons beneath?

S-D, per channel (awake z reference; sliding baseline within pieces; reference = consensus awake motif from the
other mice):
  baseline shift      mean baseline under isoflurane minus awake (10-band vector): size, and its projection on the
                      motif and on broadband
  pole separation     gap between the upper and lower halves of the unsplit projection (z . motif), isoflurane / awake
  spread              SD of the fast signal (residual . motif), isoflurane / awake
Spatial structure is tested for reproducibility across mice: region (and cortical layer) means in the discovery mice
vs the confirmation mice, Spearman r over the regions (layers) both groups have, against a null that permutes region
labels across channels within each mouse.
S-E, per EEG electrode (common-average referenced): the same state changes, against the neural change beneath it
(units within 1 mm horizontally): median log2 firing-rate ratio isoflurane / awake, and the share of those units that
join or leave the local motif's ensembles (S-C). Spearman r across electrodes within each mouse, then averaged; null =
electrode positions permuted within mouse.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from . import channel_tracker as ct
from . import position_pole as pp

REPO = Path(__file__).resolve().parents[3]


def channel_changes(npz, ref):
    d = np.load(npz, allow_pickle=True)
    t, st, piece, env = d["t"], d["state"], d["piece"], d["env"].astype(float)
    if not (st == 1).any():
        return pd.DataFrame()
    live = ~np.isnan(env).all(axis=(0, 2))
    bb = np.ones(env.shape[2]) / np.sqrt(env.shape[2])
    rows = []
    for j in np.flatnonzero(live):
        Z, base, resid = ct.signal_of(t, env[:, j], piece, st == 0)
        ok = np.isfinite(resid).all(1)
        a, i = ok & (st == 0), ok & (st == 1)
        if a.sum() < 120 or i.sum() < 120:
            continue
        shift = base[i].mean(0) - base[a].mean(0)
        U = Z @ ref
        sep = lambda m: U[m][U[m] > np.median(U[m])].mean() - U[m][U[m] <= np.median(U[m])].mean()
        E = resid @ ref
        rows.append({"array": Path(npz).stem.split("_")[1], "channel": int(d["channel"][j]), "depth_um": float(d["depth_um"][j]),
                     "region": str(d["region"][j]), "ccf_x": d["ccf"][j][0], "ccf_y": d["ccf"][j][1], "ccf_z": d["ccf"][j][2],
                     "shift_size": float(np.linalg.norm(shift)), "shift_on_motif": float(shift @ ref),
                     "shift_on_broadband": float(shift @ bb), "separation_ratio": float(sep(i) / sep(a)),
                     "spread_ratio": float(E[i].std() / E[a].std())})
    return pd.DataFrame(rows)


def reproducibility(C, by, measure, groups, rng, n_perm=1000, min_channels=3):
    """Spearman r between group means per `by` (region or layer) in discovery vs confirmation mice; null permutes the
    `by` labels across channels within each mouse."""
    def means(D):
        g = D.groupby(by)[measure].agg(["mean", "size"])
        return g["mean"][g["size"] >= min_channels]

    def stat(D):
        a, b = means(D[D.mouse.isin(groups["discovery"])]), means(D[D.mouse.isin(groups["confirmation"])])
        k = a.index.intersection(b.index)
        if len(k) < 4:
            return np.nan, len(k)
        return float(pd.Series(a[k]).rank().corr(pd.Series(b[k]).rank())), len(k)

    real, n = stat(C)
    null = []
    for _ in range(n_perm):
        P = C.copy()
        P[by] = P.groupby("mouse")[by].transform(lambda v: rng.permutation(v.to_numpy()))
        null.append(stat(P)[0])
    null = np.array(null, float)
    return {"measure": measure, "by": by, "shared": n, "spearman_r": real,
            "p": float((1 + np.sum(null[np.isfinite(null)] >= real)) / (np.isfinite(null).sum() + 1)) if np.isfinite(real) else np.nan}
