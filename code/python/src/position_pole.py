"""S-C: does a neuron's position across the whole probe decide its pole and whether it joins / leaves across states?

Per session: every well-isolated grey-matter unit; its "local" motif signal = the fast signal (consensus motif
direction) of the nearest re-referenced (CSD) channel on its own probe. Per state, each unit's coupling r (population
mean removed) with that signal, tested against circular shifts (same shifts for all units), BH across units ->
label +, -, none. Across states: stayed / flipped / left / joined / never.
Position predictors: depth along the probe (um, deeper = larger), cortical layer (from the region name: 1, 2/3, 4, 5,
6a/6b), cortical vs subcortical, region.
Tests (pooled over probes, permutation of labels within each probe, so probe sampling is kept):
  pole ~ depth          mean depth of + units minus - units (within probe, standardized)
  pole ~ layer          share of + among coupled units per layer, spread across layers (chi-square-like statistic)
  turnover ~ depth / layer  the same for left vs stayed and joined vs never.
"""
import re

import numpy as np
import pandas as pd

from . import awake_grounding as ag
from . import ensembles as en
from . import locality as lc


def layer_of(region):
    m = re.match(r"^[A-Za-z-]+?(1|2/3|4|5|6a|6b)$", region)
    return m.group(1) if m else "subcortical"


def labels_for_state(S, rng, n_shift=200):
    U = S["units"][S["keep"]].reset_index(drop=True)
    R = ag.remove_global(S["rate"][:, S["keep"]])
    arrays = {a["name"]: a for a in S["arrays"] if a["name"] != "EEG"}
    n = len(S["sec"])
    x = np.full((n, len(U)), np.nan)
    for i, row in U.iterrows():
        a = arrays.get(row.probe)
        if a is None:
            continue
        j = int(np.argmin(np.abs(a["depth_um"] - row.depth_um)))
        x[:, i] = a["fast"][:, j]
    ok = np.isfinite(x).all(0)
    Rz = (R - R.mean(0)) / (R.std(0) + 1e-12)
    Xz = (x - np.nanmean(x, 0)) / (np.nanstd(x, 0) + 1e-12)
    r = np.where(ok, (Rz * np.nan_to_num(Xz)).mean(0), np.nan)
    ms = min(300, n // 3)
    null = np.array([np.where(ok, (Rz * np.roll(np.nan_to_num(Xz), s, axis=0)).mean(0), np.nan)
                     for s in rng.integers(ms, max(ms + 1, n - ms), n_shift)])
    p = (1 + (np.abs(null) >= np.abs(r)).sum(0)) / (n_shift + 1)
    p[~ok] = np.nan
    lab = np.where(en.bh(p), np.sign(r), 0).astype(int)
    return U.assign(r=r, p=p, label=lab)


def session_table(session, ref, rng):
    out = {}
    for st, name in ((0, "awake"), (1, "isoflurane")):
        S = lc.run_session(session, ref, st, "multichannel_local")
        if S is not None:
            out[name] = labels_for_state(S, rng)[["unit_id", "region", "probe", "depth_um", "r", "label"]]
    if "awake" not in out:
        return pd.DataFrame()
    T = out["awake"].rename(columns={"r": "r_awake", "label": "label_awake"})
    if "isoflurane" in out:
        T = T.merge(out["isoflurane"][["unit_id", "r", "label"]].rename(columns={"r": "r_iso", "label": "label_iso"}),
                    on="unit_id", how="left")
    T["layer"] = T.region.map(layer_of)
    T["session"] = session
    return T


def category(a, i):
    return np.select([(a != 0) & (i == 0), (a == 0) & (i != 0), (a != 0) & (i != 0) & (np.sign(a) != np.sign(i)),
                      (a != 0) & (i != 0)], ["left", "joined", "flipped", "stayed"], "never")


def depth_test(T, mask_a, mask_b, rng, n_perm=2000):
    """Mean within-probe standardized depth of group a minus group b; labels permuted within probe."""
    Q = T[mask_a | mask_b].copy()
    Q["zd"] = Q.groupby(["session", "probe"]).depth_um.transform(lambda d: (d - d.mean()) / (d.std(ddof=0) + 1e-9))
    Q = Q[Q.zd.notna()]
    lab = mask_a[Q.index].to_numpy()
    groups = [np.flatnonzero(((Q.session == s) & (Q.probe == p)).to_numpy()) for s, p in Q[["session", "probe"]].drop_duplicates().to_numpy()]
    z = Q.zd.to_numpy()
    real = z[lab].mean() - z[~lab].mean()
    null = []
    for _ in range(n_perm):
        L = lab.copy()
        for g in groups:
            L[g] = rng.permutation(L[g])
        null.append(z[L].mean() - z[~L].mean())
    null = np.array(null)
    return real, float((1 + (np.abs(null) >= abs(real)).sum()) / (n_perm + 1)), int(lab.sum()), int((~lab).sum())


def layer_test(T, mask_a, mask_b, rng, n_perm=2000):
    """Spread across cortical layers of the share of group a (vs b): variance of per-layer shares, weighted by
    count; labels permuted within probe."""
    Q = T[(mask_a | mask_b) & (T.layer != "subcortical")].copy()
    lab = mask_a[Q.index].to_numpy()
    lay = Q.layer.to_numpy()
    groups = [np.flatnonzero(((Q.session == s) & (Q.probe == p)).to_numpy()) for s, p in Q[["session", "probe"]].drop_duplicates().to_numpy()]

    def stat(L):
        tot = L.mean()
        return sum((lay == l).sum() * (L[lay == l].mean() - tot) ** 2 for l in np.unique(lay))

    real = stat(lab)
    null = []
    for _ in range(n_perm):
        L = lab.copy()
        for g in groups:
            L[g] = rng.permutation(L[g])
        null.append(stat(L))
    shares = {l: float(lab[lay == l].mean()) for l in sorted(np.unique(lay))}
    return real, float((1 + (np.array(null) >= real).sum()) / (n_perm + 1)), shares
