"""E3: which neurons join or leave the motif's ensembles between awake and isoflurane?

Per neuron (population mean removed, equal-length comparison from iso_state.cross_state): awake label and
isoflurane label (+, -, none) give a category: stayed, flipped, left (coupled awake, not under isoflurane), joined
(not coupled awake, coupled under isoflurane), never. Two comparisons:
  left vs stayed   among neurons coupled awake
  joined vs never  among neurons not coupled awake
Predictors (standardized within signal): narrow waveform (putative fast-spiking interneuron), awake rate (log),
rate change (log2 isoflurane / awake), awake |coupling|, population coupling, CV2, and relative depth within the
region (0 = shallowest unit, 1 = deepest).
Test: per predictor, mean difference between the two groups pooled over signals; null = category labels permuted
within each signal (keeps group sizes per signal). Depth is a spatial predictor, so the spatial mouse split applies.
"""
import numpy as np
import pandas as pd

PREDICTORS = ["narrow", "log_rate_awake", "log_rate_ratio", "abs_r_awake", "pop_coupling", "cv2", "rel_depth"]


def table(per_neuron, units):
    """per_neuron: pooled iso cross-state table (with sig, unit_id); units: Phase 1 unit descriptors."""
    P = per_neuron.copy()
    P["session"], P["region"] = P.sig.str[:15], P.sig.str[16:].str.replace("-", "/")
    U = units.copy()
    P = P.merge(U[["session", "region", "unit_id", "narrow", "pop_coupling", "cv2", "depth_um", "rate_hz"]],
                on=["session", "region", "unit_id"], how="left")
    a, i = P["label_awake (population removed)"], P["label_iso (population removed)"]
    P["category"] = np.select([(a != 0) & (i == 0), (a == 0) & (i != 0), (a != 0) & (i != 0) & (np.sign(a) != np.sign(i)),
                               (a != 0) & (i != 0)], ["left", "joined", "flipped", "stayed"], "never")
    P["log_rate_awake"] = np.log10(P.rate_awake + 0.05)
    P["abs_r_awake"] = P["r_awake (population removed)"].abs()
    P["narrow"] = P.narrow.astype(float)
    P["rel_depth"] = P.groupby("sig").depth_um.transform(lambda d: (d - d.min()) / (d.max() - d.min() + 1e-9))
    return P


def compare(P, a, b, n_perm=2000, seed=0):
    """Mean of each standardized predictor in group a minus group b, pooled over signals; within-signal permutation."""
    rng = np.random.default_rng(seed)
    Q = P[P.category.isin([a, b])].copy()
    for c in PREDICTORS:
        Q[c] = Q.groupby("sig")[c].transform(lambda v: (v - v.mean()) / (v.std() + 1e-12))
    Q = Q.dropna(subset=PREDICTORS)
    lab = (Q.category == a).to_numpy()
    groups = [np.flatnonzero((Q.sig == s).to_numpy()) for s in Q.sig.unique()]
    X = Q[PREDICTORS].to_numpy()

    def diff(L):
        return X[L].mean(0) - X[~L].mean(0)

    real = diff(lab)
    null = []
    for _ in range(n_perm):
        L = lab.copy()
        for g in groups:
            L[g] = rng.permutation(L[g])
        null.append(diff(L))
    null = np.array(null)
    p = (1 + (np.abs(null) >= np.abs(real)).sum(0)) / (n_perm + 1)
    per_mouse = {m: diff_m for m, diff_m in
                 ((m, X[(Q.sig.str[:6] == m).to_numpy() & lab].mean(0) - X[(Q.sig.str[:6] == m).to_numpy() & ~lab].mean(0))
                  for m in sorted(Q.sig.str[:6].unique()))}
    return pd.DataFrame({"predictor": PREDICTORS, "difference (SD)": real, "p": p,
                         **{f"mouse {m}": v for m, v in per_mouse.items()}}).assign(comparison=f"{a} vs {b}",
                                                                                    n_a=int(lab.sum()), n_b=int((~lab).sum()))
