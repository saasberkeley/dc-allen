"""E1: is "pole separation <-> coupling strength" (R5) about the motif, or about how much LFP signal a window has?

Per signal and state, 2-min windows (as in relationships.py). For each direction v in band space:
  separation  = gap between the upper and lower halves of the unsplit projection Z @ v in the window;
  coupling, three ways (fast part x = residual @ v):
    corr      mean |r(x, neuron rate)| over neurons beating the window's shift threshold (as in R5);
    corr_all  mean |r(x, neuron rate)| over all neurons (no threshold);
    slope     mean |slope| of each neuron's z-scored rate on x in x's own units (x is NOT re-standardized per
              window): unlike a correlation, a slope does not grow just because x varies more.
  relationship = partial Spearman r across windows (covariates partialled out).
Directions: the motif, band PC1, broadband, and N_RANDOM random directions.
Spike-shift control: every neuron's rate is circularly shifted (>= 60 s) so it has no true relation to the LFP; any
remaining separation <-> coupling relation is the pure "more LFP variance" effect.
"""
import numpy as np
import pandas as pd

from . import relationships as rl

N_RANDOM = 50


def unit(v):
    return v / np.linalg.norm(v)


def window_measures(sep_src, x, Zr, rng, n_null=30):
    rows = []
    for w in rl.windows(len(x)):
        U = sep_src[w]; med = np.median(U)
        sep = U[U > med].mean() - U[U <= med].mean()
        xw, Rw = x[w], Zr[w]
        xc = xw - xw.mean()
        Rc = Rw - Rw.mean(0)
        r = (Rc.T @ xc) / (np.sqrt((Rc ** 2).sum(0) * (xc @ xc)) + 1e-12)
        slope = (Rc.T @ xc) / (xc @ xc + 1e-12)
        null = np.concatenate([(Rc.T @ np.roll(xc, rng.integers(20, len(xc) - 20))) /
                               (np.sqrt((Rc ** 2).sum(0) * (xc @ xc)) + 1e-12) for _ in range(n_null)])
        mem = np.abs(r) > np.nanpercentile(np.abs(null), 95)
        rows.append({"separation": sep, "corr": np.abs(r[mem]).mean() if mem.any() else np.nan,
                     "corr_all": np.nanmean(np.abs(r)), "slope": np.nanmean(np.abs(slope))})
    return pd.DataFrame(rows)


def window_cov(d):
    return np.array([d["cov"][w].mean(0) for w in rl.windows(len(d["t"]))])


def partial_spearman(a, b, C):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 6:
        return np.nan
    cov = np.column_stack([pd.Series(c).rank().to_numpy() for c in C[ok].T])
    ar = rl.residualize(pd.Series(a[ok]).rank().to_numpy(), cov)
    br = rl.residualize(pd.Series(b[ok]).rank().to_numpy(), cov)
    return rl._rsmall(ar, br)


def run(d, awake_Z, seed=0):
    """d: an assembled signal (awake or isoflurane). awake_Z: awake band z-scores (for band PC1)."""
    rng = np.random.default_rng(seed)
    Zr = rl.zs(d["rate"])
    C = window_cov(d)
    nb = d["Z"].shape[1]
    w, V = np.linalg.eigh(np.cov(awake_Z.T))
    dirs = {"motif": d["u"], "band PC1": V[:, -1], "broadband": unit(np.ones(nb))}
    dirs.update({f"random {i}": unit(rng.standard_normal(nb)) for i in range(N_RANDOM)})
    shifted = np.column_stack([np.roll(Zr[:, i], rng.integers(60, len(Zr) - 60)) for i in range(Zr.shape[1])])
    rows = []
    for name, v in dirs.items():
        x, sep_src = d["resid"] @ v, d["Z"] @ v
        for spikes, R in (("real", Zr), ("shifted (no true coupling)", shifted)):
            if spikes != "real" and name.startswith("random"):
                continue
            W = window_measures(sep_src, x, R, rng)
            rows.append({"direction": name, "spikes": spikes, "abs_cos_motif": float(abs(v @ d["u"])),
                         **{f"sep~{m}": partial_spearman(W.separation.to_numpy(), W[m].to_numpy(), C)
                            for m in ("corr", "corr_all", "slope")}})
    return pd.DataFrame(rows).assign(session=d["meta"]["session"], region=d["meta"]["region"])
