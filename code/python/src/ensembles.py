"""Phase 1 of the state <-> ensembles program: defining neurons, firing activity and ensembles (awake reference).

Everything here uses awake, stimulation-free seconds only (the same seconds as awake_grounding), so it can serve as
the reference that the isoflurane analysis later reuses unchanged.

Units
  Allen quality filters (in session_data.good_units), then
  - rate floor: at least MIN_RATE Hz over the awake seconds;
  - duplicates: pairs on nearby channels (<= DUP_UM apart) whose spikes coincide within +-0.5 ms far above chance
    and whose mean waveforms match are probably one neuron split in two; the lower-rate unit is dropped.
Firing descriptors per unit
  rate, CV2 (local spike-time irregularity; high = bursty), Fano factor of 1-s counts, population coupling (r with the
  summed rate of all other units), waveform duration (narrow < 0.4 ms = putative fast-spiking interneuron), depth.
Ensembles, three independent definitions; every method may leave a neuron unassigned
  lfp      each neuron's correlation with a motif's expression, tested against a circular-shift null (same shifts
           for every neuron), Benjamini-Hochberg across neurons: "+", "-" or "none"
  getclust Mishler's getClust on spiking alone; neurons in no cluster are unassigned
  corr     community detection on the significant positive correlations between neurons (population mean removed;
           null = each neuron independently circularly shifted), modularity by Newman's leading-eigenvector method;
           communities of >= 3 neurons are ensembles, all other neurons unassigned
Validation
  split-half reliability (even vs odd 60-s blocks) of each method = the noise ceiling for "membership changed";
  agreement between methods; both against label-permutation nulls.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from . import awake_grounding as ag
from . import session_data as sd
from .get_clust import get_clust, preprocess

MIN_RATE = 0.5          # Hz
DUP_UM = 60.0           # um along the probe
DUP_SYNC = 0.2          # share of the lower-rate unit's spikes with a partner spike within +-0.5 ms
DUP_WAVE_R = 0.9
NARROW_MS = 0.4
BLOCK = 60


# ---------------------------------------------------------------- units and descriptors

def unit_table(nwb, region):
    """Good units of a region with spike trains, depth (um along the probe), waveform and duration."""
    trains, ids, _ = sd.good_units(nwb, region)
    u = nwb.units.to_dataframe().loc[ids]
    el = nwb.electrodes.to_dataframe()
    eid = u.electrodes.map(lambda e: e.index[0] if hasattr(e, "index") else int(np.atleast_1d(e)[0]))
    wf = np.stack([np.asarray(w, float).ravel()[:82] for w in u.waveform_mean])
    return pd.DataFrame({"unit_id": ids, "depth_um": el.loc[eid.values, "probe_vertical_position"].to_numpy(float),
                         "waveform_ms": u.waveform_duration.to_numpy(float), "train": trains,
                         "waveform": list(wf)})


def in_intervals(s, intervals):
    m = np.zeros(s.size, bool)
    for a, b in intervals:
        m |= (s >= a) & (s < b)
    return s[m]


def duplicates(U, intervals, window=0.0005):
    """Pairs that look like one neuron split in two. Returns (rows, set of unit_ids to drop)."""
    trains = [in_intervals(s, intervals) for s in U.train]
    rows, drop = [], set()
    for i in range(len(U)):
        for j in range(i + 1, len(U)):
            if abs(U.depth_um.iat[i] - U.depth_um.iat[j]) > DUP_UM:
                continue
            a, b = trains[i], trains[j]
            if a.size == 0 or b.size == 0:
                continue
            lo, hi = (a, b) if a.size <= b.size else (b, a)
            k = np.searchsorted(hi, lo)
            near = np.minimum(np.abs(lo - hi[np.clip(k - 1, 0, hi.size - 1)]), np.abs(lo - hi[np.clip(k, 0, hi.size - 1)]))
            sync = float(np.mean(near <= window))
            wr = float(np.corrcoef(U.waveform.iat[i], U.waveform.iat[j])[0, 1])
            if sync >= DUP_SYNC and wr >= DUP_WAVE_R:
                lower = U.unit_id.iat[i] if a.size < b.size else U.unit_id.iat[j]
                drop.add(lower)
                rows.append({"a": U.unit_id.iat[i], "b": U.unit_id.iat[j], "sync": sync, "waveform_r": wr, "dropped": lower})
    return rows, drop


def binned(trains, t, piece, sigma=1.0):
    counts = np.stack([np.searchsorted(s, t + 0.5) - np.searchsorted(s, t - 0.5) for s in trains], 1).astype(float)
    rate = np.empty_like(counts)
    for p in np.unique(piece):
        m = piece == p
        rate[m] = gaussian_filter1d(counts[m], sigma, axis=0, mode="nearest")
    return counts, rate


def cv2(s):
    isi = np.diff(s)
    if isi.size < 3:
        return np.nan
    return float(np.mean(2 * np.abs(isi[1:] - isi[:-1]) / (isi[1:] + isi[:-1])))


def descriptors(U, counts, rate, intervals):
    total = rate.sum(1)
    out = []
    for k in range(len(U)):
        s = in_intervals(U.train.iat[k], intervals)
        c = counts[:, k]
        others = total - rate[:, k]
        out.append({"unit_id": U.unit_id.iat[k], "rate_hz": float(c.mean()), "cv2": cv2(s),
                    "fano": float(c.var() / c.mean()) if c.mean() > 0 else np.nan,
                    "pop_coupling": ag._r(rate[:, k], others), "waveform_ms": U.waveform_ms.iat[k],
                    "narrow": bool(U.waveform_ms.iat[k] < NARROW_MS), "depth_um": U.depth_um.iat[k]})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- ensemble definitions

def bh(p, q=0.05):
    """Benjamini-Hochberg: boolean mask of discoveries."""
    p = np.asarray(p, float)
    n = np.sum(np.isfinite(p))
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    passed = np.zeros(p.size, bool)
    ranked = p[order]
    ok = ranked <= q * (np.arange(1, p.size + 1) / max(n, 1))
    if ok.any():
        passed[order[: np.max(np.flatnonzero(ok)) + 1]] = True
    return passed & np.isfinite(p)


def lfp_membership(x, R, rng, min_shift, n_shift=500, q=0.05):
    """Per neuron: r(x, rate), two-sided p against circular shifts of x (same shifts for all neurons), BH across
    neurons. Labels +1 / -1 / 0."""
    Rz = (R - R.mean(0)) / (R.std(0) + 1e-12)
    xz = (x - x.mean()) / x.std()
    r = Rz.T @ xz / len(x)
    null = np.array([Rz.T @ np.roll(xz, s) / len(x) for s in ag.shift_values(len(x), rng, n_shift, min_shift)])
    p = (1 + (np.abs(null) >= np.abs(r)).sum(0)) / (n_shift + 1)
    sig = bh(p, q)
    return np.where(sig, np.sign(r), 0).astype(int), r, p


def getclust_membership(R, seed=0):
    """Cluster index (1..k) per neuron, 0 = in no cluster. Overlapping memberships go to the first cluster."""
    clusters = get_clust(preprocess(R), 1.0, seed=seed)
    lab = np.zeros(R.shape[1], int)
    for k, c in enumerate(clusters):
        lab[[i for i in c if lab[i] == 0]] = k + 1
    return lab


def _leading_eigen_split(B, idx):
    """Newman 2006: split the group idx by the sign of the leading eigenvector of its modularity matrix."""
    Bg = B[np.ix_(idx, idx)].copy()
    Bg[np.diag_indices_from(Bg)] -= Bg.sum(1)
    w, V = np.linalg.eigh(Bg)
    if w[-1] <= 1e-10:
        return None
    s = np.where(V[:, -1] >= 0, 1, -1)
    if abs(s.sum()) == s.size:
        return None
    dq = s @ Bg @ s
    if dq <= 1e-10:
        return None
    return idx[s > 0], idx[s < 0]


def modularity_communities(A):
    """Communities of a weighted undirected graph by recursive leading-eigenvector bisection."""
    k = A.sum(1)
    m2 = k.sum()
    if m2 == 0:
        return []
    B = A - np.outer(k, k) / m2
    todo, done = [np.flatnonzero(k > 0)], []
    while todo:
        g = todo.pop()
        sp = _leading_eigen_split(B, g) if g.size >= 2 else None
        if sp is None:
            done.append(g)
        else:
            todo += list(sp)
    return done


def corr_membership(R, rng, n_shift=100, q=0.05, min_size=3):
    """Community index (1..k) per neuron, 0 = unassigned. Edges = positive correlations of population-removed rates
    that beat a null in which every neuron is independently circularly shifted (pooled over pairs), BH-corrected."""
    Z = ag.remove_global(R)
    n, N = Z.shape
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    C = Zs.T @ Zs / n
    iu = np.triu_indices(N, 1)
    null = []
    for _ in range(n_shift):
        sh = np.column_stack([np.roll(Zs[:, i], rng.integers(60, n - 60)) for i in range(N)])
        null.append((sh.T @ sh / n)[iu])
    null = np.sort(np.concatenate(null))
    p = 1 - np.searchsorted(null, C[iu]) / null.size
    sig = bh(np.maximum(p, 1 / null.size), q) & (C[iu] > 0)
    A = np.zeros((N, N))
    A[iu[0][sig], iu[1][sig]] = C[iu][sig]
    A = A + A.T
    lab = np.zeros(N, int)
    k = 0
    for g in modularity_communities(A):
        if g.size >= min_size:
            k += 1
            lab[g] = k
    return lab, int(sig.sum()), float(sig.mean()) if sig.size else np.nan


# ---------------------------------------------------------------- agreement

def ari(a, b):
    """Adjusted Rand index between two labelings (0 counts as its own label)."""
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a, b))


def assigned_ari(a, b):
    """ARI on neurons assigned by both methods (NaN if fewer than 4)."""
    m = (a != 0) & (b != 0)
    return ari(a[m], b[m]) if m.sum() >= 4 and len(np.unique(a[m])) > 1 and len(np.unique(b[m])) > 1 else np.nan


def kappa_assigned(a, b):
    """Cohen's kappa for 'assigned vs unassigned' between two labelings."""
    from sklearn.metrics import cohen_kappa_score
    x, y = a != 0, b != 0
    if x.all() == x.any() and y.all() == y.any() and x.all() == y.all():
        return np.nan
    return float(cohen_kappa_score(x, y))


def agreement(a, b, rng, n_perm=500):
    """ARI and its permutation null (labels of b shuffled across neurons)."""
    real = ari(a, b)
    null = np.array([ari(a, rng.permutation(b)) for _ in range(n_perm)])
    return real, float(np.mean(null)), float((1 + np.sum(null >= real)) / (n_perm + 1))


# ---------------------------------------------------------------- one signal

def run_signal(path, r, seed=0):
    """r: the signal's awake_grounding result (seconds, motif outputs, min_shift)."""
    rng = np.random.default_rng(seed)
    m = r["meta"]
    io, nwb = sd.open_nwb(path)
    intervals, _ = ag.awake_intervals(nwb)
    U = unit_table(nwb, m["region"])
    io.close()
    t = r["t"]
    piece = np.r_[0, np.cumsum(np.diff(t) > 1.5)]
    counts, rate = binned(list(U.train), t, piece)
    dup_rows, drop = duplicates(U, intervals)
    keep = (counts.mean(0) >= MIN_RATE) & ~U.unit_id.isin(drop).to_numpy()
    qc = {"n_good": len(U), "n_low_rate": int((counts.mean(0) < MIN_RATE).sum()), "n_duplicates": len(drop),
          "n_kept": int(keep.sum())}
    U, counts, rate = U[keep].reset_index(drop=True), counts[:, keep], rate[:, keep]
    D = descriptors(U, counts, rate, intervals)

    half = (np.arange(len(t)) // BLOCK) % 2 == 0
    readouts = {}
    if r["outputs"].get("original") is not None and r["outputs"]["original"].shape[1]:
        readouts["original M1"] = r["outputs"]["original"][:, 0]
    if r["outputs"].get("tracker expression") is not None and r["outputs"]["tracker expression"].shape[1]:
        readouts["tracker expression M1"] = r["outputs"]["tracker expression"][:, 0]
    Rglob = ag.remove_global(rate)

    labels, extra = {}, {}
    for name, x in readouts.items():
        for kind, RR in (("raw", rate), ("population removed", Rglob)):
            lab, rr, pp = lfp_membership(x, RR, rng, m["min_shift"])
            labels[f"lfp {name} ({kind})"] = lab
            extra[f"lfp {name} ({kind})"] = (rr, pp)
    labels["getclust"] = getclust_membership(rate, seed)
    labels["corr"], n_edges, frac_edges = corr_membership(rate, rng)

    # split halves: each method rerun on even and odd 60-s blocks
    rel = []
    for name, lab in labels.items():
        halves = []
        for h in (half, ~half):
            if name.startswith("lfp"):
                x = readouts[name.split("lfp ")[1].split(" (")[0]]
                RR = rate if "(raw)" in name else Rglob
                halves.append(lfp_membership(x[h], RR[h], rng, min(m["min_shift"], h.sum() // 3))[0])
            elif name == "getclust":
                halves.append(getclust_membership(rate[h], seed))
            else:
                halves.append(corr_membership(rate[h], rng)[0])
        rel.append({"method": name, "n_assigned": int((lab != 0).sum()), "share_assigned": float(np.mean(lab != 0)),
                    "n_groups": int(len(set(lab[lab != 0]))),
                    "half_ari": ari(*halves), "half_ari_assigned": assigned_ari(*halves),
                    "half_kappa_assigned": kappa_assigned(*halves),
                    "half_share_assigned": [float(np.mean(h_ != 0)) for h_ in halves]})
    names = list(labels)
    agr = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            real, null, p = agreement(labels[names[i]], labels[names[j]], rng)
            agr.append({"a": names[i], "b": names[j], "ari": real, "null_mean": null, "p": p,
                        "ari_assigned": assigned_ari(labels[names[i]], labels[names[j]])})
    key = {"session": m["session"], "region": m["region"]}
    return {"meta": {**key, **qc, "n_seconds": len(t), "corr_edges": n_edges, "corr_edge_share": frac_edges},
            "units": D.assign(**key), "labels": pd.DataFrame(labels).assign(unit_id=U.unit_id.to_numpy(), **key),
            "lfp_stats": {k: {"r": v[0], "p": v[1]} for k, v in extra.items()},
            "reliability": pd.DataFrame(rel).assign(**key), "agreement": pd.DataFrame(agr).assign(**key),
            "duplicates": pd.DataFrame(dup_rows).assign(**key), "rate": rate, "t": t}
