"""S-A: how local is the coupling between LFP/EEG tracker signals and single neurons, and does isoflurane change it?

Per session and state (awake, isoflurane), from the multichannel envelopes (every ~40 um on every probe, every EEG
electrode) and every well-isolated grey-matter unit of the session (>= 0.5 Hz awake):
  signals per channel (awake z reference; sliding baseline within pieces):
    fast    residual projected on the reference motif (the cross-mouse consensus awake LFP motif, learned without
            this mouse, so every channel, probe and mouse uses the same direction)
    fast<30 the same with bands up to 30 Hz only (no 35-80 Hz bands, which can carry spike-waveform leakage)
    slow    baseline projected on the reference motif
    pc1     residual projected on the channel array's awake band PC1
  coupling = |r| between a unit's rate (z-scored; also with the population mean of all units removed) and a channel
  signal, per state;
  distance: same probe = difference in probe depth (um); other probe / EEG = CCF distance (voxels of 25 um).
  null = the same with the channel signals circularly shifted (>= 5 min), so autocorrelation is kept.
  reference curve = |r| between pairs of channels' fast signals vs distance (how fast the LFP itself decorrelates).
Output: binned coupling (real and null) per distance bin and kind of pair.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from . import awake_grounding as ag
from . import channel_tracker as ct
from . import session_data as sd

PROBE_BINS = [0, 50, 100, 200, 400, 800, 1600, 4000]
CCF_BINS_MM = [0, 1, 2, 3, 4, 6, 10]
LOW = 7                    # bands 0..6 end at <= 30 Hz


def units_of(nwb, min_rate_t=None):
    u = nwb.units.to_dataframe()
    good = ((u.isi_violations < sd.GOOD_UNIT["isi_violations"]) & (u.amplitude_cutoff < sd.GOOD_UNIT["amplitude_cutoff"])
            & (u.presence_ratio > sd.GOOD_UNIT["presence_ratio"]) & u.location.map(ag.is_grey))
    u = u[good]
    el = nwb.electrodes.to_dataframe()
    eid = u.electrodes.map(lambda e: e.index[0] if hasattr(e, "index") else int(np.atleast_1d(e)[0]))
    return pd.DataFrame({"unit_id": u.index, "region": u.location.to_numpy(), "train": [np.asarray(s) for s in u.spike_times],
                         "probe": el.loc[eid.values, "group_name"].to_numpy().astype(str),
                         "depth_um": el.loc[eid.values, "probe_vertical_position"].to_numpy(float),
                         "ccf": list(u[["x", "y", "z"]].to_numpy(float)),
                         "waveform_ms": u.waveform_duration.to_numpy(float)})


def array_signals(npz, ref, state):
    d = np.load(npz, allow_pickle=True)
    t, st, piece, env = d["t"], d["state"], d["piece"], d["env"].astype(float)
    live = ~np.isnan(env).all(axis=(0, 2))           # CSD end channels (local referencing) are all-NaN
    env = env[:, live]
    sec = np.round(t).astype(int)
    out = {k: [] for k in ("fast", "fast<30", "slow", "pc1")}
    pc1 = None
    for j in range(env.shape[1]):
        Z, base, resid = ct.signal_of(t, env[:, j], piece, st == 0)
        if pc1 is None:
            aw = np.isfinite(resid).all(1) & (st == 0)
            pc1 = np.linalg.eigh(np.cov(Z[aw].T))[1][:, -1]
        out["fast"].append(resid @ ref)
        lo = ref[:LOW] / np.linalg.norm(ref[:LOW])
        out["fast<30"].append(resid[:, :LOW] @ lo)
        out["slow"].append(base @ ref)
        out["pc1"].append(resid @ pc1)
    m = st == state
    return {"sec": sec[m], **{k: np.array(v).T[m] for k, v in out.items()}, "depth_um": d["depth_um"][live],
            "ccf": d["ccf"][live], "region": d["region"][live], "name": Path(npz).stem.split("_")[1]}


def run_session(session, ref, state, folder="multichannel"):
    REPO = Path(__file__).resolve().parents[3]
    path = glob.glob(str(REPO / f"data/raw/{session}*.nwb"))[0]
    io, nwb = sd.open_nwb(path)
    U = units_of(nwb)
    io.close()
    arrays = [array_signals(f, ref, state) for f in sorted(glob.glob(str(REPO / f"data/processed/{folder}/{session}_*.npz")))]
    common = set(arrays[0]["sec"])
    for a in arrays[1:]:
        common &= set(a["sec"])
    common = np.array(sorted(common))
    if len(common) < 300:
        return None
    for a in arrays:
        idx = np.searchsorted(a["sec"], common)
        for k in ("fast", "fast<30", "slow", "pc1"):
            a[k] = a[k][idx]
    ok = np.ones(len(common), bool)
    for a in arrays:
        ok &= np.isfinite(a["fast"]).all(1)
    common = common[ok]
    for a in arrays:
        for k in ("fast", "fast<30", "slow", "pc1"):
            a[k] = a[k][ok]
    piece = np.r_[0, np.cumsum(np.diff(common) > 1)]
    counts = np.stack([np.searchsorted(s, common + 0.5) - np.searchsorted(s, common - 0.5) for s in U.train], 1).astype(float)
    keep = counts.mean(0) >= 0.5 if state == 0 else np.ones(counts.shape[1], bool)
    rate = np.empty_like(counts)
    for p in np.unique(piece):
        k = piece == p
        rate[k] = gaussian_filter1d(counts[k], 1.0, axis=0, mode="nearest")
    return {"sec": common, "rate": rate, "keep": keep, "units": U.drop(columns="train"), "arrays": arrays}


def zcols(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-12)


def coupling_table(S, rng, n_shift=20, min_shift=300):
    """Binned |r| (real and shift null) of every unit with every channel, by kind of pair and distance."""
    U = S["units"][S["keep"]].reset_index(drop=True)
    R = S["rate"][:, S["keep"]]
    rates = {"raw": zcols(R), "population removed": zcols(ag.remove_global(R))}
    n = len(S["sec"])
    min_shift = min(min_shift, n // 3)
    shifts = rng.integers(min_shift, max(min_shift + 1, n - min_shift), n_shift)
    rows = []
    for a in S["arrays"]:
        is_eeg = a["name"] == "EEG"
        if is_eeg:
            dist = np.linalg.norm(np.stack(U.ccf)[:, None] - a["ccf"][None], axis=2) * 0.025
            kind = np.full(dist.shape, "EEG")
        else:
            same = np.broadcast_to(U.probe.to_numpy()[:, None] == a["name"], (len(U), len(a["depth_um"])))
            dprobe = np.abs(U.depth_um.to_numpy()[:, None] - a["depth_um"][None])
            dccf = np.linalg.norm(np.stack(U.ccf)[:, None] - a["ccf"][None], axis=2) * 0.025
            kind = np.where(same, "same probe", "other probe")
            dist = np.where(same, dprobe, dccf)
        for sig in ("fast", "fast<30", "slow", "pc1"):
            X = zcols(a[sig])
            for rk, Rz in rates.items():
                C = np.abs(Rz.T @ X / n)
                N = np.mean([np.abs(Rz.T @ np.roll(X, s, axis=0) / n) for s in shifts], axis=0)
                for kd in np.unique(kind):
                    m = kind == kd
                    bins = PROBE_BINS if kd == "same probe" else CCF_BINS_MM
                    b = np.digitize(dist[m], bins) - 1
                    for bi in range(len(bins) - 1):
                        sel = b == bi
                        if sel.sum() < 20:
                            continue
                        rows.append({"array": a["name"], "pair": kd, "signal": sig, "rates": rk,
                                     "bin": f"{bins[bi]}-{bins[bi + 1]}", "bin_lo": bins[bi], "n_pairs": int(sel.sum()),
                                     "abs_r": float(C[m][sel].mean()), "null_abs_r": float(N[m][sel].mean())})
    return pd.DataFrame(rows)


def lfp_decay(S):
    """|r| between channels' fast signals on the same probe vs depth difference (the LFP's own spatial reach)."""
    rows = []
    for a in S["arrays"]:
        if a["name"] == "EEG":
            continue
        X = zcols(a["fast"])
        C = np.abs(X.T @ X / len(X))
        D = np.abs(a["depth_um"][:, None] - a["depth_um"][None])
        iu = np.triu_indices(len(D), 1)
        b = np.digitize(D[iu], PROBE_BINS) - 1
        for bi in range(len(PROBE_BINS) - 1):
            sel = b == bi
            if sel.sum() >= 5:
                rows.append({"array": a["name"], "bin": f"{PROBE_BINS[bi]}-{PROBE_BINS[bi + 1]}", "bin_lo": PROBE_BINS[bi],
                             "lfp_abs_r": float(C[iu][sel].mean())})
    return pd.DataFrame(rows)
