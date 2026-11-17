"""Step 0 of the spatial program: band envelopes for many channels per probe, and for every EEG channel.

For each probe: every STRIDE-th valid LFP channel (sorted by depth; ~40 um apart with STRIDE = 4, since adjacent
channels are nearly identical). For the EEG: every valid electrode. Only awake stimulation-free seconds and
stimulation-free isoflurane maintenance are kept (same definitions as awake_grounding and iso_state). The probe is
read once in time chunks (all channels at a time), anti-alias decimated to 250 Hz, then each channel goes through the
Mishler envelope (smooth_env, report bands) and is sampled at 1 Hz.

Output per probe / EEG array: one .npz with
  t (s), state (0 awake, 1 isoflurane), piece (contiguous interval id),
  env (seconds, channels, bands) float32, not z-scored,
  channel (index into the series), depth_um (probe vertical position; -1 for EEG), region, ccf (x, y, z).
"""

from pathlib import Path

import numpy as np
from scipy import signal

from . import awake_grounding as ag
from . import iso_state as iso
from . import session_data as sd
from .smooth_env import smooth_env
from .state_motifs import BANDS, FS

STRIDE = 4
CHUNK_S = 120.0
PAD_S = 2.0


def intervals_of(nwb, path):
    aw, _ = ag.awake_intervals(nwb)
    out = [(a, b, 0) for a, b in aw]
    if "isoflurane_level" in nwb.acquisition:
        info = sd.read_info(nwb, str(path))
        out += [(a, b, 1) for a, b in iso.iso_intervals(info)]
    return out


def pick_channels(el, stride=STRIDE, eeg=False):
    """Column positions (within the series' data) of the channels to keep: every valid EEG electrode, or every
    stride-th valid probe channel ordered by depth. el must be in column order (series.electrodes.to_dataframe())."""
    pos = np.arange(len(el))
    valid = el.is_data_valid.to_numpy(bool)
    if eeg:
        return pos[valid]
    order = np.argsort(el.probe_vertical_position.to_numpy(float)[valid], kind="stable")
    return pos[valid][order][::stride]


def decimated(series, chans, a, b, fs_raw, q):
    """Channels `chans` of `series` between times a and b, decimated by q (chunked, padded, zero-phase)."""
    ts = series.timestamps
    i0, i1 = int(np.searchsorted(ts, a)), int(np.searchsorted(ts, b))
    pad = int(PAD_S * fs_raw)
    step = int(CHUNK_S * fs_raw)
    order = np.argsort(chans)
    inv = np.argsort(order)
    sc = np.asarray(chans)[order]
    T, X = [], []
    for s in range(i0, i1, step):
        e = min(s + step, i1)
        lo, hi = max(s - pad, 0), min(e + pad, len(ts))
        block = series.data[lo:hi, sc[0]: sc[-1] + 1][:, sc - sc[0]].astype(np.float32) * series.conversion * 1e6
        block = block[:, inv]
        dec = signal.decimate(block, q, axis=0, zero_phase=True).astype(np.float32)
        keep = slice((s - lo) // q, (s - lo) // q + (e - s) // q)
        X.append(dec[keep]); T.append(ts[s:e:q][: dec[keep].shape[0]])
    T = np.concatenate(T); X = np.concatenate(X)
    n = min(len(T), len(X))
    return T[:n], X[:n]


def rereference(X, mode):
    """Local re-referencing of decimated traces (time, channels ordered by depth / EEG electrodes).
    csd: second spatial difference along the probe (x[j-1] - 2 x[j] + x[j+1]); the end channels are dropped (NaN).
    car: subtract the mean of all electrodes (EEG)."""
    if mode == "csd":
        Y = np.full_like(X, np.nan)
        Y[:, 1:-1] = X[:, :-2] - 2 * X[:, 1:-1] + X[:, 2:]
        return Y
    if mode == "car":
        return X - X.mean(1, keepdims=True)
    return X


def extract(path, which, out_dir, mode="as recorded"):
    """which: probe name (e.g. 'probeF') or 'EEG'. mode: 'as recorded', or 'local' (CSD for probes, common average
    for the EEG)."""
    io, nwb = sd.open_nwb(path)
    ivs = intervals_of(nwb, path)
    series = nwb.acquisition["ElectricalSeriesEEG"] if which == "EEG" else nwb.acquisition["LFP"][f"ElectricalSeries{which}"]
    el = series.electrodes.to_dataframe()
    chans = pick_channels(el, eeg=(which == "EEG"))
    fs_raw = 1 / np.median(np.diff(series.timestamps[:100_000]))
    q = int(round(fs_raw / FS))
    fs = fs_raw / q
    hop = int(round(fs))
    Ts, St, Pc, Es = [], [], [], []
    for k, (a, b, st) in enumerate(ivs):
        t, X = decimated(series, chans, a, b, fs_raw, q)
        if mode == "local":
            X = rereference(X, "car" if which == "EEG" else "csd")
        nan_env = np.full((X.shape[0], len(BANDS)), np.nan)[::hop]
        envs = [smooth_env(X[:, j].astype(float), fs, BANDS, zscore=False)[::hop] if np.isfinite(X[:, j]).all()
                else nan_env for j in range(X.shape[1])]
        E = np.stack(envs, axis=1)                                  # (seconds, channels, bands)
        live = ~np.isnan(E).all(axis=(0, 2))                        # CSD end channels are all-NaN: ignore them
        ok = np.isfinite(E[:, live]).all(axis=(1, 2))
        Ts.append(t[::hop][: E.shape[0]][ok]); Es.append(E[ok].astype(np.float32))
        St.append(np.full(ok.sum(), st, np.int8)); Pc.append(np.full(ok.sum(), k, np.int16))
    sel = el.iloc[chans]
    depth = sel.probe_vertical_position.to_numpy(float) if "probe_vertical_position" in sel else np.full(len(chans), -1.0)
    out = Path(out_dir) / f"{Path(path).name[:15]}_{which}.npz"
    np.savez_compressed(out, t=np.concatenate(Ts), state=np.concatenate(St), piece=np.concatenate(Pc),
                        env=np.concatenate(Es), channel=np.asarray(chans), electrode_id=sel.index.to_numpy(),
                        depth_um=depth,
                        region=sel.location.to_numpy().astype(str), ccf=sel[["x", "y", "z"]].to_numpy(float))
    io.close()
    return out
