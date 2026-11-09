"""Cut short, time-aligned windows of EEG, cortex/hippocampus LFP and spikes.

For each NWB session this writes one HDF5 file per window to ``data/probe_samples/``:

    {session}_{window}.h5
        attrs   session, window, anchor, window_start_abs, common_start,
                aligned_end, fs, n_stim_in_window, ...
        eeg/    data (T, 30) float32 uV, ts (T,), location, is_data_valid, x, y, z
        lfp/    data (T, C) float32 uV (valid channels only), ts (T,), probe, location, electrode_id,
                probe_vertical_position
        spikes/ times (concatenated), index (end offset per unit), unit_id,
                location, probe, x, y, z, isi_violations, amplitude_cutoff, presence_ratio
        behavior/ running_speed/ts, pupil_width/ts, iso_level/ts (if present)

Timeline (Jonathan's time-shifting, from pythonNPExtractionAnes.ipynb):
    common_start = max(window_start, latest start time across all streams)
    every stream is clipped to [common_start, window_end) and shifted so that
    t = 0 at common_start. EEG, LFP, spikes and behaviour therefore share one
    timeline; each stream keeps its own (Allen-synchronised) timestamps.

Windows are anchored to experimenter annotations stored in the NWB file
(isoflurane-level log, injection epochs). If Irene's timestamps differ, edit
``WINDOWS`` / ``ANCHORS`` below.

Usage (from code/python/):
    python -m src.extract_windows                      # all data/raw/*.nwb
    python -m src.extract_windows data/raw/688277-20230921-v001.nwb --fs 250
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
from scipy import signal

WINDOW_S = 120.0
DEFAULT_FS = 500.0          # target sample rate (Hz), as in Jonathan's notebook
DEFAULT_CHANNEL_STRIDE = 4  # keep every 4th LFP channel (~80 um spacing)
PAD_S = 2.0                 # extra data read on each side so decimation edges are clean

# CCF acronym prefixes kept for LFP
CORTEX = ("SSp", "SSs", "MOs", "MOp", "RSP", "VIS", "ACA", "PL", "ILA", "ORB", "AUD", "TEa", "FRP")
HIPPOCAMPUS = ("CA1", "CA2", "CA3", "DG")

# Session type -> list of (window name, anchor, offset of window start from anchor in s)
WINDOWS = {
    "isoflurane": [
        # Stimulus-free awake period between the last awake trial and induction
        ("awake", "iso_onset", -WINDOW_S - 5.0),
        # Anesthetized: starts at the EEG power drop (+268 s after onset, when
        # 5% induction steps down to maintenance; see results/power_drop.csv)
        ("iso_anesthetized", "iso_onset", 270.0),
    ],
    "psilocybin": [
        # 2 min immediately before the psilocybin injection epoch
        ("psilocybin_before", "psilocybin_injection_start", -WINDOW_S),
        # 12.5 min after the injection starts, ending before stimulation
        # trials resume (4043 s in 692644)
        ("psilocybin_after", "psilocybin_injection_start", 750.0),
    ],
}


# --------------------------------------------------------------------------- #
# Anchors
# --------------------------------------------------------------------------- #

def _injection(nwb, drug: str, edge: str) -> float:
    if nwb.epochs is None:
        raise KeyError("session has no epochs table")
    ep = nwb.epochs.to_dataframe()
    rows = ep[ep.tags.apply(lambda tags: any(drug in t for t in tags))]
    if rows.empty:
        raise KeyError(f"no {drug} injection epoch")
    return float(rows.iloc[0][f"{edge}_time"])


def _iso_onset(nwb, threshold: float = 1.0) -> float:
    """First time the isoflurane level exceeds ``threshold`` (%)."""
    iso = nwb.acquisition["isoflurane_level"]
    level = iso.data[:]
    idx = int(np.argmax(level > threshold))
    if level[idx] <= threshold:
        raise KeyError("isoflurane level never exceeds threshold")
    return float(iso.starting_time + idx / iso.rate)


ANCHORS = {
    "iso_onset": _iso_onset,
    "psilocybin_injection_start": lambda nwb: _injection(nwb, "psilocybin", "start"),
    "psilocybin_injection_stop": lambda nwb: _injection(nwb, "psilocybin", "stop"),
}


def session_type(nwb) -> str:
    pharm = nwb.pharmacology.lower()
    for kind in ("ketanserin", "psilocybin", "isoflurane", "urethane"):
        if kind in pharm:
            return kind
    return "unknown"


# --------------------------------------------------------------------------- #
# Stream helpers
# --------------------------------------------------------------------------- #

def _timestamps(series) -> np.ndarray:
    if series.timestamps is not None:
        return np.asarray(series.timestamps[:])
    n = series.data.shape[0]
    return np.arange(n) / series.rate + series.starting_time


def _decimate_window(series, ts, t0, t1, fs_out, channels=None):
    """Read [t0, t1) from an ElectricalSeries, convert to uV, decimate to ~fs_out.

    Returns (data (T, C) float32, timestamps (T,) absolute).
    Reads PAD_S of extra data on both sides so the zero-phase IIR filter has
    no edge transients inside the window.
    """
    fs_in = 1.0 / np.median(np.diff(ts[:10000]))
    q = max(int(round(fs_in / fs_out)), 1)
    i0, i1 = np.searchsorted(ts, [t0, t1], side="left")
    pad = int(PAD_S * fs_in) // q * q
    front = min(pad, i0 // q * q)          # multiple of q so samples stay on ts[i0::q]
    r0, r1 = i0 - front, min(i1 + pad, len(ts))

    raw = series.data[r0:r1]               # (N, all channels) int16
    if channels is not None:
        raw = raw[:, channels]
    x = raw.astype(np.float64) * series.conversion * 1e6   # volts -> uV
    if q > 1:
        x = signal.decimate(x, q, ftype="iir", zero_phase=True, axis=0)
    first = front // q
    out_ts = ts[i0:i1:q]
    data = x[first:first + len(out_ts)]
    out_ts = out_ts[:len(data)]
    return data.astype(np.float32), out_ts, fs_in / q


def _lfp_channels(es, stride):
    """Indices (into es.data columns) of valid cortex/hippocampus channels, every ``stride``-th per region."""
    el = es.electrodes.to_dataframe().reset_index()
    loc = el.location.astype(str)
    valid = el.is_data_valid.to_numpy(bool)
    keep = []
    for region in loc.unique():
        if region.startswith(CORTEX) or region.startswith(HIPPOCAMPUS):
            idx = np.where((loc == region) & valid)[0]
            keep.extend(idx[::stride])
    keep = np.sort(np.array(keep, dtype=int))
    return keep, el.iloc[keep]


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def extract_window(nwb, name, anchor, offset, out_dir: Path, fs_out=DEFAULT_FS,
                   stride=DEFAULT_CHANNEL_STRIDE, duration=WINDOW_S) -> Path:
    anchor_t = ANCHORS[anchor](nwb)
    window_start = anchor_t + offset
    window_end = window_start + duration

    eeg = nwb.acquisition["ElectricalSeriesEEG"]
    probes = dict(nwb.acquisition["LFP"].electrical_series)
    eeg_ts = _timestamps(eeg)
    lfp_ts = {p: _timestamps(es) for p, es in probes.items()}
    behavior = nwb.processing["behavior"]
    running = behavior["BehavioralTimeSeries"]["running_speed"]
    running_ts = _timestamps(running)
    pupil = pupil_ts = None
    if "PupilTracking" in behavior.data_interfaces:
        pupil = behavior["PupilTracking"]["pupil_width"]
        pupil_ts = _timestamps(pupil)
    iso = nwb.acquisition.get("isoflurane_level")
    iso_ts = _timestamps(iso) if iso is not None else None

    # ---- Jonathan's common start ------------------------------------------
    stream_starts = [eeg_ts[0], running_ts[0], *(t[0] for t in lfp_ts.values())]
    if pupil_ts is not None:
        stream_starts.append(pupil_ts[0])
    if iso_ts is not None:
        stream_starts.append(iso_ts[0])
    common_start = max(window_start, max(stream_starts))
    if common_start >= window_end:
        raise RuntimeError(f"{name}: no data in window ({common_start=} >= {window_end=})")
    aligned_end = window_end - common_start

    # ---- EEG ------------------------------------------------------------------
    eeg_data, eeg_t, fs_eeg = _decimate_window(eeg, eeg_ts, common_start, window_end, fs_out)
    eeg_el = eeg.electrodes.to_dataframe()

    # ---- LFP (cortex + hippocampus) ------------------------------------------
    lfp_blocks, lfp_meta, lfp_times = [], [], []
    for p, es in probes.items():
        cols, meta = _lfp_channels(es, stride)
        if len(cols) == 0:
            continue
        d, t, fs_lfp = _decimate_window(es, lfp_ts[p], common_start, window_end, fs_out, cols)
        lfp_blocks.append(d)
        lfp_times.append(t)
        meta = meta.assign(probe=p.replace("ElectricalSeries", ""))
        lfp_meta.append(meta)
    n = min(len(t) for t in lfp_times)
    lfp_data = np.hstack([b[:n] for b in lfp_blocks])
    lfp_t = lfp_times[0][:n]
    probe_offset = max(np.max(np.abs(t[:n] - lfp_t)) for t in lfp_times)
    lfp_meta = pd.concat(lfp_meta, ignore_index=True)

    # ---- Spikes -------------------------------------------------------------
    units = nwb.units
    spike_index = units["spike_times_index"].data[:]
    all_spikes = units["spike_times_index"].target.data[:]  # flat spike times (not the index)
    starts = np.concatenate([[0], spike_index[:-1]])
    clipped = []
    for a, b in zip(starts, spike_index):
        st = all_spikes[a:b]
        clipped.append(st[(st >= common_start) & (st < window_end)] - common_start)
    spike_end = np.cumsum([len(s) for s in clipped])
    spike_times = np.concatenate(clipped) if clipped else np.array([])

    # ---- Stimuli in window (EEG/LFP contamination check) ----------------------
    stim_times = np.array([], dtype=float)
    if nwb.trials is not None:
        tr = nwb.trials.to_dataframe()
        in_window = (tr.start_time >= common_start) & (tr.start_time < window_end)
        if "stimulus_type" in tr.columns:
            in_window &= tr.stimulus_type.astype(str).str.lower().eq("electrical")
        stim_times = tr.loc[in_window, "start_time"].to_numpy(float) - common_start
    n_stim = int(len(stim_times))

    # ---- Write ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{nwb.identifier}_{name}.h5"

    def strs(x):
        return np.asarray([str(v) for v in x], dtype=h5py.string_dtype())

    def clip(ts, t0=common_start, t1=window_end):
        i0, i1 = np.searchsorted(ts, [t0, t1], side="left")
        return i0, i1

    with h5py.File(out, "w") as f:
        f.attrs.update({
            "session": nwb.identifier,
            "subject": nwb.subject.subject_id,
            "pharmacology": nwb.pharmacology,
            "window": name,
            "anchor": anchor,
            "anchor_time_abs": anchor_t,
            "window_start_abs": window_start,
            "window_end_abs": window_end,
            "common_start": common_start,
            "aligned_end": aligned_end,
            "fs": fs_out,
            "n_stim_in_window": n_stim,
            "lfp_channel_stride": stride,
            "max_probe_ts_offset_s": probe_offset,
            "timeline": "all *ts / spike times are seconds since common_start (absolute NWB time)",
            "units": "EEG/LFP in microvolts",
        })
        g = f.create_group("eeg")
        g.create_dataset("data", data=eeg_data, compression="gzip")
        g.create_dataset("ts", data=eeg_t - common_start)
        g.attrs["fs"] = fs_eeg
        g.create_dataset("location", data=strs(eeg_el.location))
        g.create_dataset("is_data_valid", data=eeg_el.is_data_valid.to_numpy(bool))
        for c in ("x", "y", "z"):
            g.create_dataset(c, data=eeg_el[c].to_numpy(float))

        g = f.create_group("lfp")
        g.create_dataset("data", data=lfp_data, compression="gzip")
        g.create_dataset("ts", data=lfp_t - common_start)
        g.attrs["fs"] = fs_lfp
        g.create_dataset("probe", data=strs(lfp_meta.probe))
        g.create_dataset("location", data=strs(lfp_meta.location))
        g.create_dataset("electrode_id", data=lfp_meta["id"].to_numpy(int))
        g.create_dataset("probe_vertical_position", data=lfp_meta.probe_vertical_position.to_numpy(int))

        g = f.create_group("spikes")
        g.create_dataset("times", data=spike_times, compression="gzip")
        g.create_dataset("index", data=spike_end)
        g.create_dataset("unit_id", data=units.id[:])
        g.create_dataset("location", data=strs(units["location"][:]))
        g.create_dataset("probe", data=strs(e.name for e in units["electrode_group"][:]))
        for c in ("x", "y", "z"):   # CCF position; y increases ventrally
            g.create_dataset(c, data=np.asarray(units[c][:], float))
        for c in ("isi_violations", "amplitude_cutoff", "presence_ratio"):
            g.create_dataset(c, data=np.asarray(units[c][:], float))

        g = f.create_group("behavior")
        g.create_dataset("stim_ts", data=stim_times)
        i0, i1 = clip(running_ts)
        g.create_dataset("running_speed", data=np.asarray(running.data[i0:i1], np.float32))
        g.create_dataset("running_ts", data=running_ts[i0:i1] - common_start)
        if pupil is not None:
            i0, i1 = clip(pupil_ts)
            g.create_dataset("pupil_width", data=np.asarray(pupil.data[i0:i1], np.float32))
            g.create_dataset("pupil_ts", data=pupil_ts[i0:i1] - common_start)
        if iso is not None:
            i0, i1 = clip(iso_ts)
            g.create_dataset("iso_level", data=np.asarray(iso.data[i0:i1], np.float32))
            g.create_dataset("iso_ts", data=iso_ts[i0:i1] - common_start)

    print(f"  {name:18s} {window_start:8.1f}-{window_end:8.1f} s  "
          f"EEG {eeg_data.shape}  LFP {lfp_data.shape}  "
          f"spikes {len(spike_times)}  stim in window {n_stim}  -> {out}")
    return out


def extract_session(path: Path, out_dir: Path, fs_out=DEFAULT_FS, stride=DEFAULT_CHANNEL_STRIDE):
    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        kind = session_type(nwb)
        print(f"{path.name}: {kind} session")
        if kind not in WINDOWS:
            print(f"  no windows defined for '{kind}' sessions; add them to WINDOWS")
            return []
        return [extract_window(nwb, name, anchor, offset, out_dir, fs_out, stride)
                for name, anchor, offset in WINDOWS[kind]]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("nwb", nargs="*", type=Path)
    p.add_argument("--out", type=Path, default=Path("data/probe_samples"))
    p.add_argument("--fs", type=float, default=DEFAULT_FS, help="target sample rate (Hz)")
    p.add_argument("--stride", type=int, default=DEFAULT_CHANNEL_STRIDE,
                   help="keep every Nth LFP channel within each region")
    args = p.parse_args(argv)
    paths = args.nwb or sorted(Path("data/raw").glob("*.nwb"))
    for path in paths:
        extract_session(path, args.out, args.fs, args.stride)


if __name__ == "__main__":
    main()
