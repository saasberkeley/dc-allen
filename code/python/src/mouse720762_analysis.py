"""Reusable analysis helpers for the mouse 720762 awake-to-isoflurane notebook.

The functions in this module intentionally read one continuous channel at a time.
That keeps memory use reasonable for the roughly 24 GB NWB recording.  Expensive
band-envelope preprocessing can be cached as compressed NumPy files by the
notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
from scipy import signal
from scipy.optimize import linear_sum_assignment

from .get_motifs import MotifResult, get_motifs
from .smooth_env import DEFAULT_FREQ_BANDS, smooth_env


MOUSE_ID = "720762"
NWB_FILENAME = "720762-20240221-v001.nwb"


@dataclass(frozen=True)
class LoadedTrace:
    values_uv: np.ndarray
    time_s: np.ndarray
    fs_hz: float
    source: str
    channel: int
    location: str
    electrode_id: int


def _sampling_rate(series) -> float:
    if series.rate is not None:
        return float(series.rate)
    timestamps = np.asarray(series.timestamps[: min(100_000, len(series.timestamps))], dtype=float)
    return float(1.0 / np.median(np.diff(timestamps)))


def _time_bounds(series) -> tuple[float, float]:
    if series.timestamps is not None:
        return float(series.timestamps[0]), float(series.timestamps[-1])
    start = float(series.starting_time)
    stop = start + (series.data.shape[0] - 1) / _sampling_rate(series)
    return start, stop


def _lfp_series(nwb, probe: str):
    """Return one LFP ElectricalSeries across old and new PyNWB interfaces."""
    container = nwb.acquisition["LFP"]
    key = f"ElectricalSeries{probe}"
    if hasattr(container, "electrical_series"):
        return container.electrical_series[key]
    return container[key]


def _sample_slice(series, start_s: float | None, stop_s: float | None) -> tuple[slice, float, float]:
    fs = _sampling_rate(series)
    first, last = _time_bounds(series)
    start = first if start_s is None else max(float(start_s), first)
    stop = last if stop_s is None else min(float(stop_s), last)
    if stop <= start:
        raise ValueError(f"empty time window after clipping: {start:g} to {stop:g} s")

    # Templeton EEG/LFP timestamps are regular even when stored explicitly.  Use
    # their measured rate rather than materializing millions of timestamps for
    # every short window request.
    def index_at_or_after(time_s: float) -> int:
        position = (time_s - first) * fs
        nearest = round(position)
        if np.isclose(position, nearest, rtol=0, atol=1e-6):
            return int(nearest)
        return int(np.ceil(position))

    i0 = max(0, index_at_or_after(start))
    i1 = min(series.data.shape[0], index_at_or_after(stop))
    return slice(i0, i1), start, fs


def _resample(values: np.ndarray, fs: float, target_fs: float | None) -> tuple[np.ndarray, float]:
    if target_fs is None or fs <= target_fs * 1.001:
        return np.asarray(values, dtype=float), float(fs)
    ratio = Fraction(float(target_fs) / float(fs)).limit_denominator(1_000)
    resampled = signal.resample_poly(values, ratio.numerator, ratio.denominator, axis=0)
    return resampled, float(fs * ratio.numerator / ratio.denominator)


def inspect_nwb(path: str | Path) -> dict:
    """Return lightweight metadata without loading the continuous recordings."""
    path = Path(path)
    with NWBHDF5IO(str(path), mode="r") as io:
        nwb = io.read()
        trials = nwb.trials.to_dataframe() if nwb.trials is not None else pd.DataFrame()
        states = []
        if "behavioral_epoch" in trials:
            states = [
                {
                    "state": str(label),
                    "start_s": float(group.start_time.min()),
                    "stop_s": float(group.stop_time.max()),
                }
                for label, group in trials.groupby("behavioral_epoch")
            ]
        probes = sorted(name for name in nwb.electrode_groups if "probe" in name.lower())
        return {
            "file": path.name,
            "subject": str(nwb.subject.subject_id),
            "session_id": str(nwb.session_id),
            "session_description": str(nwb.session_description),
            "pharmacology": str(nwb.pharmacology),
            "acquisition_keys": list(nwb.acquisition.keys()),
            "probes": probes,
            "states": sorted(states, key=lambda row: row["start_s"]),
        }


def infer_state_windows(path: str | Path) -> dict[str, tuple[float, float]]:
    """Infer awake and isoflurane intervals from the NWB trial table."""
    metadata = inspect_nwb(path)
    windows: dict[str, tuple[float, float]] = {}
    for row in metadata["states"]:
        label = row["state"].lower()
        if "awake" in label and "awake" not in windows:
            windows["awake"] = (row["start_s"], row["stop_s"])
        if ("isoflurane" in label or label == "iso") and "isoflurane" not in windows:
            windows["isoflurane"] = (row["start_s"], row["stop_s"])
    if {"awake", "isoflurane"} - windows.keys():
        missing = sorted({"awake", "isoflurane"} - windows.keys())
        raise ValueError(
            f"could not infer {missing} from behavioral_epoch labels; inspect metadata['states'] "
            "and set WINDOWS manually in the notebook"
        )
    return windows


def eeg_channel_table(path: str | Path) -> pd.DataFrame:
    with NWBHDF5IO(str(path), mode="r") as io:
        nwb = io.read()
        series = nwb.acquisition["ElectricalSeriesEEG"]
        table = series.electrodes.to_dataframe().reset_index(names="electrode_id")
    table.insert(0, "channel", np.arange(len(table), dtype=int))
    return table


def lfp_region_table(path: str | Path) -> pd.DataFrame:
    rows = []
    with NWBHDF5IO(str(path), mode="r") as io:
        nwb = io.read()
        lfp = nwb.acquisition["LFP"]
        series_map = lfp.electrical_series if hasattr(lfp, "electrical_series") else lfp
        for key in sorted(series_map.keys()):
            series = series_map[key]
            probe = key.removeprefix("ElectricalSeries")
            table = series.electrodes.to_dataframe().reset_index(names="electrode_id")
            valid = table["is_data_valid"].astype(bool) if "is_data_valid" in table else True
            for location, group in table[valid].groupby("location", sort=False):
                local_channels = group.index.to_numpy(dtype=int)
                rows.append(
                    {
                        "probe": probe,
                        "region": str(location),
                        "n_valid_channels": int(len(group)),
                        "first_channel": int(local_channels.min()),
                        "last_channel": int(local_channels.max()),
                        "middle_channel": int(local_channels[len(local_channels) // 2]),
                    }
                )
    return pd.DataFrame(rows)


def load_eeg_trace(
    path: str | Path,
    channel: int,
    start_s: float | None = None,
    stop_s: float | None = None,
    target_fs: float = 250.0,
) -> LoadedTrace:
    with NWBHDF5IO(str(path), mode="r") as io:
        nwb = io.read()
        series = nwb.acquisition["ElectricalSeriesEEG"]
        electrodes = series.electrodes.to_dataframe()
        if not 0 <= channel < series.data.shape[1]:
            raise IndexError(f"EEG channel {channel} is outside 0..{series.data.shape[1] - 1}")
        row = electrodes.iloc[channel]
        if "is_data_valid" in electrodes and not bool(row.is_data_valid):
            raise ValueError(f"EEG channel {channel} is marked invalid")
        sl, start, fs = _sample_slice(series, start_s, stop_s)
        raw = np.asarray(series.data[sl, channel], dtype=float) * float(series.conversion) * 1e6
        electrode_id = int(electrodes.index[channel])
        location = str(row.location)

    values, actual_fs = _resample(raw, fs, target_fs)
    time_s = start + np.arange(values.size) / actual_fs
    return LoadedTrace(values, time_s, actual_fs, "EEG", channel, location, electrode_id)


def choose_lfp_channel(path: str | Path, probe: str, region: str) -> tuple[int, int, str]:
    with NWBHDF5IO(str(path), mode="r") as io:
        nwb = io.read()
        series = _lfp_series(nwb, probe)
        electrodes = series.electrodes.to_dataframe()
        valid = np.ones(len(electrodes), dtype=bool)
        if "is_data_valid" in electrodes:
            valid &= electrodes.is_data_valid.to_numpy(dtype=bool)
        match = np.flatnonzero(valid & (electrodes.location.astype(str).to_numpy() == region))
        if not match.size:
            available = sorted(electrodes.loc[valid, "location"].astype(str).unique())
            raise ValueError(f"no valid {probe} channels in {region}; available regions: {available}")
        channel = int(match[len(match) // 2])
        return channel, int(electrodes.index[channel]), str(electrodes.iloc[channel].location)


def load_lfp_trace(
    path: str | Path,
    probe: str,
    region: str,
    start_s: float | None = None,
    stop_s: float | None = None,
    target_fs: float = 250.0,
) -> LoadedTrace:
    channel, electrode_id, location = choose_lfp_channel(path, probe, region)
    with NWBHDF5IO(str(path), mode="r") as io:
        nwb = io.read()
        series = _lfp_series(nwb, probe)
        sl, start, fs = _sample_slice(series, start_s, stop_s)
        raw = np.asarray(series.data[sl, channel], dtype=float) * float(series.conversion) * 1e6
    values, actual_fs = _resample(raw, fs, target_fs)
    time_s = start + np.arange(values.size) / actual_fs
    return LoadedTrace(values, time_s, actual_fs, probe, channel, location, electrode_id)


def load_lfp_matrix(
    path: str | Path,
    probe: str,
    start_s: float,
    stop_s: float,
    channels: np.ndarray | list[int] | None = None,
    max_channels: int = 48,
    target_fs: float = 250.0,
) -> dict:
    """Load a short multi-channel LFP window for a channel-by-time heatmap."""
    with NWBHDF5IO(str(path), mode="r") as io:
        nwb = io.read()
        series = _lfp_series(nwb, probe)
        electrodes = series.electrodes.to_dataframe()
        valid = np.ones(len(electrodes), dtype=bool)
        if "is_data_valid" in electrodes:
            valid &= electrodes.is_data_valid.to_numpy(dtype=bool)
        if channels is None:
            available = np.flatnonzero(valid)
            if not available.size:
                raise ValueError(f"{probe} has no valid LFP channels")
            pick = np.linspace(0, len(available) - 1, min(max_channels, len(available))).astype(int)
            channels_array = available[pick]
        else:
            channels_array = np.asarray(channels, dtype=int)
            if np.any((channels_array < 0) | (channels_array >= len(electrodes))):
                raise IndexError("one or more LFP channels are outside the probe's channel range")
        sl, start, fs = _sample_slice(series, start_s, stop_s)
        raw = np.asarray(series.data[sl, channels_array], dtype=float) * float(series.conversion) * 1e6
        locations = electrodes.iloc[channels_array].location.astype(str).to_numpy()
    values, actual_fs = _resample(raw, fs, target_fs)
    return {
        "values_uv": values,
        "time_s": start + np.arange(values.shape[0]) / actual_fs,
        "fs_hz": actual_fs,
        "channels": channels_array,
        "locations": locations,
        "probe": probe,
    }


def preprocess_envelope(
    trace: LoadedTrace,
    bands: np.ndarray = DEFAULT_FREQ_BANDS,
    notch_60hz: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a time x bands envelope, preserving a common full-session z-score."""
    values = np.asarray(trace.values_uv, dtype=float)
    bands = np.asarray(bands, dtype=float)
    if bands.max() >= trace.fs_hz / 2:
        raise ValueError(
            f"highest band edge {bands.max():g} Hz must be below Nyquist "
            f"({trace.fs_hz / 2:g} Hz)"
        )
    if notch_60hz and trace.fs_hz > 122:
        notch = signal.butter(3, [59, 61], btype="bandstop", fs=trace.fs_hz, output="sos")
        values = signal.sosfiltfilt(notch, values)
    envelope = smooth_env(values, trace.fs_hz, bands)
    valid = np.isfinite(envelope).all(axis=1)
    return envelope[valid], trace.time_s[valid]


def save_envelope_cache(
    path: str | Path,
    envelope: np.ndarray,
    time_s: np.ndarray,
    trace: LoadedTrace,
    bands: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        envelope=np.asarray(envelope, dtype=np.float32),
        time_s=np.asarray(time_s, dtype=np.float64),
        fs_hz=float(trace.fs_hz),
        source=trace.source,
        channel=int(trace.channel),
        location=trace.location,
        electrode_id=int(trace.electrode_id),
        bands=np.asarray(bands, dtype=float),
    )


def load_envelope_cache(path: str | Path) -> dict:
    with np.load(path) as cached:
        return {key: cached[key].copy() for key in cached.files}


def discover_state_motifs(
    envelope: np.ndarray,
    time_s: np.ndarray,
    fs_hz: float,
    windows: dict[str, tuple[float, float]],
    n_null: int = 20,
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, MotifResult]:
    """Run independent motif discovery on awake, isoflurane, and their union."""
    selections = {
        "awake": windows["awake"],
        "isoflurane": windows["isoflurane"],
        "full": (
            min(windows["awake"][0], windows["isoflurane"][0]),
            max(windows["awake"][1], windows["isoflurane"][1]),
        ),
    }
    results = {}
    for name, (start, stop) in selections.items():
        keep = (time_s >= start) & (time_s <= stop)
        if keep.sum() < fs_hz * 60:
            raise ValueError(f"{name} contains less than 60 seconds after preprocessing")
        results[name] = get_motifs(
            [envelope[keep].T], fs=fs_hz, n_null=n_null, seed=seed, verbose=verbose
        )
    return results


def match_motif_pairs(a: MotifResult, b: MotifResult) -> pd.DataFrame:
    """Match motif pairs by absolute Pearson correlation of their positive poles."""
    if not a.n_motifs or not b.n_motifs:
        return pd.DataFrame(columns=["motif_a", "motif_b", "correlation", "abs_correlation", "flipped"])
    a_plus = a.motifs[:, : a.n_motifs].T
    b_plus = b.motifs[:, : b.n_motifs].T
    corr = np.array([[np.corrcoef(x, y)[0, 1] for y in b_plus] for x in a_plus])
    score = np.nan_to_num(np.abs(corr), nan=0.0)
    row, col = linear_sum_assignment(-score)
    return pd.DataFrame(
        {
            "motif_a": row + 1,
            "motif_b": col + 1,
            "correlation": corr[row, col],
            "abs_correlation": np.abs(corr[row, col]),
            "flipped": corr[row, col] < 0,
        }
    )


def blocked_state_decoder(
    envelope: np.ndarray,
    time_s: np.ndarray,
    fs_hz: float,
    windows: dict[str, tuple[float, float]],
    block_seconds: int = 30,
    n_splits: int = 5,
) -> dict:
    """Decode awake vs. isoflurane with block-held-out logistic regression.

    The envelope is sampled at approximately 1 Hz.  Entire contiguous blocks are
    held out together, which avoids the inflated accuracy caused by randomly
    splitting highly autocorrelated neighboring seconds.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    hop = max(1, int(round(fs_hz)))
    env_1hz = envelope[::hop]
    t_1hz = time_s[::hop]
    rows = []
    labels = []
    groups = []
    group_offset = 0
    for label, state in enumerate(("awake", "isoflurane")):
        start, stop = windows[state]
        keep = (t_1hz >= start) & (t_1hz <= stop)
        rows.append(env_1hz[keep])
        labels.append(np.full(keep.sum(), label, dtype=int))
        local_groups = ((t_1hz[keep] - start) // block_seconds).astype(int)
        groups.append(local_groups + group_offset)
        group_offset += int(local_groups.max(initial=-1)) + 2

    X = np.vstack(rows)
    y = np.concatenate(labels)
    group = np.concatenate(groups)
    splitter = StratifiedGroupKFold(
        n_splits=min(n_splits, len(np.unique(group))), shuffle=True, random_state=0
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=0),
    )
    prediction = cross_val_predict(model, X, y, groups=group, cv=splitter)
    score = float(balanced_accuracy_score(y, prediction))
    matrix = confusion_matrix(y, prediction, normalize="true")
    model.fit(X, y)
    coefficients = model[-1].coef_[0]
    return {
        "balanced_accuracy": score,
        "confusion_matrix": matrix,
        "coefficients": coefficients,
        "n_seconds": int(len(y)),
        "n_blocks": int(len(np.unique(group))),
        "labels": np.array(["awake", "isoflurane"]),
    }


def load_spike_raster(
    path: str | Path,
    start_s: float,
    stop_s: float,
    max_units: int = 60,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Return the most active units and their spike times inside one time window."""
    with NWBHDF5IO(str(path), mode="r") as io:
        nwb = io.read()
        units = nwb.units.to_dataframe()
    clipped = []
    counts = []
    for spikes in units.spike_times:
        spikes = np.asarray(spikes, dtype=float)
        inside = spikes[(spikes >= start_s) & (spikes <= stop_s)]
        clipped.append(inside)
        counts.append(inside.size)
    selected = np.argsort(counts)[::-1][:max_units]
    selected = selected[np.asarray(counts)[selected] > 0]
    metadata = units.iloc[selected].copy()
    metadata.insert(0, "spikes_in_window", np.asarray(counts)[selected])
    return [clipped[i] for i in selected], metadata


def load_matched_spike_rasters(
    path: str | Path,
    windows: dict[str, tuple[float, float]],
    max_units: int = 60,
) -> tuple[dict[str, list[np.ndarray]], pd.DataFrame]:
    """Use the same most-active units for every requested state window."""
    with NWBHDF5IO(str(path), mode="r") as io:
        nwb = io.read()
        units = nwb.units.to_dataframe()

    per_state: dict[str, list[np.ndarray]] = {state: [] for state in windows}
    total_counts = np.zeros(len(units), dtype=int)
    all_spikes = [np.asarray(spikes, dtype=float) for spikes in units.spike_times]
    for state, (start, stop) in windows.items():
        for index, spikes in enumerate(all_spikes):
            inside = spikes[(spikes >= start) & (spikes <= stop)]
            per_state[state].append(inside)
            total_counts[index] += inside.size

    selected = np.argsort(total_counts)[::-1][:max_units]
    selected = selected[total_counts[selected] > 0]
    matched = {
        state: [trains[index] for index in selected]
        for state, trains in per_state.items()
    }
    metadata = units.iloc[selected].copy()
    metadata.insert(0, "spikes_across_windows", total_counts[selected])
    return matched, metadata
