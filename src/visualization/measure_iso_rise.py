"""Measure the initial broadband EEG rise around isoflurane onset.

This intentionally matches Gautam's power-time-course settings: 500 Hz target
rate, two-second spectrogram windows with one-second steps, median across valid
EEG channels, 1--100 Hz broadband power excluding 55--65 Hz, and ten-second
smoothing. Output is written only when the script is run explicitly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
from scipy import signal


def first_sustained(mask: pd.Series, samples: int) -> float:
    sustained = mask.rolling(samples).sum().shift(-(samples - 1)) == samples
    hits = sustained[sustained].index
    return float(hits[0]) if len(hits) else np.nan


def analyze(path: Path, pre: float = 120, post: float = 600) -> tuple[pd.DataFrame, dict]:
    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        iso = nwb.acquisition["isoflurane_level"]
        levels = np.asarray(iso.data[:])
        onset_i = int(np.argmax(levels > 1.0))
        if levels[onset_i] <= 1.0:
            raise RuntimeError("Isoflurane level never exceeds 1%")
        onset = float(iso.starting_time + onset_i / iso.rate)

        eeg = nwb.acquisition["ElectricalSeriesEEG"]
        valid = eeg.electrodes.to_dataframe().is_data_valid.to_numpy(bool)
        ts = np.asarray(eeg.timestamps[:])
        i0, i1 = np.searchsorted(ts, [onset - pre, onset + post])
        x = eeg.data[i0:i1][:, valid].astype(np.float64) * eeg.conversion * 1e6
        fs_in = 1 / np.median(np.diff(ts[i0:i0 + 10_000]))
        q = int(round(fs_in / 500.0))
        x = signal.decimate(x, q, ftype="iir", zero_phase=True, axis=0)
        fs = fs_in / q
        t0 = ts[i0] - onset

    f, t, spectrum = signal.spectrogram(
        x, fs=fs, nperseg=int(2 * fs), noverlap=int(fs), axis=0
    )
    spectrum = np.median(spectrum, axis=1)
    t = t + t0
    keep = (f >= 1) & (f < 100) & ~((f >= 55) & (f < 65))
    power = spectrum[keep].sum(0)
    change = 10 * np.log10(power / np.median(power[t < 0]))
    frame = pd.DataFrame({"broadband_db": change}, index=pd.Index(t, name="t_rel_iso_s"))

    dt = float(np.median(np.diff(frame.index)))
    smooth_n = max(int(round(10 / dt)), 1)
    smoothed = frame.broadband_db.rolling(smooth_n, center=True, min_periods=1).mean()
    frame["broadband_smoothed_db"] = smoothed
    baseline = smoothed[smoothed.index < 0]
    rise_threshold = float(baseline.mean() + 3 * baseline.std())
    run_n = max(int(round(20 / dt)), 1)
    rise_time = first_sustained((smoothed > rise_threshold) & (smoothed.index >= 0), run_n)

    drop_time = 268.0
    pre_drop = smoothed[(smoothed.index >= 0) & (smoothed.index < drop_time)]
    peak_time = float(pre_drop.idxmax())
    peak_db = float(pre_drop.max())
    first_minute = smoothed[(smoothed.index >= 0) & (smoothed.index <= 60)]
    first_minute_peak_time = float(first_minute.idxmax())
    first_minute_peak_db = float(first_minute.max())
    result = {
        "iso_onset_abs_s": onset,
        "rise_threshold_db": rise_threshold,
        "sustained_rise_onset_s": rise_time,
        "pre_drop_peak_time_s": peak_time,
        "pre_drop_peak_db": peak_db,
        "first_minute_peak_time_s": first_minute_peak_time,
        "first_minute_peak_db": first_minute_peak_db,
        "verified_drop_time_s": drop_time,
        "peak_to_drop_delay_s": float(drop_time - peak_time),
        "rise_to_drop_delay_s": float(drop_time - rise_time) if np.isfinite(rise_time) else np.nan,
        "valid_eeg_channels": int(valid.sum()),
    }
    return frame, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("nwb", type=Path)
    parser.add_argument("--timecourse", type=Path, default=Path("src/visualization/results/iso_power_timecourse.csv"))
    parser.add_argument("--summary", type=Path, default=Path("src/visualization/results/iso_initial_rise.csv"))
    args = parser.parse_args()
    frame, result = analyze(args.nwb)
    args.timecourse.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.timecourse)
    pd.DataFrame([result]).to_csv(args.summary, index=False)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
