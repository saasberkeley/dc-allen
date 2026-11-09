"""All-channel EEG motif diagnostics for extracted Templeton HDF5 windows.

This is the audit pipeline for the "clean LFP motif, missing awake EEG motif"
question.  It compares every valid EEG channel under four conditions:

* original vs. common-average-referenced (CAR) voltage
* MATLAB-compatible binary signs vs. continuous band-envelope values

It also saves plots of every major representation change, channel-level noise
and spectral metrics, motif counts, and running-speed/theta coupling.

Run from the repository root, for example::

    PYTHONPATH=code/python python -m src.eeg_motif_diagnostics \
        data/sample_nwb/688277-20230921-v001_awake.h5 \
        --out data/out/eeg_motif_diagnostics/awake
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import signal  # noqa: E402

from .get_motifs import get_motifs  # noqa: E402
from .smooth_env import DEFAULT_FREQ_BANDS, smooth_env  # noqa: E402


def _decode(values: np.ndarray) -> np.ndarray:
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values])


def load_window(path: str | Path) -> dict:
    """Load the EEG and behavior arrays needed by this diagnostic."""
    with h5py.File(path, "r") as f:
        eeg = np.asarray(f["eeg/data"], dtype=float)
        ts = np.asarray(f["eeg/ts"], dtype=float)
        valid = np.asarray(f["eeg/is_data_valid"], dtype=bool)
        location = _decode(f["eeg/location"][()])
        xyz = {
            axis: np.asarray(f[f"eeg/{axis}"], dtype=float)
            for axis in ("x", "y", "z")
            if f"eeg/{axis}" in f
        }
        running = np.asarray(f["behavior/running_speed"], dtype=float)
        running_ts = np.asarray(f["behavior/running_ts"], dtype=float)
        attrs = {key: value.item() if hasattr(value, "item") else value for key, value in f.attrs.items()}
    fs = float(1 / np.median(np.diff(ts)))
    return {
        "eeg": eeg,
        "ts": ts,
        "fs": fs,
        "valid": valid,
        "location": location,
        "xyz": xyz,
        "running": running,
        "running_ts": running_ts,
        "attrs": attrs,
        "path": str(path),
    }


def common_average_reference(data: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the valid-channel mean at every sample; invalid channels are unchanged."""
    data = np.asarray(data, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if data.ndim != 2 or valid.shape != (data.shape[1],):
        raise ValueError("data must be time x channels and valid must have one entry per channel")
    if not valid.any():
        raise ValueError("CAR needs at least one valid channel")
    reference = data[:, valid].mean(axis=1)
    out = data.copy()
    out[:, valid] -= reference[:, None]
    return out, reference


def channel_metrics(data: np.ndarray, fs: float) -> pd.DataFrame:
    """Noise, spectral-shape, theta, and line-noise summaries for each channel."""
    f, p = signal.welch(data, fs=fs, nperseg=min(int(4 * fs), data.shape[0]), axis=0)
    p = np.maximum(p, np.finfo(float).tiny)

    def power(lo: float, hi: float) -> np.ndarray:
        keep = (f >= lo) & (f < hi)
        return np.trapezoid(p[keep], f[keep], axis=0)

    broadband = power(1, min(100, fs / 2))
    theta = power(6, 10)
    line = power(59, min(61, fs / 2)) if fs / 2 > 59 else np.full(data.shape[1], np.nan)
    slope_keep = (f >= 2) & (f <= min(40, fs / 2 - 1))
    logf = np.log10(f[slope_keep])
    slopes = np.array([np.polyfit(logf, np.log10(p[slope_keep, ch]), 1)[0] for ch in range(data.shape[1])])
    centered = data - np.median(data, axis=0)
    return pd.DataFrame(
        {
            "robust_sd_uv": 1.4826 * np.median(np.abs(centered), axis=0),
            "peak_to_peak_uv": np.ptp(data, axis=0),
            "theta_fraction": theta / broadband,
            "line_noise_fraction": line / broadband,
            "spectral_slope_2_40hz": slopes,
        }
    )


def _one_hz(env: np.ndarray, fs: float) -> np.ndarray:
    return env[:: max(1, int(round(fs)))]


def _save_raw_plot(window: dict, car: np.ndarray, reference: np.ndarray, out: Path) -> None:
    valid = np.flatnonzero(window["valid"])
    picks = valid[np.linspace(0, len(valid) - 1, min(5, len(valid))).round().astype(int)]
    n = min(len(window["ts"]), int(15 * window["fs"]))
    t = window["ts"][:n] - window["ts"][0]
    fig, axes = plt.subplots(len(picks) + 1, 1, figsize=(12, 1.6 * (len(picks) + 1)), sharex=True)
    for ax, ch in zip(axes, picks):
        scale = max(np.std(window["eeg"][:n, ch]), 1e-9)
        ax.plot(t, window["eeg"][:n, ch] / scale, color="#777", lw=0.6, label="original")
        ax.plot(t, car[:n, ch] / scale, color="#1769aa", lw=0.6, label="CAR")
        ax.set_ylabel(f"ch {ch}\n{window['location'][ch]}")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    axes[-1].plot(t, reference[:n], color="#d95f02", lw=0.7)
    axes[-1].set_ylabel("common avg\n(µV)")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Step 1 — raw EEG and common-average-referenced EEG (scaled per channel)")
    fig.tight_layout()
    fig.savefig(out / "01_raw_vs_car.png", dpi=160)
    plt.close(fig)


def _save_spectra_plot(window: dict, original: pd.DataFrame, car_metrics: pd.DataFrame, out: Path) -> None:
    valid = window["valid"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    fields = [
        ("robust_sd_uv", "Robust SD (µV)"),
        ("theta_fraction", "6–10 Hz / 1–100 Hz power"),
        ("line_noise_fraction", "59–61 Hz / 1–100 Hz power"),
    ]
    x = np.flatnonzero(valid)
    for ax, (field, label) in zip(axes, fields):
        ax.plot(x, original.loc[valid, field], "o-", ms=3, lw=1, color="#777", label="original")
        ax.plot(x, car_metrics.loc[valid, field], "o-", ms=3, lw=1, color="#1769aa", label="CAR")
        ax.set_xlabel("EEG channel")
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("Step 2 — channel noise and spectral shape")
    fig.savefig(out / "02_channel_eda.png", dpi=160)
    plt.close(fig)


def _save_transform_plot(env: np.ndarray, fs: float, channel: int, location: str, out: Path) -> None:
    one = _one_hz(env, fs)
    signs = np.sign(one)
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, layout="constrained")
    axes[0].imshow(env.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    axes[0].set_title("Step 3 — smoothed, z-scored band envelopes")
    axes[1].imshow(one.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    axes[1].set_title("Step 4 — continuous envelopes sampled at ~1 Hz")
    axes[2].imshow(signs.T, aspect="auto", origin="lower", cmap="bwr", vmin=-1, vmax=1)
    axes[2].set_title("Step 5 — binary sign representation used by the MATLAB algorithm")
    for ax in axes:
        ax.set_ylabel("frequency band")
    axes[-1].set_xlabel("time bin")
    fig.suptitle(f"EEG channel {channel} ({location}): representation audit", fontsize=12)
    fig.savefig(out / "03_representation_steps.png", dpi=160)
    plt.close(fig)


def _save_movement_plot(window: dict, env: np.ndarray, out: Path) -> dict:
    theta_index = int(np.argmin(np.abs(DEFAULT_FREQ_BANDS.mean(axis=1) - 7)))
    theta = env[:, theta_index]
    good = np.isfinite(theta)
    speed = np.interp(window["ts"][good], window["running_ts"], window["running"])
    rho = float(pd.Series(theta[good]).corr(pd.Series(speed), method="spearman"))
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True, layout="constrained")
    axes[0].plot(window["ts"][good] - window["ts"][0], theta[good], color="#6a3d9a", lw=1)
    axes[0].set_ylabel("4–8 Hz envelope (z)")
    axes[1].plot(window["running_ts"] - window["ts"][0], window["running"], color="#1b9e77", lw=1)
    axes[1].set_ylabel("running speed")
    axes[1].set_xlabel("Time in window (s)")
    fig.suptitle(f"Movement check — Spearman rho(theta envelope, speed) = {rho:.3f}")
    fig.savefig(out / "04_theta_vs_running.png", dpi=160)
    plt.close(fig)
    return {"theta_speed_spearman": rho, "theta_band_index": theta_index}


def run_diagnostics(path: str | Path, out_dir: str | Path, n_null: int = 3, seed: int = 0) -> pd.DataFrame:
    """Run the full audit and return its channel summary table."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    window = load_window(path)
    car, reference = common_average_reference(window["eeg"], window["valid"])
    original_metrics = channel_metrics(window["eeg"], window["fs"])
    car_metrics = channel_metrics(car, window["fs"])
    _save_raw_plot(window, car, reference, out)
    _save_spectra_plot(window, original_metrics, car_metrics, out)

    rows = []
    representative_env = None
    representative_channel = None
    valid_channels = np.flatnonzero(window["valid"])
    for reference_name, values in (("original", window["eeg"]), ("car", car)):
        for ch in valid_channels:
            env = smooth_env(values[:, ch], window["fs"], DEFAULT_FREQ_BANDS)
            finite = np.isfinite(env).all(axis=1)
            clean = env[finite]
            if representative_env is None and reference_name == "original":
                representative_env, representative_channel = env, int(ch)
            result = {}
            for representation in ("binary", "continuous"):
                motifs = get_motifs(
                    [clean.T],
                    fs=window["fs"],
                    n_null=n_null,
                    seed=seed,
                    representation=representation,
                )
                result[representation] = motifs.n_motifs
            metrics = original_metrics if reference_name == "original" else car_metrics
            row = {
                "channel": int(ch),
                "location": window["location"][ch],
                "reference": reference_name,
                "binary_motif_pairs": result["binary"],
                "continuous_motif_pairs": result["continuous"],
                **metrics.iloc[ch].to_dict(),
            }
            row.update({axis: float(values_[ch]) for axis, values_ in window["xyz"].items()})
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "channel_summary.csv", index=False)
    _save_transform_plot(
        representative_env,
        window["fs"],
        representative_channel,
        window["location"][representative_channel],
        out,
    )
    movement = _save_movement_plot(window, representative_env, out)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True, layout="constrained")
    for ax, representation in zip(axes, ("binary", "continuous")):
        field = f"{representation}_motif_pairs"
        for name, group in summary.groupby("reference"):
            ax.plot(group.channel, group[field], "o-", ms=4, label=name)
        ax.set_title(f"{representation.title()} input")
        ax.set_xlabel("EEG channel")
        ax.set_ylabel("Significant motif pairs")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Step 6 — motif recovery across all valid EEG channels")
    fig.savefig(out / "05_motif_counts_all_channels.png", dpi=160)
    plt.close(fig)

    payload = {
        "input": str(path),
        "sampling_rate_hz": window["fs"],
        "n_channels": int(window["eeg"].shape[1]),
        "n_valid_channels": int(window["valid"].sum()),
        "n_null": n_null,
        "seed": seed,
        **movement,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-null", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    table = run_diagnostics(args.input, args.out, args.n_null, args.seed)
    detected = table.groupby("reference")[["binary_motif_pairs", "continuous_motif_pairs"]].apply(
        lambda frame: (frame > 0).sum()
    )
    print(detected.to_string())
    print(f"Outputs -> {args.out}")


if __name__ == "__main__":
    main()
