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
import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import signal  # noqa: E402
from scipy.spatial.distance import jensenshannon  # noqa: E402

from .get_motifs import get_motifs  # noqa: E402
from .gmm_motifs import bic_sweep, fit_gmm_motifs, seed_stability, shift_bands  # noqa: E402
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
        def optional(name: str) -> np.ndarray:
            return np.asarray(f[name], dtype=float) if name in f else np.array([], dtype=float)

        running = optional("behavior/running_speed")
        running_ts = optional("behavior/running_ts")
        pupil = optional("behavior/pupil_width")
        pupil_ts = optional("behavior/pupil_ts")
        iso_level = optional("behavior/iso_level")
        iso_ts = optional("behavior/iso_ts")
        stim_ts = optional("behavior/stim_ts")
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
        "pupil": pupil,
        "pupil_ts": pupil_ts,
        "iso_level": iso_level,
        "iso_ts": iso_ts,
        "stim_ts": stim_ts,
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


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation, returning NaN for missing or constant inputs."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.ptp(x[keep]) == 0 or np.ptp(y[keep]) == 0:
        return float("nan")
    return float(pd.Series(x[keep]).corr(pd.Series(y[keep]), method="spearman"))


def _interp_optional(values: np.ndarray, times: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Interpolate an optional time series, returning NaNs when it is unavailable."""
    if len(values) == 0 or len(times) == 0:
        return np.full(len(target), np.nan)
    keep = np.isfinite(values) & np.isfinite(times)
    if keep.sum() < 2:
        return np.full(len(target), np.nan)
    return np.interp(target, times[keep], values[keep], left=np.nan, right=np.nan)


def behavior_metrics(window: dict, env: np.ndarray) -> dict:
    """Movement, pupil, anesthesia and stimulation summaries for one channel envelope."""
    theta_index = int(np.argmin(np.abs(DEFAULT_FREQ_BANDS.mean(axis=1) - 7)))
    theta = env[:, theta_index]
    good = np.isfinite(theta)
    t = window["ts"][good]
    theta = theta[good]
    speed = _interp_optional(window["running"], window["running_ts"], t)
    pupil = _interp_optional(window["pupil"], window["pupil_ts"], t)
    iso = _interp_optional(window["iso_level"], window["iso_ts"], t)

    stim_deltas = []
    for stim in window["stim_ts"]:
        pre = (t >= stim - 2) & (t < stim)
        post = (t >= stim) & (t < stim + 2)
        if pre.any() and post.any():
            stim_deltas.append(float(np.nanmean(theta[post]) - np.nanmean(theta[pre])))

    return {
        "theta_speed_spearman": _safe_spearman(theta, speed),
        "theta_pupil_spearman": _safe_spearman(theta, pupil),
        "theta_stim_delta_z": float(np.nanmean(stim_deltas)) if stim_deltas else float("nan"),
        "median_running_speed": float(np.nanmedian(speed)) if np.isfinite(speed).any() else float("nan"),
        "fraction_running_gt_2": float(np.nanmean(speed > 2)) if np.isfinite(speed).any() else float("nan"),
        "median_pupil_width": float(np.nanmedian(pupil)) if np.isfinite(pupil).any() else float("nan"),
        "median_iso_level": float(np.nanmedian(iso)) if np.isfinite(iso).any() else float("nan"),
        "n_stim": int(len(window["stim_ts"])),
    }


def gmm_metrics(env_1hz: np.ndarray, k: int = 5, k_max: int = 8, n_seeds: int = 5) -> dict:
    """GMM model selection, assignment confidence, null comparison and seed stability."""
    env_1hz = np.asarray(env_1hz, float)
    max_k = min(int(k_max), len(env_1hz))
    fixed_k = min(int(k), max_k)
    if fixed_k < 1 or max_k < 1:
        raise ValueError("GMM needs at least one timepoint and positive K")

    ks, bics = bic_sweep(env_1hz, range(1, max_k + 1), seed=0, n_init=3)
    result = fit_gmm_motifs(env_1hz, fixed_k, seed=0, n_init=5)
    shifted = shift_bands(env_1hz)
    shifted_result = fit_gmm_motifs(shifted, fixed_k, seed=0, n_init=5)
    if n_seeds >= 2:
        scores, mean_ari = seed_stability(env_1hz, fixed_k, seeds=range(n_seeds), n_init=1)
    else:
        scores, mean_ari = np.array([], dtype=float), float("nan")

    third = max(1, len(result.labels) // 3)
    early = np.bincount(result.labels[:third], minlength=fixed_k).astype(float)
    late = np.bincount(result.labels[-third:], minlength=fixed_k).astype(float)
    early /= early.sum()
    late /= late.sum()
    occupancy_js = float(jensenshannon(early, late, base=2))

    return {
        "gmm_k": fixed_k,
        "gmm_bic_best_k": int(ks[np.argmin(bics)]),
        "gmm_bic_at_k": float(result.bic),
        "gmm_shifted_bic_minus_real": float(shifted_result.bic - result.bic),
        "gmm_median_confidence": float(np.median(result.confidence)),
        "gmm_mean_seed_ari": mean_ari,
        "gmm_min_seed_ari": float(scores.min()) if scores.size else float("nan"),
        "gmm_early_late_occupancy_js": occupancy_js,
        "gmm_centers_json": json.dumps(result.centers.tolist()),
        "gmm_early_occupancy_json": json.dumps(early.tolist()),
        "gmm_late_occupancy_json": json.dumps(late.tolist()),
    }


def transition_metrics(clean_env: np.ndarray, fs: float, n_null: int, seed: int) -> dict:
    """Compare motif recovery in matched early and late thirds of one window."""
    n = len(clean_env)
    third = n // 3
    if third < max(3, int(round(fs)) * 5):
        return {
            f"{period}_{representation}_motif_pairs": float("nan")
            for period in ("early", "late")
            for representation in ("binary", "continuous")
        }

    output = {}
    for period, values in (("early", clean_env[:third]), ("late", clean_env[-third:])):
        for representation in ("binary", "continuous"):
            try:
                result = get_motifs(
                    [values.T], fs=fs, n_null=n_null, seed=seed, representation=representation
                )
                output[f"{period}_{representation}_motif_pairs"] = int(result.n_motifs)
            except (ValueError, FloatingPointError) as exc:
                warnings.warn(f"{period} {representation} transition analysis failed: {exc}")
                output[f"{period}_{representation}_motif_pairs"] = float("nan")
    return output


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
    t = window["ts"][good]
    metrics = behavior_metrics(window, env)
    rows = [("theta", t, theta[good])]
    if len(window["running"]):
        rows.append(("running", window["running_ts"], window["running"]))
    if len(window["pupil"]):
        rows.append(("pupil", window["pupil_ts"], window["pupil"]))
    if len(window["iso_level"]):
        rows.append(("isoflurane", window["iso_ts"], window["iso_level"]))

    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 2.1 * len(rows)), sharex=True, layout="constrained")
    axes = np.atleast_1d(axes)
    axes[0].plot(t - window["ts"][0], theta[good], color="#6a3d9a", lw=1)
    axes[0].set_ylabel("4–8 Hz envelope (z)")
    row_index = 1
    if len(window["running"]):
        axes[row_index].plot(window["running_ts"] - window["ts"][0], window["running"], color="#1b9e77", lw=1)
        axes[row_index].set_ylabel("running speed")
        row_index += 1
    if len(window["pupil"]):
        axes[row_index].plot(window["pupil_ts"] - window["ts"][0], window["pupil"], color="#e7298a", lw=1)
        axes[row_index].set_ylabel("pupil width")
        row_index += 1
    if len(window["iso_level"]):
        axes[row_index].plot(window["iso_ts"] - window["ts"][0], window["iso_level"], color="#d95f02", lw=1)
        axes[row_index].set_ylabel("isoflurane")
    for ax in axes:
        for stim in window["stim_ts"]:
            ax.axvline(stim - window["ts"][0], color="C3", alpha=0.35, lw=0.7)
    axes[-1].set_xlabel("Time in window (s)")
    speed_rho = metrics["theta_speed_spearman"]
    title_rho = "undefined" if not np.isfinite(speed_rho) else f"{speed_rho:.3f}"
    fig.suptitle(f"Behavior check — Spearman rho(theta envelope, speed) = {title_rho}")
    fig.savefig(out / "04_theta_vs_running.png", dpi=160)
    plt.close(fig)
    return {**metrics, "theta_band_index": theta_index}


def run_diagnostics(
    path: str | Path,
    out_dir: str | Path,
    n_null: int = 3,
    seed: int = 0,
    *,
    gmm_k: int | None = None,
    gmm_k_max: int = 8,
    gmm_seeds: int = 5,
    transition: bool = False,
) -> pd.DataFrame:
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
                **behavior_metrics(window, env),
            }
            if gmm_k is not None:
                row.update(gmm_metrics(_one_hz(clean, window["fs"]), gmm_k, gmm_k_max, gmm_seeds))
            if transition:
                row.update(transition_metrics(clean, window["fs"], n_null, seed))
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

    if gmm_k is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
        for name, group in summary.groupby("reference"):
            axes[0].plot(group.channel, group.gmm_mean_seed_ari, "o-", ms=3, label=name)
            axes[1].plot(group.channel, group.gmm_median_confidence, "o-", ms=3, label=name)
            axes[2].plot(group.channel, group.gmm_early_late_occupancy_js, "o-", ms=3, label=name)
        axes[0].set(ylabel="Mean pairwise ARI", title=f"GMM K={gmm_k}: seed stability", ylim=(-0.05, 1.05))
        axes[1].set(ylabel="Median assignment probability", title="Assignment confidence", ylim=(-0.05, 1.05))
        axes[2].set(ylabel="Jensen–Shannon distance", title="Early vs late occupancy", ylim=(-0.05, 1.05))
        for ax in axes:
            ax.set_xlabel("EEG channel")
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].legend(frameon=False)
        fig.suptitle("Step 7 — GMM quality checks across all valid EEG channels")
        fig.savefig(out / "06_gmm_quality_all_channels.png", dpi=160)
        plt.close(fig)

    if transition:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True, layout="constrained")
        for ax, representation in zip(axes, ("binary", "continuous")):
            for reference_name, group in summary.groupby("reference"):
                ax.plot(group.channel, group[f"early_{representation}_motif_pairs"], "o-", ms=3,
                        label=f"{reference_name}: early")
                ax.plot(group.channel, group[f"late_{representation}_motif_pairs"], "s--", ms=3,
                        label=f"{reference_name}: late")
            ax.set(xlabel="EEG channel", ylabel="Significant motif pairs", title=representation.title())
            ax.legend(frameon=False, fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle("Step 8 — early versus late window motif recovery")
        fig.savefig(out / "07_early_vs_late_all_channels.png", dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), layout="constrained")
    fields = [
        ("robust_sd_uv", "Robust SD"),
        ("theta_fraction", "Theta fraction"),
        ("line_noise_fraction", "60 Hz fraction"),
        ("spectral_slope_2_40hz", "Spectral slope"),
    ]
    original = summary[summary.reference == "original"].copy()
    original["continuous_detected"] = original.continuous_motif_pairs > 0
    for ax, (field, label) in zip(axes, fields):
        groups = [original.loc[~original.continuous_detected, field].dropna(),
                  original.loc[original.continuous_detected, field].dropna()]
        if all(len(group) for group in groups):
            ax.boxplot(groups, tick_labels=["no motif", "motif"], showmeans=True)
        else:
            ax.text(0.5, 0.5, "Both outcome groups\nnot present", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set(title=label, ylabel=label)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Step 9 — do continuous-motif successes differ from failures?")
    fig.savefig(out / "08_success_vs_failure_eda.png", dpi=160)
    plt.close(fig)

    payload = {
        "input": str(path),
        "sampling_rate_hz": window["fs"],
        "n_channels": int(window["eeg"].shape[1]),
        "n_valid_channels": int(window["valid"].sum()),
        "n_null": n_null,
        "seed": seed,
        "gmm_k": gmm_k,
        "gmm_k_max": gmm_k_max if gmm_k is not None else None,
        "gmm_seeds": gmm_seeds if gmm_k is not None else None,
        "transition": transition,
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
    parser.add_argument("--gmm-k", type=int, default=0, help="fixed K for GMM diagnostics; 0 disables")
    parser.add_argument("--gmm-k-max", type=int, default=8, help="largest K in the BIC sweep")
    parser.add_argument("--gmm-seeds", type=int, default=5)
    parser.add_argument("--transition", action="store_true", help="compare early and late thirds")
    args = parser.parse_args(argv)
    table = run_diagnostics(
        args.input,
        args.out,
        args.n_null,
        args.seed,
        gmm_k=args.gmm_k or None,
        gmm_k_max=args.gmm_k_max,
        gmm_seeds=args.gmm_seeds,
        transition=args.transition,
    )
    detected = table.groupby("reference")[["binary_motif_pairs", "continuous_motif_pairs"]].apply(
        lambda frame: (frame > 0).sum()
    )
    print(detected.to_string())
    print(f"Outputs -> {args.out}")


if __name__ == "__main__":
    main()
