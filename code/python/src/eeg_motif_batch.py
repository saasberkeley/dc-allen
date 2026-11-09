"""Batch runner for the complete all-channel EEG motif-recovery analysis.

The inputs are extracted HDF5 windows produced by ``extract_windows.py``. Each
window is analyzed independently, then combined into cross-window CSV/JSON
summaries and overview figures. This script is ready for full mouse files once
their awake/iso windows have been extracted.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .eeg_motif_diagnostics import run_diagnostics


def _window_metadata(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        attrs = {key: value.item() if hasattr(value, "item") else value for key, value in f.attrs.items()}
    return {
        "subject": str(attrs.get("subject", "unknown")),
        "session": str(attrs.get("session", path.stem)),
        "state": str(attrs.get("window", path.stem)),
    }


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def detection_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Counts and fractions of channels with motifs for every analysis condition."""
    rows = []
    keys = ["subject", "session", "state", "reference"]
    for values, group in results.groupby(keys, dropna=False):
        base = dict(zip(keys, values))
        for method in ("binary", "continuous"):
            detected = group[f"{method}_motif_pairs"] > 0
            rows.append({
                **base,
                "method": method,
                "channels_with_motif": int(detected.sum()),
                "total_channels": int(len(group)),
                "fraction_with_motif": float(detected.mean()),
            })
    return pd.DataFrame(rows)


def outcome_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Classify exact channels as continuous rescue, binary only, both or neither."""
    table = results[["subject", "session", "state", "reference", "channel", "location",
                     "binary_motif_pairs", "continuous_motif_pairs"]].copy()
    binary = table.binary_motif_pairs > 0
    continuous = table.continuous_motif_pairs > 0
    table["outcome"] = np.select(
        [~binary & continuous, binary & ~continuous, binary & continuous],
        ["continuous rescue", "binary only", "both"],
        default="neither",
    )
    return table


def _save_overview_plots(results: pd.DataFrame, detections: pd.DataFrame, out: Path) -> None:
    labels = detections.subject + " | " + detections.state
    unique = list(dict.fromkeys(labels))
    fig, axes = plt.subplots(1, len(unique), figsize=(max(7, 5 * len(unique)), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, label in zip(axes, unique):
        subset = detections[labels == label].copy()
        subset["condition"] = subset.reference + " + " + subset.method
        ax.bar(subset.condition, subset.fraction_with_motif, color=["#777", "#4c78a8", "#aaa", "#2f9e44"])
        ax.set(title=label, ylabel="Fraction of valid channels with motif", ylim=(0, 1))
        ax.tick_params(axis="x", rotation=35)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("All-channel EEG motif recovery across windows")
    fig.tight_layout()
    fig.savefig(out / "batch_detection_fraction.png", dpi=170)
    plt.close(fig)

    if "gmm_mean_seed_ari" in results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        for (subject, state, reference), group in results.groupby(["subject", "state", "reference"]):
            label = f"{subject} {state} {reference}"
            axes[0].plot(group.channel, group.gmm_mean_seed_ari, "o-", ms=3, label=label)
            axes[1].plot(group.channel, group.gmm_early_late_occupancy_js, "o-", ms=3, label=label)
        axes[0].set(title="GMM repeatability", xlabel="EEG channel", ylabel="Mean pairwise ARI", ylim=(-0.05, 1.05))
        axes[1].set(title="Early/late GMM occupancy change", xlabel="EEG channel",
                    ylabel="Jensen–Shannon distance", ylim=(-0.05, 1.05))
        axes[0].legend(frameon=False, fontsize=7)
        fig.savefig(out / "batch_gmm_quality.png", dpi=170)
        plt.close(fig)


def run_batch(
    inputs: list[Path],
    out_dir: Path,
    *,
    n_null: int = 20,
    seed: int = 0,
    gmm_k: int | None = 5,
    gmm_k_max: int = 8,
    gmm_seeds: int = 5,
    transition: bool = True,
    skip_existing: bool = False,
) -> pd.DataFrame:
    """Analyze all windows and return the combined per-channel table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    manifest = []
    for input_path in inputs:
        input_path = Path(input_path)
        meta = _window_metadata(input_path)
        label = _safe_name(f"{meta['subject']}_{meta['state']}")
        window_out = out_dir / label
        csv_path = window_out / "channel_summary.csv"
        if skip_existing and csv_path.exists():
            frame = pd.read_csv(csv_path)
        else:
            frame = run_diagnostics(
                input_path,
                window_out,
                n_null=n_null,
                seed=seed,
                gmm_k=gmm_k,
                gmm_k_max=gmm_k_max,
                gmm_seeds=gmm_seeds,
                transition=transition,
            )
        for key, value in meta.items():
            frame[key] = value
        frame["input"] = str(input_path)
        frames.append(frame)
        manifest.append({**meta, "input": str(input_path), "output": str(window_out)})

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_dir / "all_windows_channel_summary.csv", index=False)
    detections = detection_summary(combined)
    detections.to_csv(out_dir / "detection_summary.csv", index=False)
    outcomes = outcome_summary(combined)
    outcomes.to_csv(out_dir / "channel_outcomes.csv", index=False)
    _save_overview_plots(combined, detections, out_dir)

    payload = {
        "inputs": manifest,
        "parameters": {
            "n_null": n_null,
            "seed": seed,
            "gmm_k": gmm_k,
            "gmm_k_max": gmm_k_max if gmm_k is not None else None,
            "gmm_seeds": gmm_seeds if gmm_k is not None else None,
            "transition": transition,
        },
        "detection_summary": detections.to_dict(orient="records"),
    }
    (out_dir / "batch_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return combined


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("inputs", nargs="+", type=Path, help="extracted HDF5 windows")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-null", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gmm-k", type=int, default=5, help="0 disables GMM diagnostics")
    parser.add_argument("--gmm-k-max", type=int, default=8)
    parser.add_argument("--gmm-seeds", type=int, default=5)
    parser.add_argument("--no-transition", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)
    results = run_batch(
        args.inputs,
        args.out,
        n_null=args.n_null,
        seed=args.seed,
        gmm_k=args.gmm_k or None,
        gmm_k_max=args.gmm_k_max,
        gmm_seeds=args.gmm_seeds,
        transition=not args.no_transition,
        skip_existing=args.skip_existing,
    )
    print(detection_summary(results).to_string(index=False))
    print(f"Outputs -> {args.out}")


if __name__ == "__main__":
    main()
