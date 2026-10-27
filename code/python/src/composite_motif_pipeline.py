"""Composite Mishler spectral-motif pipeline for MATLAB .mat files.

This module processes LFP data from MATLAB .mat files through the full
Mishler spectral-motif pipeline:

1. Load LFP and fs from a .mat file
2. Flatten LFP exactly like MATLAB LFP(:), using column-major order
3. Apply a zero-phase 59-61 Hz bandstop to each raw LFP trace
4. Run smooth_env: [1,5], [2,6], ..., [76,80] Hz bands, Hilbert amplitude,
   5-s Gaussian smoothing, first/last 10 s set to NaN, within-band z-scoring
5. After smooth_env, identify noisy times from the bandstopped LFP using
   filter_lfp_noise and set those time indices in smooth_env output to NaN
6. Remove every time point containing any NaN
7. Run get_motifs on the cleaned sessions
8. Save smoothLFPNorm, finalMotifs, fs, and diagnostics to a new .mat file

Usage (from code/python/):
    python -m src.composite_motif_pipeline --mat /path/to/LFP_Data.mat --out /path/to/output
    python -m src.composite_motif_pipeline --help

Dependencies: numpy, scipy
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import time
import warnings

import numpy as np
from scipy import signal
from scipy.cluster.hierarchy import linkage
from scipy.io import loadmat, savemat
from scipy.spatial.distance import pdist

# Reuse existing modules where possible
from .smooth_env import (
    mround,
    butterfilter,
    gausswin,
    fir_filtfilt,
)
from .get_motifs import (
    get_motifs,
    MotifResult,
    CORR_THRESHOLDS,
)


# -------------------------------------------------------------------------
# Default configuration
# -------------------------------------------------------------------------

# The 76 overlapping bands from [1,5] to [76,80] Hz, matching the MATLAB pipeline
DEFAULT_FREQ_BANDS = np.column_stack((
    np.arange(1.0, 77.0),
    np.arange(5.0, 81.0),
))

# Matches the MATLAB filterLFPNoise usage
DEFAULT_STDEV_THRESH = 10.0

# Default getMotifs settings
DEFAULT_N_NULL = 20


# -------------------------------------------------------------------------
# Filtering / Preprocessing
# -------------------------------------------------------------------------

def bandstop_60hz(x: np.ndarray, fs: float) -> np.ndarray:
    """Zero-phase 59-61 Hz bandstop performed immediately after LFP import.

    Parameters
    ----------
    x : np.ndarray
        Raw LFP trace (1-D)
    fs : float
        Sampling rate in Hz (must be > 122 Hz)

    Returns
    -------
    np.ndarray
        Filtered LFP trace with 60 Hz line noise removed
    """
    if fs / 2 <= 61:
        raise ValueError(
            "Sampling rate must be greater than 122 Hz for a 59-61 Hz bandstop."
        )

    sos = signal.butter(
        3,
        [59.0, 61.0],
        btype="bandstop",
        fs=fs,
        output="sos",
    )
    return signal.sosfiltfilt(sos, np.asarray(x, dtype=float))


def smooth_env(
    lfp: np.ndarray,
    fs: float,
    freq_bands: np.ndarray = DEFAULT_FREQ_BANDS,
    smooth_sec: float = 5.0,
    edge_sec: float = 10.0,
) -> np.ndarray:
    """SmoothEnv port for 76-band composite pipeline.

    Input is expected to have already received the 59-61 Hz bandstop.
    Returns time x bands.

    Parameters
    ----------
    lfp : np.ndarray
        LFP trace (1-D), already bandstop-filtered at 60 Hz
    fs : float
        Sampling rate in Hz
    freq_bands : np.ndarray
        (n_bands, 2) array of [low, high] Hz edges. Default: 76 bands from
        [1,5] to [76,80] Hz
    smooth_sec : float
        Gaussian smoothing kernel width in seconds (default: 5.0)
    edge_sec : float
        Seconds at start/end to set to NaN (default: 10.0)

    Returns
    -------
    np.ndarray
        (time, bands) array of z-scored smoothed envelopes
    """
    x = np.asarray(lfp, dtype=float).ravel()
    freq_bands = np.atleast_2d(np.asarray(freq_bands, dtype=float))

    filt = np.empty((x.size, len(freq_bands)))
    for f, (low, high) in enumerate(freq_bands):
        filt[:, f] = signal.sosfiltfilt(
            butterfilter(fs, low, high),
            x,
        )

    env = np.abs(signal.hilbert(filt, axis=0))

    w = gausswin(mround(smooth_sec * fs), 1)
    w /= w.sum()
    smooth = fir_filtfilt(w, env)

    edge = mround(edge_sec * fs)
    smooth[:edge, :] = np.nan
    smooth[-edge:, :] = np.nan

    mu = np.nanmean(smooth, axis=0)
    sigma = np.nanstd(smooth, axis=0, ddof=1)
    return (smooth - mu) / sigma


def filter_lfp_noise(
    fs: float,
    orig_lfp: np.ndarray,
    stdev_thresh: float = DEFAULT_STDEV_THRESH,
) -> np.ndarray:
    """Port of filterLFPNoise.m.

    Detects |LFP| > mean(|LFP|) + stdev_thresh * SD(|LFP|), expands each
    detected point by +/- 3 s, and also includes the first and last 10 s.

    Parameters
    ----------
    fs : float
        Sampling rate in Hz
    orig_lfp : np.ndarray
        LFP trace (1-D), typically after bandstop filtering
    stdev_thresh : float
        Threshold in standard deviations (default: 10.0)

    Returns
    -------
    np.ndarray
        Zero-based indices of noisy time points to exclude
    """
    x = np.asarray(orig_lfp, dtype=float).ravel()

    noise_samp = mround(3.0 * fs)
    snip_time = np.arange(-noise_samp, noise_samp + 1)

    mag = np.abs(x)
    threshold = np.nanmean(mag) + stdev_thresh * np.nanstd(mag, ddof=1)
    noise_idx = np.flatnonzero(mag > threshold)

    if noise_idx.size:
        artifact_idx = (noise_idx[:, None] + snip_time[None, :]).ravel()
        artifact_idx = artifact_idx[
            (artifact_idx >= 0) & (artifact_idx < x.size)
        ]
        artifact_idx = np.unique(artifact_idx)
    else:
        artifact_idx = np.array([], dtype=int)

    edge_samp = mround(10.0 * fs)
    start_idx = np.arange(0, min(edge_samp, x.size))
    end_idx = np.arange(max(0, x.size - edge_samp), x.size)

    return np.unique(
        np.concatenate((start_idx, artifact_idx, end_idx))
    )


# -------------------------------------------------------------------------
# MAT-file preprocessing
# -------------------------------------------------------------------------

def _scalar_fs(value) -> float:
    """Extract scalar sampling rate from MAT file variable."""
    arr = np.asarray(value, dtype=float).squeeze()
    if arr.size != 1:
        raise ValueError("'fs' must be a scalar sampling rate.")
    fs = float(arr)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("'fs' must be positive and finite.")
    return fs


def _flatten_lfp_cells(lfp) -> np.ndarray:
    """Flatten the cell matrix exactly like MATLAB LFP(:).

    Uses column-major (Fortran) order to match MATLAB's linear indexing.
    """
    arr = np.asarray(lfp)

    if arr.dtype != object:
        raise ValueError(
            "'LFP' must be a MATLAB cell array with one continuous trace per cell."
        )

    return arr.reshape(-1, order="F")


def preprocess_lfp_sessions(
    lfp,
    fs: float,
    stdev_thresh: float = DEFAULT_STDEV_THRESH,
    freq_bands: np.ndarray = DEFAULT_FREQ_BANDS,
    verbose: bool = True,
):
    """Preprocess LFP sessions from a MATLAB cell array.

    For each non-empty session:
    1. Apply 60 Hz bandstop
    2. Run smooth_env
    3. Detect noisy times and set to NaN
    4. Remove all NaN timepoints

    Parameters
    ----------
    lfp : np.ndarray
        MATLAB cell array of LFP traces (animals x sessions)
    fs : float
        Sampling rate in Hz
    stdev_thresh : float
        Noise detection threshold in standard deviations
    freq_bands : np.ndarray
        Frequency bands for smooth_env
    verbose : bool
        Print progress messages

    Returns
    -------
    list[np.ndarray]
        List of cleaned (bands, time) arrays, one per session
    """
    sessions = _flatten_lfp_cells(lfp)
    cleaned_sessions = []

    for i, cell in enumerate(sessions, start=1):
        raw = np.asarray(cell, dtype=float).squeeze()

        if raw.size == 0:
            raise ValueError(f"LFP session {i} is empty.")
        if raw.ndim != 1:
            raise ValueError(
                f"LFP session {i} must be one continuous 1-D trace; got {raw.shape}."
            )
        if not np.isfinite(raw).all():
            raise ValueError(
                f"LFP session {i} contains NaN or Inf before preprocessing."
            )

        # Bandstop happens immediately after the session is imported
        lfp_bandstopped = bandstop_60hz(raw, fs)

        # KEY ORDER: SmoothEnv is completed before noise detection
        smooth = smooth_env(lfp_bandstopped, fs, freq_bands)

        # Only now identify noisy times
        noise_idx = filter_lfp_noise(
            fs=fs,
            orig_lfp=lfp_bandstopped,
            stdev_thresh=stdev_thresh,
        )

        # Add noise NaNs to the NaNs already produced by SmoothEnv
        smooth[noise_idx, :] = np.nan

        # Convert time x bands -> bands x time
        smooth = smooth.T

        # Remove every time column containing a NaN
        keep = np.isfinite(smooth).all(axis=0)
        clean = smooth[:, keep]

        if clean.shape[1] == 0:
            raise ValueError(f"LFP session {i} has no samples remaining after cleaning.")

        cleaned_sessions.append(clean)

        if verbose:
            print(
                f"Preprocessed session {i}/{len(sessions)}: "
                f"{raw.size} raw -> {clean.shape[1]} clean samples"
            )

    return cleaned_sessions


# -------------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------------

def run_pipeline(
    mat_file: Path,
    save_dir: Path,
    output_file: Path | None = None,
    stdev_thresh: float = DEFAULT_STDEV_THRESH,
    n_null: int = DEFAULT_N_NULL,
    rng_seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """Run the full composite motif pipeline on a MAT file.

    Parameters
    ----------
    mat_file : Path
        Input .mat file containing 'LFP' (cell array) and 'LFPfs' (scalar)
    save_dir : Path
        Directory for output files
    output_file : Path, optional
        Custom output filename. If None, uses <input_stem>_motifs.mat
    stdev_thresh : float
        Noise detection threshold (default: 10.0)
    n_null : int
        Number of shifted null datasets for motif validation (default: 20)
    rng_seed : int, optional
        Random seed for reproducibility
    verbose : bool
        Print progress messages

    Returns
    -------
    dict
        Dictionary with keys: 'motifs', 'n_motifs', 'output_file'
    """
    mat_file = Path(mat_file)
    save_dir = Path(save_dir)

    if not mat_file.exists():
        raise FileNotFoundError(
            f"MAT file does not exist:\n{mat_file}"
        )

    try:
        mat = loadmat(mat_file, squeeze_me=False, struct_as_record=False)
    except NotImplementedError as exc:
        raise RuntimeError(
            "scipy.io.loadmat cannot read MATLAB v7.3 HDF5 MAT files. "
            "Save the input as a pre-v7.3 MAT file before running this script."
        ) from exc

    if "LFP" not in mat:
        raise KeyError("The MAT file does not contain a variable named 'LFP'.")
    if "LFPfs" not in mat:
        raise KeyError("The MAT file does not contain a variable named 'LFPfs'.")

    fs = _scalar_fs(mat["LFPfs"])

    if verbose:
        print(f"Loaded: {mat_file}")
        print(f"Sampling rate: {fs:g} Hz")

    smooth_lfp_norm = preprocess_lfp_sessions(
        lfp=mat["LFP"],
        fs=fs,
        stdev_thresh=stdev_thresh,
        verbose=verbose,
    )

    if verbose:
        print("\nRunning getMotifs...")

    motif_result = get_motifs(
        lfp_in=smooth_lfp_norm,
        fs=fs,
        n_null=n_null,
        seed=rng_seed,
        verbose=verbose,
    )

    final_motifs = motif_result.motifs

    # MATLAB-style N x 1 cell vector of cleaned sessions
    smooth_cell = np.empty((len(smooth_lfp_norm), 1), dtype=object)
    for i, session in enumerate(smooth_lfp_norm):
        smooth_cell[i, 0] = session

    save_dir.mkdir(parents=True, exist_ok=True)

    out_path = (
        save_dir / Path(output_file).name
        if output_file is not None
        else save_dir / f"{mat_file.stem}_motifs.mat"
    )

    # Variable-length candidate-size arrays are saved as a MATLAB cell array
    candidate_sizes = np.empty((len(motif_result.candidate_sizes), 1), dtype=object)
    for i, sizes in enumerate(motif_result.candidate_sizes):
        candidate_sizes[i, 0] = np.asarray(sizes)

    savemat(
        out_path,
        {
            "smoothLFPNorm": smooth_cell,
            "finalMotifs": final_motifs,
            "fs": np.array([[fs]], dtype=float),
            "corrThresholds": np.asarray(motif_result.thresholds, dtype=float),
            "nullMaxSize": motif_result.null_max_size,
            "candidateSizes": candidate_sizes,
            "nKept": motif_result.n_kept,
            "nTimepoints": np.array([[motif_result.n_timepoints]], dtype=float),
            "shifts": motif_result.shifts,
        },
        do_compression=True,
    )

    if verbose:
        print(f"\nFinal motifs: {motif_result.n_motifs}")
        print(f"Saved: {out_path}")

    return {
        "motifs": final_motifs,
        "n_motifs": motif_result.n_motifs,
        "output_file": out_path,
        "result": motif_result,
        "smooth_lfp_norm": smooth_lfp_norm,
    }


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Composite Mishler spectral-motif pipeline for MATLAB .mat files"
    )
    parser.add_argument(
        "--mat", type=Path, required=True,
        help="Input MAT file containing 'LFP' (cell array) and 'LFPfs' (scalar)"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/out/motifs"),
        help="Output directory (default: data/out/motifs)"
    )
    parser.add_argument(
        "--output-file", type=Path, default=None,
        help="Custom output filename (default: <input>_motifs.mat)"
    )
    parser.add_argument(
        "--stdev-thresh", type=float, default=DEFAULT_STDEV_THRESH,
        help=f"Noise detection threshold in stdevs (default: {DEFAULT_STDEV_THRESH})"
    )
    parser.add_argument(
        "--n-null", type=int, default=DEFAULT_N_NULL,
        help=f"Number of shifted null datasets (default: {DEFAULT_N_NULL})"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress messages"
    )

    args = parser.parse_args()

    run_pipeline(
        mat_file=args.mat,
        save_dir=args.out,
        output_file=args.output_file,
        stdev_thresh=args.stdev_thresh,
        n_null=args.n_null,
        rng_seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
