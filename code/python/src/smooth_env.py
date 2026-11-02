"""Python port of SmoothEnv.m and butterfilter.m (Mishler spectral-motif preprocessing).

Turns one continuous LFP/EEG trace into a time x bands matrix of smoothed,
z-scored Hilbert amplitude envelopes, NaN for the first and last 10 s.
"""

import numpy as np
from scipy import signal

# Placeholder bands (same as TASKS.md). Replace with the bands from the Mishler paper once confirmed.
DEFAULT_FREQ_BANDS = np.array([
    [1.0, 4.0],
    [4.0, 8.0],
    [8.0, 12.0],
    [12.0, 30.0],
    [30.0, 60.0],
    [60.0, 100.0],
], dtype=float)


def mround(x: float) -> int:
    """MATLAB round(): halves go away from zero (Python's round() goes to even)."""
    return int(np.sign(x) * np.floor(abs(x) + 0.5))


def butterfilter(fs: float, low: float, high: float, order: int = 6) -> np.ndarray:
    """Butterworth bandpass as second-order sections, matching butterfilter.m.

    MATLAB's fdesign.bandpass('N,F3dB1,F3dB2', 6, ...) means a 6th-order *bandpass*
    (6 poles). scipy.signal.butter(N, band) builds a 2N-order bandpass, so N = order / 2.
    """
    return signal.butter(order // 2, [low, high], btype="bandpass", fs=fs, output="sos")


def gausswin(n: int, alpha: float = 2.5) -> np.ndarray:
    """MATLAB gausswin(n, alpha): std = (n - 1) / (2 * alpha)."""
    return signal.windows.gaussian(n, std=(n - 1) / (2 * alpha))


def fir_filtfilt(b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Same result as scipy.signal.filtfilt(b, 1, x, axis=0), but uses FFT convolution.

    With a 5 s kernel at 250 Hz (1250 taps), filtfilt's direct-form loop takes tens of
    seconds per channel. For an FIR filter, lfilter with filtfilt's steady-state
    initial conditions is the same as convolving after holding the first sample
    constant for len(b) - 1 samples.
    """
    b = np.asarray(b, dtype=float)
    x = np.asarray(x, dtype=float)
    ntaps = b.size
    padlen = 3 * ntaps
    if x.shape[0] <= padlen:
        raise ValueError(f"signal needs more than {padlen} samples for this kernel")
    b_nd = b.reshape((-1,) + (1,) * (x.ndim - 1))

    def one_pass(y):
        held = np.concatenate([np.repeat(y[:1], ntaps - 1, axis=0), y], axis=0)
        return signal.fftconvolve(held, b_nd, mode="valid", axes=0)

    ext = np.concatenate([
        2 * x[:1] - x[padlen:0:-1],
        x,
        2 * x[-1:] - x[-2:-padlen - 2:-1],
    ], axis=0)
    y = one_pass(ext)
    y = one_pass(y[::-1])[::-1]
    return y[padlen:-padlen]


def smooth_env(
    lfp: np.ndarray,
    fs: float,
    freq_bands: np.ndarray = DEFAULT_FREQ_BANDS,
    smooth_sec: float = 5.0,
    edge_sec: float = 10.0,
    zscore: bool = True,
) -> np.ndarray:
    """Port of SmoothEnv(LFP, LFPfs, freqBands).

    zscore=False returns the smoothed envelopes before the final z-scoring step, so
    several pieces of one recording can be z-scored together.

    Returns a (time, bands) array: each band's Hilbert envelope, smoothed with a
    gausswin(round(5 * fs), 1) kernel applied forward and backward, the first and
    last `edge_sec` seconds set to NaN, then z-scored over time (ignoring NaNs).
    """
    x = np.asarray(lfp, dtype=float).ravel()
    freq_bands = np.atleast_2d(np.asarray(freq_bands, dtype=float))

    filt = np.empty((x.size, len(freq_bands)))
    for f, (low, high) in enumerate(freq_bands):
        filt[:, f] = signal.sosfiltfilt(butterfilter(fs, low, high), x)
    env = np.abs(signal.hilbert(filt, axis=0))

    w = gausswin(mround(smooth_sec * fs), 1)
    w /= w.sum()
    smooth = fir_filtfilt(w, env)

    edge = mround(edge_sec * fs)
    smooth[:edge] = np.nan
    smooth[smooth.shape[0] - edge:] = np.nan
    if not zscore:
        return smooth

    mu = np.nanmean(smooth, axis=0)
    sigma = np.nanstd(smooth, axis=0, ddof=1)
    return (smooth - mu) / sigma
