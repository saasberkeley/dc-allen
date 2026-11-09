import numpy as np
from scipy import signal

from src.smooth_env import butterfilter, fir_filtfilt, gausswin, mround, smooth_env


def test_mround_matches_matlab():
    assert [mround(x) for x in (0.5, 1.5, 2.5, -0.5, 2.4)] == [1, 2, 3, -1, 2]


def test_butterfilter_is_6th_order_with_3db_edges():
    fs, low, high = 250.0, 8.0, 12.0
    sos = butterfilter(fs, low, high)
    assert sos.shape == (3, 6)  # 6 poles, as fdesign.bandpass('N,F3dB1,F3dB2', 6, ...)
    _, h = signal.sosfreqz(sos, worN=[low, high], fs=fs)
    np.testing.assert_allclose(20 * np.log10(np.abs(h)), -3.0103, atol=1e-3)


def test_gausswin_matches_matlab_formula():
    n = 1250
    k = np.arange(n) - (n - 1) / 2
    expected = np.exp(-0.5 * (1 * k / ((n - 1) / 2)) ** 2)  # MATLAB gausswin(n, 1)
    np.testing.assert_allclose(gausswin(n, 1), expected, rtol=1e-12)


def test_fir_filtfilt_matches_scipy_filtfilt():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3000, 4)).cumsum(axis=0)
    w = gausswin(250, 1)
    w /= w.sum()
    np.testing.assert_allclose(fir_filtfilt(w, x), signal.filtfilt(w, 1, x, axis=0), atol=1e-9)


def test_smooth_env_shape_edges_and_zscore():
    fs = 250.0
    rng = np.random.default_rng(0)
    x = rng.standard_normal(int(60 * fs))
    bands = np.array([[4, 8], [8, 12], [30, 50]])
    out = smooth_env(x, fs, bands)

    edge = int(10 * fs)
    assert out.shape == (x.size, 3)
    assert np.isnan(out[:edge]).all() and np.isnan(out[-edge:]).all()
    middle = out[edge:-edge]
    assert np.isfinite(middle).all()
    np.testing.assert_allclose(middle.mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(middle.std(axis=0, ddof=1), 1, atol=1e-10)


def test_smooth_env_tracks_band_amplitude():
    """A 10 Hz burst in the second half should raise only the 8-12 Hz envelope."""
    fs = 250.0
    t = np.arange(int(400 * fs)) / fs
    rng = np.random.default_rng(0)
    x = 0.1 * rng.standard_normal(t.size) + np.where(t > 200, np.sin(2 * np.pi * 10 * t), 0)
    out = smooth_env(x, fs, np.array([[2, 4], [8, 12], [30, 50]]))

    early = np.nanmean(out[(t > 20) & (t < 190)], axis=0)
    late = np.nanmean(out[(t > 210) & (t < 380)], axis=0)
    assert late[1] - early[1] > 1.5
    assert abs(late[0] - early[0]) < 0.5 and abs(late[2] - early[2]) < 0.5
