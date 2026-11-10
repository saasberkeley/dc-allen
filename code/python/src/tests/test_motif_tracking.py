import numpy as np
import pandas as pd

from src import motif_tracking as mt


def test_local_linear_follows_a_line_to_the_edges():
    x = np.arange(200, dtype=float)
    y = np.column_stack([0.03 * x - 2, -0.01 * x + 1])
    np.testing.assert_allclose(mt.local_linear(y, 50), y, atol=1e-9)


def test_rolling_mean_lags_at_edges_but_local_linear_does_not():
    y = (0.02 * np.arange(300.0))[:, None]
    rm = pd.DataFrame(y).rolling(150, center=True, min_periods=37).mean().to_numpy()
    ll = mt.local_linear(y, 75)
    assert abs(rm[-1, 0] - y[-1, 0]) > 0.5           # the old baseline trails the trend at the end
    assert abs(ll[-1, 0] - y[-1, 0]) < 1e-9


def test_baseline_does_not_leak_across_a_cut():
    t = np.arange(600.0)
    Z = np.where(t < 300, -1.0, 2.0)[:, None] + 0.01 * np.random.default_rng(0).standard_normal((600, 1))
    seg = mt.segments_of(t, [300.0])
    base, _ = mt.split_baseline(Z, seg, 300)
    assert np.abs(base[:300] + 1).max() < 0.05
    assert np.abs(base[300:] - 2).max() < 0.05


def test_short_segment_keeps_a_moving_baseline():
    t = np.arange(120.0)
    Z = (0.05 * t)[:, None]
    base, _ = mt.split_baseline(Z, np.zeros(120, int), 300)   # segment shorter than the window
    np.testing.assert_allclose(base, Z, atol=1e-9)


def test_link_joins_similar_axes_and_starts_new_tracks():
    a = np.array([1, 1, 1, -1, -1, -1, 1, 1], float); a /= np.linalg.norm(a)
    b = np.array([1, -1, 1, -1, 1, -1, 1, -1], float); b /= np.linalg.norm(b)
    noisy = a + 0.05 * np.random.default_rng(1).standard_normal(8)
    cands = pd.DataFrame({"window": [0, 1, 1, 3], "centre": [0, 60, 60, 180], "p": [0.01] * 4,
                          "axis": [a, -noisy / np.linalg.norm(noisy), b, a]})
    out = mt.link(cands, mt.TrackConfig(max_gap=1))
    assert out.track[0] == out.track[1]           # sign flips don't matter
    assert out.track[2] != out.track[0]           # orthogonal axis starts a new track
    assert out.track[3] == out.track[0]           # a one-window gap is bridged


def _synthetic(seed=0, n_bands=8, seconds=3600):
    """Motif A switches poles all recording; motif B only in the second half; slow drift along A in the
    first half that the baseline should absorb."""
    rng = np.random.default_rng(seed)
    A = np.array([1, 1, 1, -1, -1, -1, 1, 1], float)
    B = np.array([1, -1, -1, 1, 1, -1, -1, 1], float)
    t = np.arange(seconds, dtype=float)
    env = np.empty((seconds, n_bands))
    pos = 0
    while pos < seconds:
        d = int(rng.integers(20, 60))
        pool = [A, -A] + ([B, -B] if pos >= seconds // 2 else [])
        env[pos:pos + d] = pool[rng.integers(len(pool))]
        pos += d
    env += 0.6 * rng.standard_normal(env.shape)
    drift = np.where(t < seconds // 2, 3.0 * np.sin(2 * np.pi * t / 1200), 0.0)
    env += drift[:, None] * A[None, :] / np.linalg.norm(A) * np.sqrt(n_bands)
    return t, env, A, B, drift


def test_tracker_recovers_persistent_and_late_motifs_and_moves_drift_into_baseline():
    t, env, A, B, drift = _synthetic()
    cfg = mt.TrackConfig(n_sur=5)
    sig = mt.prepare_arrays("synthetic", None, t, env, cfg, cuts=[])
    res = mt.track_prepared(sig, cfg)
    T = np.array([m.template for m in res.library])
    rA = np.abs([np.corrcoef(A, x)[0, 1] for x in T])
    rB = np.abs([np.corrcoef(B, x)[0, 1] for x in T])
    assert rA.max() > 0.9 and rB.max() > 0.9
    ia, ib = int(rA.argmax()), int(rB.argmax())

    ws = res.window_stats
    a = ws[ws.motif == ia]
    assert a.present.mean() > 0.7                                  # A lives through the recording
    b = ws[ws.motif == ib]
    late = b.centre > t[-1] / 2 + 150
    early = b.centre < t[-1] / 2 - 150
    # B appears halfway. 5-min windows catch it in about half its windows (it shares the dwells with A),
    # so the test asks for a clear late-vs-early difference rather than full detection.
    assert b[late].present.mean() > 0.4 and b[early].present.mean() < 0.15

    first = (t < t[-1] / 2 - 150) & np.isfinite(res.second[f"base_M{ia + 1}"]).to_numpy()
    base = res.second.loc[first, f"base_M{ia + 1}"].to_numpy()
    assert abs(np.corrcoef(base, drift[first])[0, 1]) > 0.9      # the drift ends up in the baseline
