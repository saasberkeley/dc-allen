import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.get_clust import event_mask, get_clust, preprocess


def _rates(seed, n_neurons=40, seconds=1500, ensembles=((0, 8), (8, 16))):
    """Independent noisy rates, plus ensembles whose members fire together in 5-15 s episodes."""
    rng = np.random.default_rng(seed)
    rate = gaussian_filter1d(rng.gamma(2.0, 1.0, (seconds, n_neurons)), 1, axis=0)
    for lo, hi in ensembles:
        on = np.zeros(seconds)
        t = 0
        while t < seconds:
            t += int(rng.integers(20, 60))
            d = int(rng.integers(5, 15))
            on[t:t + d] = 1
            t += d
        rate[:, lo:hi] += 4 * gaussian_filter1d(on, 1)[:, None]
    return rate


def test_event_mask_keeps_only_long_runs():
    s = np.zeros(100)
    s[10:12] = 5      # 2 samples: too short
    s[40:45] = 5      # 5 samples: kept
    m = event_mask(s, 3)
    assert not m[10:12].any() and m[40:45].all() and m.sum() == 5


def _jaccard_best(planted, clusters):
    return max((len(planted & set(c.tolist())) / len(planted | set(c.tolist())) for c in clusters), default=0)


def test_recovers_a_single_planted_ensemble():
    for seed in (1, 2):
        clusters = get_clust(preprocess(_rates(seed, ensembles=((0, 8),))), 1.0, seed=0)
        assert _jaccard_best(set(range(8)), clusters) == 1.0


def _opponent(seed, n=40, seconds=1500):
    """Two ensembles that alternate: one is up while the other is down (Mishler's opponent case)."""
    rng = np.random.default_rng(seed)
    rate = gaussian_filter1d(rng.gamma(2.0, 1.0, (seconds, n)), 1, axis=0)
    state, t, cur = np.zeros(seconds), 0, 0
    while t < seconds:
        d = int(rng.integers(10, 40))
        state[t:t + d] = cur
        cur, t = 1 - cur, t + d
    on = gaussian_filter1d(state, 1)
    rate[:, 0:8] += 4 * on[:, None]
    rate[:, 8:16] += 4 * (1 - on)[:, None]
    return rate


def test_recovers_opponent_ensembles():
    for seed in range(3):
        clusters = get_clust(preprocess(_opponent(seed)), 1.0, seed=0)
        assert _jaccard_best(set(range(0, 8)), clusters) >= 0.7
        assert _jaccard_best(set(range(8, 16)), clusters) >= 0.7


def test_no_ensembles_in_independent_noise():
    X = preprocess(_rates(1, ensembles=()))
    assert get_clust(X, 1.0, seed=0) == []
