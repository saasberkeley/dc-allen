import numpy as np
import pytest

from src.gmm_motifs import bic_sweep, fit_gmm_motifs, seed_stability, shift_bands


def synthetic_envelopes(seed=0):
    rng = np.random.default_rng(seed)
    slow = rng.normal([2, 2, -2, -2], 0.2, size=(100, 4))
    fast = rng.normal([-2, -2, 2, 2], 0.2, size=(100, 4))
    return np.vstack([slow, fast])


def test_fit_gmm_recovers_and_orders_two_spectral_states():
    X = synthetic_envelopes()
    result = fit_gmm_motifs(X, 2, seed=3)
    assert result.n_motifs == 2
    assert np.mean(result.labels[:100] == 0) > 0.95
    assert np.mean(result.labels[100:] == 1) > 0.95
    assert np.median(result.confidence) > 0.95


def test_occupancy_is_computed_within_group():
    result = fit_gmm_motifs(synthetic_envelopes(), 2)
    groups = np.array(["awake"] * 100 + ["iso"] * 100)
    occupancy = result.occupancy(groups)
    np.testing.assert_allclose(occupancy["awake"].sum(), 1)
    np.testing.assert_allclose(occupancy["iso"].sum(), 1)


def test_bic_and_seed_stability_support_clear_two_cluster_data():
    X = synthetic_envelopes()
    ks, bics = bic_sweep(X, [1, 2, 3])
    assert ks[np.argmin(bics)] == 2
    scores, mean_score = seed_stability(X, 2, seeds=range(3))
    assert scores.shape == (3,)
    assert mean_score > 0.99


def test_shift_bands_keeps_each_column_but_changes_alignment():
    X = np.arange(60).reshape(15, 4)
    shifted = shift_bands(X, step=2)
    np.testing.assert_array_equal(shifted[:, 0], X[:, 0])
    np.testing.assert_array_equal(shifted[:, 2], np.roll(X[:, 2], 4))


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        fit_gmm_motifs(np.ones((2, 2)), 0)
    with pytest.raises(ValueError):
        shift_bands(np.ones((4, 2)), step=0)
