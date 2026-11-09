import numpy as np

from src.eeg_motif_diagnostics import behavior_metrics, channel_metrics, common_average_reference, gmm_metrics


def test_common_average_reference_zeroes_valid_channel_mean():
    data = np.array([[1.0, 3.0, 99.0], [2.0, 6.0, 98.0]])
    referenced, common = common_average_reference(data, np.array([True, True, False]))

    np.testing.assert_allclose(common, [2.0, 4.0])
    np.testing.assert_allclose(referenced[:, :2].mean(axis=1), 0)
    np.testing.assert_array_equal(referenced[:, 2], data[:, 2])


def test_channel_metrics_identifies_theta_rich_channel():
    fs = 250.0
    t = np.arange(int(20 * fs)) / fs
    rng = np.random.default_rng(0)
    data = np.column_stack(
        [np.sin(2 * np.pi * 8 * t) + 0.05 * rng.standard_normal(t.size), rng.standard_normal(t.size)]
    )
    metrics = channel_metrics(data, fs)

    assert metrics.loc[0, "theta_fraction"] > metrics.loc[1, "theta_fraction"]


def test_behavior_metrics_aligns_optional_series_and_stimulation():
    n = 200
    t = np.arange(n) / 10
    env = np.zeros((n, 6))
    env[:, 1] = np.linspace(-2, 2, n)
    window = {
        "ts": t,
        "running": np.linspace(0, 10, n),
        "running_ts": t,
        "pupil": np.linspace(5, 1, n),
        "pupil_ts": t,
        "iso_level": np.ones(n) * 1.5,
        "iso_ts": t,
        "stim_ts": np.array([10.0]),
    }
    metrics = behavior_metrics(window, env)

    assert metrics["theta_speed_spearman"] > 0.99
    assert metrics["theta_pupil_spearman"] < -0.99
    assert metrics["theta_stim_delta_z"] > 0
    assert metrics["n_stim"] == 1


def test_gmm_metrics_reports_stability_and_early_late_change():
    rng = np.random.default_rng(3)
    early = rng.normal([2, 2, -2, -2], 0.15, size=(90, 4))
    late = rng.normal([-2, -2, 2, 2], 0.15, size=(90, 4))
    metrics = gmm_metrics(np.vstack([early, late]), k=2, k_max=3, n_seeds=3)

    assert metrics["gmm_bic_best_k"] == 2
    assert metrics["gmm_mean_seed_ari"] > 0.99
    assert metrics["gmm_median_confidence"] > 0.99
    assert metrics["gmm_early_late_occupancy_js"] > 0.9
