import numpy as np

from src.eeg_motif_diagnostics import channel_metrics, common_average_reference


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
