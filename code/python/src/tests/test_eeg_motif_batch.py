import pandas as pd

from src.eeg_motif_batch import detection_summary, outcome_summary


def sample_results():
    return pd.DataFrame(
        {
            "subject": ["m1"] * 4,
            "session": ["s1"] * 4,
            "state": ["awake"] * 4,
            "reference": ["original"] * 4,
            "channel": [0, 1, 2, 3],
            "location": ["A", "A", "B", "B"],
            "binary_motif_pairs": [0, 1, 1, 0],
            "continuous_motif_pairs": [1, 0, 1, 0],
        }
    )


def test_detection_summary_counts_each_method():
    summary = detection_summary(sample_results()).set_index("method")
    assert summary.loc["binary", "channels_with_motif"] == 2
    assert summary.loc["continuous", "channels_with_motif"] == 2
    assert summary.loc["binary", "fraction_with_motif"] == 0.5


def test_outcome_summary_classifies_exact_channels():
    outcomes = outcome_summary(sample_results()).set_index("channel").outcome
    assert outcomes.to_dict() == {
        0: "continuous rescue",
        1: "binary only",
        2: "both",
        3: "neither",
    }
