"""Load one Templeton NWB session for state-wise analysis: traces, isoflurane timeline,
electrical-stimulation blocks, stimulation-free analysis windows and sorted units."""

from dataclasses import dataclass

import numpy as np
from pynwb import NWBHDF5IO
from scipy import signal

from .smooth_env import mround

GOOD_UNIT = {"isi_violations": 0.5, "amplitude_cutoff": 0.1, "presence_ratio": 0.9}  # Allen defaults


@dataclass
class SessionInfo:
    file: str
    subject: str
    description: str
    t_start: float
    t_end: float
    iso_t: np.ndarray
    iso_level: np.ndarray
    iso_on: float
    induction_end: float
    iso_off: float
    stim_times: np.ndarray
    stim_blocks: list
    visual_blocks: list
    eeg_locations: list
    eeg_valid: np.ndarray


def _blocks(trials, kind):
    rows = trials[trials.stimulus_type == kind]
    return [(str(label), float(g.start_time.min()), float(g.stop_time.max()))
            for label, g in rows.groupby("behavioral_epoch")]


def read_info(nwb, file: str) -> SessionInfo:
    iso = nwb.acquisition["isoflurane_level"]
    level = iso.data[:]
    iso_t = iso.starting_time + np.arange(level.size) / iso.rate
    on = level > 0.5
    iso_on = float(iso_t[np.argmax(on)])
    iso_off = float(iso_t[len(on) - 1 - np.argmax(on[::-1])])
    maintenance = (iso_t > iso_on + 60) & (level < 2.0)
    trials = nwb.trials.to_dataframe()
    eeg = nwb.acquisition["ElectricalSeriesEEG"]
    electrodes = eeg.electrodes.to_dataframe()
    lfp_t = next(iter(nwb.acquisition["LFP"].electrical_series.values())).timestamps
    return SessionInfo(
        file=file,
        subject=nwb.subject.subject_id,
        description=nwb.session_description,
        t_start=float(lfp_t[0]),
        t_end=float(lfp_t[-1]),
        iso_t=iso_t,
        iso_level=level,
        iso_on=iso_on,
        induction_end=float(iso_t[np.argmax(maintenance)]),
        iso_off=iso_off,
        stim_times=trials.loc[trials.stimulus_type == "electrical", "start_time"].to_numpy(),
        stim_blocks=_blocks(trials, "electrical"),
        visual_blocks=_blocks(trials, "visual"),
        eeg_locations=list(electrodes.location.values),
        eeg_valid=electrodes.is_data_valid.to_numpy(bool),
    )


def lfp_channel(nwb, probe: str, region: str) -> int:
    """Middle valid channel of `region` on `probe`."""
    electrodes = nwb.acquisition["LFP"][f"ElectricalSeries{probe}"].electrodes.to_dataframe()
    picks = np.flatnonzero((electrodes.location.values == region) & electrodes.is_data_valid.values)
    if picks.size == 0:
        raise ValueError(f"no valid {probe} channels in {region}")
    return int(picks[picks.size // 2])


def _series(nwb, source):
    kind, *rest = source
    if kind == "eeg":
        return nwb.acquisition["ElectricalSeriesEEG"], rest[0]
    probe, channel = rest
    return nwb.acquisition["LFP"][f"ElectricalSeries{probe}"], channel


def read_raw(nwb, source):
    """Full-rate trace in µV. source: ("eeg", channel) or ("lfp", probe, channel)."""
    series, channel = _series(nwb, source)
    t = series.timestamps[:]
    x = series.data[:, channel] * series.conversion * 1e6
    return t, x, 1 / np.median(np.diff(t[:100_000]))


def decimate(t, x, fs_raw, target_fs):
    q = mround(fs_raw / target_fs)
    return t[::q], signal.decimate(x, q, zero_phase=True), fs_raw / q


def stim_free_segments(info: SessionInfo, margin: float = 5.0):
    """Stretches of [t_start, iso_off] with no electrical stimulation block (± margin)."""
    blocks = sorted((a - margin, b + margin) for _, a, b in info.stim_blocks)
    segments, start = [], info.t_start
    for a, b in blocks:
        if a > start:
            segments.append((start, min(a, info.iso_off)))
        start = max(start, b)
    if start < info.iso_off:
        segments.append((start, info.iso_off))
    return [(a, b) for a, b in segments if b > a]


def analysis_windows(info: SessionInfo, awake_gap: float = 30.0):
    """Awake = before isoflurane is switched on; isoflurane = maintenance after induction."""
    return {
        "awake": (info.t_start, info.iso_on - awake_gap),
        "isoflurane": (info.induction_end, info.iso_off),
        "full": (info.t_start, info.iso_off),
    }


def state_of_time(t, info: SessionInfo, awake_gap: float = 30.0):
    labels = np.full(np.shape(t), "induction", dtype=object)
    labels[t < info.iso_on - awake_gap] = "awake"
    labels[t >= info.induction_end] = "isoflurane"
    return labels


def good_units(nwb, region: str):
    """(depth-sorted) spike-time arrays of well-isolated units in `region`."""
    units = nwb.units.to_dataframe()
    n_in_region = int((units.location == region).sum())
    keep = ((units.location == region)
            & (units.isi_violations < GOOD_UNIT["isi_violations"])
            & (units.amplitude_cutoff < GOOD_UNIT["amplitude_cutoff"])
            & (units.presence_ratio > GOOD_UNIT["presence_ratio"]))
    units = units[keep].sort_values("y")
    return [np.asarray(s) for s in units.spike_times], units.index.to_numpy(), n_in_region


def open_nwb(path):
    io = NWBHDF5IO(str(path), mode="r")
    return io, io.read()
