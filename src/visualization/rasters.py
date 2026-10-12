"""Spike raster plots for specific units across brain states.

The five conditions in the Templeton dataset are awake, isoflurane, urethane,
psilocybin and ketanserin. A single NWB session usually holds more than one of
them (e.g. an awake baseline followed by isoflurane), labelled per trial in
``trials.behavioral_epoch``. This module finds those condition windows and plots
the *same* units in each one so they can be compared side by side.

Quick start (from the repo root)::

    python -m src.visualization.rasters data/raw/688277-20230921-v001.nwb

or, in a notebook::

    from src.visualization import rasters
    with rasters.open_session("data/raw/688277-20230921-v001.nwb") as nwb:
        units = rasters.select_units(nwb, n=30)
        rasters.plot_condition_rasters(nwb, units.index)
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

CONDITIONS = ["awake", "isoflurane", "urethane", "psilocybin", "ketanserin", "saline"]

# Fixed condition -> colour mapping so a condition is the same colour in every figure.
CONDITION_COLORS = {
    "awake": "#2a78d6",
    "isoflurane": "#eb6834",
    "urethane": "#1baf7a",
    "psilocybin": "#eda100",
    "ketanserin": "#e87ba4",
    "saline": "#8a8984",
}

# Allen-style quality thresholds (see Siegle et al. 2021).
DEFAULT_QUALITY = {
    "isi_violations": 0.5,      # max
    "amplitude_cutoff": 0.1,    # max
    "presence_ratio": 0.9,      # min
}


@contextmanager
def open_session(path):
    """Open an NWB file read-only; data stays on disk until sliced."""
    io = NWBHDF5IO(str(path), mode="r")
    try:
        yield io.read()
    finally:
        io.close()


def canonical_condition(label: str) -> str | None:
    """Map a free-text epoch/pharmacology label onto one of ``CONDITIONS``."""
    label = str(label).lower()
    for cond in CONDITIONS:
        if cond in label:
            return cond
    return None


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #

def condition_windows(nwb) -> pd.DataFrame:
    """Return one row per condition: ``condition, start, stop, n_trials``.

    Windows span the first to last trial labelled with that behavioural epoch.
    Sessions without trials fall back to the whole recording, labelled from
    ``nwb.pharmacology``.
    """
    if nwb.trials is not None and "behavioral_epoch" in nwb.trials.colnames:
        trials = nwb.trials.to_dataframe()
        rows = []
        for label, grp in trials.groupby("behavioral_epoch", sort=False):
            rows.append({
                "condition": canonical_condition(label) or str(label),
                "label": label,
                "start": grp.start_time.min(),
                "stop": grp.stop_time.max(),
                "n_trials": len(grp),
            })
        return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)

    spikes = nwb.units["spike_times"].data[:]
    return pd.DataFrame([{
        "condition": canonical_condition(nwb.pharmacology) or "unknown",
        "label": nwb.pharmacology,
        "start": float(spikes.min()),
        "stop": float(spikes.max()),
        "n_trials": 0,
    }])


def units_table(nwb) -> pd.DataFrame:
    """Unit metadata (no spike times / waveforms), indexed by unit id."""
    units = nwb.units
    skip = {"spike_times", "electrodes", "electrode_group", "waveform_mean"}
    cols = [c for c in units.colnames if c not in skip]
    df = pd.DataFrame({c: units[c][:] for c in cols}, index=units.id[:])
    df.index.name = "unit_id"
    df["probe"] = [g.name for g in units["electrode_group"][:]]
    # Spike counts from the ragged index (cheap; no spike times are read).
    idx = units["spike_times_index"].data[:]
    df["n_spikes"] = np.diff(np.concatenate([[0], idx]))
    return df


def select_units(nwb, n: int | None = 30, regions=None, probes=None,
                 quality: dict | None = DEFAULT_QUALITY, min_spikes: int = 500,
                 seed: int | None = 0) -> pd.DataFrame:
    """Pick well-isolated units, optionally restricted to brain regions/probes.

    Parameters
    ----------
    n : number of units to return (``None`` = all that pass). Units are sampled
        evenly across the passing set so the selection spans depth/regions.
    regions : iterable of CCF acronyms, matched by prefix (``"SSp"`` matches
        ``"SSp-bfd5"``). ``None`` keeps all regions.
    probes : e.g. ``["probeB"]``. ``None`` keeps all probes.
    quality : dict of thresholds (see ``DEFAULT_QUALITY``); ``None`` skips it.

    Returns the unit metadata sorted by probe then depth (dorsal at top).
    """
    df = units_table(nwb)
    keep = df.n_spikes >= min_spikes
    if quality:
        keep &= df.isi_violations <= quality.get("isi_violations", np.inf)
        keep &= df.amplitude_cutoff <= quality.get("amplitude_cutoff", np.inf)
        keep &= df.presence_ratio >= quality.get("presence_ratio", 0)
    if regions is not None:
        regions = [regions] if isinstance(regions, str) else list(regions)
        keep &= df.location.astype(str).apply(lambda loc: any(loc.startswith(r) for r in regions))
    if probes is not None:
        probes = [probes] if isinstance(probes, str) else list(probes)
        keep &= df.probe.isin(probes)
    df = df[keep].sort_values(["probe", "y"])  # CCF y increases ventrally
    if n is not None and len(df) > n:
        pick = np.linspace(0, len(df) - 1, n).round().astype(int)
        df = df.iloc[pick]
    return df


def spike_times(nwb, unit_ids, t0: float | None = None, t1: float | None = None) -> list[np.ndarray]:
    """Spike times (s) for each unit id, optionally clipped to ``[t0, t1)``."""
    ids = nwb.units.id[:]
    row_of = {uid: i for i, uid in enumerate(ids)}
    out = []
    for uid in unit_ids:
        st = np.asarray(nwb.units["spike_times"][row_of[uid]])
        if t0 is not None:
            st = st[st >= t0]
        if t1 is not None:
            st = st[st < t1]
        out.append(st)
    return out


def stim_times(nwb, t0=None, t1=None, stimulus_type: str | None = "electrical") -> np.ndarray:
    """Stimulus onset times, optionally filtered by type and time window."""
    if nwb.trials is None:
        return np.array([])
    trials = nwb.trials.to_dataframe()
    if stimulus_type is not None:
        trials = trials[trials.stimulus_type == stimulus_type]
    t = trials.start_time.to_numpy()
    if t0 is not None:
        t = t[t >= t0]
    if t1 is not None:
        t = t[t < t1]
    return t


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #

def _unit_labels(units: pd.DataFrame) -> list[str]:
    return [f"{uid} · {loc}" for uid, loc in zip(units.index, units.location)]


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, color="#888", labelcolor="#333")
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#888")


def raster(ax, trains, t0=0.0, color="k", linewidth=0.6, lineheight=0.8):
    """Draw one row per spike train, shifted by ``t0``; row 0 at the top."""
    ax.eventplot([st - t0 for st in trains], colors=color, linewidths=linewidth,
                 linelengths=lineheight)
    ax.set_ylim(len(trains) - 0.5, -0.5)
    _style(ax)
    return ax


def _resolve_conditions(nwb, conditions):
    win = condition_windows(nwb)
    if conditions is not None:
        wanted = [c.lower() for c in conditions]
        missing = [c for c in wanted if c not in set(win.condition)]
        if missing:
            print(f"[rasters] not in this session: {missing}; available: {list(win.condition)}")
        win = win[win.condition.isin(wanted)]
    if win.empty:
        raise ValueError("No matching conditions in this session.")
    return win.reset_index(drop=True)


def plot_condition_rasters(nwb, unit_ids, conditions=None, window: float = 20.0,
                           offset: str | float = "middle", rate_bin: float = 0.5,
                           mark_stim: bool = True, figsize=None, fig=None):
    """Raster of the same units in each condition, one column per condition.

    Parameters
    ----------
    unit_ids : unit ids to plot (e.g. ``select_units(nwb).index``), top to bottom.
    conditions : subset of ``CONDITIONS`` to show; default = all in the session.
    window : seconds shown per condition.
    offset : where in each condition window to start: ``"start"``, ``"middle"``,
        or seconds after the condition's first trial.
    rate_bin : bin (s) for the population-rate strip under each raster.
    mark_stim : draw electrical-stimulus onsets as ticks above each raster.
    fig : draw into this Figure/SubFigure instead of creating a new one.
    """
    units = units_table(nwb).loc[list(unit_ids)]
    win = _resolve_conditions(nwb, conditions)
    ncol = len(win)
    figsize = figsize or (4.2 * ncol + 1.2, 0.16 * len(units) + 3)
    if fig is None:
        fig = plt.figure(figsize=figsize, layout="constrained")
    axes = fig.subplots(2, ncol, sharey="row", squeeze=False,
                        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08})

    rate_max = 0
    for j, row in win.iterrows():
        length = row.stop - row.start
        if offset == "start":
            t0 = row.start
        elif offset == "middle":
            t0 = row.start + max(0.0, (length - window) / 2)
        else:
            t0 = row.start + float(offset)
        t1 = t0 + window
        trains = spike_times(nwb, units.index, t0, t1)

        ax = axes[0, j]
        raster(ax, trains, t0, linewidth=0.4)
        ax.set_xlim(0, window)
        ax.tick_params(labelbottom=False)
        color = CONDITION_COLORS.get(row.condition, "#555")
        ax.set_title(f"{row.condition}\n{t0:.0f}–{t1:.0f} s", fontsize=10, color="#222")
        ax.axhline(-0.5, color=color, lw=4, clip_on=False)
        if mark_stim:
            for s in stim_times(nwb, t0, t1):
                ax.axvline(s - t0, color="#c0392b", lw=0.6, alpha=0.5, zorder=0)

        # Population rate (spikes/s per unit)
        edges = np.arange(0, window + rate_bin, rate_bin)
        counts = sum(np.histogram(st - t0, edges)[0] for st in trains)
        rate = counts / rate_bin / max(len(trains), 1)
        rate_max = max(rate_max, rate.max())
        rax = axes[1, j]
        rax.fill_between(edges[:-1], rate, step="post", color=color, alpha=0.35, lw=0)
        rax.step(edges[:-1], rate, where="post", color=color, lw=1.2)
        rax.set_xlim(0, window)
        rax.set_xlabel("Time in window (s)")
        _style(rax)

    axes[0, 0].set_yticks(range(len(units)))
    axes[0, 0].set_yticklabels(_unit_labels(units), fontsize=6)
    axes[0, 0].set_ylabel("Unit (id · region)")
    axes[1, 0].set_ylabel("Pop. rate\n(spk/s/unit)")
    axes[1, 0].set_ylim(0, rate_max * 1.1 or 1)
    title = f"{nwb.identifier} — {len(units)} units"
    if mark_stim:
        title += "   (red lines = electrical stimulus)"
    fig.suptitle(title, fontsize=11)
    return fig


def plot_session_rasters(paths, n_units: int = 30, regions=None, unit_ids: dict | None = None,
                         window: float = 20.0, offset: str | float = "middle",
                         rate_bin: float = 0.5, figsize=None):
    """Side-by-side rasters for several sessions in ONE figure.

    Each session gets its own block of columns (one per condition it contains)
    and its own unit rows: different sessions are different mice, so rows only
    line up *within* a block. Population-rate strips share one y-scale across
    all blocks so rates are directly comparable.

    Parameters
    ----------
    paths : NWB files, plotted left to right.
    n_units, regions : passed to ``select_units`` for each session.
    unit_ids : optional ``{path: [ids]}`` to pick units per session explicitly.
    """
    paths = [Path(p) for p in paths]
    unit_ids = {Path(k): v for k, v in (unit_ids or {}).items()}
    ncols = []
    for path in paths:
        with open_session(path) as nwb:
            ncols.append(len(condition_windows(nwb)))
    figsize = figsize or (4.2 * sum(ncols) + 1.6 * len(paths), 0.16 * n_units + 3.4)
    fig = plt.figure(figsize=figsize, layout="constrained")
    subfigs = np.atleast_1d(fig.subfigures(1, len(paths), width_ratios=ncols, wspace=0.03))

    rate_axes = []
    for sub, path, ncol in zip(subfigs, paths, ncols):
        with open_session(path) as nwb:
            ids = unit_ids.get(path)
            if ids is None:
                ids = select_units(nwb, n=n_units, regions=regions).index
            plot_condition_rasters(nwb, ids, window=window, offset=offset, rate_bin=rate_bin,
                                   mark_stim=True, fig=sub)
            sub.suptitle(f"mouse {nwb.subject.subject_id} · {nwb.identifier} · {len(ids)} units",
                         fontsize=10)
        rate_axes += sub.axes[ncol:]

    top = max(ax.get_ylim()[1] for ax in rate_axes)
    for ax in rate_axes:
        ax.set_ylim(0, top)
    fig.suptitle("Spike rasters by condition  (red lines = electrical stimulus; "
                 "rows are the same units only within each mouse)", fontsize=11)
    return fig



# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None):
    p = argparse.ArgumentParser(description="Save side-by-side condition rasters for one or more NWB sessions.")
    p.add_argument("nwb", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("src/visualization/figures"))
    p.add_argument("--n-units", type=int, default=40)
    p.add_argument("--regions", nargs="*", default=None, help="CCF acronym prefixes, e.g. SSp MOs CA3")
    p.add_argument("--units", nargs="*", type=int, default=None, help="explicit unit ids (overrides selection)")
    p.add_argument("--window", type=float, default=20.0)
    p.add_argument("--no-combined", action="store_true",
                   help="skip the single figure with all sessions side by side")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    for path in args.nwb:
        with open_session(path) as nwb:
            win = condition_windows(nwb)
            print(f"{path.name}: conditions = {list(win.condition)}")
            ids = args.units or list(select_units(nwb, n=args.n_units, regions=args.regions).index)
            stem = path.stem

            plot_condition_rasters(nwb, ids, window=args.window).savefig(
                args.out / f"{stem}_rasters.png", dpi=150)
            plt.close("all")
    if len(args.nwb) > 1 and not args.no_combined:
        plot_session_rasters(args.nwb, n_units=args.n_units, regions=args.regions,
                             window=args.window).savefig(args.out / "all_sessions_rasters.png", dpi=150)
    print(f"Saved figures to {args.out}")


if __name__ == "__main__":
    main()
