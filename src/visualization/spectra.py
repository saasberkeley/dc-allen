"""EEG spectra, power-drop timing, V1/cortex band power and aligned LFP + raster figures.

Reads the time-aligned windows written by ``src/scripts/extract_windows.py``
(``data/windows/*.h5``); the power-drop time course is computed from the
continuous NWB recording, because the drop can fall outside a 2-min window.

Outputs (from the repo root: ``python -m src.visualization.spectra``):

    src/visualization/figures/eeg_spectra.png
    src/visualization/figures/band_power_change.png
    src/visualization/figures/{session}_eeg_power_timecourse.png
    src/visualization/figures/{session}_aligned.png
    src/visualization/results/band_power.csv
    src/visualization/results/power_drop.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from pynwb import NWBHDF5IO
from scipy import signal

from src.scripts.extract_windows import ANCHORS, session_type

# 55-65 Hz is skipped: 60 Hz mains noise is visible in the EEG
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "low gamma": (30, 55),
    "high gamma": (65, 100),
}
BROADBAND = (1, 100)

# Baseline vs. treatment window per session type
PAIRS = {
    "isoflurane": ("awake", "iso_anesthetized"),
    "psilocybin": ("psilocybin_before", "psilocybin_after"),
}
WINDOW_COLORS = {
    "awake": "#2a78d6",
    "iso_anesthetized": "#eb6834",
    "psilocybin_before": "#2a78d6",
    "psilocybin_after": "#eda100",
}

# Recording sites grouped for band power. EEG = skull screws, LFP = Neuropixels.
SITE_GROUPS = [
    ("EEG V1 (VISp)", "eeg", ("VISp",)),
    ("EEG visual (VISl/al/a)", "eeg", ("VISl", "VISal", "VISa")),
    ("EEG somatosensory", "eeg", ("SSp",)),
    ("EEG motor", "eeg", ("MOs", "MOp")),
    ("EEG retrosplenial", "eeg", ("RSP",)),
    ("LFP somatosensory", "lfp", ("SSp",)),
    ("LFP motor", "lfp", ("MOs",)),
    ("LFP retrosplenial", "lfp", ("RSP",)),
    ("LFP hippocampus", "lfp", ("CA1", "CA2", "CA3")),
]

QUALITY = {"isi_violations": 0.5, "amplitude_cutoff": 0.1, "presence_ratio": 0.9}
CORTEX_HIPPO = ("SSp", "MOs", "MOp", "RSP", "VIS", "CA1", "CA2", "CA3", "DG")

FIG_DIR = Path("src/visualization/figures")
RESULTS_DIR = Path("src/visualization/results")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_window(path) -> dict:
    """Read one window file into plain numpy arrays."""
    with h5py.File(path, "r") as f:
        w = {"attrs": dict(f.attrs), "path": Path(path)}
        for grp in ("eeg", "lfp", "spikes"):
            w[grp] = {k: f[grp][k][()] for k in f[grp]}
            w[grp].update(dict(f[grp].attrs))
            for k, v in w[grp].items():
                if isinstance(v, np.ndarray) and v.dtype.kind == "O":
                    w[grp][k] = v.astype(str)
    w["eeg"].setdefault("is_data_valid", np.ones(w["eeg"]["data"].shape[1], bool))
    return w


def load_windows(window_dir="data/windows") -> dict:
    """``{session: {window name: window}}`` for every file in ``window_dir``."""
    out = {}
    for p in sorted(Path(window_dir).glob("*.h5")):
        w = load_window(p)
        out.setdefault(w["attrs"]["session"], {})[w["attrs"]["window"]] = w
    return out


def _site_channels(w, kind, prefixes):
    loc = w[kind]["location"]
    keep = np.array([l.startswith(prefixes) for l in loc])
    if kind == "eeg":
        keep &= w["eeg"]["is_data_valid"]
    return np.where(keep)[0]


# --------------------------------------------------------------------------- #
# Spectra and band power
# --------------------------------------------------------------------------- #

ARTIFACT_SD = 8.0   # a 2 s segment is rejected for a channel if it peaks > 8 robust SDs
# No rejection in anesthetized windows: the animal is immobile, and the large
# events there are burst-suppression bursts (spread across channels), not artifacts.
NO_REJECTION = {"iso_anesthetized"}


def robust_sd(x):
    """Per-channel robust SD (1.4826 * MAD). x: (T, C) -> (C,)"""
    return 1.4826 * np.median(np.abs(x - np.median(x, 0)), 0)


def psd(x, fs, seg_s=2.0, reject_sd=ARTIFACT_SD, ref_sd=None):
    """Welch PSD per channel with per-channel artifact rejection.

    2 s Hann segments, 50% overlap (as ``scipy.signal.welch``). For each channel,
    segments whose peak |deviation| exceeds ``reject_sd`` robust SDs are dropped
    before averaging, so large transients (movement, chewing) on one channel
    don't distort its spectrum or the other channels'.

    ``ref_sd`` (C,) sets the SD the threshold is measured in. Pass the
    *baseline* window's ``robust_sd`` so both windows of a comparison use the
    same absolute threshold; otherwise a mostly-flat burst-suppression window
    would have its genuine bursts rejected as artifacts.

    x: (T, C) -> f, P (C, F), fraction of segments rejected (C,)
    """
    nper = int(seg_s * fs)
    f, _, S = signal.spectrogram(x, fs=fs, window="hann", nperseg=nper, noverlap=nper // 2,
                                 axis=0, scaling="density", mode="psd")   # (F, C, n_seg)
    starts = np.arange(S.shape[-1]) * (nper - nper // 2)
    segs = np.stack([x[s:s + nper] for s in starts], axis=-1)             # (nper, C, n_seg)
    dev = np.abs(segs - segs.mean(0, keepdims=True)).max(0)               # (C, n_seg)
    sd = robust_sd(x) if ref_sd is None else ref_sd                        # (C,)
    good = dev <= reject_sd * sd[:, None]
    good[:, good.sum(0) == 0] = False
    P = np.where(good[None], S, np.nan)
    P = np.nanmean(P, axis=-1).T                                           # (C, F)
    return f, P, 1 - good.mean(1)


def band_power(f, P, lo, hi):
    """Integrated power (uV^2) in [lo, hi) per channel."""
    m = (f >= lo) & (f < hi)
    return np.trapezoid(P[:, m], f[m], axis=1)


def _baseline(wins):
    """The session's baseline window (artifact thresholds are set from it)."""
    return wins[PAIRS[session_type_of(wins)][0]]


def band_power_table(windows) -> pd.DataFrame:
    """Band power per window x site group x band (mean over channels)."""
    rows = []
    for session, wins in windows.items():
        base = _baseline(wins)
        for name, w in wins.items():
            for group, kind, prefixes in SITE_GROUPS:
                ch = _site_channels(w, kind, prefixes)
                if len(ch) == 0:
                    continue
                ref = robust_sd(base[kind]["data"][:, ch])
                f, P, rejected = psd(w[kind]["data"][:, ch], w[kind]["fs"], ref_sd=ref,
                                     reject_sd=np.inf if name in NO_REJECTION else ARTIFACT_SD)
                total = band_power(f, P, *BROADBAND)
                for band, (lo, hi) in BANDS.items():
                    bp = band_power(f, P, lo, hi)
                    rows.append({
                        "session": session, "window": name, "site": group, "band": band,
                        "n_channels": len(ch),
                        "frac_segments_rejected": rejected.mean(),
                        "power_uv2": bp.mean(),
                        "relative_power": (bp / total).mean(),
                    })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#888")
    ax.tick_params(color="#888", labelcolor="#333")


def plot_eeg_spectra(windows):
    """Mean EEG PSD (valid channels) per window; shaded = 25-75% across channels."""
    sessions = list(windows)
    fig, axes = plt.subplots(1, len(sessions), figsize=(6 * len(sessions), 4.2),
                             sharey=True, layout="constrained", squeeze=False)
    for ax, session in zip(axes[0], sessions):
        kind = session_type_of(windows[session])
        for name in PAIRS.get(kind, windows[session].keys()):
            w = windows[session][name]
            ch = np.where(w["eeg"]["is_data_valid"])[0]
            ref = robust_sd(_baseline(windows[session])["eeg"]["data"][:, ch])
            f, P, rejected = psd(w["eeg"]["data"][:, ch], w["eeg"]["fs"], ref_sd=ref,
                                 reject_sd=np.inf if name in NO_REJECTION else ARTIFACT_SD)
            m = (f >= 1) & (f <= 100)
            color = WINDOW_COLORS.get(name, "#555")
            q25, q75 = np.percentile(P[:, m], [25, 75], axis=0)
            ax.fill_between(f[m], q25, q75, color=color, alpha=0.18, lw=0)
            ax.plot(f[m], P[:, m].mean(0), color=color, lw=2,
                    label=f"{name} ({len(ch)} ch, {rejected.mean():.0%} segs rejected)")
        for lo, hi in BANDS.values():
            ax.axvline(lo, color="#ddd", lw=0.8, zorder=0)
        ax.axvspan(55, 65, color="#eee", zorder=0)
        ax.text(60, 0.97, "60 Hz\nnoise", transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=7, color="#888")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1, 100)
        ax.set_xticks([1, 2, 4, 8, 13, 30, 55, 100]); ax.set_xticklabels([1, 2, 4, 8, 13, 30, 55, 100])
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(f"{session}  ({kind})", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        _style(ax)
    axes[0, 0].set_ylabel("EEG power (µV²/Hz)")
    fig.suptitle("EEG power spectra — line = mean over valid channels, band = 25–75% across channels\n"
                 "Welch, 2 s segments; artifact segments rejected per channel (threshold from baseline "
                 "window; none in anesthetized windows)", fontsize=10)
    return fig


def session_type_of(wins) -> str:
    pharm = next(iter(wins.values()))["attrs"]["pharmacology"].lower()
    return next((k for k in PAIRS if k in pharm), "unknown")


def plot_band_change(table):
    """Heatmap of band-power change (dB, treatment vs. baseline) per site group."""
    sessions = table.session.unique()
    fig, axes = plt.subplots(1, len(sessions), figsize=(6.2 * len(sessions), 4.6),
                             layout="constrained", squeeze=False)
    mats = []
    for session in sessions:
        t = table[table.session == session]
        kind = next(k for k in PAIRS if PAIRS[k][0] in set(t.window))
        base, treat = PAIRS[kind]
        piv = t.pivot_table(index=["site", "band"], columns="window", values="power_uv2")
        db = 10 * np.log10(piv[treat] / piv[base]).unstack("band")
        sites = [g for g, _, _ in SITE_GROUPS if g in db.index]
        db = db.loc[sites, list(BANDS)]
        nch = t.groupby("site").n_channels.first()
        mats.append((session, base, treat, db, nch))
    lim = max(np.nanmax(np.abs(m[3].to_numpy())) for m in mats)
    lim = max(lim, 1.0)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    for ax, (session, base, treat, db, nch) in zip(axes[0], mats):
        im = ax.imshow(db.to_numpy(), cmap="RdBu_r", norm=norm, aspect="auto")
        for i in range(db.shape[0]):
            for j in range(db.shape[1]):
                v = db.iat[i, j]
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > lim * 0.6 else "#222")
        ax.set_xticks(range(db.shape[1]))
        ax.set_xticklabels([f"{b.replace(' ', chr(10))}\n{lo}–{hi} Hz" for b, (lo, hi) in BANDS.items()],
                           fontsize=8)
        ax.set_yticks(range(db.shape[0]))
        ax.set_yticklabels([f"{s}  (n={nch[s]})" for s in db.index], fontsize=8)
        ax.set_title(f"{session}\n{treat} vs {base}", fontsize=10)
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=axes[0, -1], label="Power change (dB)", shrink=0.8)
    fig.suptitle("V1 / cortex / hippocampus band power: change from baseline window "
                 "(red = more power, blue = less)", fontsize=11)
    return fig


# --------------------------------------------------------------------------- #
# Power-drop timing (continuous recording)
# --------------------------------------------------------------------------- #

def eeg_power_timecourse(nwb_path, pre=120.0, post=600.0, fs_out=500.0, seg_s=2.0, step_s=1.0):
    """EEG band power (dB re. pre-anchor baseline) around the session's drug anchor.

    Anchor = isoflurane onset (iso sessions) or psilocybin injection start.
    Returns (DataFrame indexed by time rel. anchor, info dict).
    """
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        kind = session_type(nwb)
        anchor_name = {"isoflurane": "iso_onset", "psilocybin": "psilocybin_injection_start"}[kind]
        anchor = ANCHORS[anchor_name](nwb)
        eeg = nwb.acquisition["ElectricalSeriesEEG"]
        valid = eeg.electrodes.to_dataframe().is_data_valid.to_numpy(bool)
        ts = eeg.timestamps[:]
        i0, i1 = np.searchsorted(ts, [anchor - pre, anchor + post])
        x = eeg.data[i0:i1][:, valid].astype(np.float64) * eeg.conversion * 1e6
        fs_in = 1 / np.median(np.diff(ts[i0:i0 + 10000]))
        q = int(round(fs_in / fs_out))
        x = signal.decimate(x, q, ftype="iir", zero_phase=True, axis=0)
        fs = fs_in / q
        t0 = ts[i0] - anchor
        info = {"session": nwb.identifier, "kind": kind, "anchor": anchor_name,
                "anchor_time_abs": anchor, "n_channels": int(valid.sum())}
        if kind == "isoflurane":
            iso = nwb.acquisition["isoflurane_level"]
            lv = iso.data[:]
            it = iso.starting_time + np.arange(len(lv)) / iso.rate - anchor
            m = (it >= -pre) & (it <= post)
            info["iso"] = (it[m], lv[m])
        if nwb.epochs is not None:
            ep = nwb.epochs.to_dataframe()
            info["epochs"] = [(r.start_time - anchor, r.stop_time - anchor, r.tags[1])
                              for r in ep.itertuples()]

    f, t, S = signal.spectrogram(x, fs=fs, nperseg=int(seg_s * fs),
                                 noverlap=int((seg_s - step_s) * fs), axis=0)
    S = np.median(S, axis=1)                  # median over channels (robust to one bad channel) -> (F, T)
    t = t + t0
    out = {}
    for band, (lo, hi) in {**BANDS, "broadband": BROADBAND}.items():
        m = (f >= lo) & (f < hi)
        if band == "broadband":
            m &= ~((f >= 55) & (f < 65))
        bp = S[m].sum(0)
        out[band] = 10 * np.log10(bp / np.median(bp[t < 0]))
    return pd.DataFrame(out, index=pd.Index(t, name="t_rel_anchor_s")), info


def detect_drop(tc: pd.DataFrame, smooth_s=10, z_thresh=3.0, min_dur_s=20):
    """First sustained drop per band after t=0.

    Power is smoothed (``smooth_s`` moving mean); a drop is the first time it
    falls below baseline mean - ``z_thresh`` * baseline SD and stays there for
    at least ``min_dur_s``. Returns one row per band.
    """
    dt = np.median(np.diff(tc.index))
    win = max(int(round(smooth_s / dt)), 1)
    sm = tc.rolling(win, center=True, min_periods=1).mean()
    rows = []
    for band in tc.columns:
        s = sm[band]
        base = s[s.index < 0]
        thr = base.mean() - z_thresh * base.std()
        below = (s < thr) & (s.index >= 0)
        run = max(int(round(min_dur_s / dt)), 1)
        sustained = below.rolling(run).sum().shift(-(run - 1)) == run
        hit = sustained[sustained].index
        t_drop = float(hit[0]) if len(hit) else np.nan
        post = s[s.index >= 0]
        rows.append({"band": band, "threshold_db": thr, "drop_time_s": t_drop,
                     "min_db_after": post.min(), "t_min_s": float(post.idxmin())})
    return pd.DataFrame(rows)


def plot_power_timecourse(tc, info, drops, bands=("delta", "beta", "low gamma", "high gamma", "broadband")):
    """Band power over time around the anchor, with detected drop times marked."""
    has_iso = "iso" in info
    fig, axes = plt.subplots(2 if has_iso else 1, 1, figsize=(11, 5.6 if has_iso else 4.2), sharex=True,
                             layout="constrained", squeeze=False,
                             gridspec_kw={"height_ratios": [4, 1] if has_iso else [1]})
    ax = axes[0, 0]
    sm = tc.rolling(10, center=True, min_periods=1).mean()
    palette = ["#2a78d6", "#1baf7a", "#eb6834", "#e87ba4", "#222222"]
    for band, color in zip(bands, palette):
        lw = 2.2 if band == "broadband" else 1.4
        ax.plot(sm.index, sm[band], color=color, lw=lw, label=band)
        d = drops.set_index("band").loc[band, "drop_time_s"]
        if np.isfinite(d):
            ax.plot(d, sm[band].loc[d], "o", ms=8, mfc="white", mec=color, mew=2, zorder=5)
    ax.axhline(0, color="#999", lw=0.8)
    ax.axvline(0, color="#333", lw=1, ls="--")
    ax.text(0, 0.97, f" {info['anchor']}", transform=ax.get_xaxis_transform(), fontsize=8,
            color="#333", va="top")
    for a, b, tag in info.get("epochs", []):
        if b < tc.index[0] or a > tc.index[-1]:
            continue
        ax.axvspan(a, b, color="#ccc", alpha=0.4, lw=0)
        ax.text((a + b) / 2, 0.90, f"{tag} inj.", transform=ax.get_xaxis_transform(),
                ha="center", fontsize=8, color="#555")
    bb = drops.set_index("band").loc["broadband"]
    note = (f"broadband drop at +{bb.drop_time_s:.0f} s" if np.isfinite(bb.drop_time_s)
            else "no sustained broadband drop")
    fig.suptitle(f"{info['session']} — EEG band power vs. pre-{info['anchor']} baseline: {note}\n"
                 f"{info['n_channels']} valid channels (median), 10 s smoothing; ○ = sustained-drop onset",
                 fontsize=10)
    ax.set_ylabel("Power change (dB)")
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="lower left", bbox_to_anchor=(0, 1.0))
    _style(ax)
    if has_iso:
        it, lv = info["iso"]
        axes[1, 0].plot(it, lv, color="#eb6834", lw=1.5)
        axes[1, 0].set_ylabel("Isoflurane (%)")
        axes[1, 0].axvline(0, color="#333", lw=1, ls="--")
        _style(axes[1, 0])
    axes[-1, 0].set_xlabel(f"Time relative to {info['anchor']} (s)")
    axes[-1, 0].set_xlim(tc.index[0], tc.index[-1])
    return fig


# --------------------------------------------------------------------------- #
# Aligned LFP + raster
# --------------------------------------------------------------------------- #

def _lfp_trace_channels(w, n=8):
    """~n LFP channels spread across cortex/hippocampus groups, ordered probe then depth."""
    loc, probe = w["lfp"]["location"], w["lfp"]["probe"]
    order = np.lexsort((-w["lfp"]["probe_vertical_position"], probe))
    picks = []
    groups = [g for g in SITE_GROUPS if g[1] == "lfp"]
    per = max(n // len(groups), 1)
    for _, _, prefixes in groups:
        idx = [i for i in order if loc[i].startswith(prefixes)]
        if idx:
            picks += [idx[k] for k in np.linspace(0, len(idx) - 1, min(per, len(idx))).round().astype(int)]
    return sorted(set(picks), key=lambda i: list(order).index(i))


def _family(loc: str) -> str:
    for name, prefixes in (("somatosensory", ("SSp",)), ("motor", ("MOs", "MOp")),
                           ("retrosplenial", ("RSP",)), ("visual", ("VIS",)),
                           ("hippocampus", ("CA1", "CA2", "CA3", "DG"))):
        if loc.startswith(prefixes):
            return name
    return "other"


def plot_aligned(wins, names, t0=55.0, dur=10.0, max_units=80):
    """EEG + LFP traces + spike raster on one time axis, one column per window."""
    fig, axes = plt.subplots(3, len(names), figsize=(7.5 * len(names), 10), sharex="col",
                             layout="constrained", squeeze=False,
                             gridspec_kw={"height_ratios": [1, 3, 3]})
    for j, name in enumerate(names):
        w = wins[name]
        color = WINDOW_COLORS.get(name, "#555")

        # EEG: first valid V1 channel, else first valid visual channel
        e = w["eeg"]
        ch = _site_channels(w, "eeg", ("VISp",))
        ch = ch if len(ch) else _site_channels(w, "eeg", ("VIS",))
        m = (e["ts"] >= t0) & (e["ts"] < t0 + dur)
        ax = axes[0, j]
        ax.plot(e["ts"][m] - t0, e["data"][m, ch[0]], color="#222", lw=0.6)
        ax.set_ylabel(f"EEG {e['location'][ch[0]]}\n(µV)")
        ax.set_title(f"{name}   ({w['attrs']['common_start'] + t0:.0f}–"
                     f"{w['attrs']['common_start'] + t0 + dur:.0f} s NWB time)", fontsize=10)
        ax.axhline(ax.get_ylim()[1], color=color, lw=4, clip_on=False)
        _style(ax)

        # LFP: stacked traces, offset by a fixed spacing
        l = w["lfp"]
        picks = _lfp_trace_channels(w)
        m = (l["ts"] >= t0) & (l["ts"] < t0 + dur)
        seg = l["data"][m][:, picks]
        seg = seg - seg.mean(0)
        spacing = 4 * np.median(seg.std(0))
        ax = axes[1, j]
        for k in range(len(picks)):
            ax.plot(l["ts"][m] - t0, seg[:, k] - k * spacing, color="#222", lw=0.5)
        ax.set_yticks([-k * spacing for k in range(len(picks))])
        ax.set_yticklabels([f"{l['probe'][i]} {l['location'][i]}" for i in picks], fontsize=7)
        ax.plot([dur * 1.005] * 2, [0, -spacing / 2], color="#222", lw=2, clip_on=False)
        ax.text(dur * 1.01, -spacing / 4, f"{spacing / 2:.0f} µV", fontsize=7, va="center")
        ax.set_ylabel("LFP")
        _style(ax)

        # Raster: quality units in cortex/hippocampus, ordered by probe then depth
        s = w["spikes"]
        ok = ((s["isi_violations"] <= QUALITY["isi_violations"])
              & (s["amplitude_cutoff"] <= QUALITY["amplitude_cutoff"])
              & (s["presence_ratio"] >= QUALITY["presence_ratio"])
              & np.array([loc.startswith(CORTEX_HIPPO) for loc in s["location"]]))
        rows = np.where(ok)[0]
        family = np.array([_family(loc) for loc in s["location"]])
        rows = rows[np.lexsort((s["y"][rows], s["probe"][rows], family[rows]))]
        if len(rows) > max_units:
            rows = rows[np.linspace(0, len(rows) - 1, max_units).round().astype(int)]
        starts = np.concatenate([[0], s["index"][:-1]])
        trains = []
        for r in rows:
            st = s["times"][starts[r]:s["index"][r]]
            trains.append(st[(st >= t0) & (st < t0 + dur)] - t0)
        ax = axes[2, j]
        ax.eventplot(trains, colors="#222", linewidths=0.6, linelengths=0.8)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        # One label per region family, centred on its block, with dividers
        fam = family[rows]
        edges = [0] + [i for i in range(1, len(fam)) if fam[i] != fam[i - 1]] + [len(fam)]
        for a in edges[1:-1]:
            ax.axhline(a - 0.5, color="#bbb", lw=0.8)
        ax.set_yticks([(a + b - 1) / 2 for a, b in zip(edges[:-1], edges[1:])])
        ax.set_yticklabels([fam[a] for a in edges[:-1]], fontsize=8)
        ax.set_ylabel(f"Units (n={len(rows)})")
        ax.set_xlabel("Time (s)")
        ax.set_xlim(0, dur)
        _style(ax)
    fig.suptitle(f"{w['attrs']['session']} — EEG, LFP and spikes on one timeline "
                 f"(Jonathan's common_start alignment)", fontsize=11)
    return fig


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--windows", type=Path, default=Path("data/windows"))
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--figures", type=Path, default=FIG_DIR)
    p.add_argument("--results", type=Path, default=RESULTS_DIR)
    args = p.parse_args(argv)
    args.figures.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)

    windows = load_windows(args.windows)

    plot_eeg_spectra(windows).savefig(args.figures / "eeg_spectra.png", dpi=150)

    table = band_power_table(windows)
    table.to_csv(args.results / "band_power.csv", index=False)
    plot_band_change(table).savefig(args.figures / "band_power_change.png", dpi=150)

    drops = []
    for session, wins in windows.items():
        kind = session_type_of(wins)
        plot_aligned(wins, PAIRS[kind]).savefig(args.figures / f"{session}_aligned.png", dpi=150)
        post = 600.0 if kind == "isoflurane" else 900.0
        tc, info = eeg_power_timecourse(args.raw / f"{session}.nwb", pre=120.0, post=post)
        d = detect_drop(tc)
        plot_power_timecourse(tc, info, d).savefig(
            args.figures / f"{session}_eeg_power_timecourse.png", dpi=150)
        d.insert(0, "session", session)
        d.insert(1, "anchor", info["anchor"])
        d.insert(2, "anchor_time_abs", info["anchor_time_abs"])
        d["drop_time_abs"] = d.drop_time_s + info["anchor_time_abs"]
        drops.append(d)
        plt.close("all")
    drops = pd.concat(drops, ignore_index=True)
    drops.to_csv(args.results / "power_drop.csv", index=False)
    print(drops.round(2).to_string(index=False))
    print(f"Figures -> {args.figures}   tables -> {args.results}")


if __name__ == "__main__":
    main()
