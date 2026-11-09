"""Run the state analysis on every valid EEG channel and every grey-matter LFP region.

Per signal (all three windows, strict and relaxed versions): motifs, band-power change
from awake to isoflurane, a band-shuffled guardrail run per state, and a chunk bootstrap
per state. Across signals: EEG x LFP pole co-occurrence within each state, and PL5-style
spikes-by-pole for every region with enough well-isolated units.

Usage (from code/python/):
    python -m src.channel_sweep --nwb ../../data/raw/720762-20240221-v001.nwb \
        --out ../../data/out/2026-11-01_mouse720762/all_channels --n-boot 20
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import session_data as sd  # noqa: E402
from .get_motifs import get_motifs  # noqa: E402
from .smooth_env import mround  # noqa: E402
from .state_motifs import (  # noqa: E402
    BANDS, FS, VERSIONS, WINDOWS, assign_poles, band_labels, band_shuffle, chunk, coupling, envelope_pieces,
    minority_sign_bands, one_hz, restrict, run_motifs, spikes_by_pole, welch_window, zscore,
)

STATES = ("awake", "isoflurane")
SWEEP_VERSIONS = ("strict", "relaxed")
NOT_GREY = {"root", "null", "alv", "ccb", "cing", "fp", "scwm", "int", "fa", "em", "st", "or", "chpl", "fi",
            "ccg", "fiber tracts"}
MIN_CHANNELS = 5
MIN_GOOD_UNITS = 10
GROUPS = [("prefrontal / olfactory", ("AON", "TTd", "DP", "ORB", "ILA", "PL", "MOs")), ("motor", ("MOp",)),
          ("somatosensory", ("SSp",)), ("visual", ("VIS",)), ("retrosplenial", ("RSP",)),
          ("hippocampus", ("CA", "DG")), ("thalamus", ("AMd", "IAD", "AD", "LD", "MD", "CL", "LP")),
          ("striatum", ("STR", "CP"))]
GROUP_COLORS = dict(zip([g for g, _ in GROUPS], ["C3", "C1", "C2", "C4", "C5", "C0", "C6", "C8"]))
BAND_EDGES = ((1, 4), (4, 8), (8, 12), (12, 30), (30, 80))


def group_of(region):
    for name, prefixes in GROUPS:
        if region.startswith(prefixes):
            return name
    return "other"


def list_signals(nwb):
    eeg = nwb.acquisition["ElectricalSeriesEEG"].electrodes.to_dataframe()
    signals = [{"key": f"eeg{i}", "kind": "eeg", "label": f"EEG ch{i} ({row.location})", "source": ["eeg", int(i)],
                "location": row.location, "x": float(row.x), "z": float(row.z), "group": "EEG"}
               for i, row in enumerate(eeg.itertuples()) if row.is_data_valid]
    best = {}
    for probe, series in nwb.acquisition["LFP"].electrical_series.items():
        probe = probe.replace("ElectricalSeries", "")
        el = series.electrodes.to_dataframe()
        valid = el[el.is_data_valid]
        for region, n in valid.location.value_counts().items():
            if region in NOT_GREY or n < MIN_CHANNELS:
                continue
            if region not in best or n > best[region][1]:
                best[region] = (probe, int(n))
    for region, (probe, n) in best.items():
        signals.append({"key": f"lfp_{region.replace('/', '-')}", "kind": "lfp", "label": f"LFP {region} ({probe})",
                        "source": ["lfp", probe, region], "location": region, "probe": probe, "n_channels": n,
                        "group": group_of(region)})
    order = {g: k for k, (g, _) in enumerate(GROUPS)}
    return signals[:sum(s["kind"] == "eeg" for s in signals)] + sorted(
        [s for s in signals if s["kind"] == "lfp"], key=lambda s: (order.get(s["group"], 99), s["location"]))


def _boot_run(chunks, seed):
    rng = np.random.default_rng(seed)
    picks = rng.integers(len(chunks), size=len(chunks))
    res = get_motifs([chunks[i] for i in picks], 1.0, seed=seed, **VERSIONS["relaxed"]["kw"])
    return res.motifs[:, : res.n_motifs]


def _best_abs_r(a, b):
    if a.shape[1] == 0 or b.shape[1] == 0:
        return None
    return float(np.abs(np.corrcoef(a.T, b.T)[: a.shape[1], a.shape[1]:]).max())


def analyze_signal(spec, nwb_path, n_boot):
    t0 = time.time()
    io, nwb = sd.open_nwb(nwb_path)
    info = sd.read_info(nwb, Path(nwb_path).name)
    segments, windows = sd.stim_free_segments(info), sd.analysis_windows(info)
    src = spec["source"]
    if src[0] == "lfp":
        spec["channel"] = sd.lfp_channel(nwb, src[1], src[2])
        src = ("lfp", src[1], spec["channel"])
    else:
        spec["channel"] = src[1]
        src = ("eeg", src[1])
    t, x, fs = sd.decimate(*sd.read_raw(nwb, src), FS)
    io.close()

    psd = {w: welch_window(t, x, fs, segments, windows[w]) for w in STATES}
    band_db = {f"{lo}-{hi}": float(10 * np.log10(
        psd["isoflurane"][1][(psd["isoflurane"][0] >= lo) & (psd["isoflurane"][0] < hi)].mean()
        / psd["awake"][1][(psd["awake"][0] >= lo) & (psd["awake"][0] < hi)].mean())) for lo, hi in BAND_EDGES}

    pieces = envelope_pieces(t, x, fs, segments)
    hop = mround(fs)
    out = {**spec, "band_db": band_db, "runs": {}, "reference": {}, "guardrail": {}, "bootstrap": {}}
    boot_motifs = {}
    for win in WINDOWS:
        chunks = chunk(zscore(restrict(pieces, windows[win], fs)), fs)
        out["runs"][win] = {}
        for ver in SWEEP_VERSIONS:
            res, rt = run_motifs(chunks, ver, fs)
            n = res.n_motifs
            out["runs"][win][ver] = {
                "n_motifs": n, "motifs": res.motifs.tolist(), "n_sessions": len(chunks),
                "min_p": [min(c["p"] for c in s) for s in res.motif_support],
                "minority_sign_bands": [minority_sign_bands(res.motifs[:, i]) for i in range(n)],
            }
            if n and win not in out["reference"]:
                tt, signs, _ = one_hz(chunks, fs)
                pole = assign_poles(signs, res.motifs)
                sec = np.round(tt).astype(int)
                out["reference"][win] = {"version": ver, "motifs": res.motifs.tolist(),
                                         "seconds": sec.tolist(), "pole": pole.tolist()}
        if win in STATES:
            g, _ = run_motifs(band_shuffle(chunks, np.random.default_rng(1000)), "relaxed", fs, seed=1)
            out["guardrail"][win] = g.n_motifs
            one_hz_chunks = [c.env[::hop].T.copy() for c in chunks]
            boot_motifs[win] = [_boot_run(one_hz_chunks, 2000 + b) for b in range(n_boot)]
    for st in STATES:
        ms = boot_motifs[st]
        within = [r for i in range(len(ms)) for j in range(i + 1, len(ms)) if (r := _best_abs_r(ms[i], ms[j])) is not None]
        out["bootstrap"][st] = {"detection_rate": float(np.mean([m.shape[1] > 0 for m in ms])),
                                "within_median": float(np.median(within)) if within else None}
    across = [r for a in boot_motifs["awake"] for b in boot_motifs["isoflurane"] if (r := _best_abs_r(a, b)) is not None]
    out["bootstrap"]["across_median"] = float(np.median(across)) if across else None
    out["runtime_s"] = round(time.time() - t0)
    return out


# ---------------------------------------------------------------- figures

def is_broadband(motifs):
    """All motifs in a (bands, 2n) set have at most 1 band of the minority sign."""
    m = np.array(motifs)
    return bool(m.size) and all(minority_sign_bands(m[:, i]) <= 1 for i in range(m.shape[1] // 2))


def cell_status(sig, win):
    ref = sig["reference"].get(win)
    if not ref:
        return 0
    if is_broadband(ref["motifs"]):
        return 3
    return 2 if ref["version"] == "strict" else 1


def genuine(sig, win):
    return cell_status(sig, win) in (1, 2)


def orientation(sig, win, reference):
    """+1 if the signal's first motif (+ pole) points the same way as the PL5 motif, else -1."""
    m = np.array(sig["reference"][win]["motifs"])
    return 1.0 if np.corrcoef(m[:, 0], reference)[0, 1] >= 0 else -1.0


STATUS_COLORS = ["#eeeeee", "#9ecae1", "#08519c", "#fdae6b"]
STATUS_LABELS = ["no motif", "relaxed only", "strict (and relaxed)", "broadband-like only"]


def plot_eeg_map(eeg, invalid, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharex=True, sharey=True)
    for ax, win in zip(axes, WINDOWS):
        for s in eeg:
            st = cell_status(s, win)
            ax.scatter(s["z"], s["x"], s=260, color=STATUS_COLORS[st], edgecolor="k", lw=0.6, zorder=3)
            ax.text(s["z"], s["x"], str(s["channel"]), ha="center", va="center", fontsize=7,
                    color="white" if st == 2 else "black", zorder=4)
        for z, xx in invalid:
            ax.scatter(z, xx, s=120, marker="x", color="0.5", zorder=3)
        ax.set_title(win, fontsize=10)
        ax.set_xlabel("medial–lateral (CCF z)")
    axes[0].invert_yaxis()
    axes[0].set_ylabel("anterior → posterior (CCF x)")
    handles = [plt.Line2D([], [], marker="o", ls="", markersize=10, markerfacecolor=c, markeredgecolor="k")
               for c in STATUS_COLORS] + [plt.Line2D([], [], marker="x", ls="", color="0.5")]
    fig.legend(handles, STATUS_LABELS + ["invalid channel"], loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("EEG electrodes (top view; frontal at the top): which channels have a motif", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lfp_grid(lfp, path):
    fig, ax = plt.subplots(figsize=(8.5, 0.27 * len(lfp) + 1.6))
    grid = np.array([[cell_status(s, w) for w in WINDOWS] for s in lfp])
    from matplotlib.colors import ListedColormap
    ax.imshow(grid, cmap=ListedColormap(STATUS_COLORS), vmin=0, vmax=3, aspect="auto")
    for r, s in enumerate(lfp):
        for c, w in enumerate(WINDOWS):
            if w in STATES:
                ax.text(c, r, f"boot {s['bootstrap'][w]['detection_rate']:.0%}", ha="center", va="center", fontsize=6.5,
                        color="white" if grid[r, c] == 2 else "black")
    ax.set_xticks(range(len(WINDOWS)), WINDOWS)
    ax.set_yticks(range(len(lfp)), [f"{s['location']} ({s['probe']})" for s in lfp], fontsize=7)
    for tick, s in zip(ax.get_yticklabels(), lfp):
        tick.set_color(GROUP_COLORS.get(s["group"], "k"))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STATUS_COLORS]
    ax.legend(handles, STATUS_LABELS, loc="upper center", bbox_to_anchor=(0.5, -0.03 - 1.0 / len(lfp)), ncol=4,
              fontsize=7, frameon=False)
    ax.set_title("LFP regions: motif per window (text = share of bootstrap resamples with a motif)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def aligned_shapes(sigs, win, reference):
    rows, labels, colors = [], [], []
    for s in sigs:
        ref = s["reference"].get(win)
        if not ref or is_broadband(ref["motifs"]):
            continue
        m = np.array(ref["motifs"]).reshape(len(BANDS), -1)
        for i in range(m.shape[1] // 2):
            v = m[:, i]
            rows.append(v if np.corrcoef(v, reference)[0, 1] >= 0 else -v)
            labels.append(f"{s['label']} M{i + 1}{' *' if ref['version'] == 'relaxed' else ''}")
            colors.append(GROUP_COLORS.get(s["group"], "k"))
    return np.array(rows), labels, colors


def plot_shapes(sigs, references, path):
    fig, axes = plt.subplots(1, len(STATES), figsize=(12, 0.2 * max(10, len(sigs)) + 2))
    for ax, st in zip(axes, STATES):
        rows, labels, colors = aligned_shapes(sigs, st, references[st])
        if rows.size == 0:
            ax.axis("off")
            continue
        ax.imshow(rows, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_yticks(range(len(labels)), labels, fontsize=6)
        for tick, c in zip(ax.get_yticklabels(), colors):
            tick.set_color(c)
        ax.set_xticks(range(len(BANDS)), band_labels(), rotation=60, fontsize=7)
        ax.set_title(f"{st}: motif + pole, aligned to the PL5 {st} motif\n(* = relaxed only; broadband-like motifs omitted)",
                     fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_coupling(matrices, eeg_labels, lfp_labels, path):
    fig, axes = plt.subplots(1, len(STATES), figsize=(14, 0.22 * max(len(lfp_labels), 8) + 2.5))
    for ax, st in zip(axes, STATES):
        m = matrices[st]
        if not np.isfinite(m).any():
            ax.text(0.5, 0.5, f"{st}: no EEG channel and LFP region\nboth have a motif", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off")
            continue
        im = ax.imshow(m.T, cmap="PuOr_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(eeg_labels)), eeg_labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(lfp_labels)), lfp_labels, fontsize=6)
        ax.set_title(f"{st}: pole co-occurrence r, poles aligned to PL5 (+ = PL5-like poles together)\n"
                     "(blank = no genuine motif in one of the two)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_band_change(sigs, path):
    m = np.array([[s["band_db"][f"{lo}-{hi}"] for lo, hi in BAND_EDGES] for s in sigs])
    fig, ax = plt.subplots(figsize=(6, 0.16 * len(sigs) + 1.5))
    im = ax.imshow(m, cmap="RdBu_r", vmin=-10, vmax=10, aspect="auto")
    ax.set_xticks(range(len(BAND_EDGES)), [f"{lo}-{hi} Hz" for lo, hi in BAND_EDGES], fontsize=8)
    ax.set_yticks(range(len(sigs)), [s["label"] for s in sigs], fontsize=5.5)
    for tick, s in zip(ax.get_yticklabels(), sigs):
        tick.set_color(GROUP_COLORS.get(s["group"], "k"))
    ax.set_title("Isoflurane vs awake power (dB), every signal", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.05)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bootstrap(sigs, path):
    rows = [s for s in sigs if s["bootstrap"]["across_median"] is not None]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for s in rows:
        w = [s["bootstrap"][st]["within_median"] for st in STATES if s["bootstrap"][st]["within_median"] is not None]
        ax.scatter(np.mean(w), s["bootstrap"]["across_median"], color=GROUP_COLORS.get(s["group"], "k"), s=30,
                   edgecolor="k", lw=0.3)
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Same state: median |r| between bootstrap motifs")
    ax.set_ylabel("Awake vs isoflurane: median |r|")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c) for c in ["k", *GROUP_COLORS.values()]]
    ax.legend(handles, ["EEG", *GROUP_COLORS.keys()], fontsize=7, loc="upper left")
    ax.set_title(f"{len(rows)} signals with bootstrap motifs in both states: below the diagonal = states differ",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_spikes(rows, path):
    if not rows:
        return
    fig, axes = plt.subplots(1, len(STATES), figsize=(12, 0.25 * max(len(rows) // 2, 6) + 2), sharex=True)
    for ax, st in zip(axes, STATES):
        rs = [r for r in rows if r["state"] == st]
        for k, r in enumerate(rs):
            ax.barh(k, r["diff"], color=GROUP_COLORS.get(r["group"], "k"), alpha=1.0 if r["p"] <= 0.05 else 0.35)
            ax.text(r["diff"] + (0.02 if r["diff"] >= 0 else -0.02), k, f"p={r['p']:.3f}", va="center",
                    ha="left" if r["diff"] >= 0 else "right", fontsize=6)
        ax.set_yticks(range(len(rs)), [f"{r['region']} M{r['motif']} ({r['n_units']} units)" for r in rs], fontsize=7)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(f"{st}: local firing, PL5-like pole minus opponent\n(faded = p > 0.05; genuine motifs only)", fontsize=9)
        ax.set_xlabel("Difference in local population rate (SD)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_outputs(sigs, coupling_rows, spike_rows, invalid, out):
    """Figures and sweep_summary.json from per-signal results (broadband-like motifs set apart)."""
    by_key = {s["key"]: s for s in sigs}
    eeg = [s for s in sigs if s["kind"] == "eeg"]
    lfp = [s for s in sigs if s["kind"] == "lfp"]
    pl5 = by_key["lfp_PL5"]
    references = {st: np.array(pl5["reference"][st]["motifs"])[:, 0] for st in STATES}

    matrices = {st: np.full((len(eeg), len(lfp)), np.nan) for st in STATES}
    kept_coupling = []
    for row in coupling_rows:
        e, lf, st = by_key[row["eeg"]], by_key[row["lfp"]], row["state"]
        if not (genuine(e, st) and genuine(lf, st)) or row["r"] != row["r"]:
            continue
        r = row["r"] * orientation(e, st, references[st]) * orientation(lf, st, references[st])
        matrices[st][eeg.index(e), lfp.index(lf)] = r
        kept_coupling.append(row | {"r_aligned": r})

    kept_spikes = []
    lfp_by_region = {s["location"]: s for s in lfp}
    for row in spike_rows:
        lf, st = lfp_by_region[row["region"]], row["state"]
        if genuine(lf, st):
            kept_spikes.append(row | {"diff": row["diff"] * orientation(lf, st, references[st]), "diff_raw": row["diff"]})

    figs = out / "figures"
    plot_eeg_map(eeg, invalid, figs / "11_eeg_channel_map.png")
    plot_lfp_grid(lfp, figs / "12_lfp_region_grid.png")
    plot_shapes(sigs, references, figs / "13_motif_shapes_all_signals.png")
    plot_coupling(matrices, [f"ch{e['channel']}" for e in eeg], [f"{s['location']}" for s in lfp],
                  figs / "14_eeg_lfp_coupling.png")
    plot_band_change(sigs, figs / "15_band_change_all_signals.png")
    plot_bootstrap(sigs, figs / "16_bootstrap_all_signals.png")
    plot_spikes(kept_spikes, figs / "17_spikes_by_pole_regions.png")

    counts = {}
    for kind, group in (("eeg", eeg), ("lfp", lfp)):
        for w in WINDOWS:
            st = [cell_status(s, w) for s in group]
            counts[f"{kind}/{w}"] = {"n_signals": len(group), "strict": st.count(2), "relaxed_only": st.count(1),
                                     "broadband_only": st.count(3), "none": st.count(0),
                                     "strict_names": [s["location"] if kind == "lfp" else s["channel"]
                                                      for s in group if cell_status(s, w) == 2],
                                     "broadband_names": [s["location"] if kind == "lfp" else s["channel"]
                                                         for s in group if cell_status(s, w) == 3]}
    boot = {}
    for kind, group in (("eeg", eeg), ("lfp", lfp)):
        both = [s for s in group if s["bootstrap"]["across_median"] is not None]
        within = [np.mean([s["bootstrap"][st]["within_median"] for st in STATES if s["bootstrap"][st]["within_median"]])
                  for s in both]
        across = [s["bootstrap"]["across_median"] for s in both]
        boot[kind] = {"n_both": len(both), "within_median": float(np.median(within)),
                      "across_median": float(np.median(across)),
                      "n_across_below_within_by_0.2": int(sum(a < w - 0.2 for a, w in zip(across, within))),
                      "detection_awake_median": float(np.median([s["bootstrap"]["awake"]["detection_rate"] for s in group])),
                      "detection_iso_median": float(np.median([s["bootstrap"]["isoflurane"]["detection_rate"] for s in group]))}
    coup = {}
    for st in STATES:
        rows = sorted([r for r in kept_coupling if r["state"] == st], key=lambda r: -abs(r["r_aligned"]))
        coup[st] = {"n_pairs": len(rows), "median_abs_r": float(np.median([abs(r["r_aligned"]) for r in rows])) if rows else None,
                    "share_positive": float(np.mean([r["r_aligned"] > 0 for r in rows])) if rows else None,
                    "top": rows[:10]}
        best_eeg = {}
        for r in rows:
            best_eeg.setdefault(r["eeg"], []).append(abs(r["r_aligned"]))
        coup[st]["eeg_ranked"] = sorted(((k, float(np.median(v)), len(v)) for k, v in best_eeg.items()),
                                        key=lambda x: -x[1])[:8]
    summary = {
        "counts": counts, "bootstrap": boot, "coupling": coup, "spikes": kept_spikes,
        "spikes_excluded_broadband": [r["region"] + " " + r["state"] for r in spike_rows
                                      if not genuine(lfp_by_region[r["region"]], r["state"])],
        "guardrail_hits": [(s["label"], w) for s in sigs for w, n in s["guardrail"].items() if n],
        "guardrail_runs": sum(len(s["guardrail"]) for s in sigs),
        "n_motifs_total": sum(len(s["runs"][w][v]["minority_sign_bands"]) for s in sigs for w in WINDOWS
                              for v in SWEEP_VERSIONS),
        "n_motifs_broadband": sum(m <= 1 for s in sigs for w in WINDOWS for v in SWEEP_VERSIONS
                                  for m in s["runs"][w][v]["minority_sign_bands"]),
    }
    (out / "sweep_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nwb", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=20)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--replot", action="store_true", help="rebuild figures and summary from sweep_results.json")
    args = ap.parse_args()
    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    io, nwb = sd.open_nwb(args.nwb)
    info = sd.read_info(nwb, args.nwb.name)
    signals = list_signals(nwb)
    eeg_df = nwb.acquisition["ElectricalSeriesEEG"].electrodes.to_dataframe()
    invalid = [(float(r.z), float(r.x)) for r in eeg_df.itertuples() if not r.is_data_valid]
    if args.replot:
        io.close()
        saved = json.loads((args.out / "sweep_results.json").read_text().replace("NaN", "null"))
        rows = [r | {"r": float("nan") if r["r"] is None else r["r"]} for r in saved["eeg_lfp_coupling"]]
        make_outputs(saved["signals"], rows, saved["spikes_by_pole"], invalid, args.out)
        print("replotted from sweep_results.json", flush=True)
        return
    print(f"{len(signals)} signals ({sum(s['kind'] == 'eeg' for s in signals)} EEG, "
          f"{sum(s['kind'] == 'lfp' for s in signals)} LFP regions); {args.workers} workers", flush=True)

    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(analyze_signal, s, str(args.nwb), args.n_boot): s["key"] for s in signals}
        for k, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results[r["key"]] = r
            found = {w: f"{r['runs'][w]['strict']['n_motifs']}/{r['runs'][w]['relaxed']['n_motifs']}" for w in WINDOWS}
            print(f"[{time.time() - t0:5.0f}s] {k:2d}/{len(signals)} {r['label']:28s} strict/relaxed {found} "
                  f"guard {sum(r['guardrail'].values())} ({r['runtime_s']}s)", flush=True)
    sigs = [results[s["key"]] for s in signals]
    eeg = [s for s in sigs if s["kind"] == "eeg"]
    lfp = [s for s in sigs if s["kind"] == "lfp"]

    def poles(sig, win):
        ref = sig["reference"][win]
        return dict(zip(ref["seconds"], ref["pole"])), np.array(ref["motifs"]).shape[1] // 2

    coupling_rows = []
    for st in STATES:
        for e in eeg:
            if st not in e["reference"]:
                continue
            pe, ne = poles(e, st)
            me = np.array(e["reference"][st]["motifs"])
            for lf in lfp:
                if st not in lf["reference"]:
                    continue
                pl, nl = poles(lf, st)
                r = np.array(coupling(pe, pl, ne, nl, info)[st]["r"], dtype=float)
                best = np.unravel_index(np.nanargmax(np.abs(r)), r.shape) if np.isfinite(r).any() else None
                ml = np.array(lf["reference"][st]["motifs"])
                shape = np.abs(np.corrcoef(me[:, :ne].T, ml[:, :nl].T)[:ne, ne:]).max()
                val = float(r[best]) if best is not None else np.nan
                coupling_rows.append({"state": st, "eeg": e["key"], "lfp": lf["key"], "r": val, "shape_abs_r": float(shape)})

    io, nwb = sd.open_nwb(args.nwb)
    spike_rows = []
    for lf in lfp:
        trains, _, _ = sd.good_units(nwb, lf["location"])
        if len(trains) < MIN_GOOD_UNITS:
            continue
        for st in STATES:
            if st not in lf["reference"]:
                continue
            p, n = poles(lf, st)
            for row in spikes_by_pole(trains, p, n, info):
                if row["state"] == st:
                    spike_rows.append(row | {"region": lf["location"], "group": lf["group"], "n_units": len(trains),
                                             "version": lf["reference"][st]["version"]})
    io.close()

    make_outputs(sigs, coupling_rows, spike_rows, invalid, args.out)

    def light(s):
        return {k: v for k, v in s.items() if k != "reference"} | {
            "reference": {w: {"version": r["version"], "motifs": r["motifs"]} for w, r in s["reference"].items()}}

    (args.out / "sweep_results.json").write_text(json.dumps({
        "file": info.file, "n_boot": args.n_boot, "bands_hz": BANDS.tolist(),
        "signals": [light(s) for s in sigs], "eeg_lfp_coupling": coupling_rows, "spikes_by_pole": spike_rows,
        "runtime_s": round(time.time() - t0)}, indent=1, default=float))
    lines = ["signal,kind,group,window,strict_n,relaxed_n,relaxed_min_p,boot_detect_awake,boot_detect_iso,"
             "boot_within_awake,boot_within_iso,boot_across"]
    for s in sigs:
        b = s["bootstrap"]
        for w in WINDOWS:
            rr = s["runs"][w]
            lines.append(",".join(map(str, [s["label"].replace(",", ";"), s["kind"], s["group"], w,
                                            rr["strict"]["n_motifs"], rr["relaxed"]["n_motifs"],
                                            min(rr["relaxed"]["min_p"], default=""),
                                            b["awake"]["detection_rate"], b["isoflurane"]["detection_rate"],
                                            b["awake"]["within_median"], b["isoflurane"]["within_median"],
                                            b["across_median"]])))
    (args.out / "sweep_table.csv").write_text("\n".join(lines) + "\n")
    print(f"[{time.time() - t0:5.0f}s] done", flush=True)


if __name__ == "__main__":
    main()
