"""Task 4: map Neuropixels LFP to Mishler spectral motifs.

For each NWB session: read one LFP channel from a chosen probe/region, decimate,
run smooth_env (SmoothEnv.m), trim its NaN edges, then pass all sessions to
get_motifs (getMotifs.m). Finally measure how often each motif pole dominates in
each labelled brain state (awake / isoflurane / psilocybin ...).

Usage (from the repo root):
    python -m src.mapping.lfp_to_motifs
    python -m src.mapping.lfp_to_motifs --probe probeF --region MOs5 --out results/lfp_to_motifs
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pynwb import NWBHDF5IO  # noqa: E402
from scipy import signal  # noqa: E402

from src.mapping.get_motifs import get_motifs  # noqa: E402
from src.mapping.smooth_env import DEFAULT_FREQ_BANDS, mround, smooth_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TRIVIAL = -1


def load_lfp(path: Path, probe: str, region: str, target_fs: float) -> dict:
    """One LFP channel (middle of `region` on `probe`), decimated to ~target_fs, in µV."""
    with NWBHDF5IO(str(path), mode="r") as io:
        nwb = io.read()
        series = nwb.acquisition["LFP"][f"ElectricalSeries{probe}"]
        electrodes = series.electrodes.to_dataframe()
        in_region = np.flatnonzero((electrodes.location.values == region) & electrodes.is_data_valid.values)
        if in_region.size == 0:
            found = sorted(set(electrodes.location.values))
            raise ValueError(f"{path.name}: no valid {probe} channels in {region}. Regions on this probe: {found}")
        channel = int(in_region[in_region.size // 2])

        timestamps = series.timestamps[:]
        raw = series.data[:, channel] * series.conversion * 1e6
        fs_raw = 1 / np.median(np.diff(timestamps[:100_000]))

        trials = nwb.trials.to_dataframe()
        states = [
            (str(label), float(g.start_time.min()), float(g.stop_time.max()))
            for label, g in trials.groupby("behavioral_epoch")
        ]
        info = {
            "file": path.name,
            "subject": nwb.subject.subject_id,
            "experiment": nwb.session_description.split()[0],
            "pharmacology": nwb.pharmacology,
            "states": sorted(states, key=lambda s: s[1]),
        }

    q = mround(fs_raw / target_fs)
    return {
        **info,
        "probe": probe,
        "region": region,
        "channel": channel,
        "electrode_id": int(electrodes.index[channel]),
        "fs": fs_raw / q,
        "lfp": signal.decimate(raw, q, zero_phase=True),
        "t": timestamps[::q],
    }


def label_states(t: np.ndarray, states) -> np.ndarray:
    labels = np.full(t.size, "unlabeled", dtype=object)
    for name, start, stop in states:
        labels[(t >= start) & (t <= stop)] = name
    return labels


def pole_names(n_motifs: int):
    return [f"M{i + 1}+" for i in range(n_motifs)] + [f"M{i + 1}-" for i in range(n_motifs)]


def pole_color(pole: int, n_motifs: int) -> str:
    """Motif i's + pole is C(2i), its opponent C(2i + 1), in every figure."""
    return f"C{(2 * (pole % n_motifs) + (pole >= n_motifs)) % 10}"


def occupancy_table(sessions, session_of, state_of, pole_of, n_motifs):
    """Fraction of each state's ~1 Hz timepoints dominated by each motif pole."""
    names = pole_names(n_motifs) + ["all-same-sign"]
    codes = list(range(2 * n_motifs)) + [TRIVIAL]
    table = []
    for i, s in enumerate(sessions):
        for state, _, _ in s["states"]:
            pick = (session_of == i) & (state_of == state)
            if not pick.any():
                continue
            table.append({
                "session": f"{s['subject']} ({s['experiment']})",
                "state": state,
                "n_seconds": int(pick.sum()),
                "fraction": {name: float(np.mean(pole_of[pick] == c)) for name, c in zip(names, codes)},
            })
    return table


def band_labels(bands):
    return [f"{lo:g}-{hi:g}" for lo, hi in bands]


def plot_motifs(motifs, bands, path):
    n = motifs.shape[1] // 2
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4), squeeze=False, sharey=True)
    x = np.arange(len(bands))
    for i, ax in enumerate(axes[0]):
        ax.axhline(0, color="0.7", lw=0.8)
        ax.plot(x, motifs[:, i], "-o", color=pole_color(i, n), label=f"M{i + 1}+")
        ax.plot(x, motifs[:, i + n], "--o", color=pole_color(i + n, n), label=f"M{i + 1}- (opponent)")
        ax.set_xticks(x, band_labels(bands), rotation=60, fontsize=7)
        ax.set_title(f"Motif {i + 1}")
        ax.set_xlabel("Band (Hz)")
        ax.legend(fontsize=7, loc="lower left")
    axes[0][0].set_ylabel("Mean sign of z-scored envelope")
    fig.suptitle("LFP spectral motifs and their opponents")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_timeline(sessions, env_1hz, t_1hz, session_of, pole_of, n_motifs, bands, path):
    fig, axes = plt.subplots(
        2 * len(sessions), 1, figsize=(13, 3.2 * len(sessions)),
        gridspec_kw={"height_ratios": [4, 1] * len(sessions)},
    )
    for i, s in enumerate(sessions):
        pick = session_of == i
        t = t_1hz[pick]
        ax = axes[2 * i]
        ax.imshow(env_1hz[pick].T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                  extent=[t[0], t[-1], -0.5, len(bands) - 0.5], interpolation="none")
        ax.set_yticks(range(len(bands)), band_labels(bands), fontsize=6)
        ax.set_ylabel("Band (Hz)")
        ax.set_title(f"{s['subject']} ({s['experiment']}): z-scored band envelopes, {s['probe']} {s['region']}",
                     fontsize=9)
        for name, start, stop in s["states"]:
            ax.axvline(start, color="k", lw=0.8)
            ax.axvline(stop, color="k", lw=0.8, ls=":")
            ax.text(start, len(bands) - 0.7, f" {name}", va="top", fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.8, "pad": 1, "lw": 0})
        strip = axes[2 * i + 1]
        poles = pole_of[pick]
        colors = np.ones((1, poles.size, 3))
        for c in range(2 * n_motifs):
            colors[0, poles == c] = matplotlib.colors.to_rgb(pole_color(c, n_motifs))
        strip.imshow(colors, aspect="auto", extent=[t[0], t[-1], 0, 1], interpolation="none")
        strip.set_yticks([])
        strip.set_ylabel("Pole", fontsize=7)
        strip.set_xlim(ax.get_xlim())
    axes[-1].set_xlabel("Session time (s)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=pole_color(c, n_motifs)) for c in range(2 * n_motifs)]
    if handles:
        fig.legend(handles, pole_names(n_motifs), loc="lower center", ncol=min(2 * n_motifs, 10), fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_occupancy(table, n_motifs, path):
    names = pole_names(n_motifs) + ["all-same-sign"]
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(table) + 3), 4))
    width = 0.8 / len(names)
    x = np.arange(len(table))
    for k, name in enumerate(names):
        ax.bar(x + (k - (len(names) - 1) / 2) * width, [row["fraction"][name] for row in table], width,
               label=name, color="0.6" if name == "all-same-sign" else pole_color(k, n_motifs))
    ax.set_xticks(x, [f"{row['session']}\n{row['state']}" for row in table], fontsize=8)
    ax.set_ylabel("Fraction of seconds")
    ax.set_title("Which motif pole dominates, per brain state")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_bands(values):
    return np.array([[float(v) for v in item.split("-")] for item in values])


def prepare(nwb_paths, probe, region, fs, bands):
    """Load each session's LFP channel and run smooth_env, trimming the NaN edges."""
    sessions, envelopes = [], []
    for path in nwb_paths:
        t0 = time.time()
        s = load_lfp(Path(path), probe, region, fs)
        env = smooth_env(s.pop("lfp"), s["fs"], bands)
        valid = np.isfinite(env).all(axis=1)
        s["t"], env = s["t"][valid], env[valid]
        sessions.append(s)
        envelopes.append(env)
        print(f"[prep] {Path(path).name}: {s['experiment']}, {probe} {region} channel {s['channel']}, "
              f"{env.shape[0] / s['fs']:.0f} s at {s['fs']:.1f} Hz ({time.time() - t0:.0f}s)", flush=True)
    return {"sessions": sessions, "envelopes": envelopes, "bands": bands, "probe": probe, "region": region,
            "fs": float(np.mean([s["fs"] for s in sessions]))}


def find_motifs(prep, n_null=20, seed=0, verbose=True):
    """Run get_motifs on prepared sessions and label each ~1 Hz timepoint with its state and dominant pole."""
    sessions, envelopes, fs = prep["sessions"], prep["envelopes"], prep["fs"]
    t0 = time.time()
    result = get_motifs([e.T for e in envelopes], fs, n_null=n_null, seed=seed, verbose=verbose)
    print(f"[motifs] {prep['probe']} {prep['region']} seed {seed}: {result.n_motifs} motif pair(s) "
          f"({time.time() - t0:.0f}s)", flush=True)

    hop = mround(fs)
    env_1hz = np.concatenate(envelopes)[::hop]
    session_of = np.concatenate([np.full(e.shape[0], i) for i, e in enumerate(envelopes)])[::hop]
    state_of = np.concatenate([label_states(s["t"], s["states"]) for s in sessions])[::hop]
    signs = np.sign(env_1hz)
    pole_of = np.full(signs.shape[0], TRIVIAL)
    nontrivial = np.abs(signs.sum(axis=1)) != signs.shape[1]
    if result.n_motifs:
        pole_of[nontrivial] = np.argmax(signs[nontrivial] @ result.motifs, axis=1)
    return {
        **prep, "result": result, "n_null": n_null, "seed": seed,
        "env_1hz": env_1hz, "t_1hz": np.concatenate([s["t"] for s in sessions])[::hop],
        "session_of": session_of, "state_of": state_of, "pole_of": pole_of,
        "table": occupancy_table(sessions, session_of, state_of, pole_of, result.n_motifs),
    }


def write_outputs(run, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    result, bands, sessions = run["result"], run["bands"], run["sessions"]
    np.savez_compressed(
        out / "motifs.npz",
        motifs=result.motifs, freq_bands=bands, pole_names=np.array(pole_names(result.n_motifs)),
        env_1hz=run["env_1hz"], t_1hz=run["t_1hz"], session_of=run["session_of"],
        state_of=run["state_of"].astype(str), pole_of=run["pole_of"],
        null_max_size=result.null_max_size, n_kept=result.n_kept, null_shifts=result.shifts,
    )
    summary = {
        "params": {"probe": run["probe"], "region": run["region"], "fs": run["fs"], "freq_bands": bands.tolist(),
                   "n_null": run["n_null"], "seed": run["seed"], "corr_thresholds": list(result.thresholds)},
        "sessions": [{k: s[k] for k in ("file", "subject", "experiment", "pharmacology", "probe", "region",
                                        "channel", "electrode_id", "fs", "states")} for s in sessions],
        "n_timepoints_nontrivial": result.n_timepoints,
        "per_threshold": [
            {"corr_thresh": th, "n_candidates": int(sizes.size), "largest_candidate": int(sizes.max()),
             "null_max_size": int(nm), "kept_after_null_and_all_sessions": int(k)}
            for th, sizes, nm, k in zip(result.thresholds, result.candidate_sizes, result.null_max_size,
                                        result.n_kept)
        ],
        "n_motif_pairs": result.n_motifs,
        "motifs": {name: col.tolist() for name, col in zip(pole_names(result.n_motifs), result.motifs.T)},
        "occupancy": run["table"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    if result.n_motifs:
        plot_motifs(result.motifs, bands, out / "motifs.png")
        plot_occupancy(run["table"], result.n_motifs, out / "occupancy.png")
    plot_timeline(sessions, run["env_1hz"], run["t_1hz"], run["session_of"], run["pole_of"], result.n_motifs,
                  bands, out / "timeline.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Map NWB LFP to spectral motifs (Mishler getMotifs port)")
    parser.add_argument("--nwb", nargs="+", type=Path,
                        default=sorted((ROOT / "data" / "raw").glob("*.nwb")), help="NWB files, one per session")
    parser.add_argument("--probe", default="probeF", help="Neuropixels probe (default: probeF)")
    parser.add_argument("--region", default="MOs5", help="CCF region of the channel to use (default: MOs5)")
    parser.add_argument("--fs", type=float, default=250.0, help="sampling rate to decimate the LFP to (Hz)")
    parser.add_argument("--bands", nargs="+", help="frequency bands as lo-hi, e.g. 1-4 3-6 5-9 (Hz)")
    parser.add_argument("--n-null", type=int, default=20, help="number of shifted null datasets")
    parser.add_argument("--seed", type=int, default=0, help="seed for the null shifts")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "lfp_to_motifs", help="output folder")
    args = parser.parse_args()

    if not args.nwb:
        parser.error("no NWB files found in data/raw; run python3 -m src.scripts.download_dropbox_data first")
    bands = parse_bands(args.bands) if args.bands else DEFAULT_FREQ_BANDS
    if bands.max() >= args.fs / 2:
        parser.error(f"highest band edge {bands.max():g} Hz must be below Nyquist ({args.fs / 2:g} Hz)")

    prep = prepare(args.nwb, args.probe, args.region, args.fs, bands)
    run = find_motifs(prep, n_null=args.n_null, seed=args.seed)
    summary = write_outputs(run, args.out)

    print("\nPer threshold (candidates / largest real / largest null / kept):")
    for row in summary["per_threshold"]:
        print(f"  r={row['corr_thresh']}: {row['n_candidates']} / {row['largest_candidate']} / "
              f"{row['null_max_size']} / {row['kept_after_null_and_all_sessions']}")
    for row in run["table"]:
        top = sorted(row["fraction"].items(), key=lambda kv: -kv[1])[:3]
        print(f"  {row['session']:>24} {row['state']:<11} " + ", ".join(f"{k} {v:.0%}" for k, v in top))
    print(f"\nOutputs written to {args.out}")


if __name__ == "__main__":
    main()
