"""Run lfp_to_motifs over many brain regions and seeds, and compare the motifs found.

Usage (from the repo root):
    python -m src.mapping.region_sweep                 # 10 regions x 3 seeds, ~20 min
    python -m src.mapping.region_sweep --seeds 0       # one seed per region, ~7 min

Each region's full outputs (seed 0) go to <out>/<region>/; the cross-region
comparison goes to <out>/region_comparison.png, motif_shapes.png and sweep_summary.json.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.lfp_to_motifs import (  # noqa: E402
    ROOT, band_labels, find_motifs, pole_names, prepare, write_outputs,
)
from src.smooth_env import DEFAULT_FREQ_BANDS  # noqa: E402

# Regions with valid channels in both sessions, and the probe that records them.
# REGIONS = [
#     ("probeF", "MOs2/3", "cortex"), ("probeF", "MOs5", "cortex"), ("probeF", "MOs6a", "cortex"),
#     ("probeD", "SSp-bfd2/3", "cortex"), ("probeD", "SSp-bfd5", "cortex"), ("probeB", "SSp-tr5", "cortex"),
#     ("probeB", "CA3", "hippocampus"), ("probeB", "LD", "thalamus"), ("probeB", "VAL", "thalamus"),
#     ("probeF", "CP", "striatum"),
# ]
REGIONS = [
    ("probeF", "MOs1", "cortex"), ("probeF", "MOs2/3", "cortex"), ("probeF", "MOs5", "cortex"), ("probeF", "MOs6a", "cortex")
]
GROUP_COLORS = {"cortex": "C0", "hippocampus": "C2", "thalamus": "C4", "striatum": "C1"}


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def oriented_pairs(run, reference):
    """Each motif pair as (+pole, -pole, swapped), flipped so +pole correlates positively with reference."""
    motifs, n = run["result"].motifs, run["result"].n_motifs
    pairs = []
    for i in range(n):
        plus, minus = motifs[:, i], motifs[:, i + n]
        swapped = reference is not None and corr(plus, reference) < 0
        pairs.append((minus, plus, swapped) if swapped else (plus, minus, swapped))
    return pairs


def balance(table, motif, n_motifs, swapped):
    """(f+ - f-) / (f+ + f-) per state: +1 means the + pole always wins over its opponent."""
    plus, minus = pole_names(n_motifs)[motif], pole_names(n_motifs)[motif + n_motifs]
    if swapped:
        plus, minus = minus, plus
    out = {}
    for row in table:
        fp, fm = row["fraction"][plus], row["fraction"][minus]
        out[f"{row['session']}\n{row['state']}"] = (fp - fm) / (fp + fm) if fp + fm else np.nan
    return out


def plot_comparison(rows, bands, state_keys, path):
    n = len(rows)
    fig, (ax_shape, ax_bal, ax_txt) = plt.subplots(
        1, 3, figsize=(15, 0.42 * n + 2.4), gridspec_kw={"width_ratios": [len(bands), len(state_keys) * 1.3, 3.2]},
    )
    shapes = np.array([r["plus"] if r["plus"] is not None else np.full(len(bands), np.nan) for r in rows])
    bal = np.array([[r["balance"].get(k, np.nan) for k in state_keys] for r in rows])
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.88")

    ax_shape.imshow(np.ma.masked_invalid(shapes), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax_shape.set_xticks(range(len(bands)), band_labels(bands), rotation=60, fontsize=7)
    ax_shape.set_yticks(range(n), [r["label"] for r in rows], fontsize=8)
    for tick, r in zip(ax_shape.get_yticklabels(), rows):
        tick.set_color(GROUP_COLORS[r["group"]])
    ax_shape.set_xlabel("Band (Hz)")
    ax_shape.set_title("Motif shape (+ pole, aligned to MOs5 M1+)", fontsize=9)

    im = ax_bal.imshow(np.ma.masked_invalid(bal), cmap=plt.get_cmap("PuOr_r"), vmin=-1, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(bal):
        if np.isfinite(v):
            ax_bal.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 else "black")
    short = [f"mouse {k.split(' (')[0]}\n{k.split(chr(10))[-1]}" for k in state_keys]
    ax_bal.set_xticks(range(len(state_keys)), short, fontsize=7, rotation=30, ha="right")
    ax_bal.set_yticks([])
    ax_bal.set_title("Pole balance per state  (f+ − f−) / (f+ + f−)", fontsize=9)
    fig.colorbar(im, ax=ax_bal, fraction=0.04, pad=0.02)

    ax_txt.axis("off")
    ax_txt.set_ylim(n - 0.5, -0.5)
    ax_txt.set_title("Pairs per seed · shape r vs seed 0\n· r vs MOs5 M1+", fontsize=9)
    for i, r in enumerate(rows):
        ax_txt.text(0, i, r["note"], va="center", fontsize=7.5, family="monospace")

    handles = [plt.Line2D([], [], color=c, lw=6) for c in GROUP_COLORS.values()]
    fig.legend(handles, GROUP_COLORS.keys(), loc="lower left", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Spectral motifs across brain regions (LFP, 2 mice: isoflurane and psilocybin sessions)")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_shapes(rows, bands, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(bands))
    ax.axhline(0, color="0.7", lw=0.8)
    for r in rows:
        if r["plus"] is not None:
            ax.plot(x, r["plus"], "-o", ms=3, lw=1.2, alpha=0.8, color=GROUP_COLORS[r["group"]])
    handles = [plt.Line2D([], [], color=c, marker="o") for c in GROUP_COLORS.values()]
    ax.legend(handles, GROUP_COLORS.keys(), fontsize=8)
    ax.set_xticks(x, band_labels(bands), rotation=60, fontsize=8)
    ax.set_xlabel("Band (Hz)")
    ax.set_ylabel("Mean sign of z-scored envelope")
    ax.set_title("Every motif found (+ pole, aligned to MOs5 M1+), colored by brain area", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LFP spectral motifs across brain regions and seeds")
    parser.add_argument("--nwb", nargs="+", type=Path, default=sorted((ROOT / "data" / "raw").glob("*.nwb")))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--fs", type=float, default=250.0)
    parser.add_argument("--n-null", type=int, default=20)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "region_sweep")
    args = parser.parse_args()
    bands = DEFAULT_FREQ_BANDS

    runs = {}
    for probe, region, _ in REGIONS:
        prep = prepare(args.nwb, probe, region, args.fs, bands)
        runs[region] = [find_motifs(prep, n_null=args.n_null, seed=s, verbose=False) for s in args.seeds]
        write_outputs(runs[region][0], args.out / region.replace("/", "-"))

    ref_run = runs["MOs5"][0]
    reference = ref_run["result"].motifs[:, 0] if ref_run["result"].n_motifs else None

    rows, summary = [], []
    for probe, region, group in REGIONS:
        first = runs[region][0]
        per_seed = [r["result"].n_motifs for r in runs[region]]
        pairs = oriented_pairs(first, reference)
        if not pairs:
            rows.append({"label": f"{region} ({probe})", "group": group, "plus": None, "balance": {},
                         "note": f"{'/'.join(map(str, per_seed)):>7}  no motif"})
        for i, (plus, minus, swapped) in enumerate(pairs):
            others = [r["result"].motifs for r in runs[region][1:]]
            stability = [max((abs(corr(plus, m)) for m in o.T), default=np.nan) for o in others]
            r_ref = corr(plus, reference) if reference is not None else np.nan
            bal = balance(first["table"], i, len(pairs), swapped)
            rows.append({
                "label": f"{region} ({probe}) M{i + 1}", "group": group, "plus": plus, "balance": bal,
                "note": f"{'/'.join(map(str, per_seed)):>7}  "
                        f"{' '.join(f'{s:.2f}' for s in stability) or '-':<10}  {r_ref:+.2f}",
            })
            summary.append({
                "region": region, "probe": probe, "group": group, "motif": i + 1,
                "channels": [s["channel"] for s in first["sessions"]],
                "plus_pole": plus.tolist(), "minus_pole": minus.tolist(),
                "corr_with_MOs5_M1": r_ref, "shape_corr_other_seeds": stability,
                "pairs_per_seed": per_seed, "balance_per_state": {k.replace("\n", " "): v for k, v in bal.items()},
            })
        if not pairs:
            summary.append({"region": region, "probe": probe, "group": group, "motif": None,
                            "pairs_per_seed": per_seed})

    state_keys = list(next(r["balance"] for r in rows if r["balance"]).keys()) if summary else []
    args.out.mkdir(parents=True, exist_ok=True)
    plot_comparison(rows, bands, state_keys, args.out / "region_comparison.png")
    plot_shapes(rows, bands, args.out / "motif_shapes.png")
    (args.out / "sweep_summary.json").write_text(json.dumps(
        {"seeds": args.seeds, "n_null": args.n_null, "freq_bands": bands.tolist(), "results": summary},
        indent=2, default=float,
    ))

    print("\nregion          pairs/seed  r vs MOs5  balance per state")
    for r in rows:
        bal = "  ".join(f"{v:+.2f}" for v in r["balance"].values())
        print(f"  {r['label']:<26} {r['note']:<28} {bal}")
    print(f"\nOutputs written to {args.out}")


if __name__ == "__main__":
    main()
