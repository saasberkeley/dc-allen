"""Chunk-bootstrap stability of state motifs: are awake and isoflurane motifs the same?

For each signal and state, the ~5-min chunks are resampled with replacement and relaxed
motif detection is rerun. Within-state |r| (bootstrap motif vs bootstrap motif, same state)
shows how reproducible a state's motif is; across-state |r| (awake bootstrap vs isoflurane
bootstrap) shows how similar the two states' motifs are. Across well below within means the
states have different motifs.

Chunks are reduced to 1 Hz before resampling, which is what get_motifs analyses anyway
(shifts are whole multiples of 20 s); only the sub-second sampling phase differs.

Usage (from code/python/):
    python -m src.motif_bootstrap --nwb ../../data/raw/720762-20240221-v001.nwb \
        --out ../../data/out/2026-11-01_mouse720762 --n-boot 40
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import session_data as sd  # noqa: E402
from .get_motifs import get_motifs  # noqa: E402
from .smooth_env import mround  # noqa: E402
from .state_motifs import FS, SIGNALS, VERSIONS, chunk, envelope_pieces, restrict, zscore  # noqa: E402

STATES = ("awake", "isoflurane")


def _run(args):
    chunks, seed = args
    rng = np.random.default_rng(seed)
    picks = rng.integers(len(chunks), size=len(chunks))
    res = get_motifs([chunks[i] for i in picks], 1.0, seed=seed, **VERSIONS["relaxed"]["kw"])
    return res.motifs[:, : res.n_motifs]


def best_abs_r(a, b):
    """Best |r| between any + pole in a and any in b (None if either is empty)."""
    if a.shape[1] == 0 or b.shape[1] == 0:
        return None
    return float(np.abs(np.corrcoef(a.T, b.T)[: a.shape[1], a.shape[1]:]).max())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nwb", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=40)
    args = ap.parse_args()
    t0 = time.time()
    results = json.loads((args.out / "results.json").read_text())
    original = {(r["signal"], r["window"]): np.array(r["motifs"]).reshape(len(results["bands_hz"]), -1)
                for r in results["motif_runs"] if r["version"] == "relaxed"}

    io, nwb = sd.open_nwb(args.nwb)
    info = sd.read_info(nwb, args.nwb.name)
    segments, windows = sd.stim_free_segments(info), sd.analysis_windows(info)
    chunks = {}
    for spec in SIGNALS:
        src = spec["source"]
        if src[0] == "lfp":
            src = ("lfp", src[1], sd.lfp_channel(nwb, src[1], src[2]))
        t, x, fs = sd.decimate(*sd.read_raw(nwb, src), FS)
        pieces = envelope_pieces(t, x, fs, segments)
        hop = mround(fs)
        for st in STATES:
            chunks[(spec["key"], st)] = [c.env[::hop].T.copy() for c in chunk(zscore(restrict(pieces, windows[st], fs)), fs)]
    io.close()
    print(f"[{time.time() - t0:5.0f}s] envelopes ready; running {args.n_boot} bootstraps x {len(chunks)}", flush=True)

    boot = {}
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as pool:
        for key, ch in chunks.items():
            boot[key] = list(pool.map(_run, [(ch, 1000 + b) for b in range(args.n_boot)]))
            print(f"[{time.time() - t0:5.0f}s] {key}: motifs in {sum(m.shape[1] > 0 for m in boot[key])}/"
                  f"{args.n_boot} bootstraps", flush=True)

    summary = []
    for spec in SIGNALS:
        sig = spec["key"]
        row = {"signal": sig, "label": spec["label"]}
        for st in STATES:
            ms = boot[(sig, st)]
            orig = original.get((sig, st), np.zeros((len(results["bands_hz"]), 0)))
            orig_plus = orig[:, : orig.shape[1] // 2]
            within = [r for i in range(len(ms)) for j in range(i + 1, len(ms)) if (r := best_abs_r(ms[i], ms[j])) is not None]
            to_orig = [r for m in ms if (r := best_abs_r(m, orig_plus)) is not None]
            row[st] = {"detection_rate": float(np.mean([m.shape[1] > 0 for m in ms])),
                       "original_motifs": int(orig_plus.shape[1]),
                       "recovers_original_rate": float(np.mean([(best_abs_r(m, orig_plus) or 0) >= 0.9 for m in ms]))
                       if orig_plus.shape[1] else None,
                       "within_abs_r": within, "to_original_abs_r": to_orig}
        a, b = boot[(sig, "awake")], boot[(sig, "isoflurane")]
        row["across_abs_r"] = [r for ma in a for mb in b if (r := best_abs_r(ma, mb)) is not None]
        summary.append(row)

    (args.out / "bootstrap.json").write_text(json.dumps({"n_boot": args.n_boot, "results": summary}, indent=1))
    fig, axes = plt.subplots(1, len(summary), figsize=(4.4 * len(summary), 3.8), sharey=True)
    colors = {"awake": "#1f77b4", "isoflurane": "#d62728", "across": "#7f7f7f"}
    for ax, row in zip(axes, summary):
        groups = [("awake", row["awake"]["within_abs_r"]), ("isoflurane", row["isoflurane"]["within_abs_r"]),
                  ("across", row["across_abs_r"])]
        for k, (name, vals) in enumerate(groups):
            if vals:
                parts = ax.violinplot(vals, [k], showmedians=True, widths=0.8)
                for body in parts["bodies"]:
                    body.set_facecolor(colors[name])
                    body.set_alpha(0.5)
            else:
                ax.text(k, 0.5, "no\nmotifs", ha="center", va="center", fontsize=8, color="0.4")
        ax.set_xticks(range(3), [f"awake vs awake\n(found {row['awake']['detection_rate']:.0%})",
                                 f"iso vs iso\n(found {row['isoflurane']['detection_rate']:.0%})",
                                 "awake vs iso"], fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.set_title(row["label"], fontsize=9)
    axes[0].set_ylabel("Best |r| between bootstrap motifs")
    fig.suptitle(f"Chunk bootstrap ({args.n_boot} resamples per state, relaxed detection): "
                 "how reproducible are motifs, and do the states share them?", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out / "figures" / "10_bootstrap_stability.png", dpi=150)
    print(f"[{time.time() - t0:5.0f}s] done", flush=True)


if __name__ == "__main__":
    main()
