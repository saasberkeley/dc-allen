"""Why getMotifs.m finds few motifs within one state: null datasets keep more timepoints.

For each signal and window (original rules, one session), records the size of the largest
real candidate and of each null's largest candidate, and how many non-trivial timepoints
(not all bands the same sign) the real and the shifted datasets have.

Usage (from code/python/):
    python -m src.null_size_diagnosis --nwb ../../data/raw/720762-20240221-v001.nwb \
        --out ../../data/out/2026-11-01_mouse720762/diagnostics
"""

import argparse
import json
from pathlib import Path

import numpy as np

from . import session_data as sd
from .get_motifs import _binarize, get_motifs
from .smooth_env import mround
from .state_motifs import FS, SIGNALS, WINDOWS, envelope_pieces, joined, restrict, zscore


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nwb", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    io, nwb = sd.open_nwb(args.nwb)
    info = sd.read_info(nwb, args.nwb.name)
    segments, windows = sd.stim_free_segments(info), sd.analysis_windows(info)
    rows = []
    for spec in SIGNALS:
        src = spec["source"]
        if src[0] == "lfp":
            src = ("lfp", src[1], sd.lfp_channel(nwb, src[1], src[2]))
        t, x, fs = sd.decimate(*sd.read_raw(nwb, src), FS)
        pieces = envelope_pieces(t, x, fs, segments)
        for win in WINDOWS:
            sessions = [s.env.T for s in joined(zscore(restrict(pieces, windows[win], fs)))]
            res = get_motifs(sessions, fs, seed=0)
            hop = mround(fs)
            n_null = [_binarize(sessions, res.shifts[:, s], hop)[0].shape[0] for s in range(1, res.shifts.shape[1])]
            rows.append({
                "signal": spec["key"], "window": win,
                "n_real": res.n_timepoints, "n_total": int(sessions[0][:, ::hop].shape[1]),
                "n_null_median": float(np.median(n_null)),
                "per_threshold": [{"threshold": th, "real_largest": int(sizes.max()),
                                   "null_largest_median": float(np.median(res.null_largest[:, k])),
                                   "null_largest_max": float(res.null_largest[:, k].max())}
                                  for k, (th, sizes) in enumerate(zip(res.thresholds, res.candidate_sizes))],
            })
            print(rows[-1]["signal"], win, rows[-1]["n_real"], rows[-1]["n_null_median"], flush=True)
    io.close()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "null_size_diagnosis.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
