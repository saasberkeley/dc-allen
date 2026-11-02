"""False-positive calibration of get_motifs settings on synthetic data with no motifs.

Two kinds of motif-free data, 5 sessions x 330 s x 10 bands, smoothed like envelopes:
  independent  every band is independent noise: no coupling between bands at all.
  broadband    independent noise plus one component shared by all bands (weight w): the
               bands co-fluctuate, but there is no multi-band pattern beyond "all up/down".

Usage (from code/python/):
    python -m src.motif_calibration --out ../../data/out/2026-11-01_mouse720762/diagnostics
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .get_motifs import get_motifs
from .state_motifs import minority_sign_bands

CONFIGS = {
    "strict (getMotifs.m rules)": {"n_null": 20},
    "relaxed: p-value + half the chunks": {"n_null": 100, "null_mode": "pvalue", "alpha": 0.05,
                                           "min_session_frac": 0.5},
    "relaxed: ... + size normalization (used)": {"n_null": 100, "null_mode": "pvalue", "alpha": 0.05,
                                                 "min_session_frac": 0.5, "normalize_size": True},
    "relaxed: ... + PCA support 1 (rejected)": {"n_null": 100, "null_mode": "pvalue", "alpha": 0.05,
                                                "min_session_frac": 0.5, "min_pca_support": 1},
}
DATASETS = {"independent": 0.0, "broadband w=0.5": 0.5, "broadband w=1": 1.0, "broadband w=2": 2.0}


def synthetic(rng, weight, n_bands=10, n_sessions=5, seconds=330):
    out = []
    for _ in range(n_sessions):
        x = gaussian_filter1d(rng.standard_normal((n_bands, seconds)), sigma=3, axis=1)
        x = x / x.std(axis=1, keepdims=True)
        if weight:
            common = gaussian_filter1d(rng.standard_normal(seconds), sigma=3)
            x = x + weight * common / common.std()
        out.append((x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-datasets", type=int, default=20)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for data_name, weight in DATASETS.items():
        for cfg_name, kw in CONFIGS.items():
            hits, trivial, minority = 0, [], []
            for seed in range(args.n_datasets):
                sessions = synthetic(np.random.default_rng(seed), weight)
                signs = np.sign(np.concatenate([s.T for s in sessions]))
                trivial.append(float(np.mean(np.abs(signs.sum(axis=1)) == signs.shape[1])))
                res = get_motifs(sessions, fs=1.0, seed=seed, **kw)
                hits += res.n_motifs > 0
                minority += [minority_sign_bands(res.motifs[:, i]) for i in range(res.n_motifs)]
            flagged = sum(m <= 1 for m in minority)
            rows.append({"data": data_name, "config": cfg_name, "datasets_with_false_motif": int(hits),
                         "n_datasets": args.n_datasets, "all_same_sign_fraction": float(np.mean(trivial)),
                         "false_motifs": len(minority), "false_motifs_flagged_broadband": int(flagged),
                         "false_motif_minority_sign_bands": minority})
            print(f"[{time.time() - t0:5.0f}s] {data_name:16s} {cfg_name:42s} {hits}/{args.n_datasets} datasets; "
                  f"{flagged}/{len(minority)} false motifs flagged broadband-like", flush=True)
    (args.out / "calibration.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
