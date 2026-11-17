"""S-B runner. Usage (from the repo root): python code/python/src/run_cross_region.py"""
import glob, sys, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code/python"))
from src import cross_region as cr

def job(args):
    warnings.filterwarnings("ignore")
    s, st = args
    return cr.run_session(s, st, np.random.default_rng(0))

if __name__ == "__main__":
    sessions = sorted({Path(f).stem[:15] for f in glob.glob(str(REPO / "data/processed/multichannel/*_probe*.npz"))})
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(job, [(s, st) for s in sessions for st in (0, 1)]))
    pd.concat(res, ignore_index=True).to_pickle(REPO / "data/processed/spatial/cross_region.pkl"); print("done")
