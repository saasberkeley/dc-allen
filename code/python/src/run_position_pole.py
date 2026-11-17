"""S-C runner. Usage (from the repo root): python code/python/src/run_position_pole.py"""
import glob, json, sys, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code/python"))
from src import awake_grounding as ag, motif_value as mv, position_pole as pp

def job(session):
    warnings.filterwarnings("ignore")
    R = [ag.load(f) for f in glob.glob(str(REPO / "data/processed/awake_grounding/*.pkl"))]
    ref = mv.consensus_templates([u for r in R if r["meta"]["session"][:6] != session[:6] for u in r["templates"]["original"]])[0][0]
    return pp.session_table(session, ref, np.random.default_rng(0))

if __name__ == "__main__":
    sessions = sorted({Path(f).stem[:15] for f in glob.glob(str(REPO / "data/processed/multichannel_local/*_probe*.npz"))})
    with ProcessPoolExecutor(2) as ex:
        T = pd.concat(list(ex.map(job, sessions)), ignore_index=True)
    T.to_pickle(REPO / "data/processed/spatial/position_pole.pkl"); print("done", len(T), "units")
