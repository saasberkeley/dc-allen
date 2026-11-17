"""E2 runner. Usage (from the repo root): python code/python/src/run_transfer_states.py"""
import glob, sys, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code/python"))
from src import awake_grounding as ag, relationships as rl, iso_state as iso, transfer_states as ts

def job(k):
    warnings.filterwarnings("ignore")
    r = ag.load(REPO / f"data/processed/awake_grounding/{k}.pkl")
    e = ag.load(REPO / f"data/processed/ensembles_awake/{k}.pkl")
    path = glob.glob(str(REPO / f"data/raw/{r['meta']['session']}*.nwb"))[0]
    da = rl.assemble(path, r, e)
    di = iso.assemble_iso(path, r, e) if da is not None else None
    return ts.run(da, di) if di is not None else None

if __name__ == "__main__":
    keys = [Path(f).stem for f in sorted(glob.glob(str(REPO / "data/processed/awake_grounding/*.pkl")))]
    with ProcessPoolExecutor(2) as ex:
        res = [d for d in ex.map(job, keys) if d is not None]
    pd.concat(res, ignore_index=True).to_pickle(REPO / "data/processed/relationships_iso/transfer_states.pkl")
    print("done", len(res))
