"""E1 runner: R5 control on every signal with a tracker motif, awake and isoflurane.
Usage (from the repo root): python code/python/src/run_r5_control.py"""
import glob, sys, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code/python"))
from src import awake_grounding as ag, relationships as rl, iso_state as iso, r5_control as rc

def job(k):
    warnings.filterwarnings("ignore")
    r = ag.load(REPO / f"data/processed/awake_grounding/{k}.pkl")
    e = ag.load(REPO / f"data/processed/ensembles_awake/{k}.pkl")
    path = glob.glob(str(REPO / f"data/raw/{r['meta']['session']}*.nwb"))[0]
    da = rl.assemble(path, r, e)
    if da is None:
        return []
    out = [rc.run(da, da["Z"]).assign(state="awake")]
    di = iso.assemble_iso(path, r, e)
    if di is not None:
        out.append(rc.run(di, da["Z"]).assign(state="isoflurane"))
    return out

if __name__ == "__main__":
    keys = [Path(f).stem for f in sorted(glob.glob(str(REPO / "data/processed/awake_grounding/*.pkl")))]
    with ProcessPoolExecutor(3) as ex:
        res = [d for out in ex.map(job, keys) for d in out]
    pd.concat(res, ignore_index=True).to_pickle(REPO / "data/processed/relationships_iso/r5_control.pkl")
    print("done", len(res))
