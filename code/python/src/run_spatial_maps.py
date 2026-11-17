"""S-D runner: per-channel state changes (awake -> isoflurane) on the re-referenced data.
Usage (from the repo root): python code/python/src/run_spatial_maps.py"""
import glob, sys, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code/python"))
from src import awake_grounding as ag, motif_value as mv, spatial_maps as sm

def job(f):
    warnings.filterwarnings("ignore")
    s = Path(f).stem[:15]
    R = [ag.load(x) for x in glob.glob(str(REPO / "data/processed/awake_grounding/*.pkl"))]
    ref = mv.consensus_templates([u for r in R if r["meta"]["session"][:6] != s[:6] for u in r["templates"]["original"]])[0][0]
    return sm.channel_changes(f, ref).assign(session=s)

if __name__ == "__main__":
    files = sorted(glob.glob(str(REPO / "data/processed/multichannel_local/*.npz")))
    with ProcessPoolExecutor(2) as ex:
        C = pd.concat(list(ex.map(job, files)), ignore_index=True)
    C.to_pickle(REPO / "data/processed/spatial/channel_changes.pkl"); print("done", len(C), "channels")
