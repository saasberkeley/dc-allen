"""Phase 1 (awake reference): unit QC, firing descriptors and three ensemble definitions for every awake grounding
signal. Usage (from the repo root): python code/python/src/run_ensembles.py"""
import glob, sys, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code/python"))
from src import awake_grounding as ag, ensembles as en
OUT = REPO / "data/processed/ensembles_awake"

def job(f):
    warnings.filterwarnings("ignore")
    r = ag.load(f)
    path = glob.glob(str(REPO / f"data/raw/{r['meta']['session']}*.nwb"))[0]
    o = en.run_signal(path, r)
    ag.save(o, OUT / Path(f).name)
    return Path(f).stem

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(REPO / "data/processed/awake_grounding/*.pkl")))
    with ProcessPoolExecutor(6) as ex:
        for name in ex.map(job, files):
            print(name, flush=True)
