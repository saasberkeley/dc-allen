import sys, glob, warnings, pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code/python"))
from src import awake_grounding as ag, motif_value as mv
OUT = REPO / "data/processed/motif_value"; OUT.mkdir(parents=True, exist_ok=True)
FILES = sorted(glob.glob(str(REPO / "data/processed/awake_grounding/*.pkl")))

def one(f):
    warnings.filterwarnings("ignore")
    r = ag.load(f)
    return mv.directions_test(r, np.random.default_rng(abs(hash(Path(f).stem)) % 2**32))

if __name__ == "__main__":
    with ProcessPoolExecutor(6) as ex:
        D = pd.concat(list(ex.map(one, FILES)), ignore_index=True)
    D.to_pickle(OUT / "directions.pkl"); print("directions", D.shape, flush=True)
    R = [ag.load(f) for f in FILES]
    T, Cn = mv.transfer_test(R, np.random.default_rng(1))
    T.to_pickle(OUT / "transfer.pkl"); Cn.to_pickle(OUT / "consensus.pkl"); print("transfer", T.shape)
