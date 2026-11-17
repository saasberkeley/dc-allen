"""S-A runner: locality of tracker-signal <-> neuron coupling, every session and state.
Usage (from the repo root): python code/python/src/run_locality.py ['multichannel'|'multichannel_local']"""
import glob, sys, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code/python"))
from src import awake_grounding as ag, motif_value as mv, locality as lc

def job(args):
    session, folder = args
    warnings.filterwarnings("ignore")
    R = [ag.load(f) for f in glob.glob(str(REPO / "data/processed/awake_grounding/*.pkl"))]
    ref = mv.consensus_templates([u for r in R if r["meta"]["session"][:6] != session[:6] for u in r["templates"]["original"]])[0][0]
    out_t, out_d = [], []
    for st in (0, 1):
        S = lc.run_session(session, ref, st, folder)
        if S is None:
            continue
        name = "awake" if st == 0 else "isoflurane"
        out_t.append(lc.coupling_table(S, np.random.default_rng(0)).assign(session=session, state=name, n_seconds=len(S["sec"])))
        out_d.append(lc.lfp_decay(S).assign(session=session, state=name))
    return out_t, out_d

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "multichannel"
    sessions = sorted({Path(f).stem[:15] for f in glob.glob(str(REPO / f"data/processed/{folder}/*_probe*.npz"))})
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(job, [(s, folder) for s in sessions]))
    T = pd.concat([t for a, _ in res for t in a], ignore_index=True)
    D = pd.concat([d for _, b in res for d in b], ignore_index=True)
    T.to_pickle(REPO / f"data/processed/spatial/locality_{folder}.pkl"); D.to_pickle(REPO / f"data/processed/spatial/lfp_decay_{folder}.pkl")
    print("done", len(sessions), "sessions")
