import sys, glob, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code/python"))
from src import awake_grounding as ag, motif_value as mv, session_data as sd
OUT = REPO / "data/processed/motif_value"
FILES = sorted(glob.glob(str(REPO / "data/processed/awake_grounding/*.pkl")))

def env_job(f):
    warnings.filterwarnings("ignore")
    r = ag.load(f); m = r["meta"]
    path = glob.glob(str(REPO / f"data/raw/{m['session']}*.nwb"))[0]
    io, nwb = sd.open_nwb(path); ch = mv.eeg_channels(nwb, m["region"]); io.close()
    return Path(f).stem, ch, {w: mv.eeg_envelope(path, ch[w][0], r["t"]) for w in ("near", "far")}

if __name__ == "__main__":
    with ProcessPoolExecutor(5) as ex:
        envs = {k: (ch, e) for k, ch, e in ex.map(env_job, FILES)}
    pd.to_pickle(envs, OUT / "eeg_envelopes.pkl"); print("envelopes done", flush=True)
    R = {Path(f).stem: ag.load(f) for f in FILES}
    mouse = lambda k: k[:6]
    rng = np.random.default_rng(3)
    rows = []
    for k, r in R.items():
        others = [kk for kk in R if mouse(kk) != mouse(k)]
        cons = [c for c, _ in mv.consensus_templates([u for kk in others for u in R[kk]["templates"]["original"]])]
        # EEG regression: near-electrode EEG bands -> population rate, pooled over the other mice
        X = np.concatenate([envs[kk][1]["near"] for kk in others]); y = np.concatenate([mv._pop(R[kk]["rate"]) for kk in others])
        ok = np.isfinite(X).all(1)
        w = np.linalg.solve(X[ok].T @ X[ok] + 10 * np.eye(X.shape[1]), X[ok].T @ y[ok])
        ch, e = envs[k]
        d = mv.eeg_test(r, e, cons, w, rng)
        d["eeg_channel"] = d.electrode.map(lambda wh: ch[wh][0]); d["eeg_location"] = d.electrode.map(lambda wh: ch[wh][1])
        d["distance_mm"] = d.electrode.map(lambda wh: ch[wh][2])
        rows.append(d); print(k, flush=True)
    pd.concat(rows, ignore_index=True).to_pickle(OUT / "eeg.pkl"); print("done")
