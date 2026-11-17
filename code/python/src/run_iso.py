"""Isoflurane step: the full relationship catalog and the per-neuron awake vs isoflurane comparison, on the awake
references. Usage (from the repo root): python code/python/src/run_iso.py discovery|confirmation"""
import glob, json, sys, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code/python"))
from src import awake_grounding as ag, relationships as rl, iso_state as iso

def job(k):
    warnings.filterwarnings("ignore")
    r = ag.load(REPO / f"data/processed/awake_grounding/{k}.pkl")
    e = ag.load(REPO / f"data/processed/ensembles_awake/{k}.pkl")
    d = iso.assemble_iso(glob.glob(str(REPO / f"data/raw/{r['meta']['session']}*.nwb"))[0], r, e)
    if d is None:
        return k, None
    o = rl.run_signal(d)
    o["cross"] = iso.cross_state(r, e, d, np.random.default_rng(0))
    o["iso_meta"] = d["meta"]
    o["baseline_shift_vec"] = d["iso_baseline_vec_mean"] - d["awake_baseline_vec_mean"]
    return k, o

if __name__ == "__main__":
    group = sys.argv[1]
    mice = json.load(open(REPO / "data/processed/mouse_split.json"))[group]
    keys = [Path(f).stem for f in sorted(glob.glob(str(REPO / "data/processed/awake_grounding/*.pkl"))) if Path(f).stem[:6] in mice]
    out = {}
    with ProcessPoolExecutor(5) as ex:
        for k, o in ex.map(job, keys):
            if o is not None:
                out[k] = o
    ag.save(out, REPO / f"data/processed/relationships_iso/{group}.pkl")
    print(group, len(keys), "signals,", len(out), "with isoflurane + an awake tracker motif")
