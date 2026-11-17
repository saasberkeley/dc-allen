"""Phase 2: run the relationship catalog on one group of mice (discovery or confirmation, from
data/processed/mouse_split.json). Usage (from the repo root): python code/python/src/run_relationships.py discovery"""
import glob, json, sys, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code/python"))
from src import awake_grounding as ag, relationships as rl

def job(k):
    warnings.filterwarnings("ignore")
    r = ag.load(REPO / f"data/processed/awake_grounding/{k}.pkl")
    e = ag.load(REPO / f"data/processed/ensembles_awake/{k}.pkl")
    d = rl.assemble(glob.glob(str(REPO / f"data/raw/{r['meta']['session']}*.nwb"))[0], r, e)
    if d is None:
        return k, None
    o = rl.run_signal(d)
    return k, o

if __name__ == "__main__":
    group = sys.argv[1]
    mice = json.load(open(REPO / "data/processed/mouse_split.json"))[group]
    keys = [Path(f).stem for f in sorted(glob.glob(str(REPO / "data/processed/awake_grounding/*.pkl"))) if Path(f).stem[:6] in mice]
    out = {}
    with ProcessPoolExecutor(6) as ex:
        for k, o in ex.map(job, keys):
            if o is not None:
                out[k] = o
    ag.save(out, REPO / f"data/processed/relationships_awake/{group}.pkl")
    print(group, len(keys), "signals,", len(out), "with a tracker motif")
