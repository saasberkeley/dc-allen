"""Run the awake grounding analysis on every signal of every session given; one pickle per signal."""
import sys, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code/python"))
from src import awake_grounding as ag, session_data as sd
OUT = REPO / "data/processed/awake_grounding"

def job(path, probe, region):
    warnings.filterwarnings("ignore")
    f = OUT / f"{Path(path).name[:15]}_{region.replace('/', '-')}.pkl"
    if f.exists():
        return f"skip {f.name}"
    t0 = time.time()
    res = ag.run_signal(path, probe, region, n_shift=200)
    ag.save(res, f)
    m = res["meta"]
    return f"{f.name}: {m['n_units']} units, {m['n_seconds']} s, motifs o/r/t {m['n_motifs_original']}/{m['n_motifs_relaxed']}/{m['n_motifs_tracker']} ({time.time()-t0:.0f}s)"

if __name__ == "__main__":
    jobs = []
    for p in sys.argv[1:]:
        io, nwb = sd.open_nwb(p)
        if "LFP" in nwb.acquisition and nwb.units is not None:
            jobs += [(p, pr, r) for pr, r, _ in ag.signals_in(nwb)]
        io.close()
    print(len(jobs), "signals", flush=True)
    with ProcessPoolExecutor(4) as ex:
        for f in as_completed([ex.submit(job, *j) for j in jobs]):
            try: print(f.result(), flush=True)
            except Exception as e: print("ERROR", repr(e), flush=True)
