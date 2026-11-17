"""Step 0: multichannel envelopes for every probe and the EEG of every session.
Usage (from the repo root): python code/python/src/run_multichannel.py [workers] ['as recorded'|local]
  local = CSD along each probe, common average reference for the EEG (data/processed/multichannel_local)."""
import glob, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code/python"))
from src import multichannel as mc, session_data as sd
MODE = sys.argv[2] if len(sys.argv) > 2 else "as recorded"
OUT = REPO / ("data/processed/multichannel" if MODE == "as recorded" else "data/processed/multichannel_local")

def job(path, which):
    warnings.filterwarnings("ignore")
    f = OUT / f"{Path(path).name[:15]}_{which}.npz"
    if f.exists():
        return f"skip {f.name}"
    t0 = time.time()
    mc.extract(path, which, OUT, MODE)
    return f"{f.name} ({time.time() - t0:.0f} s)"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for p in sorted(glob.glob(str(REPO / "data/raw/*.nwb"))):
        io, nwb = sd.open_nwb(p)
        jobs.append((p, "EEG"))
        if "LFP" in nwb.acquisition:
            jobs += [(p, k.replace("ElectricalSeries", "")) for k in nwb.acquisition["LFP"].electrical_series]
        io.close()
    print(len(jobs), "jobs", flush=True)
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as ex:
        for f in as_completed([ex.submit(job, *j) for j in jobs]):
            try: print(f.result(), flush=True)
            except Exception as e: print("ERROR", repr(e), flush=True)
