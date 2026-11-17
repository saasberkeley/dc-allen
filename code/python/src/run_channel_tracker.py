"""D2 runner: the tracker on the extracted channels, at two densities.
  coarse (every COARSE-th extracted channel, ~160 um): each channel's own awake motif library and presence;
  dense (every extracted channel, ~40 um): z-scored envelopes (awake reference), sliding baseline and residual,
    projected on the probe's reference motifs (consensus of the probe's coarse awake libraries; if the probe has
    none, the consensus of the other mice's libraries of the same kind) and on band PC1 of the probe's awake data.
Usage (from the repo root): python code/python/src/run_channel_tracker.py [workers] [local]
  local: libraries only (no presence, no dense pass) on the locally re-referenced channels (CSD on probes, common
  average on EEG; data/processed/multichannel_local) -> data/processed/channel_tracker_local/_coarse
Resumable: finished arrays are skipped on a rerun."""
import glob, sys, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code/python"))
from src import awake_grounding as ag, channel_tracker as ct
IN, OUT = REPO / "data/processed/multichannel", REPO / "data/processed/channel_tracker"
COARSE = 4
LOCAL = len(sys.argv) > 2 and sys.argv[2] == "local"
if LOCAL:
    IN, OUT = REPO / "data/processed/multichannel_local", REPO / "data/processed/channel_tracker_local"

def coarse(f):
    """Each array's coarse result is saved as soon as it is done, so a restart skips finished arrays."""
    warnings.filterwarnings("ignore")
    done = OUT / "_coarse" / f"{Path(f).stem}.pkl"
    if done.exists():
        return Path(f).stem, ag.load(done)
    d = np.load(f, allow_pickle=True)
    t, state, piece, env = d["t"], d["state"], d["piece"], d["env"].astype(float)
    step = 1 if Path(f).stem.endswith("EEG") else COARSE
    res = {}
    for j in range(0, env.shape[1], step):
        Z, base, resid = ct.signal_of(t, env[:, j], piece, state == 0)
        ok = np.isfinite(resid).all(1)
        aw = ok & (state == 0)
        U = ct.awake_library(Z[aw], base[aw], resid[aw], piece[aw])
        P = np.full((len(t), len(U)), np.nan)
        for st in (0, 1):
            m = ok & (state == st)
            if m.sum() and len(U) and not LOCAL:
                P[m] = ct.presence(U, Z[m], base[m], resid[m], piece[m])
        res[j] = {"U": U.astype(np.float32), "P": P.astype(np.float32)}
    ag.save(res, done)
    return Path(f).stem, res

def dense(f, refs):
    warnings.filterwarnings("ignore")
    d = np.load(f, allow_pickle=True)
    t, state, piece, env = d["t"], d["state"], d["piece"], d["env"].astype(float)
    U = np.array(refs) if len(refs) else np.zeros((0, env.shape[2]))
    chans = []
    pc1 = None
    for j in range(env.shape[1]):
        Z, base, resid = ct.signal_of(t, env[:, j], piece, state == 0)
        ok = np.isfinite(resid).all(1)
        if pc1 is None:
            aw = ok & (state == 0)
            pc1 = np.linalg.eigh(np.cov(Z[aw].T))[1][:, -1]
        D = np.vstack([U, pc1]) if len(U) else pc1[None]
        E = np.where(ok[:, None], resid, 0.0) @ D.T
        chans.append({"ok": ok, "E": E.astype(np.float32), "B": (base @ D.T).astype(np.float32),
                      "S": ct.rolling_spread(E, piece).astype(np.float32), "Uproj": (Z @ D.T).astype(np.float32),
                      "base_vec": base.astype(np.float32)})
    return {"t": t, "state": state, "piece": piece, "directions": D.astype(np.float32),
            "direction_names": [f"ref M{i + 1}" for i in range(len(U))] + ["band PC1"], "channels": chans,
            **{k: d[k] for k in ("channel", "electrode_id", "depth_um", "region", "ccf") if k in d}}

def finish(args):
    f, refs, coarse_res, src = args
    if (OUT / f"{Path(f).stem}.pkl").exists():
        return Path(f).stem + " (already done)"
    out = dense(f, refs)
    out["coarse"] = coarse_res
    out["reference_source"] = src
    ag.save(out, OUT / f"{Path(f).stem}.pkl")
    return Path(f).stem

if __name__ == "__main__":
    (OUT / "_coarse").mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(IN / "*.npz")))
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    with ProcessPoolExecutor(W) as ex:
        C = dict(ex.map(coarse, files))
    ag.save(C, OUT / "_coarse_libraries.pkl"); print("coarse libraries done", flush=True)
    if LOCAL:
        sys.exit(0)
    jobs = []
    for f in files:
        k = Path(f).stem
        own = ct.consensus_for([v["U"] for v in C[k].values()])
        if own:
            refs, src = own, "consensus of this probe's awake channel libraries"
        else:
            others = [v["U"] for kk, cc in C.items() if kk[:6] != k[:6] and kk.endswith("EEG") == k.endswith("EEG") for v in cc.values()]
            refs, src = ct.consensus_for(others), "consensus of other mice (no awake library on this array)"
        jobs.append((f, refs, C[k], src))
    with ProcessPoolExecutor(W) as ex:
        for k in ex.map(finish, jobs):
            print(k, flush=True)
