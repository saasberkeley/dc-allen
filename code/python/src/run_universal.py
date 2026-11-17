"""Runner for U1-U5 (universal.py). Spatial mouse split (mouse_split.json, key "spatial"): every decision (number of
motifs, motif identity, sign of each relationship) is made on the discovery mice; the confirmation mice test it,
one-sided in the discovery direction.
Usage (from the repo root): python code/python/src/run_universal.py [workers]
Output: data/processed/universal/*.pkl"""
import glob, json, sys, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code/python"))
from src import awake_grounding as ag, universal as uv
OUT = REPO / "data/processed/universal"
SPLIT = json.load(open(REPO / "data/processed/mouse_split.json"))["spatial"]
DISC, CONF = SPLIT["discovery"], SPLIT["confirmation"]
KMAX = 5


def channels(workers):
    f = OUT / "channels.pkl"
    if f.exists():
        return ag.load(f)
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for rs in ex.map(uv.array_channels, sorted(glob.glob(str(uv.MC / "*.npz")))) for r in rs]
    Ch = pd.DataFrame(rows)
    ag.save(Ch, f)
    return Ch


def fit(Ch, mice, k, seed):
    T, w, _ = uv.template_pool(Ch, mice)
    return uv.fit_lines(T, w, k, np.random.default_rng(seed))


def align(D, ref):
    """Reorder / re-sign D to match ref (Hungarian on |cos|)."""
    cos, mp = uv.match(ref, D)
    out = np.full_like(ref, np.nan)
    for i, j in mp.items():
        out[i] = D[j] * np.sign(D[j] @ ref[i])
    return out, cos


def u1(Ch, rng, ell):
    rows, folds = [], {}
    for k in range(1, KMAX + 1):
        Dall, share = fit(Ch, DISC, k, 1)
        folds[k] = {"all": Dall, "share": share}
        prev_all = folds[k - 1]["all"] if k > 1 else None
        for m in DISC:
            train = [x for x in DISC if x != m]
            D, _ = fit(Ch, train, k, 2)
            Da, cos = align(D, Dall)
            folds[k][m] = Da
            # gain of the k-th motif over a (k-1) fit of the same training mice, vs a random smooth k-th direction
            Dp = fit(Ch, train, k - 1, 2)[0] if k > 1 else np.zeros((0, 10))
            R = [np.vstack([Dp, uv.smooth_directions(1, ell, rng)]) for _ in range(50)]
            for _, c in Ch[Ch.mouse == m].iterrows():
                C = c.C_awake
                cap = uv.captured(D, C)[0]
                base = uv.captured(Dp, C)[0] if k > 1 else 0.0
                rand = float(np.mean([uv.captured(r, C)[0] for r in R]))
                rows.append({"k": k, "mouse": m, "kind": "EEG" if c.is_eeg else "LFP", "captured": cap,
                             "gain": cap - base, "random_gain": rand - base,
                             "own_topk": uv.captured(D, C)[1], "fold_cos_kth": cos[k - 1]})
    G = pd.DataFrame(rows)
    per_mouse = G.groupby(["k", "mouse"]).agg(gain=("gain", "median"), random_gain=("random_gain", "median"),
                                              captured=("captured", "median"), own=("own_topk", "median"),
                                              fold_cos_kth=("fold_cos_kth", "first")).reset_index()
    chosen = 0
    for k in range(1, KMAX + 1):
        P = per_mouse[per_mouse.k == k]
        if ((P.gain > P.random_gain).sum() >= 3) and ((P.fold_cos_kth >= 0.9).sum() >= 3):
            chosen = k
        else:
            break
    return G, per_mouse, folds, max(chosen, 1), chosen


def u1_confirm(Ch, D, k, rng, ell):
    Dc, share_c = fit(Ch, CONF, k, 3)
    Dc_al, cos = align(Dc, D)
    null = []
    for _ in range(1000):
        null.append(np.sort(uv.match(D, uv.smooth_directions(k, ell, rng))[0]))
    null = np.array(null)
    p = [float((1 + (null.max(1) >= c).sum()) / (len(null) + 1)) for c in cos]   # vs best random match
    H = uv.heldout_reconstruction(Ch, DISC, CONF, D, rng, ell)
    return {"D_conf": Dc_al, "share_conf": share_c, "match_cos": cos, "match_p": p, "reconstruction": H}


def where_motifs_live(Ch, D, mice):
    T, w, meta = uv.template_pool(Ch, mice)
    c = np.abs(T @ D.T)
    meta = meta.assign(motif=c.argmax(1), cos=c.max(1), w=w)
    return meta


def u2(Ch, rng, ell):
    P, src, tgt = uv.transfer_pairs(Ch)
    own = uv.own_scores(tgt, rng, ell)
    return P, src[["session", "mouse", "array", "j", "class", "area", "is_eeg"]], tgt[["session", "mouse", "array", "j", "class", "area", "is_eeg"]], own


def u3_job(args):
    session, D, seed = args
    warnings.filterwarnings("ignore")
    return uv.grounding_session(session, D, np.random.default_rng(seed)), uv.eeg_beneath_session(session, D, np.random.default_rng(seed + 1))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    rng = np.random.default_rng(20261117)
    Ch = channels(W)
    print("channels", len(Ch), "with library", int(Ch.has_library.sum()), flush=True)
    T, _, _ = uv.template_pool(Ch, DISC)
    ell = uv.fit_ell(T, rng)
    print("null length scale", ell, flush=True)
    G, per_mouse, folds, k, chosen = u1(Ch, rng, ell)
    D = folds[k]["all"]
    print("k chosen", chosen, "(used", k, ")", flush=True)
    conf = u1_confirm(Ch, D, k, rng, ell)
    Dall, share_all = fit(Ch, DISC + CONF, k, 4)
    ag.save({"G": G, "per_mouse": per_mouse, "folds": folds, "k": k, "chosen": chosen, "D": D, "ell": ell,
             "confirm": conf, "D_allmice": align(Dall, D)[0], "share_allmice": share_all,
             "where": where_motifs_live(Ch, D, DISC + CONF),
             "heldout_disc": pd.concat([uv.heldout_reconstruction(Ch, [x for x in DISC if x != m], [m], folds[k][m], rng, ell)
                                        for m in DISC])}, OUT / "u1.pkl")
    print("U1 done", flush=True)
    P, src, tgt, own = u2(Ch, rng, ell)
    ag.save({"pairs": P, "src": src, "tgt": tgt, "own": own}, OUT / "u2.pkl")
    print("U2 done", len(P), flush=True)
    # U4 (a, b): awake dictionary on isoflurane; isoflurane principal directions vs awake principal directions
    iso_rec = pd.concat([uv.heldout_reconstruction(Ch, [x for x in DISC if x != m], [m], folds[k][m], rng, ell, cov=cv)
                         .assign(mouse=m, state=cv[2:]) for m in DISC for cv in ("C_awake", "C_iso")] +
                        [uv.heldout_reconstruction(Ch, DISC, CONF, D, rng, ell, cov=cv).assign(state=cv[2:])
                         for cv in ("C_awake", "C_iso")])
    pdirs = {}
    for cv in ("C_awake", "C_iso"):
        for grp, mice in (("discovery", DISC), ("confirmation", CONF)):
            S = Ch[Ch.mouse.isin(mice) & Ch[cv].map(lambda c: c is not None)]
            V = np.array([np.linalg.eigh(c)[1][:, -1] for c in S[cv]])
            w = 1.0 / S.groupby("mouse").mouse.transform("size").to_numpy()
            pdirs[(cv[2:], grp)] = align(uv.fit_lines(V, w, k, np.random.default_rng(5))[0], D)
    ag.save({"reconstruction": iso_rec, "principal_dictionaries": pdirs}, OUT / "u4.pkl")
    print("U4 a/b done", flush=True)
    # U3 / U5: discovery mice use their leave-one-mouse-out dictionary, confirmation mice the discovery dictionary
    sessions = sorted({Path(f).stem.split("_")[0] for f in glob.glob(str(uv.MC / "*.npz"))})
    sessions = [s for s in sessions if s[:6] in DISC + CONF and glob.glob(str(uv.MC / f"{s}_probe*.npz"))]
    jobs = [(s, folds[k][s[:6]] if s[:6] in DISC else D, int(s.replace("-", "")) % 2 ** 31) for s in sessions]
    with ProcessPoolExecutor(W) as ex:
        res = list(ex.map(u3_job, jobs))
    ag.save({"grounding": [g for r in res for g in r[0]], "eeg": [e for r in res for e in r[1]]}, OUT / "u3_u5.pkl")
    print("U3 / U5 done", flush=True)
