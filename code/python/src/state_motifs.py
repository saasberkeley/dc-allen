"""Awake vs isoflurane vs full-recording analysis of one awake -> isoflurane session.

For each signal (frontal EEG, somatosensory EEG, deep prefrontal LFP) this runs three
versions of Mishler motif detection on stimulation-free data, power spectra, PL5 spike
rasters, and cross-state statistics, and writes figures plus results.json to --out.

Versions:
  original  one "session" per window (all its stimulation-free stretches joined), 20 nulls,
            getMotifs.m rules.
  strict    the window cut into ~5-min chunks, each chunk a "session"; getMotifs.m rules.
  relaxed   same chunks; keep candidates with permutation p <= 0.05 (100 nulls), sizes compared
            as a fraction of non-trivial timepoints, present in at least half the chunks. The
            PCA support rule is kept (see report).
Every motif is also flagged "broadband-like" if 9 or more of its bands share one sign: the
shuffle null of every version rejects broadband co-fluctuation, not only multi-band motifs.

Usage (from code/python/):
    python -m src.state_motifs --nwb ../../data/raw/720762-20240221-v001.nwb \
        --out ../../data/out/2026-11-01_mouse720762
"""

import argparse
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402

from . import session_data as sd  # noqa: E402
from .get_motifs import get_motifs  # noqa: E402
from .smooth_env import mround, smooth_env  # noqa: E402

BANDS = np.array([[1, 4], [3, 6], [5, 9], [8, 12], [10, 15], [13, 20],
                  [18, 30], [25, 40], [35, 55], [50, 80]], dtype=float)
SIGNALS = [
    {"key": "eeg13", "label": "EEG ch13 (MOs, frontal)", "source": ("eeg", 13)},
    {"key": "eeg10", "label": "EEG ch10 (SSp-ll, somatosensory)", "source": ("eeg", 10)},
    {"key": "lfp_PL5", "label": "LFP PL5 (probeF, deep prelimbic)", "source": ("lfp", "probeF", "PL5")},
]
WINDOWS = ("awake", "isoflurane", "full")
VERSIONS = {
    "original": {"chunked": False, "kw": {"n_null": 20}},
    "strict": {"chunked": True, "kw": {"n_null": 20}},
    "relaxed": {"chunked": True, "kw": {"n_null": 100, "null_mode": "pvalue", "alpha": 0.05,
                                        "min_session_frac": 0.5, "normalize_size": True}},
}
FS = 250.0
CHUNK_SEC = 300.0
SEED = 0
COLORS = {"awake": "#1f77b4", "isoflurane": "#d62728", "full": "#444444", "induction": "#ff7f0e"}


@dataclass
class Piece:
    t: np.ndarray
    env: np.ndarray  # (time, bands)


def band_labels():
    return [f"{lo:g}-{hi:g}" for lo, hi in BANDS]


def envelope_pieces(t, x, fs, segments):
    pieces = []
    for a, b in segments:
        k = (t >= a) & (t < b)
        if k.sum() < 60 * fs:
            continue
        env = smooth_env(x[k], fs, BANDS, zscore=False)
        ok = np.isfinite(env).all(axis=1)
        pieces.append(Piece(t[k][ok], env[ok]))
    return pieces


def restrict(pieces, window, fs, min_sec=60.0):
    a, b = window
    out = []
    for p in pieces:
        k = (p.t >= a) & (p.t < b)
        if k.sum() >= min_sec * fs:
            out.append(Piece(p.t[k], p.env[k]))
    return out


def zscore(pieces):
    env = np.concatenate([p.env for p in pieces])
    mu, sd_ = env.mean(axis=0), env.std(axis=0, ddof=1)
    return [Piece(p.t, (p.env - mu) / sd_) for p in pieces]


def chunk(pieces, fs, chunk_sec=CHUNK_SEC):
    out = []
    for p in pieces:
        n = int(p.t.size / fs // chunk_sec)
        if n == 0 and p.t.size >= chunk_sec * fs / 2:
            n = 1
        for idx in (np.array_split(np.arange(p.t.size), n) if n else []):
            out.append(Piece(p.t[idx], p.env[idx]))
    return out


def joined(pieces):
    return [Piece(np.concatenate([p.t for p in pieces]), np.concatenate([p.env for p in pieces]))]


def one_hz(sessions, fs):
    hop = mround(fs)
    t = np.concatenate([s.t[::hop] for s in sessions])
    signs = np.sign(np.concatenate([s.env[::hop] for s in sessions]))
    which = np.concatenate([np.full(s.t[::hop].size, i) for i, s in enumerate(sessions)])
    return t, signs, which


def assign_poles(signs, motifs):
    pole = np.full(signs.shape[0], -1)
    nontrivial = np.abs(signs.sum(axis=1)) != signs.shape[1]
    if motifs.shape[1]:
        pole[nontrivial] = np.argmax(signs[nontrivial] @ motifs, axis=1)
    return pole


def run_motifs(sessions, version, fs, seed=SEED):
    t0 = time.time()
    res = get_motifs([s.env.T for s in sessions], fs, seed=seed, **VERSIONS[version]["kw"])
    return res, time.time() - t0


def band_shuffle(sessions, rng):
    out = []
    for s in sessions:
        env = s.env.copy()
        for f in range(env.shape[1]):
            env[:, f] = np.roll(env[:, f], rng.integers(env.shape[0]))
        out.append(Piece(s.t, env))
    return out


def abs_corr_matrix(a, b):
    """|corr| between the + poles (columns) of two motif sets."""
    if a.shape[1] == 0 or b.shape[1] == 0:
        return np.zeros((a.shape[1], b.shape[1]))
    return np.abs(np.corrcoef(a.T, b.T)[: a.shape[1], a.shape[1]:])


def plus_poles(motifs):
    return motifs[:, : motifs.shape[1] // 2]


# ---------------------------------------------------------------- statistics

def chunk_state(chunk_t, info):
    labels = set(sd.state_of_time(chunk_t, info))
    return labels.pop() if len(labels) == 1 and labels != {"induction"} else None


def permutation_test(full_chunks, motifs, info, fs, max_perm=20000, seed=SEED):
    """Do awake and isoflurane chunks differ in a motif's pole balance and in its shape?

    Unit of analysis = 5-min chunk. Balance = (f+ - f-) / (f+ + f-); shape = chunk's own
    re-estimate of the motif (Mishler's pooled +/- template). Exact permutation of the
    awake/isoflurane labels over chunks.
    """
    keep = [(c, s) for c in full_chunks if (s := chunk_state(c.t, info)) is not None]
    labels = np.array([s == "awake" for _, s in keep])
    n_motifs = motifs.shape[1] // 2
    hop = mround(fs)
    balance = np.full((len(keep), n_motifs), np.nan)
    shape = np.full((len(keep), n_motifs, motifs.shape[0]), np.nan)
    for c, (piece, _) in enumerate(keep):
        signs = np.sign(piece.env[::hop])
        pole = assign_poles(signs, motifs)
        for i in range(n_motifs):
            plus, minus = pole == i, pole == i + n_motifs
            if plus.sum() + minus.sum():
                balance[c, i] = (plus.mean() - minus.mean()) / (plus.mean() + minus.mean())
                shape[c, i] = np.vstack([signs[plus], -signs[minus]]).mean(axis=0)

    n, k = labels.size, int(labels.sum())
    if math.comb(n, k) <= max_perm:
        perms = [np.isin(np.arange(n), idx) for idx in itertools.combinations(range(n), k)]
    else:
        rng = np.random.default_rng(seed)
        perms = [np.isin(np.arange(n), rng.choice(n, k, replace=False)) for _ in range(max_perm)]
    perms = np.array(perms)

    out = []
    for i in range(n_motifs):
        b = balance[:, i]
        d_obs = np.nanmean(b[labels]) - np.nanmean(b[~labels])
        d_perm = np.array([np.nanmean(b[p]) - np.nanmean(b[~p]) for p in perms])
        dist = 1 - np.corrcoef(np.nan_to_num(shape[:, i]))
        same = labels[:, None] == labels[None, :]
        off = ~np.eye(n, dtype=bool)

        def gap(lab):
            s = lab[:, None] == lab[None, :]
            return dist[~s].mean() - dist[s & off].mean()

        g_obs = gap(labels)
        g_perm = np.array([gap(p) for p in perms])
        out.append({
            "motif": i + 1,
            "n_awake_chunks": k, "n_iso_chunks": n - k, "n_permutations": len(perms),
            "balance_awake": b[labels].tolist(), "balance_iso": b[~labels].tolist(),
            "balance_diff": float(d_obs), "balance_p": float(np.mean(np.abs(d_perm) >= abs(d_obs) - 1e-12)),
            "shape_within": float(dist[same & off].mean()), "shape_between": float(dist[~same].mean()),
            "shape_gap": float(g_obs), "shape_p": float(np.mean(g_perm >= g_obs - 1e-12)),
        })
    return out


def pole_by_second(sessions, motifs, fs):
    t, signs, _ = one_hz(sessions, fs)
    pole = assign_poles(signs, motifs)
    sec = np.round(t).astype(int)
    _, first = np.unique(sec, return_index=True)
    return dict(zip(sec[first].tolist(), pole[first].tolist()))


def signed(pole, i, n):
    return np.where(pole == i, 1.0, np.where(pole == i + n, -1.0, 0.0))


def coupling(a_poles, b_poles, n_a, n_b, info):
    """Correlation of pole time courses of two signals, overall and within each state."""
    secs = np.array(sorted(set(a_poles) & set(b_poles)))
    pa = np.array([a_poles[s] for s in secs])
    pb = np.array([b_poles[s] for s in secs])
    state = sd.state_of_time(secs.astype(float), info)
    out = {}
    for name, mask in [("full", np.ones(secs.size, bool)), ("awake", state == "awake"),
                       ("isoflurane", state == "isoflurane")]:
        r = np.full((n_a, n_b), np.nan)
        for i in range(n_a):
            for j in range(n_b):
                x, y = signed(pa[mask], i, n_a), signed(pb[mask], j, n_b)
                if x.std() > 0 and y.std() > 0:
                    r[i, j] = np.corrcoef(x, y)[0, 1]
        out[name] = {"r": r.tolist(), "n_seconds": int(mask.sum())}
    return out


def spikes_by_pole(spike_trains, poles, n_motifs, info, n_shift=1000, seed=SEED):
    """PL5 population rate when each pole of an LFP motif dominates, within each state.

    Null: circularly shift the pole sequence in time (>= 60 s), which keeps its
    autocorrelation but breaks its alignment with spiking.
    """
    secs = np.array(sorted(poles))
    pole = np.array([poles[s] for s in secs])
    spikes = np.sort(np.concatenate(spike_trains))
    rate = (np.searchsorted(spikes, secs + 0.5) - np.searchsorted(spikes, secs - 0.5)) / len(spike_trains)
    state = sd.state_of_time(secs.astype(float), info)
    rng = np.random.default_rng(seed)
    out = []
    for st in ("awake", "isoflurane"):
        m = state == st
        if m.sum() < 60:
            continue
        z = (rate[m] - rate[m].mean()) / rate[m].std()
        for i in range(n_motifs):
            s = signed(pole[m], i, n_motifs)
            if (s == 1).sum() < 10 or (s == -1).sum() < 10:
                continue
            diff = z[s == 1].mean() - z[s == -1].mean()
            shifts = rng.integers(60, max(61, s.size - 60), n_shift)
            null = np.array([z[np.roll(s, k) == 1].mean() - z[np.roll(s, k) == -1].mean() for k in shifts])
            out.append({"state": st, "motif": i + 1, "rate_plus_z": float(z[s == 1].mean()),
                        "rate_minus_z": float(z[s == -1].mean()), "diff": float(diff),
                        "p": float((1 + np.sum(np.abs(null) >= abs(diff))) / (n_shift + 1)),
                        "n_plus": int((s == 1).sum()), "n_minus": int((s == -1).sum()),
                        "mean_rate_hz": float(rate[m].mean())})
    return out


# ---------------------------------------------------------------- spectra & rasters

def welch_window(t, x, fs, segments, window):
    a, b = window
    total, weight = None, 0
    for s0, s1 in segments:
        lo, hi = max(a, s0), min(b, s1)
        k = (t >= lo) & (t < hi)
        if k.sum() < 8 * fs:
            continue
        f, p = signal.welch(x[k], fs, nperseg=int(4 * fs))
        w = k.sum()
        total = p * w if total is None else total + p * w
        weight += w
    return f, total / weight


def stim_average(t, x, fs, times, pre=0.2, post=1.0):
    w = np.arange(int(-pre * fs), int(post * fs))
    idx = np.searchsorted(t, times)
    idx = idx[(idx + w[0] > 0) & (idx + w[-1] < x.size)]
    seg = np.stack([x[i + w] for i in idx])
    seg -= seg[:, w < 0].mean(axis=1, keepdims=True)
    return w / fs, seg.mean(axis=0), seg[:, w < 0].std()


def shade_states(ax, info):
    windows = sd.analysis_windows(info)
    for name in ("awake", "isoflurane"):
        a, b = windows[name]
        ax.axvspan(a, b, color=COLORS[name], alpha=0.08, lw=0)
    for _, a, b in info.stim_blocks:
        ax.axvspan(a, b, facecolor="none", hatch="///", edgecolor="0.6", lw=0)


# ---------------------------------------------------------------- figures

def plot_overview(info, segments, specs, path):
    fig, axes = plt.subplots(len(specs) + 1, 1, figsize=(13, 2.2 + 2.6 * len(specs)), sharex=True,
                             gridspec_kw={"height_ratios": [1] + [2.2] * len(specs)})
    ax = axes[0]
    ax.plot(info.iso_t, info.iso_level, color="k", lw=1)
    ax.set_ylabel("Isoflurane %")
    shade_states(ax, info)
    for a, b in segments:
        ax.plot([a, b], [-0.4, -0.4], color="seagreen", lw=4, solid_capstyle="butt")
    for _, a, b in info.visual_blocks:
        ax.plot([a, b], [-0.9, -0.9], color="purple", lw=4, solid_capstyle="butt")
    ax.set_ylim(-1.3, 5.5)
    ax.set_title("Isoflurane level. Green = stimulation-free stretches used; hatched = electrical stimulation; "
                 "purple = visual stimuli; blue / red tint = awake / isoflurane windows", fontsize=8, loc="left")
    for ax, (label, (tt, f, db)) in zip(axes[1:], specs.items()):
        im = ax.pcolormesh(tt, f, db, cmap="RdBu_r", vmin=-8, vmax=8, shading="auto", rasterized=True)
        ax.set_ylabel("Hz")
        ax.set_title(f"{label}: power relative to each frequency's median (dB, 30-s averages)", fontsize=9,
                     loc="left")
        shade_states(ax, info)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    axes[-1].set_xlabel("Session time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_stim_response(curves, path):
    fig, axes = plt.subplots(1, len(curves), figsize=(5 * len(curves), 3.4))
    for ax, (label, by_state) in zip(np.atleast_1d(axes), curves.items()):
        for st, (w, mean, sd_) in by_state.items():
            ax.plot(w * 1000, mean, color=COLORS[st], label=f"{st} (pre-pulse SD {sd_:.0f} µV)")
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("Time from electrical pulse (ms)")
        ax.set_ylabel("Mean voltage (µV)")
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_spectra(spectra, path_psd, path_ratio):
    names = list(spectra)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.6), sharey=False)
    for ax, name in zip(axes, names):
        for win in WINDOWS:
            f, p = spectra[name][win]
            k = (f >= 1) & (f <= 100)
            ax.loglog(f[k], p[k], color=COLORS[win], label=win, lw=1.3 if win != "full" else 1, ls="-" if win != "full" else "--")
        for lo, hi in ((1, 4), (4, 8), (8, 12), (12, 30), (30, 80)):
            ax.axvline(lo, color="0.85", lw=0.6)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Power (µV²/Hz)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_psd, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for name in names:
        f, pa = spectra[name]["awake"]
        _, pi = spectra[name]["isoflurane"]
        k = (f >= 1) & (f <= 100)
        ax.semilogx(f[k], 10 * np.log10(pi[k] / pa[k]), label=name)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Isoflurane / awake power (dB)")
    ax.set_title("How isoflurane changes power at each frequency")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_ratio, dpi=150)
    plt.close(fig)


def plot_raster(spike_trains, n_total, info, snippets, path):
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, len(snippets), height_ratios=[3, 1.2, 1.6])
    for c, (name, (a, b)) in enumerate(snippets.items()):
        ax = fig.add_subplot(gs[0, c])
        for row, st in enumerate(spike_trains):
            s = st[(st >= a) & (st < b)]
            ax.vlines(s - a, row, row + 0.8, color=COLORS.get(name, "k"), lw=0.4)
        ax.set_title(f"{name}: {a:.0f}-{b:.0f} s", fontsize=9)
        ax.set_xlabel("Time in window (s)")
        if c == 0:
            ax.set_ylabel(f"PL5 unit (depth-sorted)\n{len(spike_trains)} of {n_total} pass quality filters")
        ax.set_ylim(0, len(spike_trains))
    edges = np.arange(info.t_start, info.iso_off + 10, 10.0)
    counts = np.array([np.histogram(st, edges)[0] for st in spike_trains]) / 10.0
    ax = fig.add_subplot(gs[1, :])
    ax.plot(edges[:-1], counts.mean(axis=0), color="k", lw=0.8)
    shade_states(ax, info)
    ax.set_ylabel("Mean rate (Hz)")
    ax.set_title("Full recording: PL5 population firing rate (10-s bins); hatched = electrical stimulation",
                 fontsize=9, loc="left")
    ax.set_xlim(edges[0], edges[-1])
    ax = fig.add_subplot(gs[2, :], sharex=ax)
    z = (counts - counts.mean(axis=1, keepdims=True)) / (counts.std(axis=1, keepdims=True) + 1e-9)
    ax.imshow(z, aspect="auto", cmap="magma", vmin=-2, vmax=2, extent=[edges[0], edges[-1], len(spike_trains), 0],
              interpolation="none")
    ax.set_ylabel("Unit")
    ax.set_xlabel("Session time (s)")
    ax.set_title("Full recording: each unit's rate (z-scored), 10-s bins", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_motif_grid(label, runs, path):
    fig, axes = plt.subplots(len(WINDOWS), len(VERSIONS), figsize=(4.2 * len(VERSIONS), 3.0 * len(WINDOWS)),
                             sharex=True, sharey=True)
    x = np.arange(len(BANDS))
    for r, win in enumerate(WINDOWS):
        for c, ver in enumerate(VERSIONS):
            ax = axes[r, c]
            run = runs[(win, ver)]
            m = np.array(run["motifs"]).reshape(len(BANDS), -1)
            n = m.shape[1] // 2
            ax.axhline(0, color="0.8", lw=0.6)
            for i in range(n):
                sup = run["motif_support"][i]
                flag = ", broadband-like" if run["broadband_like"][i] else ""
                ax.plot(x, m[:, i], "-o", ms=3, color=f"C{i}", label=f"M{i + 1} (min p {sup['min_p']:.3f}{flag})")
                ax.plot(x, m[:, i + n], "--", lw=0.8, color=f"C{i}")
            if n == 0:
                ax.text(0.5, 0.5, "no motif survived", ha="center", va="center", transform=ax.transAxes,
                        color="0.4")
            else:
                ax.legend(fontsize=6.5, loc="lower left")
            ax.set_title(f"{win} | {ver}: {n} pair(s), {run['n_sessions']} session(s)", fontsize=8.5)
            if r == len(WINDOWS) - 1:
                ax.set_xticks(x, band_labels(), rotation=60, fontsize=7)
            if c == 0:
                ax.set_ylabel("Mean sign")
    fig.suptitle(f"{label}: motifs (solid = + pole, dashed = opponent)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_similarity(sim, path):
    """sim[(signal_label, version)] = {(win_a, win_b): matrix}."""
    keys = list(sim)
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.6), squeeze=False)
    for ax, key in zip(axes[0], keys):
        mats = sim[key]
        lines = []
        for (wa, wb), m in mats.items():
            best = np.max(m, axis=1) if m.size else np.array([])
            lines.append(f"{wa} vs {wb}: " + (", ".join(f"{v:.2f}" for v in best) if best.size else "n/a"))
        ax.axis("off")
        ax.text(0, 1, f"{key[0]}\n[{key[1]}]\n\n" + "\n".join(lines), va="top", fontsize=8, family="monospace")
    fig.suptitle("Best |r| match of each motif in the first window to any motif in the second", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_permutation(perm, path):
    entries = [(sig, m) for sig, res in perm.items() for m in res["tests"]]
    if not entries:
        return
    fig, axes = plt.subplots(1, len(entries), figsize=(3.3 * len(entries), 3.8), squeeze=False)
    for ax, (sig, m) in zip(axes[0], entries):
        rng = np.random.default_rng(0)
        for xpos, key, st in ((0, "balance_awake", "awake"), (1, "balance_iso", "isoflurane")):
            v = np.array(m[key], dtype=float)
            ax.scatter(xpos + rng.uniform(-0.08, 0.08, v.size), v, color=COLORS[st], s=25, zorder=3)
            ax.plot([xpos - 0.2, xpos + 0.2], [np.nanmean(v)] * 2, color="k", lw=1.5)
        ax.axhline(0, color="0.8", lw=0.6)
        ax.set_xticks([0, 1], ["awake", "iso"])
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"{sig} M{m['motif']} [{perm[sig]['version']}]\nbalance p={m['balance_p']:.3f}, "
                     f"shape p={m['shape_p']:.3f}", fontsize=8)
    axes[0][0].set_ylabel("Pole balance per 5-min chunk\n(f+ − f−)/(f+ + f−)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_spikes_by_pole(rows, path):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(1.6 * len(rows) + 2, 3.6))
    for k, r in enumerate(rows):
        ax.bar(k - 0.18, r["rate_plus_z"], 0.36, color=COLORS[r["state"]], alpha=0.9)
        ax.bar(k + 0.18, r["rate_minus_z"], 0.36, color=COLORS[r["state"]], alpha=0.4)
        ax.text(k, max(r["rate_plus_z"], r["rate_minus_z"], 0) + 0.03, f"p={r['p']:.3f}", ha="center", fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(rows)), [f"{r['state']}\nM{r['motif']}  (+ | −)" for r in rows], fontsize=8)
    ax.set_ylabel("PL5 population rate (z within state)")
    ax.set_title("Spiking when each LFP-motif pole dominates (dark = +, light = −)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main

def minority_sign_bands(motif):
    return int(min((motif > 0).sum(), (motif < 0).sum()))


def motif_record(sig, win, ver, res, sessions, runtime, fs):
    t, signs, _ = one_hz(sessions, fs)
    pole = assign_poles(signs, res.motifs)
    n = res.n_motifs
    occ = {f"M{i + 1}{'+-'[k]}": float(np.mean(pole == i + k * n)) for i in range(n) for k in (0, 1)}
    occ["all-same-sign"] = float(np.mean(pole == -1))
    return {
        "signal": sig, "window": win, "version": ver,
        "n_sessions": len(sessions), "session_seconds": [round(float(s.t[-1] - s.t[0]), 1) for s in sessions],
        "n_timepoints": res.n_timepoints, "n_motifs": n, "motifs": res.motifs.tolist(),
        "null_max_size": res.null_max_size.tolist(), "n_kept": res.n_kept.tolist(),
        "kept_candidates": res.kept_candidates,
        "motif_support": [{"n_candidates": len(s), "min_p": min(c["p"] for c in s),
                           "thresholds": sorted({c["threshold"] for c in s})} for s in res.motif_support],
        "minority_sign_bands": [minority_sign_bands(res.motifs[:, i]) for i in range(n)],
        "broadband_like": [minority_sign_bands(res.motifs[:, i]) <= 1 for i in range(n)],
        "occupancy": occ, "runtime_s": round(runtime, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nwb", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--guardrail-repeats", type=int, default=2)
    args = ap.parse_args()
    out = args.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    T0 = time.time()

    def log(msg):
        print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)

    io, nwb = sd.open_nwb(args.nwb)
    info = sd.read_info(nwb, args.nwb.name)
    segments = sd.stim_free_segments(info)
    windows = sd.analysis_windows(info)
    log(f"iso on {info.iso_on:.0f}s, maintenance from {info.induction_end:.0f}s, off {info.iso_off:.0f}s; "
        f"stim-free segments {[(round(a), round(b)) for a, b in segments]}")

    traces, stim_curves, spectra, spectro = {}, {}, {}, {}
    for spec in SIGNALS:
        src = spec["source"]
        if src[0] == "lfp":
            channel = sd.lfp_channel(nwb, src[1], src[2])
            spec["channel"], spec["location"] = channel, src[2]
            src = ("lfp", src[1], channel)
        else:
            spec["channel"] = src[1]
            spec["location"] = info.eeg_locations[src[1]]
            assert info.eeg_valid[src[1]], f"EEG channel {src[1]} is flagged invalid"
        t_raw, x_raw, fs_raw = sd.read_raw(nwb, src)
        stim_curves[spec["label"]] = {
            st: stim_average(t_raw, x_raw, fs_raw, info.stim_times[sd.state_of_time(info.stim_times, info) == st])
            for st in ("awake", "isoflurane")}
        t, x, fs = sd.decimate(t_raw, x_raw, fs_raw, FS)
        del t_raw, x_raw
        traces[spec["key"]] = (t, x, fs)
        spectra[spec["label"]] = {w: welch_window(t, x, fs, segments, windows[w]) for w in WINDOWS}
        log(f"loaded {spec['label']} (channel {spec['channel']}), fs {fs:.2f}")

    valid = np.flatnonzero(info.eeg_valid)
    acc = {w: [] for w in WINDOWS}
    for ch in valid:
        t_raw, x_raw, fs_raw = sd.read_raw(nwb, ("eeg", int(ch)))
        t, x, fs = sd.decimate(t_raw, x_raw, fs_raw, FS)
        for w in WINDOWS:
            acc[w].append(welch_window(t, x, fs, segments, windows[w]))
    spectra[f"EEG mean of {valid.size} valid channels"] = {
        w: (acc[w][0][0], np.mean([p for _, p in acc[w]], axis=0)) for w in WINDOWS}
    log("all-channel EEG spectra done")

    for key in ("eeg13", "lfp_PL5"):
        t, x, fs = traces[key]
        f, tt, sxx = signal.spectrogram(x, fs, nperseg=int(4 * fs), noverlap=int(2 * fs))
        k = (f >= 1) & (f <= 80)
        block = 15  # 15 x 2-s steps = 30-s averages, so slow state changes are readable
        n = sxx.shape[1] // block
        sxx = sxx[k, : n * block].reshape(k.sum(), n, block).mean(axis=2)
        tt = tt[: n * block].reshape(n, block).mean(axis=1)
        db = 10 * np.log10(sxx / np.median(sxx, axis=1, keepdims=True))
        label = next(s["label"] for s in SIGNALS if s["key"] == key)
        spectro[label] = (t[0] + tt, f[k], db)

    spike_trains, unit_ids, n_units_total = sd.good_units(nwb, "PL5")
    io.close()
    log(f"PL5 units: {len(spike_trains)} of {n_units_total} pass quality filters")

    figs = out / "figures"
    plot_overview(info, segments, spectro, figs / "01_overview_timeline.png")
    plot_stim_response(stim_curves, figs / "02_stimulation_response.png")
    plot_spectra(spectra, figs / "03_power_spectra.png", figs / "04_power_ratio_iso_vs_awake.png")
    stim_free_awake = next((a, b) for a, b in segments if b <= windows["awake"][1])
    iso_seg = next((max(a, windows["isoflurane"][0]), b) for a, b in segments if b > windows["isoflurane"][0])
    mid = lambda ab: (ab[0] + ab[1]) / 2  # noqa: E731
    snippets = {"awake": (mid(stim_free_awake) - 30, mid(stim_free_awake) + 30),
                "induction": (info.iso_on + 60, info.iso_on + 120),
                "isoflurane": (mid(iso_seg) - 30, mid(iso_seg) + 30)}
    plot_raster(spike_trains, n_units_total, info, snippets, figs / "05_PL5_raster.png")

    runs, guard, window_sessions = [], [], {}
    for spec in SIGNALS:
        t, x, fs = traces[spec["key"]]
        pieces = envelope_pieces(t, x, fs, segments)
        by_key = {}
        for win in WINDOWS:
            pz = zscore(restrict(pieces, windows[win], fs))
            chunks = chunk(pz, fs)
            window_sessions[(spec["key"], win)] = chunks
            for ver, cfg in VERSIONS.items():
                sessions = chunks if cfg["chunked"] else joined(pz)
                res, rt = run_motifs(sessions, ver, fs)
                rec = motif_record(spec["key"], win, ver, res, sessions, rt, fs)
                runs.append(rec)
                by_key[(win, ver)] = rec
                log(f"{spec['key']:8s} {win:10s} {ver:8s}: {rec['n_motifs']} pair(s), "
                    f"{rec['n_sessions']} session(s), {rec['n_timepoints']} timepoints ({rt:.0f}s)")
            rng = np.random.default_rng(1000)
            for rep in range(args.guardrail_repeats):
                res, rt = run_motifs(band_shuffle(chunks, rng), "relaxed", fs, seed=SEED + 1 + rep)
                guard.append({"signal": spec["key"], "window": win, "repeat": rep, "n_motifs": res.n_motifs})
                log(f"{spec['key']:8s} {win:10s} guardrail {rep}: {res.n_motifs} pair(s) on band-shuffled data")
        plot_motif_grid(spec["label"], by_key, figs / f"06_motifs_{spec['key']}.png")

    def get(sig, win, ver):
        return next(r for r in runs if r["signal"] == sig and r["window"] == win and r["version"] == ver)

    def motifs_of(rec):
        return np.array(rec["motifs"]).reshape(len(BANDS), -1)

    for r in runs:
        if r["version"] in ("relaxed", "original") and r["n_motifs"]:
            strict = motifs_of(get(r["signal"], r["window"], "strict"))
            m = abs_corr_matrix(plus_poles(motifs_of(r)), plus_poles(strict))
            r["matches_strict"] = [bool(row.size and row.max() >= 0.9) for row in m] if m.size else [False] * r["n_motifs"]

    sim = {}
    for spec in SIGNALS:
        for ver in VERSIONS:
            mats = {}
            for wa, wb in (("awake", "isoflurane"), ("awake", "full"), ("isoflurane", "full")):
                mats[(wa, wb)] = abs_corr_matrix(plus_poles(motifs_of(get(spec["key"], wa, ver))),
                                                 plus_poles(motifs_of(get(spec["key"], wb, ver))))
            sim[(spec["label"], ver)] = mats
    plot_similarity({k: v for k, v in sim.items() if k[1] != "original"}, figs / "07_cross_state_similarity.png")

    def reference(sig, win):
        """The window's motifs from the strictest version that found any."""
        for ver in ("strict", "relaxed"):
            rec = get(sig, win, ver)
            if rec["n_motifs"]:
                return ver, motifs_of(rec)
        return None

    perm = {}
    for spec in SIGNALS:
        ref = reference(spec["key"], "full")
        if ref:
            perm[spec["key"]] = {"version": ref[0], "tests": permutation_test(
                window_sessions[(spec["key"], "full")], ref[1], info, traces[spec["key"]][2])}
    plot_permutation(perm, figs / "08_permutation_test.png")
    log(f"permutation tests: {list(perm)}")

    eeg_lfp = []
    for win in WINDOWS:
        lfp_ref = reference("lfp_PL5", win)
        for eeg_key in ("eeg13", "eeg10"):
            eeg_ref = reference(eeg_key, win)
            if not (lfp_ref and eeg_ref):
                continue
            me, ml = eeg_ref[1], lfp_ref[1]
            pe = pole_by_second(window_sessions[(eeg_key, win)], me, traces[eeg_key][2])
            pl = pole_by_second(window_sessions[("lfp_PL5", win)], ml, traces["lfp_PL5"][2])
            temporal = coupling(pe, pl, me.shape[1] // 2, ml.shape[1] // 2, info)
            eeg_lfp.append({"window": win, "eeg": eeg_key, "eeg_version": eeg_ref[0], "lfp_version": lfp_ref[0],
                            "shape_abs_r": abs_corr_matrix(plus_poles(me), plus_poles(ml)).tolist(),
                            "temporal": {k: v for k, v in temporal.items() if v["n_seconds"] >= 60}})

    spikes = []
    for st in ("awake", "isoflurane"):
        ref = reference("lfp_PL5", st)
        if ref:
            poles = pole_by_second(window_sessions[("lfp_PL5", st)], ref[1], traces["lfp_PL5"][2])
            spikes += [r | {"version": ref[0]} for r in
                       spikes_by_pole(spike_trains, poles, ref[1].shape[1] // 2, info) if r["state"] == st]
    plot_spikes_by_pole(spikes, figs / "09_spikes_by_LFP_pole.png")

    results = {
        "file": info.file, "subject": info.subject, "description": info.description,
        "bands_hz": BANDS.tolist(), "fs": FS, "chunk_sec": CHUNK_SEC, "seed": SEED,
        "versions": {k: {"chunked": v["chunked"], **v["kw"]} for k, v in VERSIONS.items()},
        "timeline": {"t_start": info.t_start, "t_end": info.t_end, "iso_on": info.iso_on,
                     "induction_end": info.induction_end, "iso_off": info.iso_off,
                     "electrical_blocks": info.stim_blocks, "visual_blocks": info.visual_blocks,
                     "stim_free_segments": segments, "windows": windows},
        "signals": [{k: v for k, v in s.items() if k != "source"} | {"source": list(map(str, s["source"]))}
                    for s in SIGNALS],
        "eeg_channels": [{"id": i, "location": loc, "valid": bool(v)}
                         for i, (loc, v) in enumerate(zip(info.eeg_locations, info.eeg_valid))],
        "pl5_units": {"n_total": n_units_total, "n_good": len(spike_trains), "criteria": sd.GOOD_UNIT,
                      "raster_snippets": snippets},
        "stim_response": {lab: {st: {"pre_sd_uv": float(v[2]), "peak_abs_uv_0_500ms":
                                     float(np.abs(v[1][(v[0] >= 0) & (v[0] < 0.5)]).max())}
                                for st, v in d.items()} for lab, d in stim_curves.items()},
        "spectra_band_power_db_change": {
            name: {f"{lo:g}-{hi:g}": float(10 * np.log10(
                d["isoflurane"][1][(d["isoflurane"][0] >= lo) & (d["isoflurane"][0] < hi)].mean()
                / d["awake"][1][(d["awake"][0] >= lo) & (d["awake"][0] < hi)].mean()))
                for lo, hi in ((1, 4), (4, 8), (8, 12), (12, 30), (30, 80))}
            for name, d in spectra.items()},
        "motif_runs": runs,
        "guardrail": guard,
        "cross_state_similarity": {f"{k[0]} | {k[1]}": {f"{a} vs {b}": m.tolist() for (a, b), m in v.items()}
                                   for k, v in sim.items()},
        "permutation_tests": perm,
        "eeg_vs_lfp": eeg_lfp,
        "spikes_by_lfp_pole": spikes,
        "runtime_s": round(time.time() - T0),
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, default=lambda o: o.tolist()
                                                 if isinstance(o, np.ndarray) else float(o)))
    rows = ["signal,window,version,n_sessions,n_timepoints,n_motif_pairs,min_p_per_motif"]
    for r in runs:
        rows.append(f"{r['signal']},{r['window']},{r['version']},{r['n_sessions']},{r['n_timepoints']},"
                    f"{r['n_motifs']},{' '.join(f'{s['min_p']:.3f}' for s in r['motif_support'])}")
    (out / "motif_counts.csv").write_text("\n".join(rows) + "\n")
    log(f"done; outputs in {out}")


if __name__ == "__main__":
    main()
