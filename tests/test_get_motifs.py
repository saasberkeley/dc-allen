import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from src.mapping.get_motifs import CORR_THRESHOLDS, _pca_uncentered, candidate_partition, get_motifs


def _matlab_chains(Z, n):
    """The `while 1` loop that builds links{i}: each leaf's path of node ids to the root."""
    links = [[i] for i in range(n)]
    while True:
        stop_tally = 0
        stop_all = False
        for i in range(n):
            rows = np.nonzero(Z[:, :2] == links[i][-1])[0]
            if rows.size:
                links[i] = links[i] + [n + int(rows[0])]
            else:
                stop_tally += 1
            if stop_tally == n:
                stop_all = True
                break
        if stop_all:
            return links


def _corr_linkage(points):
    with np.errstate(invalid="ignore", divide="ignore"):
        d = pdist(points, "correlation")
    return linkage(np.nan_to_num(d, nan=1.0), "average")


def literal_candidates(all_corr, thresh):
    """Line-by-line translation of getMotifs.m's consolidation for one threshold.

    all_corr is bands x time, as in MATLAB. Returns finalMotifsTrack (0-based).
    Deviation: a chain that has reached the root (length 1) has no x(2); MATLAB
    would error, here it just finds no sibling.
    """
    n = all_corr.shape[1]
    links = _matlab_chains(_corr_linkage(all_corr.T), n)
    track = [[i] for i in range(n)]
    link_size = [len(links)]
    outer_size = []
    n_outer = 1
    while True:
        store, links_store = [], []
        m = 1
        while True:
            new_links = [None] * len(links)
            low_corr = np.zeros(len(links), dtype=bool)
            sec_link = [x[1] if len(x) > 1 else None for x in links]
            for i in range(len(links)):
                target = sec_link[i]
                matches = [j for j, s in enumerate(sec_link) if s is not None and s == target]
                if len(matches) > 1:
                    temp = np.column_stack([all_corr[:, track[j]].mean(axis=1) for j in matches])
                    proj = all_corr.T @ temp
                    with np.errstate(invalid="ignore", divide="ignore"):
                        r = np.corrcoef(proj[:, 0], proj[:, 1])[0, 1]
                    if r >= thresh:
                        temp_links = sorted(set(sum((links[j] for j in matches), [])))[len(matches):]
                        temp_track = sum((track[j] for j in matches), [])
                        for j in matches:
                            new_links[j] = temp_links
                            track[j] = temp_track
                    else:
                        for j in matches:
                            new_links[j] = links[j]
                            low_corr[j] = True
                else:
                    j = matches[0] if matches else i
                    new_links[j] = links[j]
            track = [sorted(set(x)) for x in track]
            store += [t for t, low in zip(track, low_corr) if low]
            track = [t for t, low in zip(track, low_corr) if not low]
            links_store += [x for x, low in zip(new_links, low_corr) if low]
            new_links = [x for x, low in zip(new_links, low_corr) if not low]
            keys = [",".join(map(str, sorted(x))) for x in new_links]
            first = sorted({k: i for i, k in reversed(list(enumerate(keys)))}.values())
            links = [new_links[i] for i in first]
            track = [track[i] for i in first]
            link_size.append(len(links))
            if m > 1 and link_size[-1] == link_size[-2]:
                track = track + store
                links = links + links_store
                break
            m += 1
        outer_size.append(len(links))
        if n_outer > 1 and outer_size[-1] == outer_size[-2]:
            return track
        if len(track) < 2:
            return track
        templates = np.column_stack([all_corr[:, x].mean(axis=1) for x in track])
        links = _matlab_chains(_corr_linkage(templates.T), len(track))
        n_outer += 1


def _random_signs(rng, n_time, n_bands, n_states=4):
    """Sign data with some recurring structure, so that merges and freezes both happen."""
    states = rng.choice([-1.0, 1.0], size=(n_states, n_bands))
    x = states[rng.integers(n_states, size=n_time)]
    flip = rng.random(x.shape) < 0.25
    x[flip] *= -1
    x = x[np.abs(x.sum(axis=1)) != n_bands]
    return x


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("thresh", CORR_THRESHOLDS)
def test_consolidation_matches_literal_matlab_loop(seed, thresh):
    rng = np.random.default_rng(seed)
    X = _random_signs(rng, n_time=70, n_bands=6)

    expected = literal_candidates(X.T, thresh)
    sizes, labels = candidate_partition(X, thresh)

    got = [sorted(np.flatnonzero(labels == k).tolist()) for k in range(sizes.size)]
    assert got == expected
    assert sizes.tolist() == [len(t) for t in expected]


def test_pca_sign_convention_matches_matlab():
    rng = np.random.default_rng(0)
    F = rng.standard_normal((7, 5))
    coeff, score = _pca_uncentered(F)
    biggest = coeff[np.argmax(np.abs(coeff), axis=0), np.arange(coeff.shape[1])]
    assert (biggest > 0).all()
    np.testing.assert_allclose(score @ coeff.T, F, atol=1e-12)


def _planted_sessions(rng, motifs, n_sessions=2, seconds=1500, fs=1.0):
    """Envelopes that dwell in one of the planted motifs (or its opponent) for 20-60 s at a time."""
    poles = np.vstack([motifs, -motifs])
    sessions = []
    for _ in range(n_sessions):
        n = int(seconds * fs)
        env = np.empty((motifs.shape[1], n))
        t = 0
        while t < n:
            dwell = int(rng.integers(20, 60) * fs)
            env[:, t:t + dwell] = poles[rng.integers(len(poles))][:, None]
            t += dwell
        env += 0.6 * rng.standard_normal(env.shape)
        sessions.append((env - env.mean(1, keepdims=True)) / env.std(1, keepdims=True))
    return sessions


def test_recovers_planted_opponent_motifs():
    rng = np.random.default_rng(1)
    planted = np.array([
        [1, 1, 1, -1, -1, -1, 1, 1],
        [1, -1, -1, 1, 1, -1, -1, 1],
    ], dtype=float)
    result = get_motifs(_planted_sessions(rng, planted), fs=1.0, seed=0)

    assert result.n_motifs >= 1
    n = result.n_motifs
    np.testing.assert_array_equal(result.motifs[:, n:], -result.motifs[:, :n])
    for motif in planted:
        best = max(abs(np.corrcoef(motif, m)[0, 1]) for m in result.motifs.T)
        assert best > 0.9, f"planted motif {motif} not recovered (best |r| = {best:.2f})"


def test_no_motifs_in_structureless_noise():
    rng = np.random.default_rng(2)
    sessions = [rng.standard_normal((8, 1200)) for _ in range(2)]
    result = get_motifs(sessions, fs=1.0, seed=0)
    assert result.n_motifs == 0


def test_rejects_nan_input():
    x = np.zeros((4, 100))
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        get_motifs([x], fs=1.0)
