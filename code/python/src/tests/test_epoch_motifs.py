import numpy as np
import pandas as pd
import pytest

from src import epoch_motifs as em


def _synthetic(seed=0, n_bands=8, seconds=1800, drift_amp=3.0):
    """Two epochs. Both switch between a planted motif's poles (20-60 s dwells); the second also has a
    slow drift along the motif axis (period 1500 s), which the sliding baseline should absorb."""
    rng = np.random.default_rng(seed)
    motif = np.array([1, 1, 1, -1, -1, -1, 1, 1], float)[:n_bands]
    u = motif / np.linalg.norm(motif)
    t = np.arange(2 * seconds, dtype=float)
    env = np.empty((t.size, n_bands))
    drift = np.zeros(t.size)
    pos = 0
    while pos < t.size:
        d = int(rng.integers(20, 60))
        env[pos:pos + d] = rng.choice([-1, 1]) * motif
        pos += d
    env += 0.6 * rng.standard_normal(env.shape)
    second = t >= seconds
    drift[second] = drift_amp * np.sin(2 * np.pi * (t[second] - seconds) / 1500)
    env += drift[:, None] * u[None, :] * np.sqrt(n_bands)
    epoch = np.where(second, "iso_early", "awake").astype(object)
    Z = (env - env.mean(0)) / env.std(0, ddof=1)
    pieces = {}
    for e in ("awake", "iso_early"):
        sel = epoch == e
        z = (env[sel] - env[sel].mean(0)) / env[sel].std(0, ddof=1)
        pieces[e] = [(t[sel], z)]
    data = em.Data(None, t, env, Z, epoch, pieces)
    return data, motif, drift


@pytest.mark.xfail(reason="epoch_motifs.pipeline (paused, superseded by motif_tracking) uses a rolling-mean baseline, "
                          "which leaves drift in the expression here (r = 0.71 vs the planted drift); motif_tracking "
                          "replaced it with a local-linear baseline. Kept as a record; not fixed.", strict=False)
def test_pipeline_finds_motif_and_moves_drift_into_baseline():
    data, motif, drift = _synthetic()
    cfg = em.Config(epochs=("awake", "iso_early"), n_sur=3)
    res = em.pipeline(data, cfg)

    best = max(abs(np.corrcoef(motif, m.template)[0, 1]) for m in res.library)
    assert best > 0.9

    mi = int(np.argmax([abs(np.corrcoef(motif, m.template)[0, 1]) for m in res.library]))
    iso = (data.epoch == "iso_early") & np.isfinite(res.second[f"base_M{mi + 1}"]).to_numpy()
    r_base = abs(np.corrcoef(res.second.loc[iso, f"base_M{mi + 1}"], drift[iso])[0, 1])
    assert r_base > 0.9, f"baseline does not follow the drift (r = {r_base:.2f})"

    slow = pd.Series(res.second.loc[iso, f"M{mi + 1}"].to_numpy()).rolling(300, center=True, min_periods=75).mean()
    ok = slow.notna().to_numpy()
    r_expr = abs(np.corrcoef(slow[ok], drift[iso][ok])[0, 1])
    assert r_expr < 0.3, f"drift leaked into the expression (r = {r_expr:.2f})"

    # the planted poles are discrete, so the motif's expression is two-peaked in both epochs
    peaks = res.kfit[res.kfit.motif == mi].set_index("epoch")["peaks"]
    assert (peaks >= 2).all()
