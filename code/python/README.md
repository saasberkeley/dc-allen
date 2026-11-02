# Mapping: LFP to spectral motifs (task 4)

Python port of Mishler's MATLAB motif code (`matlab/`), and a script that runs it on the Neuropixels LFP.

| File | What it is |
|---|---|
| `smooth_env.py` | Port of `SmoothEnv.m` + `butterfilter.m`. Works for any 1-D trace (LFP or EEG). |
| `get_motifs.py` | Port of `getMotifs.m`. Takes a list of `(bands, time)` arrays, one per session. |
| `lfp_to_motifs.py` | Task 4 script: NWB LFP → `smooth_env` → `get_motifs` → figures + per-state motif occupancy. |
| `region_sweep.py` | Runs the same pipeline on 10 regions (cortex, hippocampus, thalamus, striatum) × 3 seeds and compares them. |
| `../../tests/` | Tests, including a check that `get_motifs` matches a line-by-line translation of the MATLAB loop. |

The EEG task (5) can reuse `smooth_env` and `get_motifs` unchanged. Only the loading step differs.

## Run it

From the repo root, after downloading the data:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pynwb h5py numpy scipy scikit-learn matplotlib pytest

python -m src.mapping.lfp_to_motifs                       # both sessions, probeF MOs5, ~1 min
python -m src.mapping.lfp_to_motifs --region MOs2/3 --out results/lfp_mos23
python -m src.mapping.lfp_to_motifs --bands 1-4 3-6 5-9 8-12 10-15 13-20 18-30 25-40 35-55
python -m src.mapping.region_sweep                        # 10 regions x 3 seeds, ~20 min
python -m pytest                                          # ~5 s
```

`region_sweep` writes each region's full outputs to `results/region_sweep/<region>/`, plus `region_comparison.png`
(motif shape and per-state pole balance for every region), `motif_shapes.png` and `sweep_summary.json`.

Outputs go to `results/lfp_to_motifs/` (change with `--out`):

- `motifs.png`: each motif and its opponent, as mean sign per frequency band.
- `timeline.png`: z-scored band envelopes over each session, with the dominant motif pole each second.
- `occupancy.png`: fraction of seconds each pole dominates, per brain state.
- `summary.json`: parameters, channel used, per-threshold candidate/null counts, motif templates, occupancy.
- `motifs.npz`: the arrays (motifs, 1 Hz envelopes, per-second state and pole labels).

## Porting notes

- `butterfilter`: MATLAB's `fdesign.bandpass` with N = 6 is a 6-pole bandpass, i.e. `scipy.signal.butter(3, ...)`. `butter(6, ...)` would be 12 poles.
- `gausswin(N, 1)` is a Gaussian with std = (N - 1) / 2, so N is not forced to be odd. The `filtfilt` smoothing uses FFT convolution and matches `scipy.signal.filtfilt` to about 1e-13.
- `SmoothEnv` leaves 10 s of NaN at each end. The script trims these before `get_motifs`, which rejects NaN input. **Still to be confirmed with Mishler.**
- The consolidation loop is computed in one pass over the tree instead of sweep by sweep. It gives the same candidates in the same order (checked on 200 random cases). See the docstring in `get_motifs.py` for the edge cases where MATLAB itself would error.
- Null shifts use a fixed seed (`--seed`). MATLAB's are unseeded, and tied linkage distances may break differently, so results match MATLAB in distribution, not bit for bit.

## Open questions for Irene / Jonathan

1. **Frequency bands.** The defaults are placeholders taken from TASKS.md. Which bands did the paper use?
2. **NaN edges.** Is trimming them before `getMotifs` the intended use?
3. **Channel choice.** We use one mid-layer-5 channel of secondary motor cortex (MOs5, probeF), the closest match to rat mPFC that is in both sessions. Should we use one channel, average across the layer, or treat each region separately?
4. **Sessions.** There are only two NWB files (one isoflurane, one psilocybin), so the rule that a motif must appear in every session is weak here. Should each state epoch count as its own "session"?
5. **Line noise.** The 50–80 Hz band contains 60 Hz mains noise. Notch it first?

## State analysis: awake vs isoflurane vs full recording (mouse 720762)

| File | What it is |
|---|---|
| `src/session_data.py` | Loads a session: traces, isoflurane timeline, stimulation blocks, stimulation-free windows, sorted units. |
| `src/state_motifs.py` | EEG and LFP motifs (original / strict-chunked / relaxed), spectra, PL5 raster, permutation test, EEG-vs-LFP coupling, spikes by motif pole. |
| `src/motif_bootstrap.py` | Chunk bootstrap: how reproducible each state's motif is, and whether the states share motifs. |
| `src/motif_calibration.py` | False-positive calibration of each version on synthetic motif-free data. |
| `src/null_size_diagnosis.py` | Shows why the original rules find few motifs within one state (shuffled nulls keep more timepoints). |
| `src/channel_sweep.py` | The same analysis on every valid EEG channel and every grey-matter LFP region (parallel), plus EEG x LFP coupling and spikes-by-pole per region. |

`get_motifs` gained options for the relaxed version (`null_mode="pvalue"`, `alpha`, `min_session_frac`,
`min_pca_support`, `normalize_size`); the defaults still reproduce getMotifs.m. `smooth_env` gained `zscore=False`.

```bash
python -m src.state_motifs --nwb ../../data/raw/720762-20240221-v001.nwb --out ../../data/out/2026-11-01_mouse720762
python -m src.motif_bootstrap --nwb ../../data/raw/720762-20240221-v001.nwb --out ../../data/out/2026-11-01_mouse720762
python -m src.motif_calibration --out ../../data/out/2026-11-01_mouse720762/diagnostics
python -m src.channel_sweep --nwb ../../data/raw/720762-20240221-v001.nwb --out ../../data/out/2026-11-01_mouse720762/all_channels
```

Results, figures and the write-up (`report_mouse720762_2026-11-01.docx`) are in `data/out/2026-11-01_mouse720762/`;
the notebook is `notebooks/mouse720762_states_naman.ipynb`.

The Word report is generated from the saved results: `cd report && npm install && node build_report.js ../../../data/out/2026-11-01_mouse720762`.
