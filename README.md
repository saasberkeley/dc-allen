# SAAS x Allen Institute: Brain Recordings Across States

This repo holds code for a Berkeley SAAS (Student Association for Applied Statistics) project with the Allen Institute, analyzing neural recordings across different brain states. **You don't need any biology or neuroscience background to get started.**

The full project proposal is in [`docs/sow.pdf`](docs/sow.pdf).

---

## 1. The project in plain language

### What was recorded?
Researchers at the Allen Institute recorded the electrical activity of **mouse brains**, using probes called **Neuropixels**. A Neuropixels probe is a thin needle with hundreds of tiny electrodes. It is inserted into the brain and listens to many brain regions at once.

The mice were recorded in different **states**:

| State | What it is |
|---|---|
| **Awake** | Normal, alert baseline. |
| **Anesthetized** (isoflurane) | The animal is sedated. |
| **Psilocybin** | The animal was given the psychedelic compound found in "magic mushrooms". |
| **Ketanserin** | A drug that *blocks* the receptor psilocybin acts on. It is the control. |

### What kinds of signals are in the data?
Each recording contains several streams, all recorded at the same time:

- **Spikes.** Each neuron "fires" brief electrical pulses, called spikes. The data records the time of each spike and which neuron produced it.
- **LFP (local field potential).** The smooth, wavy signal from averaging electrical activity near each electrode. Contains rhythms ("brain waves") at different speeds (1–100 Hz).
- **EEG.** A coarser brain signal measured from the surface of the head.
- **Pupil size and behavior.** Video-derived measurements of the animal's pupil and movement.

### What are we trying to figure out?
A recent paper (Mishler et al., 2026) describes a method for finding **spectral motifs**—recurring patterns of brain-wave activity across several frequencies at once. Motifs tend to come in **opposing pairs**: when one goes up, neurons tend to fire more; when its partner goes up, they fire less.

**Our question is: how do these motifs change when the brain is in a different state (awake, sedated, or on psilocybin)?**

### What have we built?
1. **Data access layer.** Code that loads recordings into clean arrays with the state labelled.
2. **Verified motif pipeline.** Python implementation of the motif method, validated against the original MATLAB code.
3. **Analysis tools.** Visualization, spectral analysis, and comparison scripts.

---

## 2. Glossary

| Term | Meaning |
|---|---|
| **NWB** (Neurodata Without Borders) | A standard file format (`.nwb`) for neuroscience data. One file holds one recording session. |
| **Session** | One recording of one mouse on one day. |
| **Neuropixels** | The high-density recording probe. |
| **Unit** | One neuron, as identified from the spike data. |
| **LFP** | Local field potential: the slow wavy electrical signal. |
| **Hz** | Cycles per second, the speed of a brain wave. |
| **Epoch** | A labelled stretch of time within a recording (e.g., "awake" or "post-drug"). |
| **Motif** | A recurring multi-frequency pattern in the LFP. |

---

## 3. Setup

You need a Mac or Linux machine (Windows: use WSL), **Python 3.9+**, and **~30 GB free disk space** for full data.

### Step 1: Clone the repo
```bash
git clone git@github.com:saasberkeley/dc-allen.git
cd dc-allen
```

### Step 2: Set up Python environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib h5py pynwb pandas jupyter
```

### Step 3: Download data (optional)
Sample data is already in `data/sample_nwb/` and `data/sample_lfp/`. For full NWB files:

1. Get the Dropbox link from project leads
2. Create `.env` from template: `cp .env.example .env`
3. Add the link to `.env`
4. Run: `python3 code/python/src/download_dropbox_data.py`

---

## 4. Repo layout

```
dc-allen/
├── code/
│   ├── python/
│   │   ├── src/                    # Python source code
│   │   │   ├── smooth_env.py               # SmoothEnv port (preprocessing)
│   │   │   ├── get_motifs.py               # getMotifs port (motif discovery)
│   │   │   ├── composite_motif_pipeline.py # Full pipeline for MAT files
│   │   │   ├── extract_windows.py          # Extract time windows from NWB
│   │   │   ├── lfp_to_motifs.py            # End-to-end LFP → motifs (NWB)
│   │   │   ├── region_sweep.py             # Run motifs across brain regions
│   │   │   ├── spectra.py                  # Spectral analysis tools
│   │   │   ├── rasters.py                  # Spike raster visualization
│   │   │   └── tests/                      # Unit tests
│   │   └── notebooks/              # Jupyter notebooks for analysis
│   └── matlab/
│       ├── src/                    # Original MATLAB reference code
│       │   ├── SmoothEnv.m         # Preprocessing
│       │   ├── butterfilter.m      # Butterworth filter helper
│       │   └── getMotifs.m         # Motif-finding algorithm
│       └── notebooks/              # Reference notebooks from Allen
│
├── data/
│   ├── sample_nwb/                 # Sample H5 files (extracted windows)
│   ├── sample_lfp/                 # Sample LFP data for testing
│   └── out/                        # Analysis outputs
│       ├── motifs/                 # Discovered motifs
│       ├── region_sweep/           # Per-region motif results
│       └── motifs_comparison/      # MATLAB vs Python comparison
│
├── docs/
│   ├── sow.pdf                     # Project proposal
│   ├── papers/                     # Background reading
│   └── preprints/                  # Mishler & Momi preprints
│
├── TASKS.md                        # Task assignments
└── README.md                       # This file
```

---

## 5. Running the motif pipeline

### Option 1: Use the composite pipeline for MAT files (recommended)
The `composite_motif_pipeline.py` module processes MATLAB `.mat` files through the full Mishler pipeline:

```bash
cd code/python
python -m src.composite_motif_pipeline --mat /path/to/LFP_Data.mat --out /path/to/output
```

This runs the complete pipeline:
- 60 Hz bandstop filtering
- 76-band smooth envelope ([1,5] to [76,80] Hz)
- Noise detection and removal
- Motif discovery with null hypothesis testing
- Saves results to a new MAT file

See all options with `python -m src.composite_motif_pipeline --help`.

### Option 2: Use the modular Python code
```python
from src.smooth_env import smooth_env
from src.get_motifs import get_motifs

# Preprocess LFP
envelope = smooth_env(lfp_trace, fs=500, freq_bands=my_bands)

# Find motifs (remove NaN edges first)
valid = ~np.isnan(envelope).any(axis=1)
result = get_motifs([envelope[valid].T], fs=500)
print(f"Found {result.n_motifs} motifs")
```

### Option 3: Use composite pipeline as a library
```python
from src.composite_motif_pipeline import run_pipeline, smooth_env, filter_lfp_noise

# Run full pipeline on MAT file
result = run_pipeline(
    mat_file="/path/to/LFP_Data.mat",
    save_dir="/path/to/output",
    stdev_thresh=10.0,
    n_null=20,
)
print(f"Found {result['n_motifs']} motifs")
```

---

## 6. Key algorithms

### SmoothEnv (preprocessing)
Converts raw LFP into smoothed, z-scored band-amplitude envelopes:

1. **Bandpass** into each frequency band (6th-order Butterworth)
2. **Hilbert envelope** for instantaneous amplitude
3. **Smooth** with 5-second Gaussian window
4. **Trim edges** (first/last 10s → NaN)
5. **Z-score** each band over time

### getMotifs (motif discovery)
Finds recurring multi-band patterns:

1. **Subsample** to ~1 Hz and **binarize** (sign of each band)
2. **Hierarchical clustering** with correlation distance
3. **Branch consolidation** at multiple correlation thresholds (0.4–0.8)
4. **Null hypothesis testing** via circular shifts
5. **PCA deduplication** across thresholds
6. **Final templates** with opposing pairs

---

## 7. Sample data

The repo includes sample data for testing:

| File | Description | Spikes | Units |
|------|-------------|--------|-------|
| `688277-20230921-v001_awake.h5` | Awake baseline | 743,622 | 1,398 |
| `688277-20230921-v001_iso_anesthetized.h5` | Isoflurane anesthesia | 68,802 | 1,398 |
| `692644-20230907-v001_psilocybin_before.h5` | Pre-psilocybin | 568,187 | 1,117 |
| `692644-20230907-v001_psilocybin_after.h5` | Post-psilocybin | 628,619 | 1,117 |

---

## 8. References

1. **Project proposal:** [`docs/sow.pdf`](docs/sow.pdf)
2. **Mishler et al. (2026)** – Spectral motif method: [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.02.20.707132v1)
3. **Momi et al. (2026)** – Psilocybin dataset: [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.08.14.744865v1)

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run from repo root, or add repo to `PYTHONPATH` |
| Download stalls | Re-run the download script (it resumes) |
| `No space left on device` | Free up ~30 GB and retry |
| HDF5 read errors | Install h5py: `pip install h5py` |
