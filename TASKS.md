# 1-Hour Work Session: Data Visualization & Spectral Motifs

Tasks for 5 team members, balanced by complexity.

---

## Setup (Everyone, first 5 min)

```bash
git clone git@github.com:saasberkeley/dc-allen-data-acquire.git
cd dc-allen-data-acquire
python3 -m venv .venv
source .venv/bin/activate
pip install pynwb h5py numpy pandas matplotlib jupyter scipy scikit-learn
```

Download data:
```bash
cp .env.example .env
# Add the Dropbox link to .env
python3 -m src.scripts.download_dropbox_data
```

Reference: `notebooks/read_Templeton_NWB.ipynb` (or view `.html` version)

---

## Task Distribution (Balanced)

| Person | Primary Task | Complexity | Stretch Goal |
|--------|--------------|------------|--------------|
| 1 | Raster plots (all 5 conditions) | Low | Add PSTH histograms |
| 2 | EEG plots + power spectra | Medium | Compare spectra across states |
| 3 | LFP plots + spectrograms | Medium | Compare spectrograms across states |
| 4 | Port `butterfilter.m` + `SmoothEnv.m` | Medium-High | Write unit tests |
| 5 | Port `getMotifs.m` (structure only) | High | Get clustering working |

**After primary tasks:** Persons 1-3 help test the spectral motifs code on their data.

---

## Person 1: Raster Plots (Spikes)

**Complexity:** Low
**Output:** `notebooks/01_raster_plots.ipynb`

Plot spike rasters for 5-10 neurons across all 5 conditions:
- [ ] Awake
- [ ] Isoflurane
- [ ] Urethane
- [ ] Psilocybin
- [ ] Ketanserin

**Starter code:**
```python
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt
import numpy as np

io = NWBHDF5IO('data/raw/YOUR_FILE.nwb', mode='r')
nwbfile = io.read()
units_df = nwbfile.units.to_dataframe()

# Raster plot
fig, ax = plt.subplots(figsize=(12, 6))
for i, (unit_id, row) in enumerate(units_df.head(10).iterrows()):
    spike_times = row['spike_times']
    ax.scatter(spike_times, np.ones_like(spike_times) * i, s=1, c='k')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Unit')
ax.set_title(f'Raster Plot - {nwbfile.pharmacology}')
io.close()
```

**Stretch goal:** Add peri-stimulus time histogram (PSTH) around stimulus events.

---

## Person 2: EEG Plots + Power Spectra

**Complexity:** Medium
**Output:** `notebooks/02_eeg_visualization.ipynb`

For each condition, create:
- [ ] Raw EEG traces (3-5 channels, 30-60 sec)
- [ ] Power spectral density (PSD) plot

**Starter code:**
```python
from pynwb import NWBHDF5IO
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

io = NWBHDF5IO('data/raw/YOUR_FILE.nwb', mode='r')
nwbfile = io.read()

EEG = nwbfile.acquisition['ElectricalSeriesEEG']
data = EEG.data[:] * EEG.conversion * 1e6  # µV
timestamps = EEG.timestamps[:]
fs = 1 / np.mean(np.diff(timestamps[:1000]))

# Raw trace (first 60 sec, channel 0)
mask = timestamps < 60
plt.figure(figsize=(12, 3))
plt.plot(timestamps[mask], data[mask, 0], lw=0.5)
plt.xlabel('Time (s)')
plt.ylabel('EEG (µV)')
plt.title(f'EEG - {nwbfile.pharmacology}')

# Power spectrum
f, psd = signal.welch(data[:, 0], fs=fs, nperseg=int(fs*2))
plt.figure(figsize=(8, 4))
plt.semilogy(f, psd)
plt.xlim(0, 100)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (µV²/Hz)')
plt.title(f'EEG Power Spectrum - {nwbfile.pharmacology}')
io.close()
```

**Stretch goal:** Overlay PSDs from all 5 conditions on one plot.

---

## Person 3: LFP Plots + Spectrograms

**Complexity:** Medium
**Output:** `notebooks/03_lfp_visualization.ipynb`

For each condition, create:
- [ ] Raw LFP traces (5-10 channels from one probe)
- [ ] Spectrogram (time-frequency plot)

**Starter code:**
```python
from pynwb import NWBHDF5IO
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

io = NWBHDF5IO('data/raw/YOUR_FILE.nwb', mode='r')
nwbfile = io.read()

probe = 'probeB'  # or probeC, probeD, probeF
LFP = nwbfile.acquisition['LFP'][f'ElectricalSeries{probe}']
data = LFP.data[:, 0] * LFP.conversion * 1e6  # µV, channel 0
timestamps = LFP.timestamps[:]
fs = 1 / np.mean(np.diff(timestamps[:1000]))

# Downsample for spectrogram
downsample = 4
data_ds = signal.decimate(data, downsample)
fs_ds = fs / downsample

# Spectrogram (first 120 sec)
f, t, Sxx = signal.spectrogram(data_ds[:int(120*fs_ds)], fs=fs_ds, nperseg=int(fs_ds*2))
plt.figure(figsize=(12, 4))
plt.pcolormesh(t, f, 10*np.log10(Sxx), shading='gouraud', vmin=-20, vmax=40)
plt.ylim(0, 100)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Power (dB)')
plt.title(f'LFP Spectrogram - {nwbfile.pharmacology}')
io.close()
```

**Stretch goal:** Compare spectrograms across conditions (side-by-side subplots).

---

## Person 4: Port Preprocessing (SmoothEnv)

**Complexity:** Medium-High
**Output:** `src/spectral_motifs/preprocessing.py`

Port these MATLAB files to Python:
- [ ] `butterfilter.m` → `butterfilter()` function
- [ ] `SmoothEnv.m` → `smooth_env()` function

**Reference:** `src/matlab/SmoothEnv.m`, `src/matlab/butterfilter.m`

**Key equivalents:**

| MATLAB | Python |
|--------|--------|
| `fdesign.bandpass` | `scipy.signal.butter(..., output='sos')` |
| `filtfilt` | `scipy.signal.sosfiltfilt` |
| `hilbert` | `scipy.signal.hilbert` |
| `gausswin(N, 1)` | `scipy.signal.windows.gaussian(N, std=(N-1)/2)` |
| `zscore` | `scipy.stats.zscore` |

**Implementation:**
```python
# src/spectral_motifs/preprocessing.py
"""Preprocessing for spectral motif analysis (ported from MATLAB)."""

import numpy as np
from scipy import signal
from scipy.stats import zscore as scipy_zscore


def butterfilter(data: np.ndarray, fs: float, low: float, high: float, order: int = 6) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    data : 1D array
        Input signal
    fs : float
        Sampling frequency (Hz)
    low, high : float
        Bandpass edges (Hz)
    order : int
        Filter order (default 6)

    Returns
    -------
    filtered : 1D array
    """
    sos = signal.butter(order, [low, high], btype='band', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, data)


def smooth_env(
    lfp: np.ndarray,
    fs: float,
    freq_bands: np.ndarray,
    smooth_window_sec: float = 5.0,
    edge_trim_sec: float = 10.0
) -> np.ndarray:
    """
    Convert raw LFP/EEG to smoothed, z-scored band envelopes.

    This replicates SmoothEnv.m:
    1. Bandpass filter into overlapping bands
    2. Hilbert envelope (instantaneous amplitude)
    3. Gaussian smoothing (~5 sec window)
    4. Trim edges (set to NaN)
    5. Z-score each band

    Parameters
    ----------
    lfp : 1D array
        Raw continuous trace
    fs : float
        Sampling frequency (Hz)
    freq_bands : (N, 2) array
        [[low1, high1], [low2, high2], ...] in Hz
    smooth_window_sec : float
        Gaussian window width in seconds
    edge_trim_sec : float
        Seconds to mark as NaN at start/end

    Returns
    -------
    envelopes : (time, bands) array
        Z-scored amplitude envelopes
    """
    n_samples = len(lfp)
    n_bands = len(freq_bands)
    envelopes = np.zeros((n_samples, n_bands))

    # Gaussian window for smoothing
    window_samples = int(smooth_window_sec * fs)
    if window_samples % 2 == 0:
        window_samples += 1  # make odd
    # gausswin(N, 1) means std = (N-1)/2
    gauss_std = (window_samples - 1) / 2
    gauss_window = signal.windows.gaussian(window_samples, std=gauss_std)
    gauss_window /= gauss_window.sum()  # normalize

    for i, (low, high) in enumerate(freq_bands):
        # 1. Bandpass
        filtered = butterfilter(lfp, fs, low, high)

        # 2. Hilbert envelope
        analytic = signal.hilbert(filtered)
        envelope = np.abs(analytic)

        # 3. Gaussian smooth (forward-backward for zero phase)
        smoothed = np.convolve(envelope, gauss_window, mode='same')

        envelopes[:, i] = smoothed

    # 4. Trim edges
    edge_samples = int(edge_trim_sec * fs)
    envelopes[:edge_samples, :] = np.nan
    envelopes[-edge_samples:, :] = np.nan

    # 5. Z-score each band (ignoring NaNs)
    for i in range(n_bands):
        valid = ~np.isnan(envelopes[:, i])
        if valid.sum() > 1:
            envelopes[valid, i] = scipy_zscore(envelopes[valid, i])

    return envelopes


# Example frequency bands (from typical spectral motif papers)
DEFAULT_FREQ_BANDS = np.array([
    [1, 4],    # delta
    [3, 6],    # theta-low
    [5, 9],    # theta-high
    [8, 12],   # alpha
    [10, 15],  # beta-low
    [13, 20],  # beta-mid
    [18, 30],  # beta-high
    [25, 40],  # gamma-low
    [35, 55],  # gamma-mid
    [50, 80],  # gamma-high
])
```

**Stretch goal:** Write unit tests comparing Python output to MATLAB on test data.

---

## Person 5: Port Motif Algorithm (getMotifs)

**Complexity:** High
**Output:** `src/spectral_motifs/motifs.py`

Port the structure of `getMotifs.m`. Focus on getting the scaffolding right; full functionality may extend beyond this session.

**Reference:** `src/matlab/getMotifs.m`, README Section 5

**Key steps in the algorithm:**
1. Subsample to ~1 Hz, concatenate sessions
2. Binarize (sign of each band)
3. Hierarchical clustering (correlation distance)
4. Merge clusters based on expression correlation
5. Null distribution (circular shifts)
6. Keep significant motifs
7. Remove duplicates with PCA

**Key equivalents:**

| MATLAB | Python |
|--------|--------|
| `pdist` | `scipy.spatial.distance.pdist` |
| `linkage` | `scipy.cluster.hierarchy.linkage` |
| `pca` | `sklearn.decomposition.PCA` |
| `corr` | `np.corrcoef` |
| `randperm` | `np.random.permutation` |

**Scaffolding:**
```python
# src/spectral_motifs/motifs.py
"""Spectral motif discovery (ported from MATLAB getMotifs.m)."""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA


def get_motifs(
    lfp_in: list,
    fs: float,
    r_thresholds: list = [0.4, 0.5, 0.6, 0.7, 0.8],
    n_null: int = 20,
    seed: int = 42
) -> np.ndarray:
    """
    Find spectral motifs from preprocessed LFP/EEG.

    Parameters
    ----------
    lfp_in : list of (bands, time) arrays
        Preprocessed envelopes, one per session
        NOTE: Input is (bands, time), transposed from SmoothEnv output
    fs : float
        Sampling frequency
    r_thresholds : list
        Correlation thresholds for merging
    n_null : int
        Number of null shuffles
    seed : int
        Random seed

    Returns
    -------
    motifs : (bands, n_motifs*2) array
        Motif templates with sign-flipped pairs
    """
    np.random.seed(seed)

    # 1. Subsample to ~1 Hz and concatenate
    target_fs = 1.0
    subsample_factor = max(1, int(fs / target_fs))

    all_data = []
    for session in lfp_in:
        # session is (bands, time)
        subsampled = session[:, ::subsample_factor]
        all_data.append(subsampled)

    concatenated = np.hstack(all_data)  # (bands, total_time)
    n_bands, n_timepoints = concatenated.shape

    # 2. Binarize: take sign of each band
    binarized = np.sign(concatenated)  # (bands, time)

    # 3. Remove trivial timepoints (all same sign)
    keep_mask = np.abs(binarized.sum(axis=0)) < n_bands
    binarized = binarized[:, keep_mask]

    # 4. Cluster with correlation distance
    # Each timepoint is a pattern across bands
    patterns = binarized.T  # (time, bands)
    distances = pdist(patterns, metric='correlation')
    Z = linkage(distances, method='average')

    # 5. TODO: Merge branches based on expression correlation

    # 6. TODO: Null distribution with circular shifts

    # 7. TODO: Keep significant motifs

    # 8. TODO: Remove duplicates with PCA

    # Placeholder return
    print(f"Processed {n_timepoints} timepoints, {n_bands} bands")
    print(f"After filtering trivial: {binarized.shape[1]} timepoints")
    return binarized  # temporary


def circular_shift_null(data: np.ndarray, shift_sec: float, fs: float) -> np.ndarray:
    """
    Create null data by circularly shifting each band independently.

    Parameters
    ----------
    data : (bands, time) array
    shift_sec : float
        Minimum shift in seconds (actual shift is random multiple of this)
    fs : float
        Sampling frequency

    Returns
    -------
    shifted : (bands, time) array
    """
    n_bands, n_time = data.shape
    shift_samples = int(shift_sec * fs)

    shifted = np.zeros_like(data)
    for i in range(n_bands):
        # Random multiple of shift_samples
        max_shifts = n_time // shift_samples
        n_shifts = np.random.randint(1, max_shifts + 1)
        actual_shift = n_shifts * shift_samples
        shifted[i, :] = np.roll(data[i, :], actual_shift)

    return shifted
```

**Stretch goal:** Get the clustering and one threshold working end-to-end.

---

## Coordination: After Primary Tasks

Once Persons 1-3 finish visualization:
1. **Help test preprocessing:** Run `smooth_env()` on your LFP/EEG data
2. **Help test motifs:** Feed preprocessed data to `get_motifs()`
3. **Document issues:** Note any bugs or edge cases

---

## Background Reading (in `docs/papers/`)

| Paper | Topic |
|-------|-------|
| Buzsaki & Koch 2012 | Origin of EEG, LFP, and spikes |
| Cohen 2017 | Where does EEG come from? |
| Jones 2018 | Neural oscillations: rhythms or bursts? |

---

## Deliverables Checklist

```
notebooks/
├── 01_raster_plots.ipynb          # Person 1
├── 02_eeg_visualization.ipynb     # Person 2
└── 03_lfp_visualization.ipynb     # Person 3

src/spectral_motifs/
├── __init__.py
├── preprocessing.py               # Person 4
└── motifs.py                      # Person 5
```

---

## Final Repo Structure

```
dc-allen-data-acquire/
├── assets/                         # Project proposal
├── data/                           # Downloaded data (git-ignored)
│   └── raw/                        # .nwb recordings
├── docs/
│   └── papers/                     # Background reading
│       ├── Buzsaki Koch 2012 - Origin of EEG, ECoG, LFP, spikes.pdf
│       ├── Where Does EEG Come From... 2017.pdf
│       └── Neural Oscillations... 2018.pdf
├── notebooks/
│   ├── read_Templeton_NWB.ipynb    # Reference (Allen Institute)
│   ├── read_Templeton_NWB.html
│   ├── 01_raster_plots.ipynb       # Person 1
│   ├── 02_eeg_visualization.ipynb  # Person 2
│   └── 03_lfp_visualization.ipynb  # Person 3
├── src/
│   ├── matlab/                     # Original MATLAB (reference only)
│   │   ├── SmoothEnv.m
│   │   ├── butterfilter.m
│   │   └── getMotifs.m
│   ├── scripts/
│   │   └── download_dropbox_data.py
│   ├── spectral_motifs/            # Python port
│   │   ├── __init__.py
│   │   ├── preprocessing.py        # Person 4
│   │   └── motifs.py               # Person 5
│   └── visualization/              # Shared viz utilities (optional)
│       └── __init__.py
├── TASKS.md                        # This file
└── README.md
```
