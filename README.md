# SAAS x Allen Institute: Brain Recordings Across States

This repo holds the data-acquisition code for a Berkeley SAAS (Student Association for Applied Statistics) project with the Allen Institute. **You don't need any biology or neuroscience background to get started.** This page explains the project in plain language and then walks you through downloading the data.

The full project proposal is in [`assets/sow.pdf`](assets/sow.pdf).

---

## 1. The project in plain language

### What was recorded?
Researchers at the Allen Institute recorded the electrical activity of **mouse brains**, using probes called **Neuropixels**. A Neuropixels probe is a thin needle with hundreds of tiny electrodes. It is inserted into the brain and listens to many brain regions at once.

The mice were recorded in different **states**:

| State | What it is |
|---|---|
| **Awake** | Normal, alert baseline. |
| **Anesthetized** (two different sedatives) | The animal is sedated. |
| **Psilocybin** | The animal was given the psychedelic compound found in "magic mushrooms". |
| **Ketanserin** | A drug that *blocks* the receptor psilocybin acts on. It is the control, to show that psilocybin's effects really come through that receptor. |

### What kinds of signals are in the data?
Each recording contains several streams, all recorded at the same time:

- **Spikes.** Each neuron "fires" brief electrical pulses, called spikes. The data records the time of each spike and which neuron produced it. Think of it as a list of timestamps per neuron.
- **LFP (local field potential).** This is the smooth, wavy signal you get by averaging the electrical activity near each electrode. It looks like a continuous squiggle and contains rhythms ("brain waves") at different speeds, measured in Hz. Slow waves are roughly 1–4 Hz, and fast ones can be 30 Hz or more.
- **EEG.** A coarser brain signal measured from the surface of the head.
- **Pupil size and behavior.** Video-derived measurements of the animal's pupil and movement. These are useful for telling what state the animal is in.

### What are we trying to figure out?
A recent paper (Mishler et al., 2026) describes a method for finding **spectral motifs**. A motif is a *recurring pattern of brain-wave activity across several frequencies at once*. Motifs tend to come in **opposing pairs**: when one goes up, the population of neurons tends to fire more, and when its partner goes up, the population tends to fire less. Each pair is linked to a different group of neurons.

The method was tested in rats learning a task. **Our question is: how do these motifs change when the brain is in a different state (awake, sedated, or on psilocybin)?**

### What will we build?
1. **D1, Data access layer.** Code that loads any recording into clean arrays with the state (awake, sedated, and so on) labelled. It also includes a written "data dictionary" explaining every field.
2. **D2, Verified motif pipeline.** Our own implementation of the motif method. We first check that it reproduces a published result, *before* running it on anything new.
3. **D3, State comparison and code release.** Statistics, figures, and a write-up comparing motifs across states.

### Where are we now?
The semester runs 14 Sep – 13 Dec 2026. We are in **Week 3 (data access)**. This repo currently contains the script that downloads the data. The loaders come next.

---

## 2. Glossary

| Term | Meaning |
|---|---|
| **NWB** (Neurodata Without Borders) | A standard file format (`.nwb`) for neuroscience data. One file holds one recording session, with spikes, LFP, and metadata inside. Files are big, often 10–20 GB. |
| **Session** | One recording of one mouse on one day. |
| **Neuropixels** | The high-density recording probe. |
| **Unit** | One neuron, as identified from the spike data. |
| **LFP** | Local field potential: the slow wavy electrical signal (see above). |
| **Hz** | Cycles per second, the speed of a brain wave. |
| **Epoch** | A labelled stretch of time within a recording (for example "awake" or "post-drug"). |
| **Motif** | A recurring multi-frequency pattern in the LFP (see above). |

---

## 3. Setup: get the code and data

You need a Mac or Linux machine (Windows users: use WSL), **Python 3.9 or newer**, and **about 30 GB of free disk space** for the data.

### Step 1: Get the repo
```bash
git clone git@github.com:saasberkeley/dc-allen-data-acquire.git
cd dc-allen-data-acquire
```

### Step 2: Check Python
```bash
python3 --version
```
You should see `Python 3.9` or higher. If not, install Python from <https://www.python.org/downloads/>.

### Step 3: Add the Dropbox link
The data lives in a private shared Dropbox folder. The link is not stored in the repo, so you have to add it yourself.

1. Ask the project leads for the **shared folder link**. It must contain an `rlkey=...` part.
2. Create your own `.env` file from the template:
   ```bash
   cp .env.example .env
   ```
3. Open `.env` and paste the link after the `=`:
   ```
   DROPBOX_FOLDER_URL=https://www.dropbox.com/scl/fo/.../...?rlkey=...&dl=0
   ```

`.env` is git-ignored, so your copy stays on your machine. **Never commit the link or post it publicly.**

### Step 4: Download the data
The download script uses only the Python standard library, so there is nothing to install for this step.

First, preview what will be downloaded, with sizes:
```bash
python3 -m src.scripts.download_dropbox_data --list
```

Then download everything:
```bash
python3 -m src.scripts.download_dropbox_data
```

Or download in pieces:
```bash
python3 -m src.scripts.download_dropbox_data --only code   # small: notebook + HTML (about 1.5 MB)
python3 -m src.scripts.download_dropbox_data --only raw    # large: two .nwb recordings (about 28.6 GB)
```

**Notes:**
- The large files take a while. The script prints progress.
- It is safe to stop it (Ctrl+C) and run it again. It resumes where it left off.
- If a file is already fully downloaded, it is skipped.
- Partial downloads end in `.part`. Don't open those. They become normal files when finished.

### Step 5: Check that it worked
```bash
ls -lh data/raw data/code
```
You should see:

```
data/
├── raw/
│   ├── 688277-20230921-v001.nwb    (12.11 GB)
│   └── 692644-20230907-v001.nwb    (16.52 GB)
└── code/
    ├── read_Templeton_NWB.ipynb    (607 KB)
    ├── read_Templeton_NWB.html     (898 KB)
    └── SmoothEnv.m, butterfilter.m, getMotifs.m   (Mishler MATLAB code)
```

The `data/` folder is git-ignored. **Never commit the data files.**

---

## 4. Optional: open the data yourself

The notebook `data/code/read_Templeton_NWB.ipynb` shows how to read these files. To run it, set up a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows (WSL): same command
pip install pynwb h5py numpy pandas matplotlib jupyter
jupyter notebook data/code/read_Templeton_NWB.ipynb
```

Don't want to run anything? Open `data/code/read_Templeton_NWB.html` in your browser to see the notebook with its outputs.

The recording files are very large. Reading them with `pynwb` loads data lazily, so you can look at small slices without filling up your memory. Avoid commands that copy a whole signal into memory.

---

## 5. The Mishler MATLAB reference code

Dr. Mishler (first author of the motif paper) shared the original MATLAB implementation. It has **not** been ported to Python yet, and porting it is our D2 task. The files are in `data/code/`:

| File | Role |
|---|---|
| `SmoothEnv.m` | Preprocessing. Turns one raw LFP/EEG trace into smoothed, z-scored band-amplitude envelopes. |
| `butterfilter.m` | Helper for `SmoothEnv`. Designs a 6th-order Butterworth bandpass filter. |
| `getMotifs.m` | The motif-finding algorithm. |

### How to run it (per Mishler)
1. For each animal/session, call `SmoothEnv(LFP, LFPfs, freqBands)`.
2. Store the resulting matrices in a **column cell array**, one cell per animal/session.
3. Pass that cell array and the sampling rate to `getMotifs(LFPIn, LFPfs)`.

```matlab
freqBands = [1 4; 3 6; ...];             % N x 2, [low high] in Hz. Bands should overlap.
LFPIn = cell(nSessions, 1);
for k = 1:nSessions
    LFPIn{k} = SmoothEnv(lfp{k}, fs, freqBands)';   % NOTE the transpose, see below
end
finalMotifs = getMotifs(LFPIn, fs);      % bands x (2 * number of motifs)
```

### Step 1: `SmoothEnv(LFP, LFPfs, freqBands)`
Input is one continuous trace and an `N x 2` matrix of band edges. Steps:
1. **Bandpass** the trace into each band (6th-order Butterworth, zero-phase `filtfilt`).
2. **Envelope.** Take the Hilbert transform and keep its magnitude, which is the instantaneous amplitude of each band.
3. **Smooth** each envelope with a Gaussian window of about 5 s, applied forward and backward.
4. **Trim edges.** The first and last 10 s are set to `NaN` to hide filter edge artifacts.
5. **Z-score** each band over time, ignoring the `NaN`s.

Output is **time x bands**. Positive means the band is stronger than its own session average, and negative means weaker.

### Step 2: `getMotifs(LFPIn, LFPfs)`
Input is a column cell array. Each cell is **bands x time**. All sessions need the same bands in the same order, but durations can differ. Output is a **bands x motifs** matrix. The first half are the motifs, and the second half are their exact sign-flipped opposites. This is the "opposing pairs" idea from the paper.

What it does:
1. **Subsample** to about 1 sample per second and concatenate all sessions.
2. **Binarize.** Take `sign()` of every band, so each time point becomes a +/- pattern across bands. Only *which bands are above or below average* matters, not by how much.
3. **Drop trivial points** where every band has the same sign (all high or all low).
4. **Cluster.** Compute correlation distance between all pairs of time points, then run average-linkage hierarchical clustering.
5. **Merge branches.** Walk up the tree. Two sibling clusters merge if their *expression over time* is correlated at or above a threshold `r`. Here, expression means the dot product of every time point with the cluster's mean pattern. Re-cluster the merged templates and repeat until the number of motifs stops changing.
6. **Repeat for thresholds** `r = 0.4, 0.5, 0.6, 0.7, 0.8`.
7. **Build null data.** Do steps 1-6 again on 20 shuffled copies. In each copy, every band is circularly shifted by a different multiple of 20 s. This breaks the relationships between bands but keeps each band's own structure.
8. **Keep only significant motifs.** A real motif must (a) have more time points than the largest motif found in any null at that threshold, and (b) appear in **every** session.
9. **Remove duplicates across thresholds** with uncentered PCA. Keep components that at least 2 candidate motifs map to.
10. **Re-estimate final templates** from the original data. Average the time points that express each axis, with the opposite pole sign-flipped. Then append the negated copies.

### Things to know before porting
- **Orientation mismatch.** `SmoothEnv` returns time x bands, but `getMotifs` expects bands x time per cell. You must transpose in between (the example above does).
- **`NaN` edges.** `SmoothEnv` leaves 10 s of `NaN` at each end, and `getMotifs` has no `NaN` handling. We expect you need to trim those samples before calling `getMotifs`. **This is unconfirmed, so ask Mishler.**
- **Randomness.** The null shifts use `randperm` with no fixed seed, so results vary a little between runs. Set a seed in the Python port.
- **Slow.** The tree-walking loops are plain MATLAB loops over every time point and are written for clarity, not speed. A Python version will likely need `scipy.cluster.hierarchy` plus some vectorizing.
- **Toolboxes.** `butterfilter.m` uses `fdesign.bandpass` (Filter Design / DSP System Toolbox). `getMotifs.m` uses `pdist`, `linkage` and `pca` (Statistics Toolbox). The Python equivalents are `scipy.signal.butter(..., output='sos')` with `sosfiltfilt`, `scipy.signal.hilbert`, and `scipy.cluster.hierarchy` / `sklearn.decomposition.PCA`.
- **Gaussian window.** `gausswin(N,1)` means the window spans only +/-1 standard deviation, which is wider than a typical "5 s Gaussian". Match this exactly in Python (for example `scipy.signal.windows.gaussian` with `std = (N-1)/2`) when checking against the MATLAB output.
- **Chosen parameters.** The band edges (`freqBands`) are not specified in the code. Take them from the paper.
- **All-sessions rule.** A motif must appear in every session, so with only two NWB recordings this is a strict filter.
- **Validation target.** D2 requires reproducing a published result first. Ask Mishler for a small example input and output to compare against.

---

## 6. Repo layout

```
assets/                 Project proposal (sow.pdf)
data/                   Downloaded data (git-ignored)
  raw/                  .nwb recordings
  code/                 Reference notebook, HTML, and Mishler MATLAB code (.m)
src/scripts/
  download_dropbox_data.py   Downloads the data
.env.example            Template for your local .env (copy to .env)
```

## 7. Reading to get up to speed (in order)

1. [`assets/sow.pdf`](assets/sow.pdf), the project proposal. It is short and explains the goals.
2. Section 5 above, and the MATLAB code it describes.
3. The rendered notebook, `data/code/read_Templeton_NWB.html`. It shows what a session's data looks like.
4. Mishler et al. (2026), the motif method: <https://www.biorxiv.org/content/10.64898/2026.02.20.707132v1>
5. Momi et al. (2026), the psilocybin dataset: <https://www.biorxiv.org/content/10.64898/2026.08.14.744865v1>
6. For deeper neuroscience background, *Principles of Neural Science* (5th ed.) is in the shared Dropbox folder. It is not downloaded by the script and is not in this repo. It is optional, so skim the chapters on neurons and electrical signaling if you're curious.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named src` | Run commands from the repo root (the folder containing `src/`), not from inside `src/`. |
| Download stalls or errors | Re-run the same command. It retries and resumes automatically. |
| `No space left on device` | Free up about 30 GB and re-run. |
| `DROPBOX_FOLDER_URL is not set` | You haven't created `.env` yet. Do Step 3. |
| Dropbox link returns 403/404 | The shared link may have expired. Ask the project leads for a new one and update `DROPBOX_FOLDER_URL` in your `.env`. |
