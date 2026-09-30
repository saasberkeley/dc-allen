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
    └── read_Templeton_NWB.html     (898 KB)
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

## 5. Repo layout

```
assets/                 Project proposal (sow.pdf)
data/                   Downloaded data (git-ignored)
  raw/                  .nwb recordings
  code/                 Reference notebook and HTML
src/scripts/
  download_dropbox_data.py   Downloads the data
.env.example            Template for your local .env (copy to .env)
```

## 6. Reading to get up to speed (in order)

1. [`assets/sow.pdf`](assets/sow.pdf), the project proposal. It is short and explains the goals.
2. The rendered notebook, `data/code/read_Templeton_NWB.html`. It shows what a session's data looks like.
3. Mishler et al. (2026), the motif method: <https://www.biorxiv.org/content/10.64898/2026.02.20.707132v1>
4. Momi et al. (2026), the psilocybin dataset: <https://www.biorxiv.org/content/10.64898/2026.08.14.744865v1>
5. For deeper neuroscience background, *Principles of Neural Science* (5th ed.) is in the shared Dropbox folder. It is not downloaded by the script and is not in this repo. It is optional, so skim the chapters on neurons and electrical signaling if you're curious.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named src` | Run commands from the repo root (the folder containing `src/`), not from inside `src/`. |
| Download stalls or errors | Re-run the same command. It retries and resumes automatically. |
| `No space left on device` | Free up about 30 GB and re-run. |
| `DROPBOX_FOLDER_URL is not set` | You haven't created `.env` yet. Do Step 3. |
| Dropbox link returns 403/404 | The shared link may have expired. Ask the project leads for a new one and update `DROPBOX_FOLDER_URL` in your `.env`. |
