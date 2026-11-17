# Tracker vs static motifs: figures and captions

Mouse 720762, deep prelimbic LFP (PL5). Made by `code/python/notebooks/tracker_vs_static.ipynb`; the numbers behind
every figure are in the CSVs one folder up.

Each entry gives a slide title, a caption, and the one-line takeaway. Q1 = states, Q2 = neurons, S = supporting
figures from 10 Nov.

**Terms**
- **Static motifs**: from the original (strict) and relaxed Mishler pipelines.
- **Tracker outputs**, for each of its 9 library motifs:
  - *expression*: the second-by-second signal after removing a 5-min sliding baseline;
  - *baseline*: that sliding baseline;
  - *spread*: the sliding SD.
- **Opponent groups**: A = neurons more active in the + pole, B = neurons more active in the − pole.

---

## Bottom line

**Q1, states.**
- The tracker follows state better than the static motifs because of its sliding baseline, not because of its motif
  directions. A 5-min baseline on the raw bands, with no motifs, does about as well.
- The new finding is in the poles. Within isoflurane maintenance, a higher dose pushes the + pole, the − pole and
  their midpoint down together (r ≈ −0.7), while their separation stays the same. So dose changes move the
  baseline, not the motif.
- Between awake and isoflurane, the separation grows. That is the known increase in variance.

**Q2, neurons.**
- The tracker's frame does not track opponent neurons better. In like-for-like tests, effect sizes match the
  static frame.
- What predicts opponency is the motif direction:
  - awake: the original/relaxed awake motif;
  - isoflurane: tracker M2, which is nearly the same direction as the relaxed isoflurane motif (cos 0.93).
- Beyond broadband power, per-second tracker expression adds little to population spiking.
- The baseline does carry neural information: the slow opponent-group drift follows the baseline for 5 of 18
  motif × state pairs after correction. Which neurons are involved does not reorganise.

**Method note.** Multi-feature regressions on slow 5-min features are not testable within one session: they fit
time-shifted targets too. Yesterday's "Tracker: both" spiking number (R² 0.67) is partly inflated by this; its
null was 0.17. The tracker's per-second expression result (R² 0.35 under isoflurane, null −0.02) stands.

---

## Q1a_state_association.png
**Title:** The tracker's baseline follows state as well as smoothed band power, and better than the static motifs

**Caption:**
- **What each bar shows:** for each method, the strongest correlation (|Spearman r|) between any one of its
  per-second outputs and isoflurane level, pupil size or running speed.
- **Significance:** the tick is the 95th percentile of the same maximum after shifting the target in time by at
  least 5 min, so methods with more outputs are not favoured. Filled bars beat that null (p ≤ 0.05); hollow bars
  do not.
- **Seconds used:** all seconds, stimulation blocks included.
- **The tracker baseline is the only method that beats the null on every target:**
  - dose over the session: r = 0.80 (null 0.70);
  - dose within maintenance: 0.72 (null 0.62);
  - pupil while awake: 0.59 (null 0.57);
  - pupil under isoflurane: 0.67 (null 0.54);
  - running: 0.26 (null 0.21).
- **The control matches it:** the same 5-min baseline applied to the raw 10 bands, with no motifs, gives
  0.79 / 0.65 / 0.51 / 0.69 / 0.28.
- **Tracker expression carries no state information.** That is by design, since the baseline is removed from it.

**Takeaway:** the state information sits in the slow baseline. The tracker separates it out cleanly, but the motif
directions themselves add little to state tracking.

## Q1b_induction_onset.png
**Title:** Who reacts first to isoflurane? A motif direction found only by the tracker changes about as early as the pupil

**Caption:**
- **What the traces are:** causal 30-s trailing means (past seconds only) around the isoflurane switch-on,
  expressed in units of the 2 minutes before it.
- **"Changes at":** the first second after which the trace stays more than 3 SD away for 30 s.
- **Order of change:**
  - tracker M2 direction: 53 s;
  - pupil: 60 s;
  - relaxed isoflurane motif: 62 s (nearly the same direction as M2);
  - broadband power: 85 s;
  - tracker M1: 88 s;
  - original awake motif: 97 s;
  - running stops: 98 s.
- **Tracker baselines (dashed)** are shown but not timed. They are centred 5-min fits cut at the known switch-on
  time, so they are told when isoflurane starts.

**Takeaway:** one induction gives no statistics, but the isoflurane-type direction (tracker M2 / relaxed
isoflurane motif) moves about 30–45 s before broadband power and the awake motif.

## Q1c_poles_M1.png
**Title:** Along the broadband motif, both poles slide down together as the dose rises

**Caption:**
- **Top panel:**
  - grey: the distribution of tracker M1 expression in each 5-min window (60-s steps, session-wide units, no
    sliding baseline), each column scaled to its peak;
  - red: the + pole (mean above the window median);
  - blue: the − pole (mean below it);
  - dashed: their midpoint.
- **Second panel:** the separation between the poles, and their asymmetry.
- **Bottom panels:** isoflurane level, pupil and running.
- **What happens:**
  - Awake, the distribution is narrow.
  - After induction it widens: the separation grows from ≈0.8 to 3–4.
  - When the dose rises to 3.5 % (about 80–97 min), the whole distribution and both poles drop by about
    6 units, then recover when the dose returns to 1.5 %.

**Takeaway:** the motif's poles move with dose, and they move together.

## Q1d_poles_M2.png
**Title:** The isoflurane-specific motif (M2): poles rise at induction and follow running while awake

**Caption:**
- Same layout as Q1c, for tracker M2, the direction that appears only from induction onward.
- Over the whole session, its + pole rises with dose (r = 0.63, p = 0.002). The − pole and midpoint move the same
  way but are not significant (0.54 and 0.60).
- Its separation grows with dose (0.53), as on every axis.
- While awake, both poles track running speed (r ≈ 0.5, p = 0.002).

**Takeaway:** a motif can be absent in a state (M2 is never detected awake) yet still have its axis position
follow behaviour in that state.

## Q1e_poles_original.png
**Title:** The original static motif's axis shows the same pole movement

**Caption:**
- Same layout as Q1c, for the original pipeline's awake motif direction (cos 0.68 with tracker M1).
- Within maintenance, its + pole, − pole and midpoint fall with dose (r = −0.68, −0.71, −0.74), and its
  separation does not change.

**Takeaway:** pole movement belongs to the data, not to the tracker. The tracker's contribution is to measure it
and separate it from moment-to-moment expression. The static pipeline z-scores it away.

## Q1f_pole_measures_vs_state.png
**Title:** Mapping each kind of pole movement onto state

**Caption:**
- **What each cell shows:** the Spearman r across 5-min windows between a pole measure and dose or behaviour.
  `*` marks p ≤ 0.05 against a null that shifts the window series by at least 10 windows.
- **Three consistent mappings:**
  1. **Awake → isoflurane = spread.** Separation rises with dose over the session: r = 0.53–0.67 on every axis.
  2. **Dose within maintenance = baseline shift.** The + pole, − pole and midpoint all fall together
     (r ≈ −0.65 to −0.77 for M1, M3 and the original), while separation does not change.
  3. **Pupil under isoflurane = the − pole.** The − pole tracks pupil (r ≈ −0.7) on M1, M3 and the original
     axis, more strongly than the + pole does.
- Asymmetry (one pole moving on its own) is mostly not significant. Exceptions:
  - M3 with awake pupil (0.64) and with pupil under isoflurane (0.37);
  - M2 and M3 with dose over the whole session (0.61, 0.55).

**Takeaway:** motif → state, baseline → dose within a state, spread → awake vs anaesthetised.

**Update from the window tests (window_tests.ipynb, slides 15–16):**
- "Baseline → dose within maintenance" mostly does not beat a 60-s block shuffle within condition segments. It
  is driven by the stimulation-block segment, where dose was also raised, so treat it as confounded.
- "− pole → pupil under isoflurane" does beat the shuffle at every window length tested.

**Caveat:** the within-maintenance dose changes happen during the isoflurane electrical-stimulation block, so
dose and stimulation overlap in time.

## Q1g_presence_by_state.png
**Title:** Motif → state map: which tracker motifs exist in which state

**Caption:**
- **What each cell shows:** the share of 5-min windows in which each library motif was detected (a window motif
  within |cos| ≥ 0.8 of it).
- **Motifs by state:**
  - **M1** (broadband): present in 100 % of isoflurane windows; 20 % of spontaneous awake windows, but 96 %
    during awake electrical stimulation.
  - **M2**: never awake; 35–75 % under isoflurane.
  - **M7**: mostly at induction (67 %).
  - **M3 and M9**: mostly during awake visual stimulation.
- **The static pipelines:** one motif awake; under isoflurane, none (strict) or one (relaxed).

**Takeaway:** the tracker gives a graded motif → state map where the static pipelines give one yes/no per state.

**Caveat:** only 3 windows fall in induction.

---

## Q2a_opponency_like_for_like.png
**Title:** Same direction, two frames: the tracker frame does not change how well a motif tracks opponent neurons

**Caption:**
- **Setup:**
  - Each direction's expression is computed in the static frame (within-state z) and the tracker frame (5-min
    baseline removed) on the same spontaneous seconds.
  - Groups A and B are chosen on training blocks.
  - The y-axis is the held-out correlation of expression with A − B.
  - Filled points: p ≤ 0.05 against a time-shift null (tick).
- **Right panels:** broadband power is regressed out of expression and neurons first.
- **Awake:** the original/relaxed awake motifs give r ≈ 0.31–0.35 in both frames. The tracker frame narrows
  the null, so more points reach significance.
- **Isoflurane:**
  - the best directions are tracker M2 (0.49 static frame, 0.45 tracker frame) and the relaxed whole-session
    motif (0.43 / 0.40);
  - both survive the broadband control (M2: 0.47 / 0.33);
  - tracker M1 (broadband) loses its opponency once broadband power is removed, as expected.

**Takeaway:** for opponency, which direction you use matters, not which frame. The tracker's useful
contribution was finding the M2 direction, which the relaxed isoflurane run also found (cos 0.93).

## Q2b_spiking_gain_over_broadband.png
**Title:** Beyond broadband power, per-second tracker expression adds little to spiking

**Caption:**
- **Left:** held-out R² of broadband power alone. It explains 0.54 of the isoflurane population rate (0.31 in
  the tracker frame) and nothing awake.
- **Right:** the R² each pipeline adds on top of broadband power in its own frame. The tick is the time-shift
  null; a black edge means above it.
- **Isoflurane population rate:**

  | Pipeline | Gain over broadband |
  |---|---|
  | Relaxed | +0.14 |
  | All 10 bands | +0.14 |
  | Original | +0.05 |
  | Tracker expression | +0.03 |

- **Awake ensemble 1 (32 units):** the original motif adds the most (+0.14), then relaxed (+0.07). Tracker
  expression makes the prediction worse.
- **Method:** per-second predictors only (see the method note).

**Takeaway:** for predicting how much the population fires, the static relaxed motifs carry as much as the full
band set, and the tracker's fast expression adds the least.

## Q2c_baseline_window_sweep.png
**Title:** Why the tracker loses the awake spiking relation: it is slow, and any sliding baseline removes it

**Caption:**
- **Setup:** residuals rebuilt with sliding baselines of 1–20 min, compared with no baseline (static frame).
  Filled points: p ≤ 0.05.
- **Top, opponency:** held-out r(A − B).
  - It barely depends on the window for the original motif awake (0.26–0.33).
  - It drops to ≈0 for the M2/M3 directions awake with any baseline.
- **Bottom, population-rate pole contrast:**
  - awake: 0.48 z with no baseline, ≈0 with any window up to 20 min;
  - isoflurane: unaffected.

**Takeaway:** the awake LFP–firing relation is a co-drift over many minutes, which the tracker assigns to the
baseline by design. A shorter window would not bring it back. It has to be read from the baseline.

## Q2d_baseline_vs_neurons_all_motifs.png
**Title:** Baseline shifts go with slow opponent-group drift for some motifs, but not with reorganised membership

**Caption:**
- **Coverage:** every tracker motif in both states, with Benjamini–Hochberg correction across 18 tests per row.
  Dark red: q ≤ 0.05; pink: p ≤ 0.05 uncorrected only; ticks: the null's 95th percentile.
- **(a) Slow opponency:** does the slow part of the A − B group difference follow the motif's baseline? It
  survives correction for:
  - awake M3, M4 and M6 (r = 0.46–0.56);
  - isoflurane M2 (0.29) and M8 (0.56).
- **(b) Reorganisation:** do different neurons go on/off at low vs high baseline? Nothing survives correction.
  The closest are isoflurane M3 and M8 (q = 0.09).
- **Caveats:**
  - Groups A and B are small for several motifs (2–13 neurons).
  - Awake has only ~1000 spontaneous seconds, i.e. a few independent 5-min stretches.
  - The significant awake motifs are rarely detected awake (Q1g).

**Takeaway:** baseline shifts partly move the same opponent groups up and down. There is no solid evidence that
they recruit different neurons.

---

## S1_opponent_neuron_split.png (from 10 Nov)
**Title:** What "opponent neurons" means

**Caption:**
- **What each bar is:** one PL5 neuron's activity in a motif's + pole minus its − pole (population mean
  removed).
- Red: significantly up in the + pole. Blue: significantly up in the − pole. The panel titles give the chance
  counts.

**Takeaway:** the motifs split PL5 neurons into two opposing groups.

## S2_opponent_groups_traces.png (from 10 Nov)
**Title:** Opponent groups follow the motif over time

**Caption:**
- A 400-s excerpt under isoflurane: tracker M2 expression (black) with the mean activity of group A (red) and
  group B (blue).
- When the motif is in its + pole, A rises and B falls, and vice versa.
- The groups here are chosen from the whole state; the held-out numbers are in Q2a.

**Takeaway:** the opponency is visible by eye, not only in statistics.
