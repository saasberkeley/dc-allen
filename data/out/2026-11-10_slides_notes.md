# Speaker notes for 2026-11-10_slides_figures/

One line per figure: what it shows and what to say. Full captions:
- figures 01–11 and 18–19: `2026-11-10_tracker_vs_static/CAPTIONS.md`;
- figures 12–17: `2026-11-10_window_tests/` (notebook `window_tests.ipynb`).

## Part 1: does the tracker follow states better?
- **01:** The tracker's baseline tracks dose, pupil and running better than the static motifs. A 5-min baseline on
  the raw bands does as well, so the gain comes from separating slow from fast, not from the motif directions.
- **02:** Along the broadband motif, both poles move. They spread apart at induction and slide down together when
  the dose goes to 3.5 %.
- **03:** Three kinds of pole movement, each mapped to state:
  - spread: awake vs isoflurane;
  - baseline: dose (but see 16);
  - − pole: pupil under isoflurane.
- **04:** Motif → state map: M1 is everywhere under isoflurane, M2 only under isoflurane, M7 at induction.
- **05:** The isoflurane-type direction (M2) changes about as early as the pupil (~55 s), before broadband power.
  This is one induction, so there is no test.

## Part 2: is it better grounded in neurons?
- **06–07:** What opponency means: motif poles split PL5 neurons into two groups that alternate.
- **08:** The same direction gives about the same opponency in the static frame and the tracker frame. The
  direction matters, not the frame. M2 is the best isoflurane direction, and it survives removing broadband power.
- **09:** Beyond broadband power, the relaxed motifs add the most to spiking (+0.14 R²); tracker expression adds
  little (+0.03).
- **10:** The awake LFP–spiking relation is a slow co-drift over many minutes. Every sliding baseline from 1 to
  20 min removes it, so it has to be read from the baseline.
- **11:** The baseline follows slow opponent-group drift for 5 of 18 motif × state pairs. The neurons involved do
  not reorganise.

## Part 3: is the 5-minute window creating these patterns?
- **12, synthetic ground truth:** every sliding window trades two errors.
  - Short windows wander even with no drift: the 5-min baseline moves by about 1/5 of the pole distance.
  - Long windows leak real drift into the expression: 43 % at 5 min, 71 % at 10 min.
  - 5 min recovers a planted drift about as well as anything (r = 0.95).
- **13:** The state associations barely change between 1- and 20-min baselines, so they are not specific to
  5 min.
- **14:** Inside a condition, the real baseline moves more than data with its 60-s blocks shuffled.
  - The extra is clear under isoflurane: M1 1.60 vs 0.97 at 5 min.
  - About 60 % of the apparent movement is the size window noise alone would produce.
- **15:** The test that matters.
  - **Pooled over a state:** most associations are no stronger than the shuffle. They come from differences
    between conditions (stimulation blocks, dose steps), not from tracking within one.
  - **Inside conditions:**
    - awake pupil and running clearly beat the shuffle with short windows (pupil 0.64 vs 0.30 at 1 min);
    - dose inside the stimulation block beats it at 1, 2 and 10 min, but not at 5 min;
    - pupil inside maintenance does not beat it.
- **16:** Pole mappings, by result:
  - **Robust:** "− pole follows pupil under isoflurane" beats the shuffle for every window on M1, M3 and the
    original axis.
  - **Not robust:** "baseline follows dose within maintenance" mostly does not beat it. It is driven by the
    stimulation-block segment, where dose was also raised, so dose and stimulation are confounded.
  - **Expected:** separation vs dose fails the shuffle by design, because it is an awake vs isoflurane
    difference.
- **17:** Re-running the tracker with baselines of 2, 10 and 20 min and detection windows of 3 and 10 min.
  - M1 and M2 come back every time (|cos| ≥ 0.94), with the same state map: M1 in every isoflurane window,
    M2 isoflurane-only.
  - The minor motifs M3–M9 depend on the window: the library has 3–10 motifs.

## Backup
- **18–19:** pole plots for M2 and for the original static motif axis. Same layout as 02.

## Bottom line on the window doubt
- **Not created by the window:** the main motifs (M1, M2), the state map, the opponency results, the − pole /
  pupil link, and the awake/isoflurane difference.
- **Partly created by window noise or condition differences:**
  - about 60 % of the size of within-condition baseline movement;
  - pooled baseline–state correlations;
  - "baseline follows dose", which is confounded with the stimulation block.
- **Depends on the window:** the minor motifs M3–M9.
- **A better default:** a 2-min baseline lowers chance correlations and keeps the real within-condition links.
  But it wanders more (figure 12), so it suits short-timescale questions.
