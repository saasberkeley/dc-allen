# EEG analysis: isoflurane and psilocybin sessions

## Scope

This report reviews the processed windows, figures, scripts, and result tables on
`gautam-branch`. The two drug conditions come from different mice, so they are
separate before/after experiments rather than a controlled head-to-head drug
comparison.

## What the EEG spectrum shows

The spectrum's horizontal axis is frequency, not time. Moving from left to right
means moving from slow EEG rhythms to fast EEG rhythms. The downward slope in
both panels is expected: biological field potentials generally contain more
low-frequency than high-frequency power.

### Isoflurane session (mouse 688277)

Compared with the awake window, the selected anesthetized window shows:

- lower power around 2–10 Hz;
- higher power through much of the beta range (13–30 Hz); and
- substantially lower power above roughly 40 Hz.

Isoflurane therefore did not uniformly reduce every frequency. It redistributed
power across bands. The narrow peak around 60 Hz is mains electrical noise and
should not be interpreted as neural activity.

### Psilocybin session (mouse 692644)

The post-psilocybin spectrum is most clearly elevated at low frequencies. The
difference between the before and after curves becomes smaller at higher
frequencies. Gautam's detector found no sustained broadband power drop comparable
to the isoflurane session.

## Isoflurane power timing

Time zero is Gautam's detected isoflurane onset: the first point at which the
recorded isoflurane level exceeds 1%. The concentration rises to approximately
5% during induction and later steps down toward a maintenance concentration.

Gautam defines a sustained drop as band power falling below the pre-onset mean
minus three baseline standard deviations for at least 20 seconds, after 10-second
smoothing. Using that definition:

| Band | Sustained-drop onset after isoflurane onset |
|---|---:|
| Theta | 237 s |
| High gamma | 244 s |
| Low gamma | 264 s |
| Beta | 267 s |
| Broadband | 268 s (4 min 28 s) |
| Alpha | 272 s |
| Delta | No sustained drop detected |

The broadband drop occurs near the change from the high induction dose to the
lower maintenance dose. This is consistent with the brain settling into a stable
anesthetized state, but EEG alone does not establish the exact moment the animal
lost consciousness.

### Initial-rise measurement still required

For this analysis, "initial rise" means the early broadband EEG change around
isoflurane onset, not the response to individual electrical stimuli. The existing
analysis detects sustained decreases only; it does not calculate a rise.

The continuous NWB recording was reanalyzed using Gautam's settings. A sustained
rise was defined as power exceeding the pre-onset mean plus three baseline
standard deviations for at least 20 seconds, with the same 10-second smoothing
used by the drop detector. The resulting threshold was +4.94 dB, and no
post-onset interval crossed it for 20 seconds. Therefore, the apparent initial
increase is a fluctuation rather than a statistically sustained rise under this
definition.

During the first minute, broadband power peaked at +1.41 dB approximately 31
seconds after onset. The largest pre-drop peak was +2.01 dB at approximately 145
seconds. The verified sustained drop began at 268 seconds, 123 seconds after that
peak. The defensible comparison is therefore: a modest, nonsustained early
increase followed by a clear sustained broadband decrease about 4 minutes 28
seconds after isoflurane onset.

## Verified V1 band changes under isoflurane

The values below compare the selected isoflurane-anesthetized window with the
awake window for the four valid EEG channels labelled VISp:

| V1 band | Change |
|---|---:|
| Delta (1–4 Hz) | -1.0 dB |
| Theta (4–8 Hz) | -2.6 dB |
| Alpha (8–13 Hz) | -0.7 dB |
| Beta (13–30 Hz) | +3.3 dB |
| Low gamma (30–55 Hz) | +0.7 dB |
| High gamma (65–100 Hz) | -4.9 dB |

The higher V1 power is driven primarily by beta, with a much smaller low-gamma
increase. Theta and high gamma decrease, so it would be inaccurate to say that
isoflurane raises V1 power at every frequency.

## Methods and limitations to report

- The spectra use Welch-style estimates with two-second Hann segments and 50%
  overlap.
- Artifact rejection uses a threshold derived from the baseline window; Gautam
  disables rejection in the anesthetized window to avoid deleting genuine
  burst-suppression events.
- Frequencies from 55–65 Hz are excluded from band totals because of 60 Hz noise.
- The processed state windows are two minutes long and downsampled to 500 Hz.
- The selected windows strongly influence the comparison.
- Each drug session contains one mouse, preventing population-level conclusions.
- Irene's final timestamps are not present on any currently fetched repository
  branch and should be requested directly, then checked against the NWB-derived
  anchors.
- Gautam's current multi-channel aligned LFP/raster figure is accepted as the
  working final format. Identical neuron ordering across states should still be
  confirmed.

## Short meeting summary

In the isoflurane session, EEG power did not simply decrease everywhere. The
selected anesthetized period had greater beta power but lower theta and high-gamma
power, and the sustained broadband decrease began approximately 268 seconds after
isoflurane onset. An apparent early rise peaked at +1.41 dB in the first minute
but did not satisfy the sustained-rise criterion. In V1, beta was the principal
band driving the increase. The
psilocybin session instead showed its clearest post-injection increases at low
frequencies and no comparable sustained broadband drop. These are single-mouse,
separate-session observations and should be treated as preliminary.
