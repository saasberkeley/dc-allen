const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun, Table, TableRow, TableCell,
  WidthType, ShadingType, AlignmentType, LevelFormat, BorderStyle, PageBreak, Footer, PageNumber,
  TableOfContents,
} = require("docx");

const OUT = process.argv[2];
// Python's json writes NaN, which is not valid JSON
const load = (...p) => JSON.parse(fs.readFileSync(path.join(OUT, ...p), "utf8").replace(/\bNaN\b/g, "null"));
const R = load("results.json");
const B = load("bootstrap.json");
const C = load("diagnostics", "calibration.json");
const D = load("diagnostics", "null_size_diagnosis.json");
const SW = load("all_channels", "sweep_summary.json");
const FIG = path.join(OUT, "figures");
const sweepSection = require("./sweep_section.js");

const W = 9360; // content width in DXA (US Letter, 1" margins)
const f2 = (x) => Number(x).toFixed(2);
const f3 = (x) => Number(x).toFixed(3);
const f0 = (x) => Math.round(Number(x)).toString();
const pct = (x) => `${Math.round(100 * x)}%`;
const median = (v) => { const s = [...v].sort((a, b) => a - b); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };

// --- inline markup: **bold**, *italic*
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text), spacing: { after: 120, line: 276 }, ...opts });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const bullets = (items, level = 0) => items.map((t) => new Paragraph({ numbering: { reference: "bullets", level }, children: runs(t), spacing: { after: 60, line: 276 } }));
const numbered = (items) => items.map((t) => new Paragraph({ numbering: { reference: "numbers", level: 0 }, children: runs(t), spacing: { after: 60, line: 276 } }));

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, i, header) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: header ? { fill: "DCE6F1", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: runs(String(text), { size: 18, bold: header || undefined }) })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, i, false)) }))],
  });
}
const gap = () => new Paragraph({ children: [], spacing: { after: 60 } });

let figNo = 0;
function figure(file, caption, widthPx = 620) {
  const buf = fs.readFileSync(file.includes("/") ? path.join(OUT, file) : path.join(FIG, file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  figNo += 1;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: widthPx, height: Math.round(widthPx * h / w) },
        altText: { title: `Figure ${figNo}`, description: caption, name: file } })] }),
    new Paragraph({ spacing: { after: 200 }, children: runs(`**Figure ${figNo}.** ${caption}`, { size: 18, color: "404040" }) }),
  ];
}

// ---------------------------------------------------------------- data
const tl = R.timeline;
const runsBy = (sig, win, ver) => R.motif_runs.find((r) => r.signal === sig && r.window === win && r.version === ver);
const label = { eeg13: "EEG ch13 (frontal, MOs)", eeg10: "EEG ch10 (somatosensory, SSp-ll)", lfp_PL5: "LFP PL5 (deep prelimbic)" };
const sigs = ["eeg13", "eeg10", "lfp_PL5"];
const wins = ["awake", "isoflurane", "full"];
const usable = (win) => { const r = runsBy("eeg13", win, "strict"); return { sec: r.session_seconds.reduce((a, b) => a + b, 0), chunks: r.n_sessions }; };
const cell = (r) => r.n_motifs === 0 ? "0" : r.motif_support.map((s, i) => `${r.n_motifs} (p ${f3(s.min_p)}${r.broadband_like[i] ? ", broadband" : ""})`).join("; ");
const lfpChan = R.signals.find((s) => s.key === "lfp_PL5").channel;
const units = R.pl5_units;
const stim = R.stim_response;
const db = R.spectra_band_power_db_change;
const guard = R.guardrail;
const guardHits = guard.reduce((a, g) => a + g.n_motifs, 0);
const perm = R.permutation_tests;
const spikes = R.spikes_by_lfp_pole;
const eegLfp = R.eeg_vs_lfp;
const boot = Object.fromEntries(B.results.map((r) => [r.signal, r]));
const diag = (sig, win) => D.find((d) => d.signal === sig && d.window === win);
const cal = (data, cfg) => C.find((c) => c.data === data && c.config.startsWith(cfg));
const iso13 = eegLfp.find((e) => e.window === "isoflurane" && e.eeg === "eeg13");
const iso10 = eegLfp.find((e) => e.window === "isoflurane" && e.eeg === "eeg10");
const spA = spikes.find((s) => s.state === "awake"), spI = spikes.find((s) => s.state === "isoflurane");
const bandPeaks = (sig) => db[Object.keys(db).find((k) => k.startsWith(sig))];

// ---------------------------------------------------------------- content
const c = [];
c.push(new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("Spectral motifs in mouse 720762: awake, isoflurane and the full recording")] }));
c.push(P(`Naman Rudrakshi · SAAS × Allen Institute · 1 November 2026`, { spacing: { after: 40 } }));
c.push(P(`Data: ${R.file} (awake → isoflurane, EEG + Neuropixels). Code: code/python/src/ on branch naman-mouse720762-states. All outputs: data/out/2026-11-01_mouse720762/.`, { spacing: { after: 240 } }));

c.push(H1("Summary"));
const a = usable("awake"), i = usable("isoflurane"), fu = usable("full");
const pl5boot = boot.lfp_PL5, e13boot = boot.eeg13, e10boot = boot.eeg10;
c.push(...bullets([
  `**All mandatory deliverables are done for the awake, isoflurane and full-recording windows:** EEG motif detection (frontal channel 13 and somatosensory channel 10), LFP motif detection (deep prelimbic cortex, PL5), EEG and LFP power spectra, a spike raster of the PL5 units, and statistical tools that extend the original analysis.`,
  `**Electrical stimulation contaminates both EEG and LFP,** so every analysis uses only stimulation-free data: ${f0(a.sec / 60)} min awake and ${f0(i.sec / 60)} min under isoflurane. Our state windows differ from Justin's, which spanned the stimulation and visual-stimulus blocks.`,
  `**Mishler's original algorithm finds a motif in only 1 of 9 signal × window combinations.** We traced part of this to a bias: under isoflurane, shuffled data keeps up to ${pct(diag("lfp_PL5", "isoflurane").n_null_median / diag("lfp_PL5", "isoflurane").n_real - 1)} more usable seconds than the real data, so it grows bigger clusters. The relaxed version corrects this and finds motifs in ${R.motif_runs.filter((r) => r.version === "relaxed" && r.n_motifs).length} of 9 combinations, without finding any in motif-free test data.`,
  `**Deep prefrontal LFP has a reproducible motif in both states** (found in ${pct(pl5boot.awake.detection_rate)} of awake and ${pct(pl5boot.isoflurane.detection_rate)} of isoflurane bootstrap resamples). EEG has one under isoflurane (${pct(e13boot.isoflurane.detection_rate)} frontal, ${pct(e10boot.isoflurane.detection_rate)} somatosensory). The frontal and somatosensory channels show only a weak awake pattern, but the all-channel sweep finds awake EEG motifs on ${SW.counts["eeg/awake"].strict} of 26 channels (section 5).`,
  `**Awake and isoflurane motifs are different.** Motifs from the same state agree almost perfectly across resamples (median |r| ${f2(median(pl5boot.awake.within_abs_r))}–${f2(median(e13boot.isoflurane.within_abs_r))}), while awake and isoflurane motifs agree only weakly (median |r| ${f2(median(e13boot.across_abs_r))}–${f2(median(pl5boot.across_abs_r))}).`,
  `**Under isoflurane, frontal EEG, somatosensory EEG and deep LFP share one motif** (theta/alpha down while beta–gamma goes up, and its opponent): shape |r| = ${f2(iso13.shape_abs_r[0][0])} and ${f2(iso10.shape_abs_r[0][0])} between EEG and LFP, and the active pole co-occurs second by second (r = ${f2(iso13.temporal.isoflurane.r[0][0])}). This is the most promising starting point for the EEG → LFP ML extension.`,
  `**The awake LFP motif tracks spiking:** PL5 population firing is ${f2(spA.diff)} SD higher when its + pole dominates than when its opponent does (p = ${f3(spA.p)}). This supports the paper's claim that motifs reflect neural population activity.`,
  `**All 26 EEG channels and 45 LFP regions (section 5):** each state has one motif shared brain-wide; awake and isoflurane motifs differ in ${SW.bootstrap.lfp["n_across_below_within_by_0.2"]} of ${SW.bootstrap.lfp.n_both} LFP regions and ${SW.bootstrap.eeg["n_across_below_within_by_0.2"]} of ${SW.bootstrap.eeg.n_both} EEG channels; EEG and LFP motif poles co-occur in every pair tested (median |r| ${f2(SW.coupling.awake.median_abs_r)} awake, ${f2(SW.coupling.isoflurane.median_abs_r)} isoflurane); and in every region tested the PL5-like pole goes with more local firing (significant in ${SW.spikes.filter((r) => r.p <= 0.05).length} of ${SW.spikes.length}). ${SW.n_motifs_broadband} of ${SW.n_motifs_total} motifs, mostly in ventral prefrontal and olfactory LFP, are broadband-like and set aside.`,
  `**Caveats:** one mouse; a shared skull-screw reference for EEG and LFP; placeholder frequency bands (Mishler's reference motifs use 76 bands); the method cannot tell broadband co-fluctuation from motifs, so each motif is checked separately; 60 Hz line noise in the LFP.`,
]));

c.push(H1("1. Background"));
c.push(P(`A spectral motif is a recurring pattern of which frequency bands are above or below their own average at the same moment. Mishler et al. (2026) report that motifs come in opponent pairs (a pattern and its mirror image) that track opposing groups of neurons. Their algorithm, getMotifs, z-scores the smoothed amplitude envelope of each band, reduces each second to a pattern of signs across bands, clusters those patterns hierarchically, merges clusters that are expressed at the same times, and keeps only clusters bigger than anything found in shuffled ("null") data in which each band is shifted in time relative to the others. The team's question for this mouse is whether motifs differ between the awake and isoflurane states.`));

c.push(H1("2. Data and preprocessing"));
c.push(H2("2.1 The recording and its timeline"));
c.push(P(`The session lasts ${f0((tl.t_end - tl.t_start) / 60)} minutes. The NWB file contains the isoflurane level recorded at 100 Hz, which we used to define the states precisely: isoflurane is switched on at ${f0(tl.iso_on)} s (5% induction), drops to maintenance (<2%) at ${f0(tl.induction_end)} s, and is switched off at ${f0(tl.iso_off)} s. Two blocks of single electrical pulses (every ~7 s) were delivered, one awake (${f0(tl.electrical_blocks[0][1])}–${f0(tl.electrical_blocks[0][2])} s) and one under isoflurane (${f0(tl.electrical_blocks[1][1])}–${f0(tl.electrical_blocks[1][2])} s), each followed by a block of natural-scene images.`));
c.push(...figure("01_overview_timeline.png", `Session timeline. Top: isoflurane level, with the stimulation blocks (hatched), visual stimuli (purple) and the stimulation-free stretches used for analysis (green). Middle and bottom: spectrograms of frontal EEG and deep prelimbic LFP, each frequency shown relative to its own median, in 30-s averages. Induction causes an abrupt change in both; the two isoflurane stretches look different (see Limitations).`));
c.push(table(["Window", "Definition", "Time range (s)", "Stimulation-free seconds used", "5-min chunks"],
  wins.map((w) => [w, { awake: "Before isoflurane is switched on (30-s margin)", isoflurane: "Maintenance, after induction", full: "Whole recording, awake through isoflurane" }[w],
    `${f0(tl.windows[w][0])}–${f0(tl.windows[w][1])}`, f0(usable(w).sec), usable(w).chunks]),
  [1300, 3400, 1500, 1800, 1360]));
c.push(gap());

c.push(H2("2.2 Excluding electrical stimulation"));
const s13 = stim[Object.keys(stim).find((k) => k.startsWith("EEG ch13"))].awake;
const sPL = stim[Object.keys(stim).find((k) => k.startsWith("LFP"))].awake;
c.push(P(`The stimulating electrode's tip sits in PL6a, directly below the PL5 channel we use. Averaged over pulses, the awake evoked response (after a brief pulse artifact) reaches about 400 µV in PL5 against a background fluctuation of ${f0(sPL.pre_sd_uv)} µV, and about 130 µV against ${f0(s13.pre_sd_uv)} µV in frontal EEG, lasting about 300 ms. With a pulse every ~7 s and a 5-s smoothing window, every second inside a stimulation block is contaminated, so motifs found there would partly describe the response to stimulation. We therefore drop both stimulation blocks (plus 5 s either side) and analyse only the stimulation-free stretches. Visual stimuli were kept: they evoke far smaller responses in frontal areas.`));
c.push(...figure("02_stimulation_response.png", `Average response to an electrical pulse in each signal, awake (blue) and under isoflurane (red). The awake response is several times the background fluctuation; under isoflurane only the brief artifact remains.`));

c.push(H2("2.3 Channels"));
c.push(P(`**EEG numbering.** The experiment log lists bad EEG channels 6, 7, 14 and 23 for this session, and the NWB file flags exactly those four as invalid, which confirms that the log counts channels from 0. In this mouse, channel 7 (Irene's suggested somatosensory channel) is in visual area VISa and is invalid, and channel 8 is retrosplenial. We used **channel 13** (MOs, frontal, left hemisphere) and **channel 10** (SSp-ll, somatosensory, left hemisphere). Justin used 13 and 9, so channel 13 allows a direct comparison.`));
c.push(P(`**LFP.** Probe F passes through prelimbic layer 5 (PL5, 78 channels), the deep prefrontal region the notes asked for and the region with the most sorted units in this recording (${units.n_total}). We used the middle PL5 channel (${lfpChan}). For the raster, ${units.n_good} of the ${units.n_total} PL5 units pass the Allen Institute's standard quality filters (ISI violations < 0.5, amplitude cutoff < 0.1, presence ratio > 0.9).`));

c.push(H2("2.4 From voltage to band envelopes"));
c.push(...bullets([
  `Each signal is downsampled from 2500 Hz to 250 Hz.`,
  `SmoothEnv (our Python port) runs separately on each stimulation-free stretch, so its filters never see stimulation and its 10-s blank edges fall at the ends of stretches.`,
  `**Z-scoring happens once per analysis window, across all its stretches,** not per stretch. SmoothEnv normally z-scores each call; we added an option to skip that step. Z-scoring each stretch separately would erase real differences between them, for example between awake and isoflurane in the full-recording window.`,
  `**Frequency bands:** 10 overlapping bands from 1 to 80 Hz (1–4, 3–6, 5–9, 8–12, 10–15, 13–20, 18–30, 25–40, 35–55, 50–80 Hz). The 6 bands now in the repo's default are not Mishler's: his reference file (data/finalGlutLFPMotifs.mat) holds motifs over **76 bands**, recorded at about 1017 Hz. Which 76 bands is the most important open question for Mishler.`,
]));

c.push(H1("3. Methods: three versions of motif detection"));
c.push(H2("3.1 Original"));
c.push(P(`Mishler's rules unchanged (our verified Python port of getMotifs), with each window treated as one "session": its stimulation-free stretches joined end to end. 20 shuffled null datasets. This mirrors Justin's runs, except that ours exclude stimulation.`));
c.push(H2("3.2 Strict (chunked)"));
c.push(P(`Same rules, but each window is cut into ~5-minute chunks (${a.chunks} awake, ${i.chunks} isoflurane, ${fu.chunks} full) and each chunk is passed as a separate session. getMotifs was designed for several animals or sessions at once: its null shifts each session separately, and a motif must appear in every session. With chunks, "every session" becomes "every 5-minute chunk of this state", a within-animal stability requirement.`));
c.push(P(`**A finding about that rule:** a cluster counts as present in a session if it contains even one second from it. Random noise produces every sign pattern occasionally, so the rule is much weaker than it sounds (our test suite has a case showing this).`));
c.push(H2("3.3 Why the original finds so little: a bias in the null comparison"));
c.push(P(`getMotifs discards "trivial" seconds in which every band has the same sign (everything up or everything down at once), then compares the size of real clusters with the size of null clusters. But the null is made by shifting bands relative to each other, which destroys exactly that broadband co-movement. Null datasets therefore keep more seconds, and bigger datasets grow bigger clusters. Real clusters must beat nulls that had more material to work with.`));
const diagRows = [];
sigs.forEach((s) => wins.forEach((w) => { const d = diag(s, w); const t = d.per_threshold[2]; diagRows.push([label[s], w, d.n_real, f0(d.n_null_median), `${pct(d.n_null_median / d.n_real - 1)}`, `${t.real_largest} / ${f0(t.null_largest_median)} / ${f0(t.null_largest_max)}`]); }));
c.push(table(["Signal", "Window", "Usable seconds (real)", "Usable seconds (null, median)", "Null excess", "Largest cluster at r = 0.6: real / null median / null max"], diagRows, [2200, 1100, 1300, 1500, 1100, 2160]));
c.push(P(`The bias is large under isoflurane (null datasets keep 59–92% more seconds), where cortex alternates between bursts and silence that move every band together (see the raster below), and moderate for the full recording. It is absent awake, where almost no seconds are trivial. There, the real awake EEG clusters are bigger than a typical null cluster but not bigger than the largest ones, so they fail the original rule, and they also fail the relaxed p-value test. The empty awake EEG results are genuine: no awake EEG cluster beats chance at this sample size.`, { spacing: { before: 120, after: 120 } }));

c.push(H2("3.4 Relaxed"));
c.push(P(`Same chunks as strict, with three changes to the filters:`));
c.push(...numbered([
  `**Permutation p-value instead of "beat every null".** The original keeps a candidate only if it is bigger than the largest cluster of every one of 20 nulls. We run 100 nulls and give each candidate p = (1 + number of nulls whose largest cluster is at least as big) / 101, keeping p ≤ 0.05. Given the same nulls, anything the original rule keeps also passes this test (checked in the test suite).`,
  `**Present in at least half the chunks** instead of all of them.`,
  `**Size as a fraction of usable seconds.** Null cluster sizes are rescaled by (real usable seconds / null usable seconds) before comparing, which removes the bias in 3.3.`,
]));
c.push(P(`We did not relax the original's last filter: the PCA step that merges candidates found at the different thresholds and only accepts a motif supported by at least two of them. Relaxing it to one produced false motifs in pure noise (next section).`));

c.push(H2("3.5 Calibration and guardrails"));
c.push(P(`A relaxed method is only useful if it does not find motifs where there are none. We ran every version on synthetic data with no motifs (20 datasets each, 5 sessions × 330 s × 10 bands, smoothed like real envelopes). In *independent* data every band is unrelated noise. In *broadband* data all bands additionally share a common slow component of weight w, so they co-fluctuate without any multi-band pattern.`));
const dataCols = ["independent", "broadband w=0.5", "broadband w=1", "broadband w=2"];
const cfgs = ["strict", "relaxed: p-value + half", "relaxed: ... + size normalization", "relaxed: ... + PCA support 1"];
const cfgNames = ["Strict (original rules)", "Relaxed: p-value + half the chunks", "Relaxed: + size normalization (used)", "Relaxed: + PCA support 1 (rejected)"];
c.push(table(["Version", ...dataCols.map((d) => d.replace("broadband", "Broadband"))],
  cfgs.map((cf, k) => [cfgNames[k], ...dataCols.map((d) => { const x = cal(d, cf); return x ? `${x.datasets_with_false_motif}/20 (${x.false_motifs_flagged_broadband}/${x.false_motifs} flagged)` : "n/a"; })]),
  [2760, 1650, 1650, 1650, 1650]));
c.push(P(`Cells give the datasets in which a false motif appeared, and in brackets how many of those false motifs the broadband flag (below) catches. Share of all-same-sign seconds in the broadband data: ${dataCols.slice(1).map((d) => `${d.replace("broadband ", "")}: ${pct(cal(d, "strict").all_same_sign_fraction)}`).join(", ")}.`, { spacing: { before: 120 } }));
c.push(...bullets([
  `**On independent noise** the original rules produce a false motif in 1 dataset of 20, both relaxed versions in 0, and relaxing the PCA step in 3. That is why the PCA step was kept.`,
  `**On broadband data every version fails, including Mishler's original.** The shuffle null tests "is there any coupling between bands?", and shared broadband fluctuation is coupling. This is a property of the method, not of the relaxation. The size-normalized version is somewhat more sensitive to it.`,
  `**Broadband flag.** False motifs from broadband data look like "nearly every band the same sign". We flag a motif as broadband-like if at most 1 of its 10 bands has the minority sign. The flag catches almost all false motifs (counts in the table). **Every motif we found in the real data has 2–5 minority-sign bands, so none is flagged.**`,
  `**Real-data guardrail.** For every signal and window, we also ran the relaxed version twice on the real chunks with every band circularly shifted by an independent random amount (which keeps each band's own dynamics but destroys any cross-band pattern). Result: ${guardHits} motifs in ${guard.length} runs.`,
]));

c.push(H2("3.6 Statistical tools"));
c.push(...bullets([
  `**Chunk bootstrap (are motifs the same across states?).** For each signal and state, the 5-minute chunks are resampled with replacement and relaxed detection is rerun, ${B.n_boot} times. Comparing bootstrap motifs from the same state shows how reproducible a state's motif is; comparing awake bootstrap motifs with isoflurane ones shows how similar the states' motifs are.`,
  `**Exact permutation test over chunks.** Using the motif found in the full recording, we measure in each 5-minute chunk the pole balance (f+ − f−)/(f+ + f−), where f± is the fraction of seconds that pole dominates, and test whether awake and isoflurane chunks differ, using all ${perm.lfp_PL5.tests[0].n_permutations} ways of relabelling the chunks.`,
  `**EEG vs LFP.** For each window where both EEG and LFP have a motif: the shape correlation of the motifs, and the second-by-second correlation of which pole is active (+1, −1, or 0 when neither dominates).`,
  `**Do motifs track spiking?** PL5 population firing rate per second (z-scored within the state) when the LFP motif's + pole dominates versus its − pole. Null: shift the pole sequence in time by at least 60 s (1000 shifts), which keeps its slow dynamics but breaks its alignment with spiking.`,
]));

c.push(H1("4. Results"));
c.push(H2("4.1 Power spectra (deliverable 3)"));
c.push(...figure("03_power_spectra.png", `Power spectra (Welch, 4-s windows) of frontal EEG, somatosensory EEG, deep prelimbic LFP and the average of all 26 valid EEG channels, for the awake, isoflurane and full windows (stimulation-free data only).`));
c.push(...figure("04_power_ratio_iso_vs_awake.png", `Change in power from awake to isoflurane at each frequency (dB). Above zero means more power under isoflurane.`, 520));
const bands = ["1-4", "4-8", "8-12", "12-30", "30-80"];
c.push(table(["Signal", ...bands.map((b) => `${b} Hz`)], Object.entries(db).map(([k, v]) => [k, ...bands.map((b) => `${v[b] > 0 ? "+" : ""}${v[b].toFixed(1)} dB`)]), [3360, 1200, 1200, 1200, 1200, 1200]));
c.push(P(`Awake spectra have peaks near 3 Hz and 7 Hz (theta) and a bump around 40–60 Hz (gamma). Isoflurane raises power below ~2.5 Hz (about +10 dB at 1 Hz), removes the theta peak, raises 10–40 Hz, and cuts power above ~45 Hz. The deep LFP gains much more at 8–30 Hz (${f2(bandPeaks("LFP")["8-12"])} and ${f2(bandPeaks("LFP")["12-30"])} dB) than the EEG (${f2(bandPeaks("EEG ch13")["8-12"])} and ${f2(bandPeaks("EEG ch13")["12-30"])} dB), in line with the observation from the PoCs that LFP keeps more high-frequency detail. The PL5 spectrum shows a narrow 60 Hz line-noise peak.`, { spacing: { before: 120 } }));

c.push(H2("4.2 PL5 spike raster (deliverable 4)"));
c.push(...figure("05_PL5_raster.png", `Spikes of the ${units.n_good} well-isolated PL5 units (sorted by depth) in 60-s snippets: awake, one minute into induction, and under isoflurane maintenance. Below: population rate and each unit's rate over the full recording (10-s bins).`));
c.push(P(`Awake, PL5 units fire steadily at about 6 Hz on average. Within a minute of induction, firing collapses into synchronized bursts separated by silence, and under isoflurane the population fires about 1 Hz on average, in near-simultaneous bursts across units. These population-wide bursts are why so many isoflurane seconds have every band moving together (39–50% of seconds, against about 0% awake).`));

c.push(H2("4.3 Motif detection (deliverables 1 and 2)"));
c.push(table(["Signal", "Window", "Original", "Strict", "Relaxed"],
  sigs.flatMap((s) => wins.map((w) => [label[s], w, ...["original", "strict", "relaxed"].map((v) => cell(runsBy(s, w, v)))])),
  [2600, 1100, 1700, 1700, 2260]));
c.push(P(`Each cell gives the number of motif pairs and the smallest p-value among the candidates supporting each (0.048 is the smallest possible with 20 nulls, 0.010 with 100).`, { spacing: { before: 120 } }));
c.push(...bullets([
  `**Deep LFP, awake:** all three versions find the same motif (|r| > 0.9 between versions): 1–4 Hz down while 3–6 Hz and 8–20 Hz go up, and its opponent.`,
  `**Isoflurane:** only the relaxed version finds motifs, and all three signals find essentially the same one: about 3–15 Hz down while roughly 20–80 Hz goes up (and its opponent).`,
  `**Full recording:** the relaxed version finds one motif in frontal EEG and in deep LFP. Section 4.5 shows it is essentially the awake-versus-isoflurane difference.`,
  `**Awake EEG:** nothing in any version. The bootstrap (4.4) shows a consistent but weak pattern that passes in 28–45% of resamples.`,
  `Running the relaxed version *without* size normalization reproduced the strict results exactly (saved in diagnostics/), which confirms the normalization is what recovers the isoflurane motifs.`,
]));
for (const s of sigs) c.push(...figure(`06_motifs_${s}.png`, `${label[s]}: motifs for each window (rows) and version (columns). Solid line = + pole, dashed = its opponent. Values are the average sign of each band (−1 to +1) over the seconds that express the motif.`));

c.push(H2("4.4 Are motifs the same across states? (bootstrap)"));
c.push(...figure("10_bootstrap_stability.png", `Chunk bootstrap, ${B.n_boot} resamples per state. Each violin shows the best |r| between motif pairs from two resamples: both awake, both isoflurane, or one of each. Labels give how often a resample produced any motif.`));
c.push(table(["Signal", "Motif found (awake / iso)", "Same state: median |r| (awake / iso)", "Awake vs iso: median |r| (IQR)"],
  sigs.map((s) => { const b = boot[s]; const q = (v) => { const x = [...v].sort((p, q2) => p - q2); return `${f2(median(v))} (${f2(x[Math.floor(x.length / 4)])}–${f2(x[Math.floor(3 * x.length / 4)])})`; };
    return [label[s], `${pct(b.awake.detection_rate)} / ${pct(b.isoflurane.detection_rate)}`, `${f2(median(b.awake.within_abs_r))} / ${f2(median(b.isoflurane.within_abs_r))}`, q(b.across_abs_r)]; }),
  [2700, 2000, 2400, 2260]));
c.push(P(`Whenever a state's motif appears, it has the same shape (median |r| 0.96–0.99), but motifs from awake and isoflurane are far apart (median |r| about 0.35). **The two states have different motifs** in every signal. Deep LFP motifs are the most robust (found in about 90% of resamples in both states).`, { spacing: { before: 120 } }));

c.push(H2("4.5 The full-recording motif is a state signature"));
c.push(...figure("08_permutation_test.png", `Pole balance of the full-recording motif in each 5-minute chunk. +1 means the + pole always beats its opponent in that chunk.`, 480));
const pt = perm.lfp_PL5.tests[0];
c.push(P(`Every awake chunk is entirely the + pole and every isoflurane chunk entirely the − pole (exact permutation p = ${f3(pt.balance_p)} for both EEG and LFP, the smallest possible with ${pt.n_awake_chunks} awake and ${pt.n_iso_chunks} isoflurane chunks). The motif's awake pole matches the spectral difference in 4.1: more theta and gamma, less delta and beta. So "motifs across the full recording" mostly rediscover the change of state rather than fluctuations within a state, which is why the per-state analyses are the informative ones.`));

c.push(H2("4.6 Frontal and somatosensory EEG versus deep LFP"));
c.push(table(["Window", "EEG channel", "Shape |r| with LFP motif", "Pole co-occurrence r (seconds)"],
  eegLfp.map((e) => [e.window, e.eeg, f2(e.shape_abs_r[0][0]),
    Object.entries(e.temporal).filter(([k]) => e.window === "full" || k === e.window).map(([k, v]) => `${k}: ${v.r[0][0] === null ? "n/a" : f2(v.r[0][0])} (${v.n_seconds})`).join("; ")]),
  [1500, 1500, 2400, 3960]));
c.push(P(`Under isoflurane, both EEG channels and the deep LFP express the same motif (shape |r| ${f2(iso13.shape_abs_r[0][0])} and ${f2(iso10.shape_abs_r[0][0])}), and they switch poles together second by second (r = ${f2(iso13.temporal.isoflurane.r[0][0])}). Awake, no comparison was possible because neither EEG channel has a significant awake motif. The deep LFP awake motif has no detectable EEG counterpart, which is consistent with the notes that LFP is less noisy and carries more high-frequency detail.`, { spacing: { before: 120 } }));

c.push(H2("4.7 Do LFP motifs track spiking?"));
c.push(...figure("09_spikes_by_LFP_pole.png", `PL5 population firing rate (z-scored within each state) in seconds when the LFP motif's + pole dominates (dark) versus its opponent (light).`, 440));
c.push(P(`Awake, firing is ${f2(spA.diff)} SD higher when the + pole dominates (${spA.n_plus} vs ${spA.n_minus} seconds, p = ${f3(spA.p)}, the smallest possible with 1000 shifts). Under isoflurane the difference is ${f2(spI.diff)} SD (p = ${f3(spI.p)}). The two poles go with different levels of population activity, as the paper claims. Arousal and movement, which change both LFP and firing, could also produce this; running speed and pupil size are in the file and should be checked next.`));

c.push(...sweepSection({ H1, H2, P, bullets, table, figure, f2, pct, SW }));

c.push(H1("6. Limitations"));
c.push(...bullets([
  `**One mouse, one session.** Chunks give within-animal stability, not generalization across animals.`,
  `**Frequency bands are placeholders.** Mishler used 76 bands; ours are 10.`,
  `**Broadband co-fluctuation.** No version of the method separates it from motifs; we flag it per motif instead.`,
  `**The isoflurane window is not uniform.** The first stretch (1.5%, after induction) has strong 15–40 Hz power; the second (1.8%, after a deep-dose stimulation block) has suppressed high frequencies (Figure 1).`,
  `**60 Hz line noise in the LFP,** inside the 50–80 Hz band.`,
  `**Shared reference.** EEG and LFP are both referenced to a skull screw. A signal common to the reference appears in every channel, so part of the brain-wide motif sharing and EEG–LFP co-occurrence in section 5 could come from the reference. Re-referencing the LFP (for example to a neighbouring channel) would test this.`,
  `**Ventral prefrontal and olfactory regions** mostly give broadband-like motifs, so this method says little about them.`,
  `**Guardrail hits.** ${SW.guardrail_hits.length} of ${SW.guardrail_runs} band-shuffled runs in the sweep found a motif, close to the expected 5%.`,
  `**The awake window includes two visual-stimulus blocks** (natural scenes), kept because their effect on frontal areas is small.`,
]));

c.push(H1("7. Open questions"));
c.push(...bullets([
  `**For Mishler:** which 76 frequency bands; whether trimming SmoothEnv's NaN edges before getMotifs is intended; whether treating chunks of one recording as sessions is acceptable.`,
  `**For Irene and Jonathan:** is excluding the electrical-stimulation blocks right, or should they be analysed as a separate condition? Is one PL5 channel enough, or should we average the layer?`,
  `**For Sara:** the filename of the third recording in the Dropbox folder (only two were found).`,
]));

c.push(H1("8. What this means for the ML extension (EEG → LFP)"));
c.push(...bullets([
  `**Under isoflurane the link is already strong:** shared motif shape and r = ${f2(iso13.temporal.isoflurane.r[0][0])} pole co-occurrence. Predicting the LFP pole from EEG should work well there, and makes a good first benchmark.`,
  `**Awake works too, with the right channels:** frontal channel 13 has no awake motif, but channels ${SW.coupling.awake.eeg_ranked.slice(0, 4).map(([k]) => k.replace("eeg", "")).join(", ")} track deep-LFP poles at a median |r| of about ${f2(SW.coupling.awake.eeg_ranked[3][1])}–${f2(SW.coupling.awake.eeg_ranked[0][1])} (best pair ${f2(SW.coupling.awake.top[0].r_aligned)}). A model using all 26 channels' band envelopes should beat any single channel.`,
  `**Rule out the shared reference first:** re-reference the LFP and check that the co-occurrence survives before treating it as EEG reading out deep activity.`,
  `**Evaluation should hold out whole 5-minute chunks** (neighbouring seconds are strongly correlated) and be done separately within each state, since across states the problem reduces to state detection.`,
]));

c.push(H1("9. Reproducing this"));
c.push(P(`From code/python/ with the project's virtual environment:`));
c.push(...bullets([
  `python -m src.state_motifs --nwb ../../data/raw/720762-20240221-v001.nwb --out ../../data/out/2026-11-01_mouse720762 (all deliverables, ~5 min)`,
  `python -m src.motif_bootstrap --nwb ... --out ... --n-boot 40 (bootstrap, ~6 min)`,
  `python -m src.motif_calibration --out .../diagnostics (synthetic calibration, ~25 min)`,
  `python -m src.null_size_diagnosis --nwb ... --out .../diagnostics (section 3.3)`,
  `python -m src.channel_sweep --nwb ... --out .../all_channels (section 5, ~1 hour; add --replot to redraw from saved results)`,
  `python -m pytest src/tests (tests, including the relaxed-mode checks)`,
  `Notebook: code/python/notebooks/mouse720762_states_naman.ipynb`,
]));

const doc = new Document({
  creator: "Naman Rudrakshi",
  title: "Spectral motifs in mouse 720762",
  styles: {
    default: { document: { run: { font: "Arial", size: 21 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", next: "Normal", run: { size: 40, bold: true, font: "Arial", color: "1F3864" }, paragraph: { spacing: { after: 160 } } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 30, bold: true, font: "Arial", color: "1F3864" }, paragraph: { spacing: { before: 320, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 25, bold: true, font: "Arial", color: "2E5597" }, paragraph: { spacing: { before: 240, after: 100 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 22, bold: true, font: "Arial" }, paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Mouse 720762 motif analysis · page ", PageNumber.CURRENT], size: 16, color: "808080" })] })] }) },
    children: c,
  }],
});
Packer.toBuffer(doc).then((buf) => {
  const file = path.join(OUT, "report_mouse720762_2026-11-01.docx");
  fs.writeFileSync(file, buf);
  console.log("wrote", file, buf.length, "bytes,", figNo, "figures");
});
