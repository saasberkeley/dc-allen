// Inserted into build_report.js: all-channel sweep section (reads all_channels/sweep_summary.json).
module.exports = function sweepSection(ctx) {
  const { H1, H2, P, bullets, table, figure, f2, pct, SW } = ctx;
  const c = [];
  const cnt = (k) => SW.counts[k];
  const ea = cnt("eeg/awake"), ei = cnt("eeg/isoflurane"), ef = cnt("eeg/full");
  const la = cnt("lfp/awake"), li = cnt("lfp/isoflurane"), lf = cnt("lfp/full");
  const be = SW.bootstrap.eeg, bl = SW.bootstrap.lfp;
  const ca = SW.coupling.awake, ci = SW.coupling.isoflurane;
  const spA = SW.spikes.filter((r) => r.state === "awake"), spI = SW.spikes.filter((r) => r.state === "isoflurane");
  const sig = (rows) => rows.filter((r) => r.p <= 0.05);
  const eegName = (k) => `ch${k.replace("eeg", "")}`;
  const lfpName = (k) => k.replace("lfp_", "").replace("-", "/");

  c.push(H1("5. All 26 EEG channels and 45 LFP regions"));
  c.push(H2("5.1 What was run"));
  c.push(...bullets([
    `**Signals:** all 26 valid EEG channels, and every LFP region that is grey matter (white matter, fibre tracts and ventricles excluded) with at least 5 valid channels: 45 regions on the four probes. For each region we used its middle valid channel; a region recorded on two probes was taken from the probe with more channels in it.`,
    `**Per signal:** the strict and relaxed versions in all three windows, one band-shuffled guardrail run per state, a 20-resample chunk bootstrap per state, and the awake-to-isoflurane band-power change. Preprocessing is identical to sections 2–3. The sweep took about an hour on 6 processor cores.`,
    `**Guardrail:** ${SW.guardrail_hits.length} of ${SW.guardrail_runs} band-shuffled runs produced a motif (${pct(SW.guardrail_hits.length / SW.guardrail_runs)}), about the rate a 5% test should give: ${SW.guardrail_hits.map(([l, w]) => `${l.replace(/ \(probe.\)/, "")} (${w})`).join(", ")}. A single relaxed-only motif in one signal should be read with that rate in mind.`,
    `**Broadband-like motifs:** ${SW.n_motifs_broadband} of the ${SW.n_motifs_total} motifs found are broadband-like (at most 1 of 10 bands with the minority sign; see 3.5). They concentrate in ventral prefrontal and olfactory LFP (${[...new Set([...la.broadband_names, ...li.broadband_names])].filter((n) => /^(DP|ILA|ORB|TTd|AON)/.test(n)).join(", ")}) and two anterior thalamic nuclei (AMd, IAD). They are shown separately in the figures and **left out of the coupling and spiking analyses below**.`,
  ]));

  c.push(H2("5.2 EEG across the skull"));
  c.push(...figure("all_channels/figures/11_eeg_channel_map.png", `Every EEG electrode seen from above (frontal at the top, CCF coordinates), coloured by whether a motif was found in each window. Crosses mark the four invalid channels.`));
  c.push(P(`**Awake:** ${ea.strict} of 26 channels pass the strict test (channels ${ea.strict_names.join(", ")}), ${ea.relaxed_only} more pass only the relaxed one, and ${ea.broadband_only} (channels ${ea.broadband_names.join(", ")}) have only a broadband-like motif. This revises the single-channel result in 4.3: awake EEG motifs do exist, but frontal channel 13 and somatosensory channel 10 are not among the channels that carry them. Note that channel 12's awake guardrail also produced a motif. **Isoflurane:** ${ei.relaxed_only} of 26 channels have a motif, all relaxed-only, and all the same shape (5.4). **Full recording:** ${ef.relaxed_only} channels, relaxed-only.`));

  c.push(H2("5.3 LFP regions"));
  c.push(...figure("all_channels/figures/12_lfp_region_grid.png", `Every LFP region (coloured by brain area) and window. Text gives the share of bootstrap resamples that produced a motif.`, 520));
  c.push(P(`**Awake:** ${la.strict} of 45 regions pass the strict test (${la.strict_names.join(", ")}), ${la.relaxed_only} more relaxed-only, ${la.broadband_only} broadband-like only. **Isoflurane:** ${li.relaxed_only} regions have a genuine motif (all relaxed-only) and ${li.broadband_only} only a broadband-like one. **Full recording:** ${lf.relaxed_only} regions. Robust awake motifs appear across very different structures: prefrontal (PL5, MOs), motor (MOp6b), visual (VISpm1), retrosplenial (RSPagl5), thalamus (AD, LD) and striatum (CP, STR).`));

  c.push(H2("5.4 One motif per state, brain-wide"));
  c.push(...figure("all_channels/figures/13_motif_shapes_all_signals.png", `The + pole of every genuine motif, each flipped if needed to point the same way as the deep prelimbic (PL5) motif of that state. Rows are EEG channels then LFP regions, coloured by area; * = relaxed-only.`));
  c.push(P(`Within each state, nearly every signal expresses the same pattern. **Awake:** 1–4 Hz down while roughly 3–20 Hz goes up (with variation above 25 Hz). **Isoflurane:** roughly 3–12 Hz down while 18–80 Hz goes up. The PL5 motifs in section 4 are therefore not local to prelimbic cortex: they are the local view of a state-specific pattern shared across cortex, hippocampus, thalamus and striatum, and visible on the skull.`));

  c.push(H2("5.5 Awake and isoflurane motifs differ everywhere"));
  c.push(...figure("all_channels/figures/16_bootstrap_all_signals.png", `Each dot is a signal with bootstrap motifs in both states: how reproducible its motifs are within a state (x) against how similar its awake and isoflurane motifs are (y). Points far below the diagonal mean the two states have different motifs.`, 520));
  c.push(P(`For EEG (${be.n_both} channels), bootstrap motifs agree at a median |r| of ${f2(be.within_median)} within a state but only ${f2(be.across_median)} across states, with ${be.n_across_below_within_by_0_2 ?? be["n_across_below_within_by_0.2"]} of ${be.n_both} channels at least 0.2 lower across states. For LFP (${bl.n_both} regions): ${f2(bl.within_median)} within against ${f2(bl.across_median)} across, ${bl["n_across_below_within_by_0.2"]} of ${bl.n_both} regions. The answer to the team's main question holds for the whole brain: **awake and isoflurane have different motifs.**`));

  c.push(H2("5.6 EEG versus LFP, every pair"));
  c.push(...figure("all_channels/figures/14_eeg_lfp_coupling.png", `Second-by-second co-occurrence of motif poles for every EEG channel (columns) and LFP region (rows) where both have a genuine motif. Each signal's poles are oriented like PL5's, so positive means the PL5-like poles occur at the same time.`));
  const topRows = (cc) => cc.top.slice(0, 5).map((r) => [eegName(r.eeg), lfpName(r.lfp), f2(r.r_aligned), f2(r.shape_abs_r)]);
  c.push(table(["State", "EEG channel", "LFP region", "Pole co-occurrence r", "Shape |r|"],
    [...topRows(ca).map((r) => ["awake", ...r]), ...topRows(ci).map((r) => ["isoflurane", ...r])], [1500, 1700, 2100, 2200, 1860]));
  c.push(P(`**Awake:** ${ca.n_pairs} EEG–LFP pairs have genuine motifs on both sides; median |r| = ${f2(ca.median_abs_r)}, and ${pct(ca.share_positive)} of pairs are positive once oriented. The EEG channels that track the deep motifs best are ${ca.eeg_ranked.slice(0, 4).map(([k, m]) => `${eegName(k)} (median ${f2(m)})`).join(", ")}. **Isoflurane:** ${ci.n_pairs} pairs, median |r| = ${f2(ci.median_abs_r)}, ${pct(ci.share_positive)} positive. EEG therefore carries a large part of the deep motif's timing in both states, provided the right channels are used, subject to the shared-reference caveat in section 6.`, { spacing: { before: 120 } }));

  c.push(H2("5.7 Spectral change in every signal"));
  c.push(...figure("all_channels/figures/15_band_change_all_signals.png", `Isoflurane versus awake power in five bands for every signal (red = more under isoflurane).`, 420));
  c.push(P(`The pattern from section 4.1 holds almost everywhere: more 1–4 Hz and 12–30 Hz power and less 4–8 Hz power under isoflurane. Two groups differ: 30–80 Hz power falls strongly in hippocampus, thalamus, motor and retrosplenial LFP but hardly changes in the EEG, and the ventral prefrontal and olfactory regions (AON, DP, ILA1, ORBm, TTd) lose rather than gain 1–4 Hz power. Those are also the regions dominated by broadband-like motifs.`));

  c.push(H2("5.8 Do motifs track local spiking?"));
  c.push(...figure("all_channels/figures/17_spikes_by_pole_regions.png", `For every region with a genuine motif and at least 10 well-isolated units: the difference in local population firing (SD) between seconds dominated by the PL5-like pole and seconds dominated by its opponent. Faded bars are not significant (circular-shift null, 1000 shifts).`));
  c.push(P(`**Awake:** ${sig(spA).length} of ${spA.length} regions tested show significantly higher firing on the PL5-like pole (${spA.map((r) => `${r.region} ${f2(r.diff)} SD`).join(", ")}). **Isoflurane:** ${sig(spI).length} of ${spI.length} are significant (${sig(spI).map((r) => r.region).join(", ")}), and every difference points the same way. Across prefrontal, somatosensory, retrosplenial and thalamic regions, the same pole goes with more local firing, which is consistent with the paper's claim and suggests a brain-wide axis of population activity. ${SW.spikes_excluded_broadband.length} region–state combinations were not tested because their motif was broadband-like (${SW.spikes_excluded_broadband.join(", ")}).`));
  return c;
};
