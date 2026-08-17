# E034 — H2d re-measured: localisation on the corrected connectome, and does the contrast occur

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**H2d** — the pallium does not form separable representations of distinct stimuli.
`SUPPORTED as a limitation`, demoted from the critical path by E019 pending
re-measurement. Blocks H2b, H2c and everything downstream of them.

## 2. Question

Two things about H2d are stale and neither has been checked since E019 (calls made
audible) and E023 (E/I bug fixed, gain re-baselined to 0.95):

**(a) Localisation.** E017 found the "saw hawk" vs "heard alarm" separability loss is a
17× fan-in dilution at the sensory→pallium projection, not pallial recurrence, on a
pallium that (unknown at the time) had **no inhibitory neurons at all**. E023 re-measured
overall separability on the corrected connectome and found it essentially unchanged
(7.4% vs 7.5%), but did **not** re-run E017's ablation breakdown (recurrence removed,
Field-L-style segregation) — its own §6 flags this as still owed. Is the loss still
localised the same way on a connectome that can now inhibit itself?

**(b) Occurrence.** The whole diagnosis rests on contrasting "saw a hawk" against "heard
an alarm call" as if a hen ever experiences one without the other. Before E019, she
didn't: the call channel was constant at 1.0 in the live coop. E019 fixed the
precondition (calls are no longer saturated) but nobody has checked the actual
consequence: in a running coop, does a hen ever find herself head-down (blind) while a
flockmate is audibly alarm-calling? If that combination essentially never occurs, H2d's
representational finding — however real — describes a situation the model rarely puts a
hen in, and the priority of fixing it should be discounted accordingly.

## 3. Prediction

**(a) Localisation:** the qualitative pattern replicates — most of the loss still occurs
at the sensory→pallium projection (fan-in dilution), recurrence remains not the cause
(zeroing it does not improve separability), and Field-L-style segregation still recovers
roughly 2× without closing the gap. Reasoning: E/I balance changes drive statistics and
gain sensitivity, not the fan-in structure (~19 stub inputs per pallial unit) that E017
identified as the mechanism, and E023's unchanged aggregate number is consistent with the
same mechanism still dominating.

**(b) Occurrence:** the auditory aerial-call channel is no longer constant (confirming
E019's fix reaches this specific channel), and the "blind AND hearing an alarm" state
occurs on a non-trivial minority of hen-steps during active predation — enough that H2d's
contrast describes something a hen actually encounters, not a hypothetical.

## 4. Falsifier

**(a)** The loss is no longer dominated by the feedforward projection — e.g. zeroing
recurrence now *improves* separability substantially, or Field-L segregation recovers
much more or much less than ~2× — meaning E017's mechanism needs revision on the
corrected connectome, not just re-confirmation.

**(b)** The auditory aerial channel is still ~constant (E019's fix didn't reach live
operation the way the unit tests suggested), or "blind and hearing" is vanishingly rare
(<0.1% of hen-steps) even while hawks are active — meaning H2d's contrast still barely
occurs and the representational finding, while numerically confirmed, is not yet shown to
matter for anything a hen does.

## 5. Design

**(a) Localisation:** re-run the existing, unmodified E017 scripts —
`scratchpad/where_it_collapses.py` (stage-by-stage separability), `scratchpad/
modality_mixing.py` (input-stage overlap), and `scratchpad/why_pallium_collapses.py`
(the recurrence-removed / Field-L-segregated ablations — the specific piece E023 §6
flagged as not yet re-run) — against `hen/connectome.build`'s current defaults (gain
0.95, E/I-corrected). All three already use matched full-amplitude percepts presented
alone (hawk overhead vs. flockmate's aerial call, nothing else) held for 2 s of settling
time, replicated across 6 genomes, exactly as E017 and E023 did. No design change; the
connectome underneath them has changed.

**(b) Occurrence:** new script, `scratchpad/e034_occurrence.py`. A 5-minute live rollout
(`run.simulate.rollout`, full per-step trace) at the standard H4 configuration — 16 hens,
hawk every 20 s, matching what E026–E033 already treat as "the coop" — with plasticity
off (a fixed hen; occurrence is a property of the world and the reflex arc, not of
learning). Reads the visual aerial channel (already head-down-gated in
`coop/sensing.py`) and the auditory aerial-call channel per hen-step, and reports the
joint distribution.

**Command:**
```bash
python -m scratchpad.where_it_collapses
python -m scratchpad.modality_mixing
python -m scratchpad.why_pallium_collapses
python -m scratchpad.e034_occurrence --minutes 5 --hawk-period 20
```

## 6. Result

**(a) Localisation**, 6 genomes, corrected connectome (gain 0.95, E/I-fixed):

```
stage-by-stage separability (RMS diff / mean activity):
  sensory       1.0631 +- 0.2273     (E017: 1.055 +- 0.221)
  pallium       0.0735 +- 0.0279     (E017: 0.062 +- 0.012)
  arcopallium   0.0701 +- 0.0264     (E017: 0.059 +- 0.013)
  motor_stub    0.0578 +- 0.0251     (E017: 0.045 +- 0.012)

input stage (modality_mixing.py):
  auditory share of stub afferent weight : 0.075 +- 0.009  (matches 4/53 = 0.075 by count)
  stub units driven by BOTH hawk and call: 0.094 +- 0.027
  cosine(hawk drive, call drive) on stub : 0.245 +- 0.127
  input-stage separability (x mean drive): 2.700 +- 0.190

ablation (why_pallium_collapses.py), pallial separability:
  intact                 0.0735 +- 0.0279   1.00x
  no recurrence           0.0641 +- 0.0150   0.87x
  targeted (Field L)      0.1069 +- 0.0292   1.45x   <- does not replicate, see below
```

**Correction, added after [E035](E035-modality-segregation-in-the-prior.md):** the
1.45× "targeted (Field L)" figure above does not survive a properly paired, re-normalised
re-measurement. `why_pallium_collapses.py` zeroes connections on an already
fan-in-normalised connectome without re-normalising afterward, confounding "segregated"
with "less total input drive." A structural implementation that re-normalises correctly,
tested on a *paired* 12-genome sample, found segregation indistinguishable from intact
(t=0.04 against threshold 2.201). **Do not cite 1.45× (or E017's 2.06×) going forward.**
The localisation finding two paragraphs up (sensory→pallium fan-in dilution, recurrence
not the cause) is unaffected — it is a within-genome comparison and was never exposed to
this confound.

**(b) Occurrence**, 5 min at H4's standard config (16 hens, hawk every 20 s), fixed hen,
480,000 hen-steps:

```
visual aerial channel 'on' (sees hawk)      : 17.956%
auditory aerial channel 'on' (hears alarm)   : 29.738%
blind to hawk AND hearing an alarm           : 11.929%
  -- of hen-steps where an alarm is audible, 40.1% are also blind

auditory aerial channel: min=0.0000 max=1.0000 mean=0.2202 std=0.3708
visual aerial channel:   min=0.0000 max=0.7549 mean=0.0680 std=0.1704

distinct flock-wide alarm-audible episodes: 11   (~15 possible at this period/duration)
distinct flock-wide hawk-visible episodes:  11
```

## 7. Interpretation

**(a) Localisation replicates qualitatively, with one real quantitative miss.** Every
number moved slightly upward from E017's (sensory 1.06 vs 1.055, pallium 0.074 vs 0.062,
etc.) — consistent with E023's aggregate finding that separability is "unchanged to
within noise" on the corrected connectome — and the ordering is identical: massive loss
at the sensory→pallium projection (~14.5×, against E017's 17×), further mild decay
downstream, recurrence removed making things slightly *worse* not better (0.87×, against
E017's 0.79× — same direction, same conclusion: **recurrence is not the cause**).

~~**Field-L segregation is where the prediction misses.** §3 predicted "roughly 2×,"
extrapolating E017's 2.06×. The corrected connectome gives **1.45×** — real, same
direction, but nearly a third smaller.~~ **Superseded by E035, run immediately after
this section was written**: neither 2.06× nor 1.45× replicates on a paired sample: see
the correction above the result table. What looked like "a real quantitative miss" was
this project repeating E017/E023's own mistake — an unpaired 6-genome ratio-of-means on
a quantity with 6× genome-to-genome spread — one measurement short of catching it.

**(b) Occurrence is a clear, strong confirmation.** The auditory aerial channel is
nowhere near constant (std 0.37, spans its full 0–1 range) — E019's fix reaches live
operation, not just the unit tests it was verified against. **"Blind and hearing" is not
an edge case: it is 11.9% of all hen-steps, and 40% of every hen-step where an alarm is
audible at all.** The scenario H2d's diagnosis is built on — a hen who cannot see the
hawk herself but could, in principle, learn something from a flockmate's call — happens
routinely whenever a hawk is present. This falsifier does not fire; prediction (b) is
confirmed outright.

## 8. Consequence

- **H2d stays `SUPPORTED as a limitation`, now on a genuinely current measurement.** The
  representational bottleneck is unchanged by the E/I fix (confirming E023), the
  mechanism (feedforward fan-in dilution, not recurrence) is unchanged by it too, and —
  new information — **the scenario this diagnosis describes is not rare**: hens
  routinely find themselves blind to a hawk while a flockmate calls, in the coop the rest
  of the tree already treats as standard (H4's config). H2d is no longer resting on an
  artificial contrast; it is resting on something that happens roughly 12% of the time
  whenever the world has a predator in it.
- ~~**The Field-L segregation number needs updating wherever it is cited**: 1.45×, not
  2.06×. It is still a real, biologically-motivated partial fix and still not sufficient
  on its own against a ~14–17× loss.~~ **Superseded by E035**: neither number is safe
  to cite. Modality segregation has no measurement supporting it as a partial fix.
- **This raises H2d's priority, not lowers it.** The concern that demoted H2d from the
  critical path — that its contrast never occurs — is now answered the other way: it
  occurs often. Combined with H2b (the pathway can modulate but not initiate) and H2c's
  correct architecture (associate into the sensory representation, not the motor output)
  being blocked specifically by insufficient separability at that representation, H2d is
  the most direct remaining lever on H2c, H3 and everything below them.
- ~~**Backlog items this strengthens rather than opens fresh**: modality-segregated
  afferents (measured here again, smaller than thought but real)~~ — **downgraded by
  E035**, not strengthened: the one paired test this has had found nothing. The
  innate auditory reflex scaffold (E018, already built, off by default) is unaffected
  by this correction and remains the live candidate — it doesn't fix separability but
  sidesteps needing it for the specific crouch-on-call behaviour, since the reflex arc
  would supply the association instead of the pallium discovering it.
- **Done in E035**: modality segregation was moved into the connectivity prior
  (`connectome.build(modality_segregated=True, aud_fraction=...)`) as originally
  intended here, and that implementation is what caught this section's own error.
