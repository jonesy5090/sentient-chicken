# E004 — does the learning effect survive replication?

> **Pre-registered.** Sections 1–5 were written and committed while the run was still
> executing. Sections 6–8 were filled in afterwards.

## 1. Parent hypothesis

**H2** — three-factor plasticity produces measurable behavioural improvement.

## 2. Question

[E003](E003-does-the-fixed-readout-rescue-learning.md) found learning-without-growth
improving drive regulation at t=2.50, short of the t=3.18 threshold at four seeds.
Does the effect clear significance at twelve?

## 3. Prediction

Made before the run, from E003's numbers.

**Primary.** Learning without growth beats the fixed control on within-run hunger
change, clearing the two-tailed t threshold at p=0.05 with 11 df (**t > 2.20**).

If E003's point estimate (−0.090) and per-seed spread (SD ≈ 0.072, from SE 0.036 at
n=4) are both representative, twelve seeds gives SE ≈ 0.021 and t ≈ 4.3. So the
prediction is that this clears comfortably, and a *marginal* pass would itself be
informative — it would suggest E003's effect size was inflated by small-sample luck.

**Secondary, stated now so it cannot be rationalised later.** Learning *with* growth
remains the weaker condition. It was weaker in both E001 (t=1.00 vs 0.78, both null)
and E003 (t=0.77 vs 2.50). Prediction: its t stays below the no-growth condition's.

**Predator exposure**: no prediction. E003 showed SEs of 2469 on a mean of 13; it is
reported for completeness, not tested.

## 4. Falsifier

- **For H2**: no-growth fails to clear t=2.20. Combined with three runs at this
  duration, that would mean the E003 result was small-sample noise and the mechanism
  story from E002, however clean, does not produce a reliable behavioural effect.
  H2 would move toward `REFUTED at this timescale` rather than staying open —
  the next move would be a longer run, but as a genuinely different hypothesis about
  timescale rather than a rescue of this one.
- **For the growth sub-finding**: growth matching or beating no-growth would kill the
  "rewiring destabilises learning" reading and make E001/E003's ordering a
  coincidence.

## 5. Design

Identical to E003 in every respect except seed count. Nothing else was touched
between the two runs — same commit, same defaults, same duration, same coops.

- **Conditions**: fixed / learning without growth / learning with growth.
- **Matched across conditions**: seed, coop layout, genome, predator arrival times.
- **Primary metric**: within-run change in mean flock hunger, `last third − first
  third`, paired against the fixed control.
- **Replicates**: 12 seeds (0–11). E003's seeds 0–3 are a subset, so this is a
  superset rather than an independent sample — stated because it means the two
  results are not fully independent.
- **Threshold**: two-tailed t at p=0.05, 11 df = 2.20.
- **Command**: `python -m run.experiment --minutes 20 --seeds 12`

## 6. Result

_Pending — run in progress._

## 7. Interpretation

_Pending._

## 8. Consequence

_Pending._
