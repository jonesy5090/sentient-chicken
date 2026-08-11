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

12 matched seeds, 20 min of chicken time, 16 hens. Wall clock 47 min.

| condition | hunger early | hunger late | change | fed % | exposure | synapses |
|---|---|---|---|---|---|---|
| fixed (innate only) | 0.303 | 0.330 | +0.027 | 6.3 | 5480 | 36373 |
| learning, no growth | 0.308 | **0.272** | **−0.036** | **7.4** | 5436 | **21148** |
| learning + growth | 0.310 | 0.302 | −0.009 | 7.2 | 4482 | 40753 |

Paired contrasts against the fixed control (11 df, threshold t=2.23):

```
hunger change   learning, no growth   -0.063 +/- 0.016 SE   t=3.93   SIGNIFICANT
hunger change   learning + growth     -0.036 +/- 0.020 SE   t=1.75   suggestive only
exposure        learning, no growth    -44.3 +/- 1460 SE    t=0.03   noise
exposure        learning + growth      -998  +/- 988 SE     t=1.01   suggestive only
```

Across all four runs of the same contrast:

| | E001 (frozen readout) | E003 (n=4) | E004 (n=12) |
|---|---|---|---|
| learning, no growth | −0.031, t=0.78 | −0.090, t=2.50 | **−0.063, t=3.93** |
| learning + growth | −0.017, t=1.00 | −0.046, t=0.77 | −0.036, t=1.75 |

## 7. Interpretation

**H2 is supported for learning without structural growth.** t=3.93 with 11 df is
p≈0.002. A hen who learns regulates her drives measurably better than a
genome-matched, coop-matched hen who cannot: mean hunger *falls* across a run where
the control's *rises*, and she feeds on 7.4% of timesteps against the control's 6.3%.

**Both pre-registered predictions held.** The primary cleared its threshold, and the
secondary — that growth stays the weaker condition — held for the third consecutive
run.

**The effect size shrank by 30%, exactly as the pre-registration flagged it might.**
E003 gave −0.090; twelve seeds give −0.063. So E003's point estimate *was* inflated
by small-sample luck. The prediction anticipated this and it still cleared, because
precision improved faster than the estimate fell (SE 0.036 → 0.016, better than the
predicted 0.021). Predicted t was 4.3; observed 3.93.

The between-seed variance is worth noting on its own: the fixed control's own change
went from +0.064 at four seeds to +0.027 at twelve. Coops differ a lot, which is
why the pairing matters and why four seeds was never going to settle this.

**A caveat on independence.** E003's seeds 0–3 are a subset of E004's 0–11, so the
two results are not independent samples. Eight of the twelve seeds are new.

**Structural growth still does not help, and the efficiency gap is now stark.** The
no-growth hen achieves a significant effect with **21,148 synapses**, having pruned
down from 36,373 innate. The growth hen ends with **40,753** — nearly twice as many —
and does not clear significance. Losing 42% of the innate connectome and getting
better at the task is a developmentally realistic outcome (massive synaptic pruning
during early development is real in birds and mammals alike), but the *ordering* here
is a finding rather than a design choice, and it was not what the architecture
anticipated.

The most plausible reading remains that continuous rewiring destabilises what has
been learned — the growth rule adds synapses on coactivity, which is orthogonal to
whether those synapses help. It should not be dismissed on three runs of a contrast
that switches growth fully on or off; a rate sweep would say much more.

**Predator exposure remains uninformative** at this flock size and duration, as in
E003. The growth condition's t=1.01 is not a signal.

## 8. Consequence

- **H2 → `SUPPORTED`**, scoped: *for learning without structural growth, at 20 min of
  chicken time*. Growth is explicitly not supported.
- **H2a opened**: does structural growth hurt learning? Weaker in all three runs, now
  with a 2x synapse cost attached. Needs a growth-*rate* sweep rather than an
  on/off contrast.
- **Phase 1 is done enough to build on.** The prerequisite for everything in
  `docs/backlog.md` is a learning rule with a demonstrated behavioural effect, and
  that now exists. H3 (does the audience effect emerge without being programmed?) is
  the next hypothesis and the first one that is about communication rather than
  machinery.
- **Default `growth_enabled` left `True`** despite the result. Three runs of a binary
  contrast is not enough to change a default that has a biological justification, and
  H2a will settle it properly. Recorded here so the choice is deliberate rather than
  inherited.
- **Retired for good**: predator exposure as a metric at this scale. Two runs, both
  uninformative.
- **No ethics review triggered.**

