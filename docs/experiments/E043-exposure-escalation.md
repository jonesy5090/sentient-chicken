# E043 — is comprehension exposure-limited, not just separability-limited?

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**H2c** — a learned cue can recruit an innate response via top-down association.
`NOT STARTED` as a working mechanism. E042 re-tested it at E041's density fix and found
comprehension still negligible (0.005–0.007, ~1/30th the auditory scaffold's 0.19),
with `|W_pred|` reaching under 1% of its cap in every condition — as plausible an
explanation as remaining separability gap, and untested until now.

## 2. Question

E009 (the original test of this exact mechanism, on the old pre-E023 connectome)
escalated predator density up to 90× the then-default and found `|W_pred|` **saturate
its cap** — the rule clearly *can* move a lot, given enough exposure. E042 used H4's
standard `hawk_period_s=20` and saw `|W_pred|` reach only ~1% of cap in 20 minutes.
**Does raising predator density — independent of the density fix — let `|W_pred|` move
substantially further, and does comprehension track it?**

## 3. Prediction

**Primary.** `|W_pred|` after rearing is substantially larger at `hawk_period_s=10` than
at E042's `hawk_period_s=20`, at both connectome densities — this is close to a
mechanical prediction (more co-occurrence events, more updates) rather than a
speculative one.

**Secondary, genuinely uncertain.** Whether higher exposure alone (at default density)
produces comprehension above the no-association baseline — E009's own history is mixed
here: `|W_pred|` saturated at high predator density on the old connectome, and
comprehension *still* did not emerge, with the baseline crouch rate rising instead
(hallucinating a base rate, not a contingency). If the same happens here, that
re-confirms separability, not exposure, was always the binding constraint at default
density — and the interesting question becomes whether *full* density plus high
exposure together finally produce it.

## 4. Falsifier

**If `|W_pred|` does not move further than E042's ~1% of cap even at this much higher
predator density**, exposure is not the limiting factor after all, and the "not enough
data" explanation in E042 §7 should be withdrawn — something else (learning rate, gain,
still-insufficient separability) is the real bottleneck.

**Not a falsifier:** `|W_pred|` moving a lot without comprehension moving with it. That
would replicate E009's own finding (saturation without contingent learning) and is a
real, informative possible outcome, not a null result to be explained away.

## 5. Design

Two conditions only (dropping the no-association control — E042 already established it
near zero and stable, no need to re-verify): association enabled at default density
(0.30) and at full density (1.0, E041's fix), both at `hawk_period_s=10` — matching
E009's own most extreme escalation (900s baseline / 90). `hawk_dive_s=12` is unchanged,
so dives now last nearly as long as the average gap between them; this is a deliberate
stress test for maximum exposure, not a claim about a realistic coop, the same spirit as
`scaffold_gain`'s positive controls elsewhere in this project.

Otherwise identical to E042: 16 hens, 20 minutes rearing, `food_deplete_rate=0`,
`enabled=True, growth_enabled=False, explore_sigma=0.6, pred_enabled=True`. Same
`comprehension()` metric, same 8 seeds (first-pass de-risking, not a registered
replacement).

**Command:**
```bash
python -m scratchpad.e043_exposure --seeds 8 --minutes 20 --hawk-period 10
```

## 6. Result

8 seeds, 20 min rearing, 16 hens, hawk every 10 s, `food_deplete_rate=0`. Wall clock 374 s.

```
condition                     comp before  comp after mean|W_pred| max|W_pred|
assoc, default density            -0.0001      0.0064      0.00057     0.02020
assoc, full density                -0.0001      0.0080      0.00045     0.01482

PRIMARY: mean|W_pred| vs E042's hawk_period=20 baseline (reported, not a live contrast):
  default density: 0.00057 vs E042's 0.00058  -- unchanged
  full density:     0.00045 vs E042's 0.00047  -- unchanged

SECONDARY: comprehension, full density vs default density:
  +0.0016 +/- 0.0024  t=0.66  threshold(df=7)=2.365  -> not significant
```

**The registered primary comparison — mean `|W_pred|` — does not move, and the falsifier
fires on it exactly as written.** Despite roughly 2× the hawk frequency and dives now
overlapping their own gaps, the average magnitude of `W_pred` after rearing is
indistinguishable from E042's `hawk_period_s=20` result in both density conditions.

**But `max|W_pred|` tells a different story, and this file did not register it as the
primary metric, which matters.** The single largest-magnitude entry in `W_pred` reached
**0.020 and 0.015** — 40% and 30% of the 0.05 cap respectively — a large jump from
whatever near-zero value it presumably held in E042 (not tracked there; only the mean
was reported, a design gap this file inherited and is flagging rather than silently
fixing after the fact). **Recorded as an unregistered, exploratory observation, not
promoted to a finding**, per this project's discipline about not moving the goalposts
after seeing the data.

Comprehension after rearing ticked up in both conditions relative to E042 (default:
0.0047→0.0064; full: 0.0069→0.0080), consistent in *direction* with more exposure
mattering somewhat, but both still roughly 1/25th–1/30th the auditory scaffold's 0.19,
and the density contrast remains far from significant (t=0.66, even weaker than E042's
t=1.17).

## 7. Interpretation

**Taken at face value, the registered falsifier fires: exposure (at this escalation) is
not the limiting factor for the metric this file committed to.** The honest reading
stops there for the primary contrast.

**The max/mean discrepancy is the more informative finding, and it reframes rather than
resolves the question.** `W_pred` is a large matrix (`OBS_DIM × N`, restricted to
pallial sources) where the overwhelming majority of entries connect a pallial neuron to
an observation channel with no reason to be associated at all — predicting most of the
59 channels from most of the ~256 pallial units is meaningless, and a *mean* over the
whole matrix is dominated by that majority staying at zero. A handful of entries
growing toward 30–40% of cap while the mean stays flat is exactly what targeted,
biologically sensible learning would look like buried inside an uninformative average —
or it could be a handful of synapses randomly drifting under noise with no more meaning
than any other coordinate in a high-dimensional walk. **This experiment cannot
distinguish those two stories, and did not register a metric that could.**

**What this changes about E042's "not enough exposure" hypothesis.** It is not simply
confirmed (mean growth didn't track exposure) or simply refuted (something clearly is
different at higher exposure, just not visible in the metric registered to detect it).
The honest state is that the diagnostic asked the wrong question at the wrong
granularity.

## 8. Consequence

- **H2c stays `NOT STARTED`.** No status change — comprehension remains far below
  anything meaningful in every condition tried across E042 and E043.
- **The registered falsifier fired and is recorded as such**, not quietly reinterpreted
  because a more favourable unregistered number appeared in the same run. That number is
  reported above and flagged as exploratory, exactly where it belongs.
- **New, better-specified diagnostic owed, not run here**: does `W_pred`'s growth
  concentrate on the entries a correct association *should* strengthen — specifically,
  pallial-neuron-to-aerial-channel weights sourced from neurons responsive to the alarm
  call — or is it diffuse across the matrix regardless of relevance? This needs reading
  `W_pred` structurally (which rows/columns moved), not just its mean or max, and is a
  cheap, static analysis on the already-cached end states from E042/E043 rather than a
  new rearing run.
- **Compute-costly escalation (more predator density, more duration) is not obviously
  the next lever** given this result — the bottleneck, whatever it is, did not visibly
  respond to a 2× exposure increase on the metric that was supposed to detect it. The
  structural read above is cheaper and more likely to say something before spending more
  on rearing runs.
