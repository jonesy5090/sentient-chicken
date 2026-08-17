# E037 — does H2's clean null survive the corrected connectome?

> **Pre-registered.** Sections 1–5 written and committed before the run starts, pooling
> declared in advance per the E029/E030 template.

## 1. Parent hypothesis

**H2** — three-factor plasticity produces measurable behavioural improvement. `UNDER
TEST`, clean null. Also feeds directly into how H2b, H2e and H2f are read (see below).

## 2. Question

H2's current headline number — learning vs fixed on within-run hunger change, **+0.011 ±
0.012, t=0.95**, pooled across 24 seeds — comes from
[E020](E020-h2-after-the-e019-fixes.md) and
[E021](E021-the-cost-of-exploration.md), both run at **gain=0.70**, before
[E023](E023-ei-fix-and-rebaseline.md) found the pallium had **no inhibitory neurons at
all** and re-baselined gain to 0.95. E023 §6 explicitly listed *"H2, on fresh seeds"* as
the first item owed and it has not been run since. Does the null hold on the connectome
every other current result (E033–E036) is actually measured on?

**Why this matters now, specifically.** E033 refuted H2e (a trained readout is not
inert), E036 opened H2f (the rule may be the wrong kind). Both of those results are
about *what the rule does or doesn't do*, and both were measured on the current,
corrected connectome — but the tree's own account of the rule's headline effect on
foraging (H2 itself) is still resting on the old one. Interpreting H2e/H2f against a
stale H2 baseline risks comparing across two different networks without saying so, the
exact mistake E023 §6 flagged for other results.

## 3. Prediction

**No strong directional prediction.** The E/I fix and gain change move the pallium's
operating point substantially (mean pallial rate 0.276 at gain 0.95 vs 0.189 at the old
0.70, per `hen/connectome.py`'s own docstring), which changes the statistics of the
cortical drive feeding into the readout — but nothing in E023 characterised how, or
whether, that affects the *learned* component of behaviour, only the *representational*
one (separability, unaffected). Registering low confidence rather than a number, per this
project's habit of stating that plainly instead of picking one to look decisive: **the
null most likely persists** (learning stays indistinguishable from fixed on hunger
change), because nothing in the causal chain to this metric — the credit window, the
reward composition, the readout's expressiveness — was touched by E023. But the E/I fix
is exactly the kind of change (H2d's own re-measurement, E034, found real if unchanged
numbers) that this project has repeatedly found moves things nobody predicted.

**Secondary, no prediction, reported for comparability only:** noise-only vs fixed
(E021 found no exploration cost on fresh seeds, t=0.01 pooled 24 seeds — not expected to
change) and learning+growth vs fixed (growth has never won in five prior comparisons).

## 4. Falsifier

**H2's status changes if the pooled 24-seed contrast clears t=2.069 (df=23) in either
direction.** A significant positive result reopens the "learning harms" finding E013
made and E020 withdrew. A significant negative result would be H2's first positive
finding and would need E013–E016's entire withdrawn interpretation revisited a second
time, plus a direct examination of what E023 changed to produce it.

**If the null persists**, H2's status is unchanged, but its supporting number moves from
a stale connectome to the current one, and every subsequent H2-family result (H2e, H2f)
can be read against a consistent baseline for the first time since E023.

## 5. Design

**Identical to E020's `PHASE1`** (`run/experiment.py`, unmodified): four conditions —
fixed, noise-only, learning-no-growth, learning+growth — `explore_sigma` explicit in
each, 20 minutes, 16 hens. No code changes: `run_condition` already calls
`connectome.build` with no gain override, so today's default (0.95, E/I-corrected)
applies automatically.

**Primary metric:** within-run hunger change, learning-no-growth minus fixed, paired
across matched seeds — E020's exact primary.

**Replicates and pooling:** **24 seeds, in two blocks (0–11, 12–23)**, matching E020's
and E021's own seed ranges for direct comparability. Per-seed paired differences from
both blocks are concatenated into one 24-length sample; pooled mean/SE/t computed
directly on that sample (not by averaging block summary statistics), threshold t=2.069
at 23 df — the E030 method, declared here before either block runs.

**Command:**
```bash
python -m scratchpad.e037_rebaseline --seeds 24 --minutes 20 --budget 100000
```

Cached per (condition, seed) cell in `scratchpad/e037_cache.json` so an interrupted run
loses nothing, matching `scratchpad/e032.py`'s pattern.

**Cost estimate.** E020 (12 seeds, 4 conditions, 20 min) took 54 min wall clock; E021
(12 seeds, 5 conditions) took 63 min. 24 seeds × 4 conditions here is roughly double
E020's cost: **~100–120 minutes**, uninterrupted.

## 6. Result

**First pass, as registered — and it does not answer the registered question.**

```
condition                    fed %    hunger early   hunger late   change
fixed (innate only)           2.59        0.3474        0.5459     +0.1985
noise only (no learning)      2.48        0.3501        0.5599     +0.2098
learning, no growth           2.60        0.3414        0.5505     +0.2091
learning + growth             2.54        0.3406        0.5558     +0.2152

PRIMARY, learning vs fixed:
  block A (0-11)   -0.0044 +/- 0.0056  t=0.79
  block B (12-23)  +0.0257 +/- 0.0119  t=2.17
  POOLED (24)      +0.0106 +/- 0.0072  t=1.49   threshold 2.069   not significant
```

**These numbers bear no resemblance to E020/E021's world.** E020 measured fed%=6.6,
hunger change +0.005 for the identical condition at the identical duration and flock
size. Here: fed%=2.6, hunger change **40× larger**. Something beyond the connectome
changed.

**Diagnostic — gain and the readout are ruled out directly.** Lesioning `W_out` (zeroing
it entirely) at both gain=0.70 (E020's own setting) and gain=0.95 (today's default), 4
seeds each:

```
gain=0.70  lesioned=False  fed%=2.45
gain=0.70  lesioned=True   fed%=2.40
gain=0.95  lesioned=False  fed%=2.49
gain=0.95  lesioned=True   fed%=2.40
```

Flat across both manipulations. Since `reflex = obs @ p.reflex.T` reads directly from
the observation with no dependency on the recurrent connectome (`hen/brain.py`), and
`cortical = W_out @ motor_stub` is the only path gain could act through for a
plasticity-off hen, this rules out both gain and the readout as the cause.

**Diagnostic — traced to `food_deplete_rate`.** A full 20-minute trace of a single fixed
hen's world shows food genuinely running low: one of four feeders ends at **0.0097
remaining** (below the 0.01 threshold at which it stops being visible at all), hunger
climbing steadily from 0.32 to 0.60 over the run despite the peck reflex firing
throughout. Setting `food_deplete_rate=0` (4 seeds, otherwise identical) recovers
fed% to **5.21** and cuts the hunger change to **+0.0723** — not all the way back to
E020's +0.005, but an order of magnitude closer, confirming the mechanism.

`git log` confirms the timeline: `food_deplete_rate` was introduced in commit `f659745`
(**E025**), which post-dates both E020 and E021. Its own commit message states the rate
was chosen so a patch "does not run out of food over a 20-minute run" — true at E025's
own 6-minute, single-seed diagnostics, false at `run/experiment.py`'s registered
20-minute, 24-seed duration. See [E025](E025-food-depletion-and-clumping.md), written
alongside this file, for the full history.

**Second pass — `food_deplete_rate=0`, isolating gain/E-I as originally intended:**

```
condition                    fed %    hunger early   hunger late   change
fixed (innate only)           6.21        0.3142        0.3261     +0.0120
noise only (no learning)      5.57        0.3227        0.3479     +0.0252
learning, no growth           6.27        0.3127        0.3250     +0.0122
learning + growth             6.08        0.3122        0.3293     +0.0171

PRIMARY, learning vs fixed:
  block A (0-11)   -0.0445 +/- 0.0150  t=2.96   (SIGNIFICANT alone, one direction)
  block B (12-23)  +0.0451 +/- 0.0206  t=2.19   (SIGNIFICANT alone, opposite direction)
  POOLED (24)      +0.0003 +/- 0.0156  t=0.02   threshold 2.069   not significant

SECONDARY:
  noise only vs fixed:       +0.0132 +/- 0.0157  t=0.84
  learning + growth vs fixed: +0.0052 +/- 0.0133  t=0.39
```

**This matches E020's world closely**: fed% 6.21 against E020's 6.6, hunger early/late
0.314/0.326 against E020's 0.315/0.320 — the residual gap is consistent with the E/I fix
itself rather than a second uncontrolled variable.

## 7. Interpretation

**The registered prediction is confirmed, but only on the second pass.** The null
persists: pooled, learning is statistically indistinguishable from fixed
(+0.0003 ± 0.0156, t=0.02) — if anything a cleaner null than E020/E021's own
+0.011 ± 0.012, t=0.95. Gain and the E/I fix do not move H2's status.

**Block A and block B disagree in *sign*, both individually significant, and this is
worth stating plainly rather than only in a table.** Taken alone, block A says learning
helps (t=2.96); taken alone, block B says learning harms (t=2.19). Either would have
been reported as a finding under this project's old habits, and each would have been
wrong in a different direction. Pooling is not a formality here — it is the only reason
this run reports a null instead of two contradictory "significant" results from the same
24 seeds split differently. This is the E021 lesson recurring in the same experiment
that is nominally about confirming a *different* lesson, and is exactly the kind of thing
worth keeping in the record even though it does not change the conclusion.

**The bigger finding is methodological, not about H2 at all.** The registered design
(§5) assumed "no code changes needed... today's default connectome" was sufficient to
isolate the gain/E-I change. It was not: `spec.DEFAULT_COOP` itself had drifted, silently,
via a change made twelve experiments ago for an unrelated hypothesis (H4's flock
dispersal), calibrated and verified only at a duration and seed count too short to reveal
its effect on a longer, more heavily-replicated harness. **This was found by inspecting
the first-pass numbers against E020's, not by an instrument check before the run** —
exactly the discipline `CLAUDE.md` §3 asks for and this file did not do until the results
looked wrong. Recorded rather than smoothed over.

## 8. Consequence

- **H2's null is confirmed on the corrected connectome, and the number to cite going
  forward is the clean-world pooled estimate: +0.0003 ± 0.0156, t=0.02 (24 seeds,
  gain=0.95, E/I-fixed, `food_deplete_rate=0`).** Not the confounded first pass
  (+0.0106, t=1.49) and not E020/E021's stale +0.011 ± 0.012.
- **`docs/hypothesis.md`'s H2 section updated** with this number and the depletion
  caveat.
- **New, higher-priority backlog item: characterise `food_deplete_rate`'s effect on
  every other tree result that uses `spec.DEFAULT_COOP` at 16+ hens for durations of 10
  minutes or more without an override**, including this session's own E032/E033/E036.
  Not performed here — see `docs/backlog.md`.
- **A standing guard is owed**: some form of assertion that a registered "no code
  changes needed, today's defaults" design has actually checked *every* relevant
  default, not just the one the experiment is nominally about. No such guard exists yet;
  filed as process debt rather than built here, since building it properly means
  deciding what "relevant" means across the whole config surface, not a quick addition.
- **E025 gains a consequence it didn't have when written**: its own stated assumption
  about `food_deplete_rate` not mattering at 20 minutes is now known to be false, cross-
  referenced in both files.
