# E039 — does H4's registered contrast survive without food depletion?

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**H4** — `SUPPORTED as written` (E030), pooled 36-seed test, L vs C? on `caught/dive`,
−0.044 ± 0.012, t=3.60. This audit checks whether that result shares the
`food_deplete_rate` confound E037/E038 found in H2's and E032/E033's harnesses.

## 2. Question

E037 found `food_deplete_rate` (added by E025 for an unrelated question, left in
`spec.DEFAULT_COOP`'s permanent defaults) substantially confounds a 20-minute, 16-hen
run. E038 found E032/E033's causal-efficacy result — built on the identical duration and
flock size — did not survive controlling for it. **H4's own ladder runs at 16 hens too**,
and a direct check (this session, before writing this file) found depletion is not
negligible at H4's actual 10-minute duration and 20-second hawk period either: one feeder
already down to **4.3%** remaining by minute 10.

H4's registered metric (`caught/dive`) is intent-to-treat with a denominator fixed at
dive onset — explicitly designed to be unmovable by treatment (E028). Unlike E032/E033's
outer trained-vs-fixed comparison, H4's ladder has **no plasticity in any condition**, so
there is no divergent learning trajectory across conditions to accumulate different
depletion by construction — the mechanism that broke E032/E033's outer comparison does
not obviously apply here. But H4's channel manipulation *does* change behaviour
(hearing a real alarm vs a decorrelated one changes whether a hen interrupts foraging),
so conditions are not guaranteed to forage identically, and a hungrier hen's risk-taking
around a predator is not ruled out as a pathway either. **Does L vs C? on `caught/dive`
hold at `food_deplete_rate=0`?**

## 3. Prediction

**Weak prior that this is more robust than E032/E033 was**, for the structural reason
above (no plasticity, no divergent rearing trajectories), but this exact reasoning
("the contrast is protected because both branches share the confound") was argued and
found insufficient for E032/E033's outer comparison, so it is registered as a weak prior
only, not a confident one. No specific magnitude predicted.

## 4. Falsifier

**If the L vs C? contrast collapses toward zero or reverses sign**, H4's `SUPPORTED`
status needs urgent re-examination on a full, properly powered clean re-measurement,
following the exact path E038 took for H2e. **If it holds** (same sign, comparable
order of magnitude, even without clearing its own threshold at a reduced seed count),
that substantially de-risks H4's headline finding without proving no confound exists.

## 5. Design

`run/h4.py`'s own `run_condition`, called directly (matching `scratchpad/e030.py`'s own
pattern) for the three conditions E030's headline contrast needs: `C? yoked`, `L
language`, `Lx lesioned`. **8 seeds (60–67)**, a subset of E030's own block-C seed range
(60–71) for direct comparability. Identical to E030 otherwise: 10 minutes, 16 hens,
hawk every 20 s, 1.5× pallium, scaffold on, `food_deplete_rate=0`.

Added `--food-deplete-rate` to `run/h4.py`'s CLI for this (previously unparametrised);
the existing cache key already hashes the whole config tuple
(`_key`'s docstring: "E024's 48 cells were cached before E025 added food depletion... The
world is part of the measurement; it belongs in the key" — written after that exact
mistake happened once already), so no cache-key changes were needed, unlike E037/E038's
scripts.

**Command:**
```bash
python -m scratchpad.e039_h4_depletion --seeds 8 --seed-offset 60 --minutes 10 \
  --hawk-period 20 --food-deplete-rate 0.0
```

**Not a full replication.** Same discipline as E038's interim check: 8 seeds is a
de-risking flag, not a registered replacement for E030's 36-seed pooled result.

## 6. Result

8 seeds (60–67), 10 minutes, 16 hens, hawk every 20 s, `food_deplete_rate=0`:

```
C? yoked       caught/dive=0.1444  dives=288.0  fed%=4.15
L  language    caught/dive=0.0677  dives=288.0  fed%=3.69
Lx lesioned    caught/dive=0.0914  dives=288.0  fed%=3.53

PRIMARY -- L vs C?, caught/dive:
  -0.0767 +/- 0.0139  t=5.52  threshold(df=7)=2.365  -> SIGNIFICANT

SECONDARY -- L vs Lx, caught/dive:
  -0.0237 +/- 0.0260  t=0.91  threshold(df=7)=2.365  -> not significant
```

**The falsifier does not fire.** Same sign as E030's pooled result (−0.044 ± 0.012),
comparable order of magnitude (−0.077, within the 2–3× block-to-block spread this
metric already showed in E029/E030's own three blocks), and — unlike E038's 8-seed
interim check on E032/E033 — **significant even at this reduced sample.** `dives` is
identical across all three conditions (288.0), confirming the denominator is still
unmoved by the treatment as designed.

`L vs Lx` stays non-significant (t=0.91, against E030's t=0.46), same qualitative
reading: the effect does not depend on the pallium. Sign differs (−0.024 here vs
E030's +0.010) but both are noise-level and not directly comparable at this n.

## 7. Interpretation

**H4's registered contrast is robust to the confound that broke E032/E033.** The
structural reasoning in §2 — no plasticity in any condition, so no divergent rearing
trajectory to accumulate different depletion the way E032/E033's trained-vs-fixed
comparison could — appears to hold here, whereas the analogous reasoning was tried and
found insufficient for E032/E033's outer comparison. The difference is not surprising in
hindsight: H4's channel manipulation changes what a hen *hears*, not her baseline
foraging drive, and with everything else innate and fixed, there is much less room for
the three conditions to diverge behaviourally over 10 minutes than there was between a
*trained* and an untrained *fixed* 20-minute rearing.

**This is a de-risking check, not a replication, and should not be oversold.** 8 seeds
is not E030's 36. A confound that shows up only at larger n, or in a different part of
the ladder (`N`, `C-`, `C0`, `Cs` were not run here), remains possible in principle.
But a falsifier that was pre-registered to fire on a sign reversal or collapse toward
zero did neither — it fired in the opposite direction, landing on a result *more*
significant than a reduced sample usually produces.

## 8. Consequence

- **H4's `SUPPORTED` status is not disturbed.** No correction needed to E026–E030.
- **The broader `food_deplete_rate` audit item in `docs/backlog.md` is substantially
  closed for H4**, alongside its earlier resolution for H2 (E037) and H2e (E038). E036
  (H2f) remains the one leg reasoned-about but not directly checked.
- **Not run here, and lower priority given this result**: a full 36-seed clean
  re-measurement of H4, or extending this check to the rest of the ladder (`N`, `C-`,
  `C0`, `Cs`). Worth doing eventually for completeness, not urgently — this is the
  opposite situation from H2e, where an interim check found a red flag demanding
  immediate full verification.
- **Pattern worth naming**: two audits of the same confound, two different outcomes.
  The confound was real and consequential for H2's own contrast and for E032/E033's;
  it does not appear consequential for H4's. "The same bug affects every experiment
  that touches this default" was not true, and finding that out required checking each
  one rather than reasoning from one result to the others.
