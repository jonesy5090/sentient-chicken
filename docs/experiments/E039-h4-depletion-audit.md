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

*Pending — filled in after the run, not before.*

## 7. Interpretation

*Pending §6.*

## 8. Consequence

*Pending §6.*
