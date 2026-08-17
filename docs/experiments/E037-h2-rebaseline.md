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

*Pending — filled in after the run, not before.*

## 7. Interpretation

*Pending §6.*

## 8. Consequence

*Pending §6.*
