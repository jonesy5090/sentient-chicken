# E013 — the clean test of H2

> **Pre-registered.** Sections 1–5 written and committed before the run was launched.

## 1. Parent hypothesis

**H2** — three-factor plasticity produces measurable behavioural improvement.
`UNDER TEST`. No run since [E009](E009-lagged-pallial-association.md) has tested it
without a confound.

## 2. Question

Does a hen that learns regulate her drives better than a genome-matched, coop-matched
hen that cannot — measured in an environment where the metric is not being swamped by
something else?

## 3. What changed first, and why

[E012](E012-corrected-phase-1-contrast.md) found that `call_energy_cost`, added in
E005 for H3, was charged directly to **hunger**, tripling the rate it accumulates and
destroying the metric H2 is measured on. A parameter added for one hypothesis had
silently changed the measurement basis of another.

The cost has now been moved to its own budget, `vigour`:

- Calling spends vigour; silence restores it (~90 s from empty).
- Vigour **enters the reward signal**, so calling stays genuinely expensive and H3
  still has a gradient for audience-sensitivity to emerge from.
- Vigour **attenuates the call flockmates hear**, because a spent bird cannot call
  loudly. Vocal effort is now self-limiting without an arbitrary cap.
- **Hunger is untouched by calling.**

Verified before running: a fixed flock's mean hunger returns to **0.323** (it was
0.630 with the cost charged to hunger; E004's regime was ~0.33), vigour cycles in
0.68–1.0 rather than pinning, and the ethogram still passes 7/7.

## 4. Prediction

**A learning advantage returns.** Learning-without-growth beats the fixed control on
within-run hunger change by more than t=2.23 (11 df).

Confidence is moderate, not high. E004's t=3.93 was measured in the saturated regime,
and the gain is now corrected — so the effect could be smaller, or absent for reasons
that have nothing to do with the call cost.

**Secondary predictions:**
- Exploration costs something: the noise-only control is worse than the fixed control.
- Growth stays the weaker of the two learning conditions (as in E001, E003, E004).

**If this is null**, H2 is in real trouble. It would be the first clean test, and a
null with no confound left to blame moves it toward `REFUTED at this timescale`.

## 5. Design

Four conditions, exploration stated explicitly in each, 12 matched seeds, 20 min,
gain 0.70, vigour budget in place. Identical to E004 in every other respect.

- **Command**: `python -m run.experiment --minutes 20 --seeds 12`

## 6. Result

_Pending._

## 7. Interpretation

_Pending._

## 8. Consequence

_Pending._
