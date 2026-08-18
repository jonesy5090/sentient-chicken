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

*Pending — filled in after the run, not before.*

## 7. Interpretation

*Pending §6.*

## 8. Consequence

*Pending §6.*
