# E111 — is a positive result reachable in this coop at all?

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **H2**. Not a test of the brain. A test of whether the
**instrument** could show a positive, which `CLAUDE.md` names as the single
highest-value habit in this repo and which H2 has never had.

---

## 2. Question

[E110](E110-postsynaptic-factor.md) closed the fifth and last mechanistic explanation for
H2's null, and left one number worth staring at: **mean hunger equilibrates at ~0.63 in
every arm, including a frozen readout that cannot learn at all.**

Five experiments have asked "why doesn't learning help". None has asked **"how much better
than this is even possible?"**

### The arithmetic says the answer is bounded, and computable

Hunger rises at `1/hunger_fill_s` per hen per second and falls at
`peck_food_rate × hunger` while feeding. Feeding depletes a patch at `food_deplete_rate`,
and a patch regrows at `(1 − amount)/food_regrow_s`, so the **maximum sustainable food
supply** for the whole flock is `n_food / food_regrow_s`, reached when the patches are
held near empty.

At the shipped defaults, 16 hens and 4 patches:

| | |
|---|---|
| hunger gained, whole flock | 16 × 1/1800 = **8.889e-3** /s |
| max sustainable food supply | 4 × 1/300 = **0.0133** units/s |
| hunger removed per food unit | 0.03 / 0.02 = **1.50 × hunger** |
| **equilibrium** | 8.889e-3 = 0.0133 × 1.50 × h → **h = 0.444** |

**No policy whatsoever can hold mean hunger below ~0.444 in this coop.** It is a
resource-limited environment and the limit is set by patch regrowth, not by the brain.

Against the observed 0.633, that leaves a window of about **0.19 hunger units**. Whether
that is enough for any experiment to detect is exactly the question, and it is answerable
by measurement rather than by this arithmetic, which idealises away travel time, crowding
and the `food_amount > 0.01` feeding threshold.

**So: what does the best possible forager actually achieve, and is the gap to the reflex
hen large enough that a learning rule could have demonstrated one?**

## 3. Prediction

1. **The oracle beats the reflex hen, but not by much.** I expect oracle hunger in the
   range **0.48–0.58**, against the reflex baseline's 0.633 — a gap of roughly
   0.05–0.15, and short of the 0.444 analytic floor because travel time and crowding
   are real.
2. **The gap is detectable but not comfortable.** E110's paired SE on hunger differences
   was ~0.02, so a 0.10 gap is about 5 SE. That means H2 is *answerable* here — but the
   entire dynamic range available to any learning rule is about five times the noise, and
   a rule that captured a third of the available headroom would produce t≈1.7 and read as
   a null.
3. **The floor arm will show the metric can move a long way upward.** A hen that never
   pecks should approach hunger 1.0, confirming the measurement is not saturated.
4. **I am unsure whether the oracle will beat the reflex hen on predation.** It ignores
   hawks entirely, so it may well be worse, and that is worth knowing because it bounds
   what a *foraging* improvement can cost elsewhere.

## 4. Falsifier

**Primary — and it changes what the project does next either way.** If oracle hunger is
not better than the reflex baseline by more than **2× the paired standard error**, there
is effectively no headroom: a reflex-only hen is already at the achievable optimum, **H2
is unanswerable in this coop rather than false**, and every null from E001 onward is
uninformative about the learning rule. The environment would have to change — more
patches, faster regrowth, or a task with room in it — before any rule is tested again.
That is a large claim and this is the measurement that would license it.

**Instrument falsifier — reported before the headline.** The oracle must actually forage
better: fraction of steps at a patch, and fraction of steps feeding, both clearly above
the reflex hen's. If they are not, the "oracle" is not one and its hunger number means
nothing.

**Model falsifier.** The oracle's hunger must not fall *below* the 0.444 analytic floor.
If it does, my model of the environment's resource limit is wrong and §2's whole argument
has to be withdrawn — which would be a more important finding than the headline.

**Triviality falsifier.** The floor arm must be clearly worse than both. If a hen that
never pecks does as well as one that does, the hunger metric is not measuring foraging.

## 5. Design

**No model changes. No flag.** The oracle is a hand-written motor vector computed
directly from world state in a probe script, driving `world.step` with the brain bypassed
entirely. Nothing in `hen/` is touched, so nothing here can affect any other result.

**The oracle policy**, per hen per step:

- find the nearest patch with `food_amount > 0.01`
- if within `peck_radius`: `M_PECK = 1`, no forward drive (`peck_stops_walking` is off at
  the defaults, but a hen standing still on a patch is the right comparison)
- otherwise: turn toward it — `M_TURN_L`/`M_TURN_R` from the sign of the heading error —
  and `M_FORWARD = 1`

It ignores hawks, water, cold and flockmates. It is a **foraging** ceiling, which is the
right ceiling for a reward that is ~83% hunger ([E107](E107-red-team-review-2026-08-24.md)).

**Arms**, matched seeds, 8 seeds, 16 hens, 30 min, `hawk_period_s=60`:

| arm | |
|---|---|
| **oracle** | the hand-written forager |
| **reflex** | the shipped hen, `eta_out=0` — E110's frozen control, re-measured here |
| **floor** | motor forced to zero: she never pecks |

**Measured**: mean hunger at the end of rearing; fraction of steps at a patch and
feeding; `caught/dive`; mean `food_amount` across patches, which says directly whether
the coop is resource-limited in practice as §2 claims in theory.

### Cost

~15 minutes.

---

## 6. Result

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*
