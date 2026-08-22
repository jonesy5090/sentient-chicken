# E090 — assays that test magnitude, and a hunger term on pecking

*Part 1 (the assay fix) is engineering and is reported as done. Part 2's sections 1–5 are
written and committed before anything is run.*

---

## Part 1 — the assays tested sign, not magnitude

### What was wrong

E089 found the gakel scaffold suppresses pecking by **3.5%** at full amplitude and that
this is behaviourally inert: a correctly planted association, with every other link
validated, moved occupancy by +0.3% against an instrument resolving 5.1%.

**The 3.5% had been printed by every ethogram run for seven experiments.** The assay
reported `gakel peck=0.954 vs contact peck=0.989`, asserted `peck_g < peck_c`, and passed.
Its own docstring warns that "a scaffold that damped everything on the audio bus would
pass a bare sign test while being useless" — and it then applied a bare sign test.

Surveying `run/probes.py`, three assays were bare sign tests and one had a thin fixed
margin:

| assay | old test | measured |
|---|---|---|
| `withdraw on hearing a gakel call` | `peck_g < peck_c` | 3.5% |
| `avoid a sick flockmate` | `away > 0 and away > toward` | 0.63 separation |
| `approach flockmates when cold` | `left_bias > 0 and closed > 0` | +0.12, +0.194 m |
| `contact call when isolated` | `alone > huddled + 0.1` | 72% relative |

The other nine test absolute thresholds (`peck > 0.5`, `crouch > 0.5`) or ratios
(`d_sick < 0.5 × d_healthy`) and were already sound.

### What changed

`MIN_MODULATION = 0.25` — a modulation assay must show a **25% relative change** in the
channel it modulates. `MIN_BIAS = 0.05` where the read-out is a bias rather than a
suppression.

**The threshold is a judgement and is documented as one.** Its grounds: 3.5%
demonstrably produces nothing, so the bar must sit far above it; and **every other
modulation assay clears 25% comfortably** — the contact call at 72% relative, the sick
flockmate at 0.63 separation — so it is not a number invented to fail one thing. That
check is what makes it defensible, and it was done before the threshold was fixed.

### The result: 12/13, and the failure is the point

`withdraw on hearing a gakel call` now **fails**, correctly. It is registered in
`probes.EXPECTED_FAILURES` with the reason and the proposed fix, and marked **`xfail`
strictly** in pytest — so if it ever starts passing, the suite goes red and someone must
update the registry. An assay softened until it passes stops being a guard, which is the
entire finding here.

Suite: **88 passed, 1 xfailed.**

---

## Part 2 — a hunger term on pecking

## 1. Parent hypothesis

`docs/hypothesis.md` → **T2** → **T2-revised**, mechanism 1. Also bears on **H1**, since
it changes an innate reflex.

## 2. Question

Two independent findings converge on the same missing wire.

**E089's design tension.** `_add_gakel_scaffold` deliberately keeps its weight below the
visual arc's so that first-hand information dominates second-hand — a defensible
principle. But with a linear reflex arc feeding a saturating sigmoid, a signal held below
the food drive cannot change behaviour *at all*. Either the call matters or it stays
subordinate; no `SCAFFOLD_WEIGHT` gives both, because there are only two terms and one
must win.

**The rotation question.** T2 rotates which feeder is poisoned, so over time every feeder
acquires an aversive association. What happens then? Measured in the wiring:

- `hunger → M_PECK` is **0.0**. Pecking is driven only by seeing food (+7.0) and water
  (+5.0), opposed by the gakel channel (−1.5). Hunger drives `M_FORWARD` (+2.0) — walking,
  not eating. **A starving hen pecks exactly as much as a sated one.**
- Aversions *do* extinguish: `W_pred` is a delta rule
  (`pred_err = observability × (obs − predicted)`), so predicting a call that does not
  arrive drives the weight down. And because E083's redesign suppresses pecking but not
  walking, she keeps visiting and keeps getting extinction trials. Had the anchor kept its
  original locomotion suppression, she would have avoided the place, never experienced its
  safety, and the aversion would have been self-sealing — the classic avoidance-learning
  trap.

So the equilibrium is set by rotation rate against extinction rate, **with no hunger term
anywhere in it**. If rotation outpaced extinction, a hen would decline every feeder and
starve rather than take a risk. That is wrong biology — risk tolerance scales with need is
one of the better-documented foraging trade-offs — and it is why the existing starvation
falsifier would mislabel the failure as "the manipulation starved her".

**Does adding `hunger → M_PECK` dissolve E089's tension by making the call matter
*conditionally*?** A sated hen defers to the warning; a hungry one overrides it. Three
terms instead of two, with a balance point rather than a winner.

## 3. Prediction

1. **The ethogram assay passes without touching `SCAFFOLD_WEIGHT`.** At low hunger the
   gakel call suppresses pecking by ≥25%; the registry entry is removed.
2. **And the response is conditional**: at high hunger (≥0.8) suppression is
   substantially weaker than at low hunger (≤0.2). This is the whole point — a flat
   improvement would just be `SCAFFOLD_WEIGHT` by another name.
3. **`peck at food` still passes.** A hen with food under her beak must still peck, which
   is E089's saturation working *for* us: the existing assay stages hunger at 0.1 and
   requires `peck > 0.5`.
4. **No other assay moves.** This adds one weight to one channel.
5. **E089 re-run shows avoidance** — occupancy at the planted target falls ≥10% — because
   the anchor can now compete.

I hold prediction 5 least firmly. It requires the conditional response to be strong enough
at the hunger levels the flock actually occupies, and E089 measured mean hunger at 0.33,
which is nearer "sated" than "starving".

## 4. Falsifier

**Primary.** The assay cannot reach 25% suppression at low hunger without a food-drive
weight that breaks `peck at food` at hunger 0.1. The two requirements would then be
incompatible in the same way E089 found, and the sigmoid — not the wiring — is the
constraint.

**Conditionality falsifier.** Suppression at hunger 0.8 is within 20% (relative) of
suppression at hunger 0.2. The term is then not buying conditional behaviour, only
strength, and `SCAFFOLD_WEIGHT` would be the honest way to get that.

**Regression falsifier.** Any other ethogram assay changes state, or the E057 H2f result
does not reproduce. This touches the innate arc, which every experiment sits on.

**Starvation falsifier.** Mean hunger in a free-running flock rises above its current
baseline by more than 0.1. A hunger term on pecking must not make hens *worse* at feeding.

## 5. Design

### The change

In `hen/innate.py`, add `w(spec.M_PECK, spec.IDX_HUNGER, +H)` alongside the existing food
and water drives. Sweep `H ∈ {0, 2, 4, 6, 8}` — 0 is the current behaviour and the
control.

`SCAFFOLD_WEIGHT` stays at 1.5. The point is to test whether a third term dissolves the
tension, not to win it by raising the second.

**Off by default is not available here** — this is the innate arc, and a config switch on
a reflex would be the eighth opt-in flag. Instead the sweep runs first and the default
moves only if Part 2 clears its falsifiers, with the E057 H2f reproduction as the
regression gate.

### Measurements

1. **Assay sweep.** Gakel peck suppression at hunger 0.2 and 0.8, per `H`. Plus the full
   ethogram at each `H`, for the regression falsifier.
2. **Free-running hunger.** 8 seeds, 20 min, mean hunger per `H`, against `H=0`.
3. **E089 re-run** at the chosen `H`, unchanged otherwise, for prediction 5.

### Cost

~15 minutes for the sweep and the ethogram; ~30 for the E089 re-run.

---

## 6. Result

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*
