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
  **This is deliberate and cited**, not an oversight: `innate.py:78–80` records that
  "neonatal pecking is famously indiscriminate: chicks peck at small objects whether or
  not they are hungry, and refine by experience." Any change here has to preserve that.
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

**Neonatal-indiscriminateness falsifier — added with the revised design.** Unwarned
pecking must differ by less than **2%** between a sated hen (hunger 0.2) and a starving
one (0.8). `innate.py` cites chicks pecking "whether or not they are hungry", and a change
that makes ordinary foraging hunger-dependent contradicts the literature the model is
built against, whatever it does for T2. The arithmetic predicts 0.9996 vs 1.0000 — well
inside — but predicting it is not measuring it.

## 5. Design

### Revised before running — the original design was unworkable, and the arithmetic says why

*§5 originally held `SCAFFOLD_WEIGHT` at 1.5 and swept the hunger term alone, on the
reasoning that raising the scaffold would be "winning the tension rather than dissolving
it". That is impossible, and it took four lines of arithmetic to see. Recorded rather than
silently amended; nothing had run.*

| scaffold | hunger w | unwarned peck (sated/starving) | warned, sated | warned, starving |
|---|---|---|---|---|
| 1.5 | 0 | 0.9991 / 0.9991 | 0.9959 | 0.9959 |
| 1.5 | 8 | 0.9998 / 1.0000 | 0.9992 | 1.0000 |
| 7.0 | 0 | 0.9991 / 0.9991 | 0.5000 | 0.5000 |
| **9.0** | **4** | **0.9996 / 1.0000** | **0.2315** | **0.7685** |
| 9.0 | 8 | 0.9998 / 1.0000 | 0.4013 | 0.9879 |

**Neither term alone can work.** With `SCAFFOLD_WEIGHT` at 1.5 the drive never leaves
saturation, so any hunger weight is swallowed (0.9992 vs 1.0000). With hunger at 0 the
suppression is identical sated and starving at *every* scaffold weight — no
conditionality is available, which is E089's tension restated.

**Together they work, and the reason is the saturation E089 identified.** Because food
drives `M_PECK` at +7.0, adding hunger is invisible when nothing is wrong — unwarned
pecking reads 0.9996 sated against 1.0000 starving — and decisive once a warning has
pushed the drive out of saturation. **The cited neonatal fact is a claim about the
unwarned case, and it survives intact.** That is what makes this a resolution rather than
a trade.

### The change

In `hen/innate.py`, add `w(spec.M_PECK, spec.IDX_HUNGER, +H)` alongside the existing food
and water drives, and make `SCAFFOLD_WEIGHT`'s gakel→`M_PECK` term a separate, larger
value `W`. Sweep the pair: `W ∈ {1.5, 5, 7, 9}` × `H ∈ {0, 4, 8}`, with (1.5, 0) as the
current behaviour and the control.

The alarm scaffold's weight is **not** touched. E089 noted the same argument applies to
it, but changing two reflexes at once would confound the regression gate.

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

### The sweep — the arithmetic holds in the running model

Peck response with food underfoot, warned (gakel call in earshot) against unwarned
(contact call, identical amplitude and position):

| W | H | unwarned .2 / .8 | warned .2 | warned .8 | supp .2 | supp .8 | conditional |
|---|---|---|---|---|---|---|---|
| 1.5 | 0 | 0.9886 / 0.9886 | 0.9538 | 0.9538 | 3.5% | 3.5% | no |
| 1.5 | 8 | 0.9977 / 1.0000 | 0.9903 | 0.9999 | 0.7% | 0.0% | useless |
| 5.0 | 0 | 0.9886 / 0.9886 | 0.4174 | 0.4174 | 57.8% | 57.8% | no |
| 7.0 | 8 | 0.9977 / 1.0000 | 0.3423 | 0.9844 | 65.7% | 1.6% | yes |
| 9.0 | 0 | 0.9886 / 0.9886 | 0.0152 | 0.0152 | 98.5% | 98.5% | no |
| **9.0** | **4** | 0.9949 / 0.9995 | 0.1895 | 0.2742 | **96.7%** | **72.6%** | marginal |
| 9.0 | 8 | 0.9977 / 1.0000 | 0.0709 | 0.9026 | 92.9% | 9.7% | strongly |

**Neither term alone works, as predicted.** At W=1.5 adding hunger makes suppression
*worse* (3.5% → 0.7%) — the drive never leaves saturation, so the hunger term only
deepens it. At H=0 suppression is bit-identical sated and starving at every weight, which
is E089's tension reproduced exactly.

**The neonatal falsifier is clear.** Unwarned pecking spreads 0.9977 → 1.0000 between
hunger 0.2 and 0.8: **0.47%** against a 2% bar. Pecking stays indiscriminate when nothing
is wrong, because saturation hides the hunger term until a warning pulls the drive out of
it. The cited literature survives.

**Starvation falsifier clear, and the term helps feeding.** Free-running, 8 seeds:

| W | H | mean hunger | p10 | p50 | p90 | vs control |
|---|---|---|---|---|---|---|
| 1.5 | 0 (control) | 0.3997 | 0.311 | 0.364 | 0.504 | — |
| 9.0 | 4 | 0.3847 | 0.295 | 0.344 | 0.507 | −0.0150 |
| 9.0 | 8 | 0.3541 | 0.285 | 0.320 | 0.433 | −0.0456 |

Hunger *falls*: hungry hens peck harder and feed better.

### 6b. The conditionality is measured in the wrong place, and it matters

*Post-hoc.* The sweep tested hunger 0.2 and 0.8. The free-running flock lives at
**p10–p90 = 0.285–0.433, median 0.320** — it never approaches 0.8. Suppression across the
range the hens actually occupy:

| hunger | | W=9 H=4 | W=9 H=8 |
|---|---|---|---|
| 0.285 | flock p10 | 95.4% | 86.9% |
| 0.320 | flock median | 94.7% | 83.4% |
| 0.433 | flock p90 | 92.0% | 67.0% |
| 0.800 | swept "starving" | 72.6% | 9.7% |
| | **range within p10–p90** | **3.4 pts** | **19.9 pts** |

**H=4 is effectively unconditional where the flock lives.** The conditionality that made
it clear the falsifier happens at a hunger level the hens never reach. H=8 varies by 20
points inside the occupied range and is genuinely conditional in practice.

**And my selection rule picked H=4** — "strongest sated suppression among those clearing",
which optimises one axis and checks the other. That is the same error as E088's, made
again, and caught only by measuring where the flock actually sits.

### 6c. But the regression falsifier fires at H=8, on the load-bearing assay

| config | head-down blindness | gakel withdraw | ethogram |
|---|---|---|---|
| 1.5 / 0 (current) | PASS — 111/288 steps | FAIL (3.5%) | 12/13 |
| **9.0 / 4** | PASS — 109/290 steps | **PASS (72.6%)** | **13/13** |
| 9.0 / 8 | **FAIL — 399 head-down / 0 head-up** | FAIL (9.7%) | 11/13 |

**At H=8 the hen pecks continuously and never lifts her head.** The assay stages hunger
0.6; H=8 adds +4.8 to the peck drive and she never stops. That removes the
vigilance/foraging alternation — the head-down gate, which `CLAUDE.md` calls "the whole
thesis in one line" and "the load-bearing wall", because without it no signal is ever
worth making at any brain size.

The second H=8 failure is different in kind: `withdraw on hearing a gakel call` stages
hunger **0.8** (`probes.py:377`), so it tests the *starving* case — exactly where a
conditional design should show weak suppression. That hunger value was arbitrary when
nothing depended on it and is now load-bearing.

## 7. Interpretation

**The hunger term resolves E089's tension, and the resolution has a ceiling set by
something other than T2.**

At W=9 H=4 the gakel anchor is behaviourally meaningful for the first time: 72.6%
suppression at hunger 0.8 against E089's 3.5%, with the full ethogram at 13/13 and the
previously-registered failure cleared. Unwarned pecking stays indiscriminate. Hens feed
slightly *better*. That is a real fix to a real defect.

**But the conditional behaviour it was built for is not available at any adoptable
weight.** Push H high enough for risk tolerance to vary across the range the flock
occupies, and the hen pecks continuously and stops looking up. **You cannot make hunger
drive pecking hard enough to produce meaningful risk-tolerance variation without
destroying the information asymmetry the entire project depends on.**

That is a second instance of E089's shape — two desiderata that cannot both be satisfied
by linear terms into a saturating sigmoid — and it is the more interesting one, because
the thing it collides with is the project's founding premise rather than a scaffold
weight.

**What would resolve it is a mechanism this model does not have**: a multiplicative gate,
where hunger scales the *aversion* rather than adding to the peck drive. Then risk
tolerance could vary without pecking increasing at all. That is a change to how the reflex
arc composes, not to a weight in it, and it should not be smuggled in as a tuning step.

## 8. Consequence

**Recommended for T2: `gakel_peck_weight=9.0`, `hunger_peck_weight=4.0`.** It clears every
pre-registered falsifier, passes 13/13, and turns a behaviourally inert anchor into a
working one. Its conditionality is weak in practice (3.4 points across the occupied range)
and it should be described as "the warning now works" rather than "the hen weighs risk
against need".

**Rejected: H=8**, on the head-down gate. Recorded with the number (399/0) rather than as
a preference, because it is the more attractive configuration on every axis except the one
that matters most.

**Defaults unchanged.** `gakel_peck_weight=None` and `hunger_peck_weight=0.0` reproduce
the current arc bit-identically; 88 passed, 1 xfailed. Per §5, the default moves only
after the E089 re-run and an E057 H2f reproduction, neither of which has been done.

**Owed: the gakel assay must test both ends.** Under a conditional design, one staged
hunger value cannot validate the response — it should require ≥25% suppression when sated
*and* meaningfully less when starving. Changing it now, having seen which end each config
fails at, would be tuning the test to the result; it should be written as a specification
first. Noting that W=9 H=4 passes at the existing hunger 0.8 anyway, so nothing depends on
the change.

**Next: re-run E089 at W=9 H=4.** Every other link is validated and the anchor can now
compete. That is prediction 5, and it is the test of whether any of this reaches
behaviour.
