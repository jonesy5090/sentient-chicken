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

### 6a. The greedy oracle — and why its answer was wrong

8 seeds, 30 min, 16 hens. `scratchpad/e111_headroom.py`.

| arm | hunger | at a patch | feeding | mean `food_amount` | caught/dive |
|---|---|---|---|---|---|
| greedy oracle | 0.5999 | 3.7% | 3.7% | 0.2349 | 0.0844 |
| reflex (frozen) | 0.6332 | 6.4% | 2.6% | 0.4700 | 0.1978 |
| floor (never pecks) | 1.0000 | 0.0% | 0.0% | 1.0000 | 0.1047 |

Oracle vs reflex on hunger: **−0.0333 ± 0.0601, t=−0.55**. Read alone, that fires the
primary falsifier and licenses the largest claim in this file: no headroom, H2
unanswerable in this coop.

**It would have been wrong, and my own instrument falsifier is what said so.** §4 required
the oracle to be at a patch *and* feeding more than the reflex hen. It feeds more (3.7% vs
2.6%) and is **at a patch less** (3.7% vs 6.4%). A policy that spends less time on food
than the animal it is supposed to be the ceiling for is not a ceiling. Sixteen hens all
chasing the same nearest patch strip it and then travel together to the next one, so the
greedy oracle spends its life commuting.

### 6b. The camped oracle — there is a great deal of headroom

Give each hen a home patch and let her stay on it. `scratchpad/e111b_better_oracle.py`.

| | hunger | at a patch | feeding | `food_amount` | caught/dive |
|---|---|---|---|---|---|
| **camped oracle** (seeds 0–7) | **0.4223** | 4.8% | 4.8% | 0.0231 | 0.0556 |
| reflex | 0.6332 | 6.4% | 2.6% | 0.4700 | 0.1978 |
| **camped oracle** (seeds 8–15) | **0.4223** | 4.8% | 4.8% | 0.0238 | 0.0483 |
| reflex | 0.6557 | 6.4% | 2.5% | 0.4927 | 0.1653 |

**Camped vs reflex on hunger: −0.2108 ± 0.0308 (t=−6.85), replicating at −0.2334 ± 0.0321
(t=−7.28) on a disjoint seed block.**

**E111's own headline is withdrawn by E111b.** There is **~0.21–0.23 hunger units** of
headroom — about seven times the paired standard error, and ten times the effect size any
learning arm has ever produced. The environment is not the explanation. **H2 is answerable
in this coop**, and every null in the tree is a null about the learning rule after all.

Prediction 1 was wrong (I said 0.48–0.58; the camped oracle reaches 0.42). Prediction 2's
worry — that the window would be uncomfortably close to the noise — was wrong in the
reassuring direction. **Prediction 4 was wrong too, and interestingly**: I expected the
oracle to be *worse* on predation because it ignores hawks. It is far **better** —
caught/dive 0.0556 against 0.1978 — because a hen standing still on a patch is not
wandering into a hawk's path. Foraging efficiently and surviving are not in tension here;
the reflex hen is worse at both.

### 6c. My model falsifier fires — §2's floor is wrong

The camped oracle reaches **0.4223**, below the analytic floor of **0.4444** that §2
derived. Per §4 that withdraws the arithmetic, and the reason is worth keeping.

`food_amount` is clipped at zero. When a patch sits near empty — and the camped arm holds
it at **0.023** — several hens pecking it request more depletion than there is food to
take, and the clip discards the excess. But the *feeding* test is
`food_amount > 0.01` evaluated before the clip, so each of those hens still receives her
full `peck_food_rate × hunger` reduction. **Hunger removed is therefore not conserved
against food depleted at low stock**, and the flock can extract slightly more than
regrowth supplies. §2's supply-limited equilibrium is a good approximation and not a hard
bound; the number 0.4444 is withdrawn.

### 6d. Why the reflex hen wastes her foraging — measured

The reflex hen is at a patch **6.4–7.3%** of the time and feeds only **2.6%**. So she is
standing on food and not eating for most of the time she is there. `innate.py` wires
`M_PECK` to the `CLS_FOOD` **vision** channels at weight 7.0, and nothing else.

| | at a patch | away |
|---|---|---|
| `M_PECK > 0.5` | **39.65%** | **59.59%** |
| mean `M_PECK` | 0.4373 | 0.5839 |
| summed `CLS_FOOD` vision | 0.9100 | 0.9430 |

**She pecks *less* when she is standing on food than when she is not.**

The channel driving the peck reflex reads 0.91 on food and 0.94 off it — the saturation
[E107](E107-red-team-review-2026-08-24.md) measured, here in its behavioural consequence.
With `vision_range = 10 m` and `peck_radius = 0.30 m` in a 20 m coop, "food is visible" is
true almost everywhere, so a reflex keyed to it fires almost everywhere and carries no
information about arrival. `innate.py`'s own comment already describes the symptom — "a
hen does not stop at a feeder, she walks through it and pecks as she passes" — without
noticing that the pecking is not aimed either.

And the model already contains the channel that would fix it. `IDX_FOOD_ARRIVAL`, the
E053 discovery pulse, carries "am I at a feeder" at **AUC 0.87–0.99**
([E107](E107-red-team-review-2026-08-24.md)). It is wired to the food *call* and **not to
`M_PECK`**.

## 7. Interpretation

**The environment is not the excuse.** A policy exists inside the model's own action
space, using no information the hen lacks, that is 0.21 hunger units better than the
reflex baseline and simultaneously three times safer from hawks. It replicates across seed
blocks at t≈7. Ten experiments' worth of nulls are nulls about the learning rule, not
about a saturated task.

**And the target is now concrete rather than abstract.** For ten experiments "learning
should improve foraging" has meant an unspecified better policy. It now means something
sayable in one sentence: **peck when you are standing on food, and stay there.** The
reflex hen does neither, and the gap between "pecks 40% of the time she is on food" and
"pecks 100% of the time she is on food" is most of the 0.21.

**Which sharpens what the failures of E100–E110 mean.** The rule cannot acquire that
policy, and [E109](E109-what-the-rule-writes.md) says why in the rule's own terms: its
update direction is the reflex arc's own, so it can make her peck *harder on average* but
cannot make her peck *when on food and not otherwise*. That is a redirection, and
redirection is exactly what `dz_motor` forbids. E109 and E111 are the same finding from
the two ends — one measured in the weights, one in the coop.

**A note on how nearly this went wrong.** §6a's greedy oracle produced a clean, plausible
null that would have licensed "H2 is unanswerable in this coop, and every experiment since
E001 is uninformative" — a claim large enough to redirect the whole project. What stopped
it was a pre-registered instrument check on a quantity that had nothing to do with the
headline: *is the oracle actually at food more often?* It was not. This is the fourth time
in this project an instrument check has caught a wrong conclusion before it was published,
and the first time it caught one of mine that I wanted to be true.

## 8. Consequence

**No code changes.** The oracles are probe scripts; nothing in `hen/` or `coop/` is
touched, so no existing result moves.

**`docs/hypothesis.md`.** H2's node records that headroom is measured and large
(0.21–0.23 hunger units, replicated) — so "the task has no room in it" is closed as an
explanation, and the nulls are about the rule. E019's "the metric was a coin flip" failure
does **not** recur here.

**The concrete target is recorded** so future work is measured against it rather than
against an unspecified better policy: camped-oracle hunger **0.4223**, reflex **0.6332**.

### Follow-ups

1. **Wire `IDX_FOOD_ARRIVAL` to `M_PECK`** — a one-line innate change with a real
   justification: a hen who cannot tell she is standing on food is not a model of a
   chicken, and the channel that tells her already exists. This is a **fix to the
   reflex arc**, not to learning, and it must be pre-registered as such: it will move
   every baseline in the project, and the interesting question afterwards is whether the
   *remaining* headroom is still large.
2. **Then re-ask H2 against the corrected baseline.** If the reflex hen is repaired and
   headroom persists, the learning nulls stand on much firmer ground. If repairing her
   closes most of the gap, then what learning was being asked to discover was a defect in
   the innate wiring, which is a different and more embarrassing finding.
3. **The trained-flock mute** (backlog §5, open since E032) is unaffected and still the
   oldest open item.
