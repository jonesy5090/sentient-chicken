# E120 — the H0 ladder: evolve an intact channel against a yoked one

> §1–5 written before the ladder ran. §6a's three pre-checks ran *first*, deliberately —
> CLAUDE.md §3 requires the instrument to be measured before the contrast, and two of the
> three failed at the density E119 pointed to. Their results are in §6a and they chose the
> configuration §5 specifies.

## 1. Parent hypothesis

**H0 — root.** This is the experiment `run/evolve.py` was built for and the one
[E116](E116-generational-selection.md) named as follow-up 3: *evolve flocks with an intact
channel against a yoked one.* It is the first route this project has to **H5**, because a
transmission bottleneck across generations is what compositional structure requires.

It is reached by a chain: E116 built the machinery, E117 found the founding flock is not a
population, E118 found the criterion rather than the search was stalling it, E119 found
predation is heritable at high predator density, and E120's own pre-checks found the
density where an alarm can actually matter.

## 2. Question

Given a flock that can be selected at all, does selection across generations discover
comprehension — a use for what flockmates say — that within-lifetime learning never did?

## 3. Prediction

**I expect a null or a very small effect, and I am running it anyway because the
pre-checks now guarantee a null would be informative.** The reasoning, stated before the
run so it can be scored:

- **The benefit exists but sits in the wrong place.** The positive control that works
  (§6a) is `auditory_scaffold`, which lives in `p.reflex`. The reflex arc is never plastic
  and `evolve._mutate` never touches it. Selection acts only on `W` and `W_out`.
  [E027](E027-h4-lesion.md) already found H4's effect surviving a full `W_out` lesion — it
  was carried by the arc. So the ladder asks the evolvable substrate to rebuild, from
  scratch, a benefit the project has only ever demonstrated through a pathway selection
  cannot reach.
- **The signal is weak.** r(heard aerial, hawk-on-me) is **+0.194**, not E026's +0.56.
- **The scoreboard is half noise even at its best.** r(caught) is **+0.179** here, against
  r(−drives) **+0.717**.
- **The search is short.** E118 found every arm doing 90% of its work by generation ~2 and
  ending at 11–13% of founding diversity.

Numerically: **intact-minus-yoked reduction in catches per dive below +0.005**, against a
hand-planted ceiling worth −1.195 catches per hen. If it clears **+0.01** I will have been
clearly wrong, and that is the result worth having.

## 4. Falsifier

**For H0:** if an intact channel produces no greater reduction in predation than a yoked
one, across generations, on a substrate where selection demonstrably works (E118: 8/8
lineages improved) and at a density where a planted comprehension demonstrably helps
(§6a: −1.195 catches, t=−2.82), then **the generational route to H0 has been tried and
failed on its own terms** — and, unlike every previous null in this project, not because
the instrument could not have shown a positive. That is the specific claim these five
experiments were built to make testable.

**Against a spurious positive:** the channel must help *predation specifically*. Hunger
improvement is reported as a matched secondary, and the channel has no mechanistic route
to it. If intact beats yoked on hunger as much as on catches, the difference is not about
communication and is recorded as a confound, not a result — the same logic that made
E058's crouch effect collapse when three unrelated channels moved with it.

**Against the arms differing in anything but information:** both carry the 4096-step call
ring buffer even though only the yoked arm reads it, so the two are on an identical code
path with identical memory traffic. Channel content is the only difference.

## 5. Design

**Configuration, chosen by §6a's pre-checks, not inherited.** `hawk_period_s=20` — the
only one of three densities where predation is heritable, the channel is informative, *and*
a planted comprehension saves hens. 16 hens, 600 s lifetimes, 16 generations, 4 lineages
per arm, `founder_sigma=2.0` (without which E117 measured nothing selectable).

**Founders carry no auditory scaffold.** Comprehension must be discovered, not supplied.

**`caught_weight` recalibrated, not inherited.** E116 chose 0.10 by equalising the two
fitness terms' raw spreads; E117/E118 showed that is the wrong principle, because a term's
spread counts for nothing if none of it is repeatable. Equalising the terms'
**repeatable** spread (`sd·√r`, using E119's r) gives **0.1049** at this density —
measured at sd(caught)=1.6775, sd(drives)=0.0879. It lands within 5% of E116's value by a
different route, which is worth recording as a coincidence rather than a vindication.

**Arms:** `channel_mode="intact"` against `channel_mode="yoked"` — the flock's real call
stream, shifted per hen by more than a hawk dive. Same bandwidth, same rate, same
amplitude distribution, same energetic cost, zero contingency with her own world. E026's
headline control, re-measured here at **8.7% information retained** rather than assumed.

**Primary metric.** Reduction in catches per dive from generation 0 to 15, intact minus
yoked, paired per lineage; 4 lineages give the error bar (df=3, bar 3.182).

**Secondary, pre-declared:** hunger improvement (the confound check from §4); and the
final flocks of *both* arms re-assayed on an **intact** world, so the last measurement is
of the evolved brain rather than of the channel it was reared under.

**Replicates.** 4 lineages per arm in block 1. E116's own bar, and thin — any positive
needs a second block before it changes a status, and this experiment will not claim
otherwise.

**Command.**

```bash
PYTHONPATH=. python scratchpad/e120c_ladder.py
```

## 6. Result

### 6a. The three pre-checks, and the two that failed

Run before the ladder, at `hawk_period_s=10` where E119's predation repeatability pointed.

| check | result at 10 s |
|---|---|
| **A.** does the aerial channel vary? | heard 0.1445 at rest, **0.2704** with a hawk on her — varies, not saturated |
| **B.** does the yoked control destroy it? | intact r=**+0.173**, yoked r=**−0.015** — **8.7% retained**, passes |
| **C.** can a planted comprehension save a hen? | **+0.227 ± 0.503, t=0.45 — null. FAILS.** |

Check C is the mandatory positive control, and its failure blocked the ladder outright: if
a hand-wired alarm→crouch mapping saves nobody, a selected one cannot either, and a null
would have been a fact about the coop.

The cause is structural and visible once looked for. `hawk_dive_s` is **12 s**, so at a
mean onset interval of 10 s dives overlap and the sky is busy **52.6%** of the time. An
alarm that is always sounding is not a warning.

Sweeping density resolves it, and produces a genuine squeeze:

| `hawk_period_s` | sky busy | r(channel) | r(caught) | planted comprehension |
|---|---|---|---|---|
| 10 | 52.6% | +0.173 ± 0.046 | **+0.478** | +0.227 ± 0.503, t=0.45 |
| **20** | **36.9%** | **+0.194 ± 0.054** | **+0.179** | **−1.195 ± 0.424, t=−2.82** |
| 50 | 18.3% | +0.042 ± 0.089 | +0.049 | +0.469 ± 0.295, t=1.59 |

**Selection needs more hawks; the alarm call needs fewer.** Only 20 s satisfies both, and
it is the only density where a planted comprehension saves hens — **1.195 catches each**.
That it is also the density E026/E030 independently chose for H4 is a convergence worth
noting.

### 6b. The ladder: null, at 24 lineages

Six blocks of 4 lineages per arm, `hawk_period_s=20`, 16 generations, founders matched
between arms (same genome key, same world key per lineage).

| intact − yoked, pooled over 24 lineages | estimate | 95% CI | t (bar 2.069) |
|---|---|---|---|
| **primary:** reduction in catches/dive | **+0.0350 ± 0.0201** | [−0.0066, +0.0765] | +1.74 — **null** |
| final flock, catches/hen | **−0.3385 ± 0.3131** | [−0.986, +0.309] | −1.08 — **null** |
| final flock, catches/dive | −0.0169 ± 0.0172 | [−0.0525, +0.0186] | −0.99 — **null** |
| confound: hunger improvement | +0.0117 ± 0.0106 | — | +1.11 — null |
| confound: final hunger | +0.0028 ± 0.0093 | — | +0.30 — null |

15 of 24 lineages favour intact on the primary; **12 of 24 on the final assay — a coin
flip.**

### 6c. The effect shrank as lineages were added, exactly as E021 warns

| block | primary | final catches/hen |
|---|---|---|
| 1 | +0.0777 | **−0.781** |
| 2 | +0.0393 | **−1.016** |
| 3 | −0.0051 | +0.094 |
| 4 | +0.0208 | +0.313 |
| 5 | −0.0142 | +0.422 |
| 6 | +0.0915 | −1.063 |

The first two blocks looked like a result — −0.78 and −1.02 catches per hen, both in H0's
direction, agreeing with each other. Three of the next four blocks had the **wrong sign**.
Pooled, it is nothing. Had I stopped at 8 lineages, as E116's own bar and this project's
recent habit would have allowed, I would have reported a promising effect. **This is E021's
lesson (a t=3.84 evaporating on fresh seeds) reproduced almost exactly, and the only reason
it was caught is that a power calculation said 8 lineages was not enough and I ran 24.**

### 6d. And this null is informative — which is the whole point of the last five experiments

Everything CLAUDE.md §3 demands was established *before* this contrast ran:

- **The manipulated variable varies.** Aerial channel 0.1445 at rest, 0.2704 with a hawk
  on her (§6a A).
- **The control destroys what it claims to.** 8.7% of the information retained, re-measured
  at this configuration (§6a B).
- **A positive result is physically reachable.** A planted comprehension is worth **−1.195
  catches per hen** here (§6a C).
- **Selection works on this substrate.** E118: 8 of 8 lineages improved on the corrected
  criterion.
- **The flock is selectable at all.** E117: repeatability +0.628 with founder variation,
  −0.014 without.

And the design had the power to see it. The per-lineage standard deviation of the
final-assay difference is **1.534**, so detecting the planted ceiling of −1.195 at 80%
power needs **13 lineages**; 24 were run. The smallest effect this design could reliably
detect is **0.877 catches per hen** — and the planted ceiling, −1.195, **falls outside the
95% confidence interval** [−0.986, +0.309].

**So the ladder can exclude an effect as large as a hand-wired comprehension provides.**
That sentence is what five experiments were built to make sayable, and no previous null in
this project could say it.

### 6e. Predictions scored

The §3 prediction was "a null or a very small effect", with a numeric bound of **below
+0.005** on the primary and a stated "if it clears +0.01 I will have been clearly wrong".

**The substance holds and the bound does not.** The primary is +0.0350 — seven times the
bound — but its confidence interval includes zero, so there is no established effect to
have been wrong about. Scored honestly: right that the ladder would return a null, wrong
to have put the bound that tight, and the CI is the number to quote rather than either.

The *mechanism* predicted in §3 survives and is now the leading explanation: the only
comprehension this project has ever demonstrated lives in `p.reflex`, which
`evolve._mutate` never touches, and E027 already found H4's effect surviving a full
`W_out` lesion. Selection was asked to rebuild in `W`/`W_out` a benefit only ever shown
through a pathway it cannot reach, and it did not.

## 7. Interpretation

**The generational route to H0 has been tried and has failed on its own terms.** Not
because the instrument was broken — it was measured working, five different ways, before
the contrast ran — and not because the flock could not be selected, which E117 and E118
established it can. Sixteen generations of selection on a substrate where selection
demonstrably produces improvement, at the one predator density where an alarm can
demonstrably save a hen, with a control that demonstrably destroys 91% of the channel's
information, produced no advantage for hearing your flockmates.

**What the null is about, and what it is not about.** It is not about whether communication
could help this flock — §6a C proves it could, by 1.195 catches per hen. It is about
whether *this evolvable substrate can find that mapping*. The distinction matters because
it points somewhere specific rather than to a general discouragement:

- Mutation touches only `W` and `W_out`. The reflex arc, `W_in`, `dale` and the biases are
  flock-wide constants and permanently outside selection's reach.
- The one demonstrated route from an alarm call to a crouch runs *through the reflex arc*
  (E027: H4's effect survives lesioning `W_out` entirely).
- So the ladder asked selection to build a second, independent route through a pallium
  that E107 measured as emitting a near-fixed direction and E109 measured as confined to
  the arc's own direction.

**Three explanations I cannot separate**, and the honest position is that they are
alternatives rather than a conclusion:

1. **The substrate.** `W`/`W_out` cannot express an alarm→crouch mapping at all, in which
   case making the reflex arc heritable is the experiment, not more generations.
2. **The signal.** r(channel) is +0.194 here against E026's +0.56; the channel may simply
   carry too little for selection to latch onto, and the 12 s dive against a 20 s interval
   still leaves the sky busy 37% of the time.
3. **The search.** E118 found 90% of all improvement arriving by generation 2 with
   diversity down to ~12%, so 16 generations may be 14 generations of nothing. Against
   this: E118 ruled out four separate attempts to extend that horizon.

**What would distinguish them**, in the order I would run them: make `p.reflex` heritable
and re-run the ladder (tests 1 directly, and it is a small change to `_mutate`); decouple
`hawk_dive_s` from `hawk_period_s` so warning intervals exist without a permanently busy
sky (tests 2, and is a change to `coop/spec.py` rather than to the brain); and a
plant-then-select control, where founders carry the scaffold and selection must merely
*retain* it — if selection cannot even hold onto a comprehension it is handed, the problem
is the criterion's grip, not the search's reach.

**A caveat that applies to the whole chain.** Being caught still costs a hen nothing
physiologically (E119 §6b) — hunger moves only with `at_food_any & pecking`. Predation
enters fitness because I put it there at a weight I chose. A reader is entitled to ask
whether selecting on a cost the organism does not experience models evolution or models a
preference, and the answer remains the latter. Giving a strike a real homeostatic cost is
the principled repair and remains unattempted.

## 8. Consequence

**`docs/hypothesis.md`.** H0's generational route is recorded as **tried and NOT
SUPPORTED**, with the power statement attached — this design excludes an effect the size
of a hand-planted comprehension. H4 is untouched: it is a separate claim about the reflex
arc and this says nothing about it.

**`docs/backlog.md` §8.** E116's follow-up 3 is **closed**. Three successor items, in
priority order: heritable reflex arc; decouple dive duration from dive frequency; and the
plant-then-select control.

**No code changes.** `run/evolve.py` is unchanged by this experiment; the recombination
added in E118 stays non-default.

**A methodological note worth more than the result.** Blocks 1–2 (8 lineages, E116's own
bar) showed −0.78 and −1.02 catches per hen in H0's direction and would have been reported
as promising. Sixteen more lineages took it to −0.34 with 12 of 24 favouring intact. **Run
the power calculation before the blocks, not after the ones that look good.**

**Code.** `scratchpad/e120_ladder_checks.py`, `scratchpad/e120b_density_tradeoff.py`,
`scratchpad/e120c_ladder.py`, `scratchpad/e120d_pool.py`.
