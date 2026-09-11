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

*(ladder result to follow)*

## 7. Interpretation

*(written after the run)*

## 8. Consequence

*(written after the run)*
