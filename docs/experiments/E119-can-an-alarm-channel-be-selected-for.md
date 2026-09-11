# E119 — can an alarm channel be selected for at all in this coop?

> Sections 1–5 fixed before the numbers were read. The three checks were specified in
> `scratchpad/e119_can_predation_be_selected.py`'s module docstring, written and committed
> in that form before it was executed; this file restates them. This is an **instrument
> experiment**, in CLAUDE.md §3's sense — the ladder it gates would be a separate one.

## 1. Parent hypothesis

**H0**, via [E116](E116-generational-selection.md) follow-up 3 — "evolve flocks with an
intact channel against a yoked one. This is the experiment `run/evolve.py` exists for, and
it is the first route to H5."

That ladder is the headline experiment of the whole generational turn. This asks whether
it can be run at all, before two hours are spent running it.

## 2. Question

Is there any fitness criterion in this coop on which a hen who hears about a hawk does
better than one who does not?

## 3. Prediction

Three facts collide, and none of them has been measured together:

- E117 measured **predation repeatability at zero** in all eleven arm-by-block
  measurements, so a criterion containing the predation term spends its selection
  differential on luck.
- E118 found the criterion that *works* is the one with predation **deleted**
  (`caught_weight=0.0`), which took selected lineages from 4-of-8 improving to 8-of-8.
- `coop/world.py` attaches **no physiological cost to being caught**. Hunger moves with
  `at_food_any & pecking` and nothing else; `n_caught_any` feeds the fitness function and
  the reward's `strike_penalty`, and no state variable of the hen herself.

So, before looking:

1. **Being caught is free.** Within-flock correlation between `n_caught_any` and final
   drives will be **|r| < 0.15** and not significant — a caught hen is no hungrier,
   colder or thirstier than an uncaught one.
2. **Avoiding predators is expensive.** Crouch rate will correlate **positively** with
   hunger (**r > +0.2**), because a crouching hen is not pecking. If so, a drives-only
   criterion selects *against* anti-predator behaviour, and an alarm channel would be
   actively penalised.
3. **More hawks will not make predation heritable.** r(caught) will stay below +0.2 at
   `hawk_period_s` of 20 and 10, because E117's block-2 lean was ~+0.15 at 50 s and E116
   found quadrupling the *lifetime* did not help either. If it does rise above +0.3 at
   10 s, that is the density the ladder should run at and this experiment has found it.

I would rather prediction 3 were wrong than right.

## 4. Falsifier

This experiment gates a larger one, so its falsifier runs in the useful direction: **if
all three predictions hold, the H0 ladder as specified cannot be run**, because no
criterion available rewards hearing about a hawk — the predation term is noise, the
drives term is indifferent or hostile to hawk avoidance, and raising predator density does
not fix the first. That is a finding about the *world*, not about the channel, and it
would send the next work to `coop/world.py` rather than to `run/evolve.py`.

Conversely, **if predation becomes repeatable at high density** (r > +0.3 at
`hawk_period_s=10`), the ladder is unblocked and should be run there, with
`caught_weight` recalibrated at that density rather than inherited from E116's 50 s
measurement.

**Against a spurious reading of check A:** a zero correlation between being caught and
being hungry is also what you would see if *neither* varied. So `caught` mean and standard
deviation are reported per condition, and the check is void if the spread is degenerate —
the same trap E117's `readout_1.00` fell into with its exactly-zero predation variance.

**Against check B being an artefact of the flock rather than the hen:** founder variation
is supplied throughout (`founder_sigma=2.0`). E117 measured that without it there are no
behavioural individual differences at all, so any of these within-flock correlations would
be correlating noise with noise, and a null would be a null about the flock rather than
about predation.

## 5. Design

Three checks, one pass over the same rollouts, 8 genome seeds each, at three predator
densities.

| | measured as | prediction |
|---|---|---|
| **A. does being caught cost her?** | within-flock r(`n_caught_any`, final drives) | \|r\| < 0.15 |
| **B. does avoiding hawks cost her?** | within-flock r(crouch rate, hunger), r(flee, hunger) | r > +0.2 |
| **C. is being caught hers?** | r(`n_caught_any`) across two replicate lifetimes | < +0.2, at every density |

**Held identical:** 16 hens, 600 s lifetimes, `founder_sigma=2.0`, plasticity off, the
coop layout and starting positions per seed. Only `hawk_period_s` varies: **50, 20, 10**.

**Primary metric:** check C's r(caught), Fisher-z averaged over 8 seeds, per density. It
is the one that decides whether the ladder runs.

**Secondary, pre-declared (not exploratory — they are predictions 1 and 2):** checks A and
B, same averaging.

**Replicates:** 8 seeds per density, matching E117 so the numbers sit on the same footing.
Roughly 12 minutes.

**Command.**

```bash
PYTHONPATH=. python scratchpad/e119_can_predation_be_selected.py
```

## 6. Result

8 genome seeds per density, `founder_sigma=2.0` throughout, 600 s lifetimes. Fisher-z
averaged; bar at df=7 is 2.365.

### 6a. Predation *is* heritable — at a density nobody had measured

| `hawk_period_s` | caught/hen | sd | **r(caught)** | r(−drives) |
|---|---|---|---|---|
| 50 | 1.75 | 1.13 | **+0.049 ± 0.093, t=0.53** | +0.501 ± 0.132, t=4.18 |
| 20 | 2.92 | 1.44 | **+0.179 ± 0.074, t=2.46** | +0.717 ± 0.074, t=12.12 |
| 10 | 4.67 | 2.21 | **+0.478 ± 0.079, t=6.57** | +0.716 ± 0.083, t=10.85 |

**Prediction 3 is wrong, and this is the best outcome available.** Whether being caught is
a property of the hen depends entirely on how often hawks come. At one dive per 50 s a hen
is caught 1.75 times in her life and the count is pure noise; at one per 10 s she is caught
4.67 times and **her rate is strongly repeatable**.

This is a resolution limit, not a fact about hens. Catches are rare events sampled around a
hen-specific rate, and at 1.75 events the sampling noise swamps the rate. **E117's "zero
predation repeatability", measured at 50 s and stated across eleven arms, is a statement
about the predator density it was measured at**, and E117 §7's conclusion that the H0
ladder is blocked was drawn one density too early. Corrected here.

Note also that r(−drives) does not rise between 20 s and 10 s (0.717, 0.716) while
r(caught) nearly triples. The two are decoupled, so this is not a general "more events,
more signal" artefact of the harness.

### 6b. Being caught costs a hen nothing she is scored on

| `hawk_period_s` | r(caught, drives) | r(crouch, hunger) | r(flee, hunger) |
|---|---|---|---|
| 50 | +0.001 ± 0.111, t=0.01 | +0.096 ± 0.109, t=0.89 | +0.060 ± 0.094, t=0.64 |
| 20 | −0.084 ± 0.083, t=−1.02 | −0.052 ± 0.099, t=−0.53 | +0.114 ± 0.099, t=1.15 |
| 10 | −0.120 ± 0.106, t=−1.13 | +0.125 ± 0.134, t=0.94 | +0.110 ± 0.117, t=0.94 |

**Prediction 1 holds.** A hen who is caught four or five times is no hungrier, colder or
thirstier than one who is never caught — r(caught, drives) is null at every density, and
if anything negative. This matches the source: `coop/world.py` moves hunger with
`at_food_any & pecking` and nothing else, and `n_caught_any` feeds the fitness function
and the reward's `strike_penalty` but no state variable of the hen herself. **Predation in
this coop is a scoring convention, not an injury.**

**Prediction 2 is wrong, and its being wrong is a relief.** Crouching does *not*
measurably cost hunger (t ≤ 0.94 everywhere). I expected a drives-only criterion to select
*against* anti-predator behaviour; it does not. It is **indifferent** to it, not hostile.
The likely reason is that crouch rates among these hens are low and the hunger dynamics are
dominated by whether she is standing on food at all, not by the fraction of time she spends
head-up.

### 6c. Predictions scored

1. Being caught is free — **right** (null at all three densities, and the source agrees).
2. Avoiding predators is expensive — **wrong**. It is free too; the criterion is blind to
   hawks, not biased against them.
3. More hawks will not make predation heritable — **wrong**, decisively: +0.049 → +0.478.

Two of three wrong, and the important one was wrong in the direction that unblocks the
work rather than the one that would have ended it.

## 7. Interpretation

**The H0 ladder is unblocked, and it must be run at `hawk_period_s=10`.** At 50 s a
generational contrast between an intact channel and a yoked one would have been scored on
a quantity carrying no information about the hens being selected, and would have returned a
null that meant nothing. E118's fix — delete predation from the criterion — was the right
call *at the density E118 ran*, and is the wrong call at 10 s, where predation is the more
heritable of the two terms available.

**This corrects E117, and the correction is of a familiar kind.** E117 measured a null and
stated it eleven times over, which made it feel robust; it was eleven measurements of the
same under-powered configuration. The project's own §3 table lists seven conclusions about
the brain that were really conclusions about a broken instrument, and "predation carries no
signal" was very nearly the eighth. What saved it was asking whether the null *could* have
been positive, at a setting the null had not been measured at.

**What this does not license.** It does not show that an alarm channel helps. It shows that
a criterion exists on which helping would be visible — which is the precondition, not the
result. In particular, being caught still costs the hen nothing physiologically, so
predation enters fitness purely as a scoring choice by me. A reader is entitled to ask
whether selecting on a cost the organism does not experience is modelling evolution or
modelling a preference, and the honest answer is that it is the latter dressed as the
former. Making a strike carry a real homeostatic cost would be the principled fix and is
not attempted here.

**`caught_weight` cannot be inherited.** E116 set it to 0.10 to equalise the two terms'
spreads at its own density. At 10 s, sd(caught) is 2.21 against 1.13 at 50 s, and — more
importantly — E117/E118 showed that equalising *spread* is the wrong principle when one
term's spread is mostly noise. The weight should equalise the terms' **repeatable**
variance, which is now measurable: r(caught)=0.478 and r(−drives)=0.716.

## 8. Consequence

**`docs/hypothesis.md`.** H0's "blocking the ladder" note is corrected: the blocker was a
predator density, not a property of the model. The ladder is live at
`hawk_period_s=10`.

**`docs/backlog.md` §8.** E116's follow-up 3 is unblocked. New item: predation is a
scoring convention with no physiological cost, and making a strike actually cost the hen
something is the principled repair.

**E117 is corrected in place**, per this project's convention, with a pointer here: its
"predation repeatability is zero" is true at `hawk_period_s=50` and false at 10.

**Next:** E120's pre-checks — does the aerial channel vary at this density, does the yoked
control still destroy the information it claims to, and can a hand-wired comprehension
(`auditory_scaffold=True`) reduce catches here at all. The last is the mandatory positive
control; without it a null ladder would again be uninterpretable.

**Code.** `scratchpad/e119_can_predation_be_selected.py`.
