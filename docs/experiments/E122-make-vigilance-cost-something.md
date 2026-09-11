# E122 — make vigilance cost something, then re-run the ladder

> §1–5 written before anything ran.

## 1. Parent hypothesis

**H0**, via [E121](E121-can-the-evolvable-pathway-see-the-alarm.md) §6d. E120's ladder
returned an informative null; E121 found the reason is not the brain:

> On the scrambled arm, which carries no information by construction, crouch 0.399 → 0.790
> takes catches 3.375 → 0.930 for 0.024 hunger — **net +0.233 fitness, information-free.**
> A communication experiment whose optimal policy is unconditional is not a communication
> experiment.

An alarm call is worth having only when vigilance is expensive enough that you want it
*selectively*. This makes it expensive, then re-runs E120 in the repaired world.

## 2. Question

Is there a coop in which unconditional vigilance stops paying — and if there is, does an
intact channel then beat a yoked one across generations?

## 3. Prediction

**Why vigilance is cheap, measured.** Patches cover **0.28%** of the floor (4 patches of
`peck_radius=0.30` in a 20×20 m run). At `hunger_fill_s=1800` and `peck_food_rate=3.0e-2`,
holding hunger at 0.46 requires **feeding only 4.0% of the time**, and E121c measured her
at a patch **4.5%** of the time. **Foraging here is search-limited, not handling-limited**:
she spends her life looking for food, not eating it, so interrupting the eating costs
almost nothing.

That is also why E112's "camped oracle" only reached 4.8% time-at-patch against the reflex
hen's 3.3% — there is very little patch to camp on.

So:

1. **Raising patch size and lowering intake rate will make foraging handling-limited**, and
   the net fitness gain of doubling the crouch rate will fall from **+0.233** toward zero.
   Predicted: `peck_radius=2.0` with `peck_food_rate=5.0e-3` brings it **below +0.05**.
2. **A setting exists where it crosses zero**, i.e. unconditional vigilance is a net loss.
3. **In that world, the ladder will still be null.** I expect intact − yoked below +0.02 on
   catches per dive. Two of the three E121 mechanisms are untouched by this repair: the
   motor stub still carries the alarm at only AUC 0.687, and planting that direction still
   buys nothing over a scramble. Removing the degenerate optimum is **necessary** for
   communication to pay; it is not obviously **sufficient**.

Prediction 3 is the one I would most like to be wrong about, and it is stated first so it
cannot be revised afterwards.

## 4. Falsifier

**The honesty constraint, stated before any tuning.** World-tuning until the channel helps
would be worthless. So:

- **The calibration target is defined without reference to the channel.** It is a property
  of the world alone: the net fitness gain from an *information-free* increase in crouching
  should be ≈ 0 rather than +0.233. The crouch increase is applied by adding a constant to
  `b_motor[M_CROUCH]` — the purest possible information-free vigilance, no alarm direction
  involved.
- **The world is frozen before the ladder runs**, on that criterion alone, and the ladder is
  run once at the frozen setting.
- **If no setting reaches ≈ 0**, that is reported and the ladder is not run.

**For H0:** if the ladder is null in a world where unconditional vigilance demonstrably
does not pay, then two separate reasons for the null have been excluded — the degenerate
optimum (here) and instrument failure (E120 §6a) — and the remaining explanation is E121's
signal-to-noise limit, which is a fact about the architecture rather than the task.

**Against a hollow calibration:** a setting could reach ≈ 0 by making predation irrelevant
rather than foraging important — e.g. if the hens end up so hungry that the hunger term
swamps everything. So mean hunger, catches, and time-at-patch are reported for every
setting, and any setting whose hunger leaves the 0.3–0.7 band is rejected regardless of its
net figure.

## 5. Design

**Part A — calibrate.** Sweep `peck_radius` ∈ {0.3, 1.0, 2.0} × `peck_food_rate` ∈
{3.0e-2, 1.0e-2, 5.0e-3}. At each setting, sweep a crouch bias added to `b_motor[M_CROUCH]`
∈ {0, 0.5, 1.0, 1.5} and record crouch rate, catches per hen, hunger, and time-at-patch.
Compute the net fitness change at the bias that roughly doubles crouch, using E120's
calibrated `caught_weight=0.1049`.

4 seeds per cell, 600 s lifetimes, `hawk_period_s=20`.

**Part B — re-verify the instrument in the frozen world.** Changing the coop invalidates
E120 §6a, so all three pre-checks are re-run: the aerial channel's contrast, the yoked
control's information retention, and the `auditory_scaffold` positive control. **If the
positive control is null in the new world, the ladder is not run** — same rule as E120,
which is what stopped a meaningless run at `hawk_period_s=10`.

**Part C — the ladder.** Intact vs yoked, 16 generations, `founder_sigma=2.0`, in the frozen
world. Lineage count set by a power calculation on Part C's own pilot variance, not by
E116's habit of 4 — E120 needed 24 and the first 8 pointed the wrong way.

**Primary metric (Part A):** net fitness change from doubling the crouch rate. **Primary
(Part C):** reduction in catches per dive, intact minus yoked.

**Command.**

```bash
PYTHONPATH=. python scratchpad/e122_vigilance_cost.py calibrate
```

## 6. Result

### 6a. The calibration failed — no coop makes unconditional vigilance stop paying

Nine settings (`peck_radius` × `peck_food_rate`), four crouch biases each, 4 seeds.
At roughly doubled crouch the net gain stays **+0.10 to +0.17** in every one. E121's
reference at the default coop is +0.233; nothing came near zero.

The informative column is not the net but **`d_driv`, the drives cost of crouching, which
is ≈ 0 or negative in almost every cell.** Hunger barely moves with crouch bias anywhere —
at `peck_radius=0.3, peck_food_rate=5.0e-3` it goes **0.6019 → 0.6016** while crouch nearly
doubles. Raising patch size and lowering intake changed the *level* of hunger (0.35 → 0.60
across settings) and not its *sensitivity* to lost feeding time.

**Predictions 1 and 2 are both wrong**, and prediction 3 is moot: no world was found to run
the ladder in.

### 6b. The scoreboard was not the problem either

The obvious next suspect was the metric. `evolve.fitness` scores **terminal drives**
against a **cumulative catch count**, and hunger is a regulated homeostat — intake is
`peck_food_rate * hunger`, so a hen who loses feeding time gets hungrier and then eats
faster per second. Catches have no such regulator. So the foraging side looked compressed
by self-correction while the predation side was not, which would make trading foraging for
safety look good at any parameter setting. `coop/world.py` already tracks **`n_fed`**, a
cumulative successful-peck counter that `evolve.fitness` never reads.

Re-scoring the *same* runs with a running total instead of an equilibrium, predation term
held identical:

| crouch bias | crouch | `n_fed` vs base | net (terminal drives) | net (cumulative intake) |
|---|---|---|---|---|
| 0.0 | 0.398 | — | — | — |
| 0.5 | 0.490 | **+4.1%** | +0.0471 | +0.0339 |
| 1.0 | 0.587 | **+5.1%** | +0.0817 | +0.0892 |
| 1.5 | 0.676 | −4.3% | +0.1234 | +0.1319 |

**Also wrong.** Crouching more leaves cumulative feeding essentially unchanged and at low
bias *raises* it. The two scorings agree to within noise. The foraging cost of vigilance is
genuinely zero in aggregate, by both measures.

### 6c. Why: crouching is camping

`coop/actuation.py` sets `mobility = 1 - crouch`. Crouching zeroes locomotion, so a
crouching hen **stays where she is** — and if she is on a patch, she keeps working it
instead of wandering off.

| crouch bias | time at food | **patch dwell** | distance/step |
|---|---|---|---|
| 0.00 | 4.34% | 15.2 chunks | 0.0405 m |
| 0.75 | 4.20% | 20.4 chunks | 0.0305 m |
| 1.50 | 4.37% | **29.3 chunks (+92%)** | 0.0216 m (−47%) |

Crouching **doubles how long she stays on a patch** while halving her movement, and
time-at-food comes out unchanged: she finds patches less often and stays much longer once
there. The two effects cancel almost exactly, which is why E121c's real local suppression
(feeding at 52.7% of the upright rate) produces no aggregate cost.

And the behaviour crouching supplies is the one this project has already named as missing.
[E112](E112-repair-the-peck-reflex.md) §7:

> **What the camped oracle does that the repaired hen still cannot is stay on its patch.**

**Crouching accidentally implements E112's persistence.**

## 7. Interpretation

**There is no vigilance/foraging trade-off in this model to calibrate.** The anti-predator
action and the best available foraging action are *the same action* — freeze, stay put,
keep working the patch. That is why nine coop settings failed, why the cumulative-intake
rescoring failed, and why unconditional vigilance is worth +0.233 with no information.

**An alarm call cannot pay when the behaviour it recommends is what you should be doing
anyway.** This is a single fact that explains the whole arc: E120's ladder null, E121's
"plant buys nothing over a scramble", and E122's failed calibration are three views of it.
It is a defect in the **world model**, not in the brain, the learning rule, or the channel
— and it is the first explanation in this sequence that is neither.

**It also unifies two findings that looked unrelated.** E112 diagnosed a foraging gap
("staying put") and E121 diagnosed a communication gap ("vigilance is free"). They are the
same fact: the model has exactly one mechanism for persistence, and it is the anti-predator
reflex.

**What this does not show.** It does not show that a repaired world would make the channel
pay — E121's other two findings (the alarm reaches the motor stub at only AUC 0.687, and a
planted direction buys nothing over a scramble) are untouched by anything here, and both
would survive the repair. Removing the degenerate optimum remains *necessary*, not
demonstrated *sufficient*.

**The falsifier fired as written, and the ladder was not re-run.** §4 said: "If no setting
reaches ≈ 0, that is reported and the ladder is not run." Nine settings, none close. Running
it anyway would have produced a null that meant nothing, which is the trap E120 §6a caught
at `hawk_period_s=10`.

**On the honesty constraint.** §4 fixed the calibration target without reference to the
channel — the net gain from an information-free crouch increase, applied as a constant on
`b_motor[M_CROUCH]`. That target was never met, so no world was frozen and no ladder was
run. Nothing here was tuned toward a communication result, and the negative outcome is the
evidence that the constraint held.

**Three predictions, three wrong.** Patch size and intake rate were the wrong levers; the
scoreboard was the wrong suspect; and the mechanism turned out to be a coupling between
two motor channels that nobody had looked at. The project's base rate on mechanism
predictions continues to be poor, and continues to be worth stating in advance.

## 8. Consequence

**`docs/hypothesis.md`.** H0 records that the predation-based route is blocked by a world
defect: crouching is simultaneously the anti-predator response and the only persistence
mechanism, so vigilance is free and unconditional vigilance is optimal. T1's "no intake
benefit" gains a mechanism.

**`docs/backlog.md` §8.** "Raise the aggregate cost of vigilance" is **closed as
attempted and failed** — it is not reachable through coop parameters. Replaced by the
repair the mechanism points to, in priority order and none of them tested:

1. **Give the hen persistence that is not crouching.** If staying on a patch were available
   without freezing, crouching would stop being a foraging aid. This is E112's own owed
   item and it now has a second, independent reason to exist.
2. **Make freezing genuinely exclude feeding** — `fed = at_food_any & pecking & ~crouched`,
   which is what `actuation.py`'s comment already claims ("not foraging"). Weaker than (1),
   because a frozen hen still holds her position and resumes when she stops.
3. **Make crouching carry a cost that is not foraging** — a metabolic term, or protection
   that depends on cover rather than being unconditional.

**No code changes.** `hen/` and `coop/` are untouched. The calibration target was not met,
so nothing is adopted.

**Code.** `scratchpad/e122_vigilance_cost.py`, `scratchpad/e122b_cumulative_intake.py`,
`scratchpad/e122c_crouch_is_camping.py`.
