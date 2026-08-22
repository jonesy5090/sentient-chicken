# E092 — hens should peck in one place, then move: depletion and a locomotion gate

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **H1** (a credible hatchling) and **T2**. Bears on every spatial
measurement in E082–E089.

---

## 2. Question

Two observations about the model's foraging, raised from watching a recording:

**(a) Movement is continuous and unrealistic.** A real hen pecks in one spot, moves,
pecks again. The model has no coupling between eating and walking: `M_FORWARD` is driven
by hunger, thirst, cold and a tonic bias, while `M_PECK` is driven by seeing food, and
nothing connects them. She walks and pecks simultaneously and indefinitely.

**(b) Feeders should deplete.** They do — `food_deplete_rate = 2.0e-2` per second per hen,
with regrowth on `food_regrow_s`, in `world.py:351`.

**But every experiment in the T2 arc set `food_deplete_rate=0.0`.** It entered in E082's
config and propagated through every scratchpad importing it: E082, E083, E084, E085, E086,
E087, E088, E089. **So the entire spatial arc was measured in a world where feeders never
run out.** That is not a realism quibble — it removes the only force that would ever make
a hen leave a patch, which is precisely the behaviour T2 is trying to measure.

The two are one mechanism. With depletion but no locomotion coupling, she wanders while
pecking and depletion just moves food around under her. With the coupling but no
depletion, she stops at the first patch forever. **Together they give the observed
cycle**: arrive, stop, peck until the patch is spent, move on.

**And it may hand T2 its missing behaviour.** E083 removed `M_FORWARD` suppression from the
gakel scaffold so a warned hen would keep walking rather than freeze. If pecking suppresses
locomotion, then *suppressing pecking restores it* — a warned hen starts moving because she
has stopped eating. Departure would fall out of the coupling rather than needing its own
wire. E089 measured occupancy at +0.3% with a correct association; this is a candidate for
why.

## 3. Prediction

1. **The cycle appears.** Speed becomes bimodal — near zero while pecking, walking
   otherwise — where it is currently unimodal. Measured as the fraction of time below 20%
   of `walk_speed`, which should rise from near 0 to **>0.3**.
2. **Dwell per visit becomes bounded.** E085 measured 17–75 s, with 279 s and 717 s on two
   seeds. Depletion should cap it: at `food_deplete_rate` 2.0e-2 a single hen strips a
   patch in ~50 s. I predict no seed above **120 s**.
3. **Aggregation weakens.** E084 measured flock spread 1.66–7.21 m in a 20 m arena, with
   occupancy at a fixed cell ranging 0.000–0.481 on world key alone. Depletion should push
   hens off shared patches. I predict mean spread rises by **≥1 m**.
4. **The ethogram survives.** `peck at food` stages a hen 0.25 m from food and requires
   `peck > 0.5`; she must still reach it. Head-down blindness needs both postures and
   should get them more cleanly, not less.
5. **The gakel call produces departure.** With the coupling, occupancy at a warned patch
   should fall — the E089 measurement, re-run.

I hold 5 least firmly. It requires the warned hen's restored locomotion to actually carry
her out of the target radius within the run.

## 4. Falsifier

**Primary.** The speed distribution does not become bimodal, or the low-speed fraction
stays below 0.2. The coupling would then not produce intermittent foraging and the
locomotion model needs a different mechanism.

**Ethogram falsifier.** Any assay changes state. `peck at food` is the one at risk: a hen
who stops moving the instant she pecks may never close the last 0.25 m.

**Starvation falsifier.** Mean hunger in a free-running flock rises by more than 0.1 above
the current baseline of 0.3997. Depletion plus a locomotion gate could plausibly make hens
worse at feeding, and that is a fair cost only if small.

**Throughput falsifier.** The real-time factor drops. `CLAUDE.md` treats this as a
correctness constraint.

## 5. Design

### The change

**(a)** Restore `food_deplete_rate` to its default in T2's config. This is a *removal* of
an override, not a new parameter.

**(b)** A locomotion gate in `coop/actuation.py`, alongside the two that already exist:

```python
mobility = jnp.clip(1.0 - crouch, 0.0, 1.0)                       # crouching freezes
mobility = jnp.where(w.sick_on, mobility * cfg.sickness_mobility_scale, mobility)
mobility = mobility * (1.0 - cfg.peck_stops_walking * head_down)  # E092
```

`head_down` is already computed there from `HEAD_DOWN_ACTIONS`, so this needs no new state
and stays inside the compiled scan. `peck_stops_walking` defaults to **0.0** — every prior
result bit-identical — and is swept.

Putting it in `actuation.py` rather than the reflex matrix is deliberate: a reflex matrix
maps observations to motor channels and cannot express motor-to-motor coupling. This is a
mechanical consequence of posture, like crouching and sickness, and belongs where those
are.

### Measurements

1. **Speed distribution**, 8 seeds, sweeping `peck_stops_walking ∈ {0, 0.5, 0.8, 1.0}`
   with depletion on. Low-speed fraction and bimodality.
2. **Dwell per visit and flock spread**, against E084/E085's figures.
3. **Full ethogram** at each setting, and free-running mean hunger.
4. **E089 re-run** at the adopted setting, for prediction 5.

### Cost

~20 minutes for 1–3; ~30 for the E089 re-run.

---

## 6. Result

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*
