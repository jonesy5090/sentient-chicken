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

### (b) Depletion — the finding, and it is a confound not a cosmetic

`food_deplete_rate` is 2.0e-2 by default with regrowth, and works. **E082 through E089 all
ran with it at 0.0.** Restoring it costs a little hunger (0.3997 → 0.4266 at `n_food=4`)
and changes what every spatial metric in that arc was measuring: with infinite food there
is no force that ever makes a hen leave a patch, which is the behaviour T2 exists to
detect.

### (a) The locomotion gate — three attempts, two of them mis-keyed

| attempt | keyed on | v@food | v away | outcome |
|---|---|---|---|---|
| 1 | `head_down` = max(peck, scratch) | 0.0864 | 0.0794 | no localisation |
| 2 | `M_PECK` | 0.0864 | 0.0794 | **identical** — no localisation |
| 3 | `M_PECK × at_food_prev` | **0.2243** | **0.2615** | localised, but weak |

**Attempts 1 and 2 keyed on signals that are on everywhere.** `M_SCRATCH` is hunger-driven
by design. And `M_PECK` became hunger-driven **in E090, two experiments ago and at my own
hand** — at hunger 0.43 that is sigmoid(4.0×0.43 − 2.5) ≈ 0.31 across the whole arena. I
keyed a location-specific mechanism to a signal I had personally made non-local. Attempt 2
produced numbers bit-identical to attempt 1, which is what exposed it.

Attempt 3, gated on pecking **and** standing at a patch:

| gate | slow frac | mean speed | dwell (s) | spread (m) | hunger | food left |
|---|---|---|---|---|---|---|
| 0.0 | 0.002 | 0.2652 | 7.5 | 2.03 | 0.4266 | 0.461 |
| 0.5 | 0.003 | 0.2592 | 7.8 | 2.02 | 0.4006 | 0.356 |
| 0.8 | 0.034 | 0.2514 | 8.3 | 1.99 | 0.3887 | 0.289 |
| 1.0 | 0.042 | 0.2485 | 8.6 | 1.97 | **0.3824** | **0.219** |

- **Primary falsifier FIRES.** Slow fraction 0.042 against a 0.3 bar. No visible
  intermittency.
- **Ethogram falsifier FIRES.** `head_down_blindness` returns **399 head-down / 0 head-up
  steps** — a hen who stops at a patch stays there and pecks continuously, so the
  vigilance/foraging alternation vanishes. Identical failure mode to E090's rejected H=8,
  on the same load-bearing assay.
- **Prediction 3 falsified.** Flock spread went *down* (2.03 → 1.97), not up. I predicted
  depletion would disperse hens off shared patches.
- **But foraging improves markedly**: hunger 0.4266 → 0.3824 and food consumed more than
  doubles (food remaining 0.461 → 0.219). Stopping to eat makes her a better forager.

**A third measurement flaw, found while writing this up.** The `v@food` band is
`peck_radius × 3.0` while the gate triggers on `at_food_prev` at `peck_radius` — the
measurement window is three times wider than the mechanism's, so the 0.2243 vs 0.2615 split
understates whatever localisation exists. The direction is trustworthy; the magnitude is
not.

## 7. Interpretation

**The observation was right and the fix is not available at this weight.** Real hens
forage intermittently, the model does not, and both ingredients were missing — one
switched off by our own config, one absent entirely. Restoring depletion and adding the
gate moves every measure in the right direction and improves feeding. It does not produce
the cycle, and pushed hard enough to try, it destroys the head-down gate.

**That is E090's collision again, on the same assay.** A hen who stops to eat cannot also
be alternating between postures on a four-second timescale, and the head-down blindness
assay requires ten of each within 400 steps. Whether that assay is too brittle or the
mechanism too blunt is a real question — real hens *do* stay at a patch for many seconds —
but it must not be settled by loosening the assay now, having seen which way it fails.
That would be the E090 goalpost problem exactly.

**Three mis-specified falsifiers in four experiments is a pattern, not bad luck.** E088's
selection rule optimised decodability and tested selectivity; E090's optimised sated
suppression and tested conditionality; E092's primary tested "is she sometimes slow" rather
than "is she slow where the food is", and *passed* at gate 0.8 while the mechanism was
doing nothing. Each time the falsifier measured a proxy that was easier to compute than the
thing itself. **Standing correction: a falsifier must name the quantity the mechanism is
supposed to move, not a correlate of it** — and where the mechanism is conditional or
local, the falsifier has to carry that condition.

## 8. Consequence

**Adopted: depletion, by removing the override.** T2 configs no longer set
`food_deplete_rate=0.0`. This changes the baseline for anything compared against E082–E089,
which is stated rather than hidden.

**Not adopted: `peck_stops_walking`. Default stays 0.0**, so every prior result is
bit-identical (88 passed, 1 xfailed). The parameter and its gate ship, documented, because
the mechanism is right in direction and the next attempt should not have to rediscover
which signal to key it on.

**Next, in order.** *(1)* Ask whether `head_down_blindness` is measuring the gate or the
assay's own 400-step window — pre-registered as a specification, before any weight is
touched. *(2)* If the window is the problem, re-run this sweep. *(3)* Re-run E089 with
depletion on regardless, because that confound applies to it whatever happens here.
