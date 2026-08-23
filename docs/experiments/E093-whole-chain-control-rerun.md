# E093 — E089's whole-chain control, re-run with three changed inputs

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **T2** → **T2-revised**. Re-runs
[E089](E089-whole-chain-control-on-the-repaired-stack.md), the backlog's staging step 3.

---

## 2. Question

E089 planted a correct place→gakel association, validated the plant with a hard gate
(84.8% held-out decoding, 5.22 selectivity, firing 1.037 at the target against 0.459
elsewhere), and measured occupancy at the target move by **+0.3%**. Its primary falsifier
was written as the stop condition: *"with the plant validated, the representation readable,
the anchor correct and the metric adequate, a null here is a statement about the
architecture and not about our instruments."*

**Three of those inputs have since changed.**

| what E089 assumed | what was actually true | fixed by |
|---|---|---|
| the anchor was correct | 3.5% peck suppression, behaviourally inert | E090 — now 96.7% sated |
| she could see the place she was warned about | scratching held the head-down gate shut; 73% vision | E091 — now 97% |
| feeders behaved normally | `food_deplete_rate` was 0.0 for the whole arc | E092 — restored |

The third is the one that matters most for *this* measurement. E089 asked whether a hen
leaves a poisoned feeder. It ran in a world where feeders never empty, which removes the
only force that would ever make her leave one — so the dependent variable had no natural
dynamics to be modulated.

**Does a hen avoid the planted feeder, now that the warning is strong, she can look at
what she was warned about, and the feeders behave like feeders?**

### Why this might work without E092's rejected locomotion gate

E092 tried to make suppressed pecking restore locomotion directly, and that failed on the
head-down assay. But the same effect is available indirectly and needs no new mechanism: a
warned hen stops eating at the target while the *other* feeders continue to deplete and
regrow. Her hunger rises, hunger drives `M_FORWARD` at +2.0, and she walks. Departure would
emerge from foraging dynamics already in the model.

That is a prediction, not a claim, and §4's agitation falsifier is what distinguishes it
from her simply moving more everywhere.

---

## 3. Prediction

1. **Occupancy at the planted target falls by ≥10% relative** at `pred_gain=2.0` — the
   threshold E089 used, against a metric E085 measured at 5.1% resolution (n=8).
2. **Monotonically** across the gain ladder.
3. **Control-cell occupancy falls by less than 5%.** The discriminator between avoidance
   and agitation.
4. **Peck rate at the target falls by ≥25%**, matching E090's ethogram bar. E089 measured
   2.9% with the inert anchor.
5. **Hunger rises modestly** — she declines one feeder of several — but stays below 0.60.

I hold prediction 1 more firmly than at E089 and still not confidently. The mechanism now
has every link measured *and* a plausible route to departure, but no behavioural effect has
ever been observed in this arc, across thirty experiments.

## 4. Falsifier

**Primary.** Occupancy at the target falls by less than 10% relative, or non-monotonically.

**Interpretation of a null is different this time, and is fixed now rather than after.**
E089's null was attributed to the architecture and that attribution was wrong — three
inputs were broken. If this fires, the honest reading is **not** "the architecture cannot
do it" but "the association reaches the motor system, the motor system responds, and the
response does not move her" — which is a claim about the *locomotion* model, the thing
E092 failed to fix. It would make E092's question the blocker rather than a side quest.

**Agitation falsifier.** Control-cell occupancy falls by ≥10% alongside the target.

**Starvation falsifier.** Mean hunger at gain 2.0 exceeds 0.60.

**Plant falsifier — a hard gate that aborts.** Unchanged from E089: ≥80% held-out
decoding, ≥2× selectivity at the target, decreasing distance profile.

**Anchor falsifier — new.** Peck suppression at the target must exceed 25%. E089's anchor
was inert and its own assay passed it on a sign test; this checks the anchor is doing
something *in this run* rather than trusting E090's staged measurement.

---

## 5. Design

E089's script, with exactly four changes, all of them adopted defaults or recorded
recommendations rather than new choices:

- `food_deplete_rate` at its **default** (the override is removed, not overridden).
- `gakel_peck_weight=9.0`, `hunger_peck_weight=4.0` — E090's adopted configuration.
- E091's scratch suppression, which is unconditional in the scaffold.
- `peck_stops_walking=0.0` — E092's gate stays **off**, as rejected. This experiment must
  not silently carry a change that failed its own falsifiers.

Everything else identical: 8 seeds, `pred_gain ∈ {0, 0.5, 1, 2}`, 20 simulated minutes,
per-seed target and control chosen from an independent selection run, plant fitted on live
hippocampal states and gated before the ladder, both arms on the test run key.

Reported per gain: occupancy at target and control, hunger, forward drive, peck rate at the
target, and `pred@gakel` split by at-target versus elsewhere.

### Cost

~30 minutes.

---

## 6. Result

### The anchor works. The run is invalid.

| `pred_gain` | occ target | occ ctrl | hunger | fwd | peck@T | pred@T | pred elsewhere |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.6758 | 0.7044 | 0.455 | 0.627 | 0.767 | 1.122 | 1.230 |
| 0.5 | 0.6726 | 0.7086 | 0.456 | 0.627 | 0.715 | 1.064 | 1.114 |
| 1.0 | 0.6321 | 0.6708 | 0.459 | 0.629 | 0.665 | 1.100 | 1.102 |
| 2.0 | 0.6760 | 0.7111 | 0.468 | 0.632 | 0.542 | 1.087 | 1.136 |

**The anchor falsifier clears, and this is a real positive.** Peck at the target falls
**0.767 → 0.542 = 29.4%**, against E089's inert 2.9%. E090's weights work in a live
free-running flock, not only on the staged assay. Agitation and starvation clear.

**The primary falsifier fires** — occupancy +0.0%, non-monotonic — **and it must not be
read**, because the plant was not selective in this run.

### 6b. The plant gate passed on a number it should never have passed on

| seed | held-out acc | selectivity ratio | decreasing |
|---|---|---|---|
| 0 | 68.1% | 3.03 | yes |
| 1 | 85.8% | 2.46 | yes |
| **2** | 65.2% | **5 909 203.50** | **no** |
| 3 | 86.7% | 2.34 | yes |
| 4 | 86.2% | 8.32 | yes |
| 5 | 94.0% | 4.97 | yes |
| 6 | 83.5% | 4.25 | yes |
| 7 | 88.5% | 3.79 | yes |
| **mean** | 82.2% | **738 654** | 7/8 |

**Two defects, both mine, both visible in the output.**

**(1) The gate aggregates by mean.** Seed 2's ratio is degenerate — the discriminant's
scores away from the target are all negative, `relu` sends them to ~0, and the ratio
explodes. One such seed carries `mean(ratios) >= 2.0` regardless of the other seven. A
per-seed minimum would have caught it immediately.

**(2) The gate measures the wrong quantity.** It checks the *fitted discriminant's*
separation on held-out samples. Behaviour is driven by the *installed, normalised*
`W_pred` row through the runtime einsum. These are different quantities, and the live one
was printed in the results table all along:

| | E089 | E093 |
|---|---|---|
| live `pred@target` | 1.037 | 1.122 |
| live `pred elsewhere` | 0.459 | 1.230 |
| **live ratio** | **2.26×** | **0.91×** |

**The plant fires slightly harder away from the target than at it.** E083 found exactly
this failure mode and my response was to add the live split as a **reported column** rather
than as the **gate**. It has been sitting in the output reading 0.91 while the gate
announced PASSED.

## 7. Interpretation

**This is the fourth invalid whole-chain control, and the third distinct reason.** E082's
plant was a matched filter. E083's was fitted on parked states and read on moving ones.
E093's is fitted and gated in one space and installed in another. Each time the gate
measured something adjacent to the thing that drives behaviour.

**What changed between E089 and E093 is depletion**, and that is the likely cause: the
selection run now has hens leaving patches, so the states the discriminant is fitted on
differ from the states it is read on. The fit-space accuracy stays respectable (82.2%)
because the discriminant still separates *sampled* states; the live projection does not,
because the normalisation and the runtime path differ. That is a hypothesis, not a
measurement, and it is written here as one.

**E093 therefore says nothing about T2's behaviour.** The primary falsifier fired against
a plant driving 0.91× at the target, which is not the manipulation the experiment intended
and cannot support the conclusion the falsifier was written to license.

**What E093 does establish** is worth separating from what it does not. E090's anchor
produces **29.4% peck suppression in a free-running flock** — the first time the gakel
response has done anything measurable outside a staged assay. That was the change E089's
null was blamed on, and it is now fixed and confirmed in situ.

## 8. Consequence

**Not adopted, nothing propagated to T2's status.** E089's node stands as it is; this does
not replace it, because a fourth invalid run is not evidence either way.

**The gate must be rebuilt to measure the live quantity, and this is now the blocker for
the whole arc.** Specifically: gate on `relu(pred@gakel)` **at the target versus
elsewhere, in a live run with the plant installed**, per seed, with a minimum rather than
a mean — the exact column that has been reported and ignored for three experiments. The
fit-space accuracy can stay as a secondary diagnostic; it must not be the gate.

**And a per-seed minimum everywhere a gate aggregates.** The mean is wrong for any gate
whose statistic can diverge, and this one can by construction, since it is a ratio with a
`relu` in the denominator.

**Recorded as a pattern, because it is now unmistakable.** E092 named three mis-specified
falsifiers in four experiments — each measuring a proxy cheaper to compute than the thing
itself. This is the fourth, and it is worse: the correct quantity was **already being
computed and printed**, and I gated on a different one anyway. The standing correction from
E092 §7 needs strengthening: *where the correct quantity is already available, the gate
must use it, and a gate that aggregates must not use a mean.*
