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

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*
