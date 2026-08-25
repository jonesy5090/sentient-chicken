# E112 — repair the peck reflex, then re-ask H2

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **H2**, and **H1** (the innate arc). A change to the *hardwired*
arc, not to learning, made because the arc is wrong.

---

## 2. Question

[E111](E111-is-there-headroom.md) measured a defect in the innate wiring:

| | on food | off food |
|---|---|---|
| `M_PECK > 0.5` | **39.65%** | **59.59%** |

**She pecks less when she is standing on food.** `innate.py:84-85` drives `M_PECK` from
the `CLS_FOOD` channels of the **two front bins only** — `_FRONT = (5, 6)`, the pair
straddling the beak — at weight 7.0. That is the right idea for a hen approaching food
from a distance and the wrong one for a hen standing on it: at 0.30 m the bearing to the
patch swings wildly with every step, so the patch spends much of the dwell in a bin the
peck reflex does not read, or behind her entirely. Meanwhile `vision_range = 10 m` in a
20 m coop keeps the same channels near-saturated everywhere else — 0.9430 summed off food
against 0.9100 on it — so the reflex fires constantly and at nothing in particular.

`innate.py:305` already describes half of this: "A hen does not stop at a feeder, she
walks through it and pecks as she passes." What it does not say is that the pecking is not
aimed either.

**And the model already has the signal that would fix it.** `IDX_FOOD_ARRIVAL`, the E053
discovery pulse, is set to 1.0 on the rising edge of arriving at a patch and decays over
`food_call_decay_s = 4 s`. It is bearing-independent, it carries "am I at a feeder" at
**AUC 0.87–0.99** ([E107](E107-red-team-review-2026-08-24.md)), and it is wired to the
food **call** and to nothing else. A hen announces that she has found food and does not
act on it.

**So: does repairing the arc close the 0.21-hunger gap E111 measured — and does anything
remain for learning to do afterwards?**

This is a change to the **hardwired** arc. `CLAUDE.md` says the reflex arc is fixed for
life and never plastic; it does not say it must stay *wrong*. But it moves every baseline
in the project, which is why it is opt-in and pre-registered rather than simply corrected.

## 3. Prediction

1. **The repair works mechanically.** `P(peck | on food)` rises from 39.65% to **above
   80%**, and the sign of the on-food/off-food difference reverses.
2. **It closes much of the gap.** Repaired hunger falls from 0.6332 to **0.50–0.55**,
   against the camped oracle's 0.4223.
3. **Headroom persists but shrinks** — I expect roughly half of E111's 0.21 to remain,
   since the oracle also *stays* on the patch and the repair only makes her peck while she
   is there.
4. **Learning still adds nothing on top.** [E110](E110-postsynaptic-factor.md) found no
   benefit over a frozen readout and I expect that to survive the repair, because
   [E109](E109-what-the-rule-writes.md)'s constraint is unchanged by it.

## 4. Falsifier

**Instrument falsifier — reported before anything else.** `P(peck | on food)` must exceed
80%. If the wiring does not change the behaviour it targets, no hunger number in this
experiment means anything.

**Regression falsifier.** Inertness bit-identical at `arrival_peck_weight = 0.0`; the
`peck_at_food` ethogram assay must still pass; no other assay may change state; the suite
must pass.

**Interpretation guard, written in advance so neither outcome can be spun afterwards.**

- If the repair closes **most** of the gap and learning still adds nothing, then what ten
  experiments were asking the learning rule to discover was **a defect in the hand-written
  wiring**. That is the uncomfortable outcome and it is the one I would report first.
- If the repair closes **little** of the gap, the arc was not the limitation, E111's
  headroom stands undiminished, and the nulls remain squarely about the rule.
- If the repair **hurts** — plausible, since a hen who stops to peck is a hen not
  travelling to the next patch — that is a real finding about the foraging trade-off and
  the flag stays off.

**Replication rule.** E021: nothing moves the tree on one seed block.

## 5. Design

**`arrival_peck_weight: float = 0.0`** in `connectome.build`, passed through to
`innate.reflex_matrix`, adding `w(M_PECK, IDX_FOOD_ARRIVAL, weight)`. Off by default, so
every existing result is untouched.

**The weight is chosen a priori against `REST_BIAS`**, in the same way and for the same
reason `SCAFFOLD_WEIGHT` is, so it cannot be tuned into the shape of a result:

- **4.0**. `sigmoid(4.0 − 2.5) = 0.82`, so arriving at a patch reliably crosses the 0.5
  action threshold on its own, without needing the visual channels to cooperate.
- It decays below threshold when the pulse falls under `2.5/4.0 = 0.625`, which at
  `food_call_decay_s = 4 s` is **1.5 s after arrival**. So it buys a *peck bout* on
  arrival, not permanent pecking — which is what real hens do and what
  `food_call_decay_s`'s own comment already argues for ("a bout, not continuous").
- It is **half** the visual crouch weight of 8.0, so seeing a hawk still dominates being
  on food. First-hand danger beats appetite, which is the correct ordering and the same
  one `SCAFFOLD_WEIGHT` is justified by.

**Arms**, matched seeds, 8 seeds, 16 hens, 30 min, `hawk_period_s = 60`:

| arm | arc | readout |
|---|---|---|
| baseline / frozen | current | `eta_out = 0` |
| **repaired / frozen** | `arrival_peck_weight = 4.0` | `eta_out = 0` |
| baseline / learning | current | instrumental |
| repaired / learning | `arrival_peck_weight = 4.0` | instrumental |

The camped oracle's **0.4223** from E111 is the reference ceiling; it is not re-run.

**Measured**: `P(peck | on food)` and `P(peck | off food)`; hunger; fraction at a patch
and feeding; `caught/dive`; the full ethogram at the repaired arc.

### Cost

~25 minutes.

---

## 6. Result

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*
