# E089 — the whole-chain positive control, on a stack where every link is now measured

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **T2** → **T2-revised**. Direct successor to
[E088](E088-frozen-centring-baseline.md).

This is **staging step 3** from `docs/backlog.md`'s T2-revised section, whose own wording
is the reason this experiment exists rather than the contrast:

> *Hand-plant a `W_pred` association (place P → gakel) and confirm a hen avoids P with no
> learning involved. **This is the step E065 skipped and three experiments paid for.** If a
> hand-wired success is undetectable, stop.*

---

## 2. Question

E082 and E083 both attempted this step and both returned no avoidance. Neither was a valid
test, and we now know why in detail:

- **The plant was anti-selective** — it drove the gakel channel at 0.656 at the target and
  1.244 elsewhere, with the innermost distance bin the *lowest* of seven (E083).
- **The metric could not resolve the predicted effect** — 18.3% minimum detectable at n=4,
  against a pre-registered 15% (E084).
- **The place signal was not readable** — position decoded at ~4 points above chance under
  movement (E085).
- **The readout removed most of what was there** — the centring baseline cost ~20 points
  (E086/E087).

Every one of those is now fixed and measured:

| link | state | evidence |
|---|---|---|
| place representable | hippocampus routed, `pred_src` extended | E086: parked 99.5% |
| place readable while moving | frozen centring baseline | E088: **89.0% on fresh seeds** |
| metric resolves the effect | per-seed target selection | E085: **MDE 5.1% at n=8** |
| anchor produces leaving | `M_PECK` only, no freeze | E083, guarded assay |
| chain conducts end to end | forward drive responds to the plant | E082 |

**So: with a correctly selective plant, on a readable representation, measured with an
instrument that can see the effect — does a hen avoid the poisoned feeder?**

If she does not *here*, T2-revised has no remaining instrumental excuse and the backlog's
"stop" applies.

---

## 3. Prediction

1. **Occupancy at the planted target falls by ≥15% relative** at `pred_gain=2.0`, against
   an instrument that resolves 5.1% at n=8 — so a real effect of the predicted size is
   comfortably visible.
2. **Monotonically** across the gain ladder.
3. **Control-cell occupancy does not fall** by more than 5%. This separates avoidance from
   agitation, and it is the discriminator, not the headline.
4. **Peck rate at the target falls**, which is the direct read on the reflex firing where
   it matters.
5. **Hunger rises modestly** — she declines food at one of several feeders — but stays
   below 0.60.

I hold prediction 1 with more confidence than at any previous attempt, and less than the
table above might suggest. Every *link* is measured; the *chain* end-to-end under a
selective plant never has been. E082 showed the chain conducts, but it conducted a signal
that was pointing at the wrong places.

## 4. Falsifier

**Primary.** Occupancy at the target falls by less than **10%** relative (roughly 2× the
metric's resolution), or falls non-monotonically. **This is the "stop" condition**: with the
plant validated, the representation readable, the anchor correct and the metric adequate,
a null here is a statement about the architecture and not about our instruments. T2-revised
would then need re-scoping rather than another repair.

**Agitation falsifier.** Control-cell occupancy falls by ≥10% alongside the target. She is
moving more, not avoiding a place.

**Starvation falsifier.** Mean hunger at gain 2.0 exceeds 0.60.

**Plant falsifier — a hard gate, run before anything else.** The plant must, on the
held-out run: decode at ≥80% balanced accuracy, fire ≥2× harder at the target than
elsewhere, and produce a distance profile decreasing across the first three bins. **The run
aborts if any of these fail.** E082 and E083 each passed an amplitude-only pre-flight while
the plant was useless; that pre-flight is replaced, not supplemented.

---

## 5. Design

### Configuration — every fix from E083–E088, on by name

`place_to_hippocampus=True`, `shared_place_map=True`, `gakel_scaffold=True`,
`place_cells_enabled=True`, `pred_bar_freeze_s=60.0` (E088's recommendation, chosen for
selectivity stability rather than the rule's 40 s), `contamination_enabled=False`.

Contamination stays off deliberately: the plant supplies the association, and a live
rotating contaminant would add a second, uncontrolled teaching signal. This is a control
for whether the anchor *can* produce avoidance, not a learning run.

### The plant — fitted on live states, gated on selectivity

Per seed, using E085/E086's machinery unchanged:

1. **Selection run** (`fold_in(k,2)`, gain 0, no plant) → target = highest-occupancy cell,
   control = second-highest. Chosen before any treatment, from an independent run.
2. Fit the discriminant on that run's **hippocampal** `z_lag − z_lag_bar` states, labelled
   by distance to the target — the configuration E088 measured at 89.0%.
3. **Gate on the held-out test run** against the plant falsifier's three conditions.
4. Only then install it into `W_pred`'s gakel row and run the behavioural ladder.

### The ladder

`pred_gain ∈ {0.0, 0.5, 1.0, 2.0}`, **8 seeds**, 20 simulated minutes, both arms of every
comparison on the test run key so selection bias cancels in the paired difference.

Reported per gain: occupancy at target and control, hunger, forward drive, peck rate at
target, and live `pred@gakel` split by at-target vs elsewhere — the split E083 had to
invent post-hoc, now standard.

### Cost

~30 minutes.

---

## 6. Result

### The plant gate passed — for the first time in this arc

| seed | target | ctrl | held-out acc | selectivity | decreasing |
|---|---|---|---|---|---|
| 0 | 2 | 7 | 68.0% | 9.48 | no |
| 1 | 6 | 5 | 86.8% | 4.34 | yes |
| 2 | 11 | 10 | 82.9% | 1.90 | yes |
| 3 | 3 | 2 | 89.9% | 2.62 | yes |
| 4 | 2 | 7 | 88.9% | 6.46 | yes |
| 5 | 2 | 7 | 93.0% | 3.99 | yes |
| 6 | 10 | 5 | 77.0% | 4.93 | yes |
| 7 | 10 | 6 | 91.6% | 8.02 | yes |
| **mean** | | | **84.8%** | **5.22** | **7/8** |

Live, the plant fires at **1.037 at the target against 0.459 elsewhere** — 2.26× — where
E083's read 0.53, i.e. *anti*-selective. This is the first validly-planted association in
the T2 arc.

### And the behaviour does not move

| `pred_gain` | occ target | occ ctrl | hunger | fwd | peck@T | pred@T | pred elsewhere |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.6997 | 0.7252 | 0.328 | 0.564 | 0.719 | 1.037 | 0.459 |
| 0.5 | 0.6832 | 0.7211 | 0.331 | 0.565 | 0.719 | 1.033 | 0.450 |
| 1.0 | 0.6938 | 0.7243 | 0.332 | 0.566 | 0.708 | 1.038 | 0.458 |
| 2.0 | 0.7020 | 0.7287 | 0.330 | 0.564 | 0.698 | 1.033 | 0.472 |

- **Primary falsifier FIRES.** Occupancy at the target runs **+0.3%** (0.6997 → 0.7020),
  non-monotonic, against a metric resolving 5.1% at n=8.
- Agitation falsifier clear (+0.5% at the control).
- Starvation falsifier clear (hunger 0.330).

### 6b. Why — and it is not the architecture

*Post-hoc, and it changes the conclusion, so it is reported before the interpretation.*

Peck at the target fell only **2.9%** across the whole ladder. That is far too little for a
percept the plant drives to saturation, so the path was measured directly rather than
reasoned about (`$CLAUDE_JOB_DIR/tmp/e089_path_check.py`, hens parked on the target with
food underfoot, settled 90 s past the freeze):

| `pred_gain` | predicted@gakel | `reflex_in[gakel]` | M_PECK | M_FORWARD |
|---|---|---|---|---|
| 0.0 | +0.5534 | 0.0000 | **0.9894** | 0.5947 |
| 2.0 | +0.5534 | **1.0000** | **0.9543** | 0.5947 |

**The pathway works perfectly.** `reflex_in[gakel]` goes 0 → 1.0, fully saturated. And a
**full-amplitude** gakel percept moves pecking by **3.5%**.

The arithmetic, available in the source without running anything: food drives `M_PECK` at
**+7.0** (`innate.py:83`); `SCAFFOLD_WEIGHT` is **1.5** (`innate.py:45`); both sit deep in
sigmoid saturation.

| scaffold weight | peck with call | peck without | suppression |
|---|---|---|---|
| **1.5 (current)** | 0.9959 | 0.9991 | **0.3%** |
| 3.0 | 0.9820 | 0.9991 | 1.7% |
| 5.0 | 0.8808 | 0.9991 | 11.8% |
| 7.0 | 0.5000 | 0.9991 | 50.0% |
| 9.0 | 0.1192 | 0.9991 | 88.1% |

**To halve pecking, the scaffold must roughly match the food drive.** At 1.5 it cannot
move behaviour at all, whatever the association does.

## 7. Interpretation

**The stop condition fired, and its premise was false.** I pre-registered that a null here
would be a statement about the architecture, because the plant, the representation, the
metric and the anchor would all have been validated. Three of those four held. **The anchor
had not been validated for magnitude — only for sign.**

**And the number was printed in every ethogram run since E083.** The assay reports
`gakel peck=0.954 vs contact peck=0.989`; it asserts `peck_g < peck_c` and passes. That
assay's own docstring — which I wrote — warns that "a scaffold that damped everything on
the audio bus would pass a bare sign test while being useless". It then applies a bare sign
test. The 3.5% was on screen every time and nobody multiplied it through to ask whether it
could move a behaviour.

**This is E026's lesson, repeated exactly.** `CLAUDE.md` records it: *"Hearing an alarm
drove crouch to sigmoid(1.5 − 2.5) = 0.269; hiding required > 0.5. Both numbers were in the
source, written by the same person, never multiplied together."* Here: the gakel scaffold
drives peck suppression to 0.3% against a food drive of 7.0, and avoidance requires far
more. Both numbers are in `innate.py`, eleven lines apart.

**But underneath the oversight is a real design tension, and it is the interesting part.**
`_add_gakel_scaffold`'s docstring states the weight is deliberately small so that "it stays
well below the visual arc's own weights so first-hand information continues to dominate
second-hand". That is a defensible principle — a hen should trust her eyes over a rumour.
The measurement says the two requirements are incompatible as built: **with a linear reflex
arc feeding a saturating sigmoid, a second-hand signal held below first-hand weights cannot
change behaviour at all.** Either the call matters or it stays subordinate. There is no
setting of `SCAFFOLD_WEIGHT` that gives both.

That is a genuine architectural finding rather than a tuning oversight, and it applies to
the alarm scaffold on the same argument.

**What E089 does establish, and it is not nothing.** The plant gate passed at 84.8% with
2.26× live selectivity, so for the first time an association *has* been correctly planted
in this model. Every link from place cells to `reflex_in` is now measured end to end. The
failure is in the last two millimetres — reflex weight to motor output.

## 8. Consequence

**I pre-committed to stopping on this falsifier, and I am not going to quietly convert that
into a seventh repair.** The honest position:

- The stop condition's *premise* was wrong, so the stop does not carry the meaning I
  assigned it. This is a diagnosable instrument-class failure, the seventh in the arc.
- But it is also the first one that reveals a **design tension rather than a defect** —
  subordinate-by-construction signals cannot change behaviour through a saturating
  reflex — and that tension is a finding worth recording whether or not T2 continues.

**The decision on whether to continue is a scope call, not mine to make silently.** The two
routes:

1. **Raise the gakel→`M_PECK` weight to ~7 and re-run E089.** One line, ~30 minutes,
   and every other link is already validated. It abandons the stated
   first-hand-dominates-second-hand principle for this call, which should be argued for
   explicitly rather than tuned into.
2. **Re-scope T2** on the finding in §7: in this architecture a referential signal that
   stays subordinate to direct perception cannot alter behaviour, which is a real result
   about the design and arguably more interesting than the avoidance measurement.

**Owed regardless: the ethogram assay must test magnitude, not sign.** Every behavioural
scaffold assay in `run/probes.py` shares this shape. A guard that would pass on a 0.3%
effect is not a guard. This is the highest-priority fix in the repo now, above T2 itself,
because it plausibly affects the alarm scaffold and anything else validated the same way.
