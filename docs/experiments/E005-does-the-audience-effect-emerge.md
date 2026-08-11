# E005 — does the audience effect emerge without being programmed?

> **Pre-registered.** Sections 1–5 written and committed before the run was launched.

## 1. Parent hypothesis

**H3** — learned usage reproduces the audience effect without being programmed.

## 2. Question

Real cockerels alarm- and food-call far more readily with a conspecific in earshot.
Call *production* is innate in chickens and is hardwired here; *usage* is learned and
deliberately is not. After rearing, does calling become conditional on an audience?

## 3. Prediction

**Primary.** The food-call audience effect — calling with flockmates in earshot minus
calling alone — increases over rearing in the learning condition and not in the fixed
control. Threshold: two-tailed t at p=0.05 for the seeds run.

**Secondary, and the more interesting one.** The *alarm*-call effect increases less
than the food-call effect, or not at all, for a reason that has nothing to do with
alarm calls being harder: **the hen gets almost no practice with them.** Hawks arrive
on a ~900 s Poisson schedule, so a 30-minute rearing contains roughly two hawk events.
Food is continuously present. Food-call usage gets something like three orders of
magnitude more learning experience than alarm-call usage within the same run.

If that reading is right, the alarm effect is not absent but starved, and it should
appear at longer rearing or higher predator frequency. That is a cheap follow-up and
a genuine prediction rather than a hedge.

**Control.** The fixed condition must show exactly zero change. It is the same
connectome assayed twice.

## 4. Falsifier

- **For H3**: no increase in either effect in the learning condition. Usage-gating
  does not emerge, and the claim that the model captures chicken development beyond
  what was wired into it fails.
- **For the "starved practice" reading**: the alarm effect increasing as much as the
  food effect, which would make the exposure asymmetry irrelevant.
- **For the assay itself**: any movement in the fixed control. That would mean the
  measurement is picking up something other than learning and nothing else in the
  experiment can be trusted.

## 5. Design

**Two mechanisms had to be added before this was testable, and neither is neutral.**
Stated here rather than buried, because they are the obvious place for this
experiment to be circular:

1. **Calling costs energy** (`call_energy_cost`, `coop/spec.py`). Without a cost
   there is no reason ever to stay quiet, so no gradient could produce
   audience-sensitivity — a hen would call whenever the reflex fired, free, forever.
2. **Kin selection** (`kin_weight`, `hen/plasticity.py`): a fraction of flockmates'
   welfare enters each hen's reward. Under purely individual reward a call can never
   repay its cost — the caller pays and the *listener* benefits — so hens would learn
   to fall silent. Kin selection is the standard explanation for why alarm calling
   evolves in group-living birds at all.

**Neither rewards calling when an audience is present.** One charges for calling; the
other rewards flockmates doing well. That audience-conditional calling is the optimal
policy given those two facts is the hypothesis under test, not an instruction.

- **Assay** (Evans & Marler's design): identical stimulus, identical hen, varying only
  whether flockmates are within earshot. Absent flockmates are parked far outside
  vision and hearing rather than deleted, so both conditions run the same flock size
  through the same compiled program.
- **Plasticity is off during measurement.** The assay measures what she has learned
  and must not let her learn while being measured.
- **Hawk staged at 7 m, not overhead.** A pilot run found the alarm reflex saturated
  at 0.97 directly underneath, leaving no headroom for any effect to appear in. At 7 m
  the call sits mid-range. This is also the ecologically meaningful case: a hen has a
  real decision about a distant threat and none about one on top of her.
- **Growth disabled**, per H2a — it is the weaker condition in all three phase 1 runs.
- **Conditions**: learning / fixed. **Seeds**: 6. **Rearing**: 30 min of chicken time.
- **Command**: `python -m run.audience --minutes 30 --seeds 6`

## 6. Result

_Pending._

## 7. Interpretation

_Pending._

## 8. Consequence

_Pending._
