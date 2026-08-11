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

6 seeds, 30 min of rearing, 16 hens. Wall clock 21 min.

| condition | when | alarm alone | alarm aud. | effect | food alone | food aud. | effect |
|---|---|---|---|---|---|---|---|
| learning | hatch | 0.321 | 0.321 | −0.000 | 0.390 | 0.396 | +0.006 |
| learning | reared | 0.390 | 0.358 | **−0.031** | 0.580 | 0.617 | **+0.038** |
| fixed | hatch | 0.321 | 0.321 | −0.000 | 0.390 | 0.396 | +0.006 |
| fixed | reared | 0.321 | 0.321 | −0.000 | 0.390 | 0.396 | +0.006 |

Change in the effect over rearing (6 seeds, threshold t=2.57):

```
learning   alarm  -0.031 +/- 0.022 SE   t=1.43   suggestive, WRONG DIRECTION
learning   food   +0.032 +/- 0.050 SE   t=0.64   noise
fixed      alarm  +0.000 +/- 0.000 SE   t=0.00
fixed      food   +0.000 +/- 0.000 SE   t=0.00
```

## 7. Interpretation

**H3 is not supported.** The primary prediction fails: the food-call audience effect
rose in the mean but the between-seed spread swamps it (t=0.64). The alarm effect
moved *against* prediction and did so more consistently than the food effect moved
with it.

**The assay itself is sound.** The fixed control is flat to three decimal places in
every cell — it is the same connectome measured twice, and it measures the same
thing twice. Whatever went wrong is in the model, not the measurement.

**A pilot at 2 seeds gave +0.034 ± 0.001. Six seeds give +0.032 ± 0.050.** Same mean,
fifty times the standard error. The pilot's tiny SE was two seeds happening to agree,
and it would have read as a strong result to anyone who wanted one. The harness now
warns below four seeds; the warning was added before this run and it was correct.

**The secondary prediction was right in direction and probably wrong in reason.** The
alarm effect did lag the food effect. But it did not merely lag — it went negative,
which practice-starvation does not explain. A starved signal stays near zero; it does
not reliably invert.

### The leading suspect: the kin term cannot assign credit

This looks like the same class of failure as E002, and it is structural rather than a
matter of run length.

The kin term adds `kin_weight × mean(other hens' reward)` to each hen's modulator.
That quantity is **very nearly identical for every hen in the flock** — it is a flock
average, and every bird receives its own near-copy of the same number. The learning
rule is a product of eligibility traces and that modulator. For a hen to learn *that
her call helped*, the modulator has to move **because she called**. A global flock
average does not: it moves the same way for the silent hen standing next to her.

So the current reward gives the flock a reason for calls to exist and gives no
individual hen a way to discover she is responsible for one. The signal is there; the
credit assignment is not. Under that reading, the only component of the reward that
*does* correlate with her own calling is the energy **cost**, which is private and
immediate — and a rule that sees only the cost of calling should learn to suppress
calls, most strongly in the conditions where calling happens most. That is consistent
with the sign we observed.

**The fix that follows from this diagnosis** is to weight the kin term by
audibility — a hen's reward should reflect the welfare of the flockmates who could
actually *hear* her, not the flock mean. `coop/sensing.py` already computes the
distance attenuation matrix used for the auditory channel, so the machinery exists.
That makes the kin bonus covary with her own calling and with who was in earshot,
which is precisely the correlation the rule needs. It is also better biology: kin
selection operates on the relatives you actually help.

This is a hypothesis about the null, not a result. It gets tested, not assumed.

## 8. Consequence

- **H3 stays `NOT STARTED` → `UNDER TEST`**, with this null recorded against it.
- **E006 opened**: audibility-weighted kin reward, then rerun this assay unchanged.
  Same shape as E002 → E003: diagnose the structural blocker, change one thing,
  re-measure.
- **If E006 also fails**, the next suspects in order are (a) practice starvation on
  the alarm channel, testable cheaply by raising hawk frequency during rearing, and
  (b) the rearing duration, which is the expensive answer and should stay last.
- **Recorded as a methodological note**: the 2-seed pilot and the 6-seed run had the
  same mean and a 50x difference in standard error. Pilots at n<4 are for checking
  that the code runs and the ceilings are clear, never for reading effects.
- **The saturation finding stands** and was worth the pilot: the alarm reflex pins at
  0.97 with a hawk overhead, so the assay stages it at 7 m. Any future assay on a
  reflex-driven channel needs the same check before it can measure anything.
- **No ethics review triggered.** The call energy cost and kin term add no tripwire —
  both are scalars in a reward, neither is a nociceptive channel.
