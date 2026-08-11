# E001 — does three-factor plasticity produce measurable behavioural improvement?

## 1. Parent hypothesis

**H2** — three-factor plasticity produces measurable behavioural improvement.

## 2. Question

Does a flock learning under a reward-prediction-error rule regulate its drives better
over a rearing run than a genome-matched, coop-matched flock that cannot learn?

## 3. Prediction

Hunger declines across a run in the learning conditions relative to the fixed
control. Stated as: `hunger(last third) - hunger(first third)` is more negative for
the learning conditions, by more than 2 SE across matched seeds.

## 4. Falsifier

No reliable difference from the fixed control. That would mean the rule is producing
connectome churn without behavioural consequence.

## 5. Design

- **Conditions**: fixed (innate only) / learning without growth / learning with growth.
- **Matched across conditions**: seed, coop layout, genome, predator arrival times.
  The only difference is whether weights change.
- **Primary metric**: within-run change in mean flock hunger.
- **Secondary (exploratory)**: cumulative predator exposure, feeding rate, live
  synapse count.
- **Replicates**: 4 seeds.
- **Command**: `python -m run.experiment --minutes 20 --seeds 4`

## 6. Result

20 minutes of chicken time, 4 matched seeds, 16 hens.

| condition | hunger early | hunger late | change | fed % | exposure | synapses |
|---|---|---|---|---|---|---|
| fixed (innate only) | 0.306 | 0.370 | +0.064 | 5.2 | 3075 | 36358 |
| learning, no growth | 0.329 | 0.362 | +0.033 | 5.2 | 1755 | 27048 |
| learning + growth | 0.326 | 0.373 | +0.047 | 5.1 | 2960 | 44442 |

Paired contrasts against the fixed control:

```
learning, no growth   hunger change  -0.031 +/- 0.040 SE   (within noise)
learning + growth     hunger change  -0.017 +/- 0.017 SE   (within noise)
```

**H2 is not supported by this run.** Both effects point in the predicted direction
and neither exceeds its standard error.

Structural change definitely occurs: ~12k synapses grown and ~12k pruned from 36.5k
innate over 20 minutes. That is churn, not evidence — it is exactly what the
hypothesis was written to distinguish from learning.

## 7. Interpretation

The null is real but this test is underpowered, and both things should be said.

**Why it may be underpowered.** The critical period is configured at 3 days; a
20-minute run covers 0.5% of it. A real chick learns its surroundings over days and
its rank over weeks. Asking for a detectable effect in 20 simulated minutes is asking
learning to work roughly 200x faster than the timescale the rule was parameterised
for. Four seeds against a between-coop variance this large is also thin.

**Why that is not a free excuse.** "Run it longer" is the standard way a null gets
talked away. The honest position is that this run does not support H2, and the next
one is pre-registered at a duration and replicate count chosen to be capable of
detecting the effect rather than chosen after seeing this result.

**Alternative explanations not ruled out:**

- The reward signal may be too sparse. Drive reduction is near-continuous and small;
  the prediction error hovers near zero (measured mean reward ~-0.02).
- Synaptic scaling may be cancelling the learning signal — it pulls row sums back
  toward innate values every consolidation, and if it dominates, weights churn
  without net drift.
- The cortical readout starts at 0.05 scale. If `eta_out` is too small for the
  pallium to gain real influence over the motor output within the run, nothing the
  pallium learns can reach behaviour regardless of how good it is.

**One exploratory signal worth a powered test.** Predator exposure differed
substantially — 1755 for learning-without-growth against 3075 for fixed, a ~43%
reduction. No SE was computed for it in this run, it was not the pre-registered
metric, and the growth condition did not show it (2960). It could easily be seed
noise. It is flagged, not claimed. Note that `n_struck` accumulates per timestep, so
these are units of exposure time, not discrete strike events.

## 8. Consequence

- **H2 stays `UNDER TEST`**, with this null recorded against it. Not refuted: the
  test was not powered to refute it.
- **Harness improved**: `run/experiment.py` now reports paired SE for predator
  exposure as well as hunger, so the secondary signal is testable next time.
- **New backlog items**:
  - E002: rerun at 1 day of chicken time, 8 seeds, pre-registered on both hunger and
    predator exposure. Roughly 4 hours of wall clock at the measured ~14x real-time
    for the plastic conditions.
  - Diagnose whether the cortical pathway ever gains influence over behaviour —
    measure the ratio of cortical to reflex drive at the motor output over a run. If
    it stays near zero, no amount of pallial learning can matter and `eta_out` or the
    readout scale is the thing to fix, not the run length.
  - Check whether synaptic scaling is cancelling the learning signal: ablate it and
    compare weight drift.
- **No ethics review triggered.** No tripwire approached.
