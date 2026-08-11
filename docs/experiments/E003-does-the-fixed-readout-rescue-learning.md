# E003 — does fixing the readout rate rescue learning?

## 1. Parent hypothesis

**H2** — three-factor plasticity produces measurable behavioural improvement.

## 2. Question

[E002](E002-can-the-pallium-reach-a-muscle.md) found the cortical readout frozen at
`eta_out=2e-3` and raised it to 2e-2. Does the [E001](E001-does-plasticity-help.md)
null move with only that changed?

## 3. Prediction

Learning conditions show hunger declining relative to the fixed control, by more than
the two-tailed t threshold at p=0.05.

## 4. Falsifier

No movement from E001's null. That would clear the readout as the explanation and
point at synaptic scaling or the reward signal.

## 5. Design

Identical to E001 in every respect except `eta_out`. Same seeds, same coops, same
genomes, same predator arrivals, same duration, same metrics.

- **Command**: `python -m run.experiment --minutes 20 --seeds 4`

## 6. Result

| condition | hunger early | hunger late | change | fed % | exposure | synapses |
|---|---|---|---|---|---|---|
| fixed (innate only) | 0.306 | 0.370 | +0.064 | 5.2 | 3075 | 36358 |
| learning, no growth | 0.321 | **0.295** | **−0.026** | **6.5** | 3062 | 24571 |
| learning + growth | 0.325 | 0.343 | +0.018 | 5.5 | 2814 | 45450 |

Paired contrasts against the fixed control (n=4, so the p=0.05 threshold is t=3.18):

```
hunger change      learning, no growth   -0.090 +/- 0.036 SE   t=2.50   suggestive only
hunger change      learning + growth     -0.046 +/- 0.060 SE   t=0.77   noise
predator exposure  learning, no growth   -13.3 +/- 2469 SE     t=0.01   noise
predator exposure  learning + growth     -260 +/- 1831 SE      t=0.14   noise
```

Comparison against E001, same seeds, only `eta_out` changed:

| | E001 (2e-3) | E003 (2e-2) |
|---|---|---|
| learning, no growth | −0.031 ± 0.040 (t=0.78) | **−0.090 ± 0.036 (t=2.50)** |
| learning + growth | −0.017 ± 0.017 (t=1.00) | −0.046 ± 0.060 (t=0.77) |

## 7. Interpretation

**The effect tripled and changed sign in absolute terms.** Under learning without
growth, mean hunger now *falls* across a run (0.321 → 0.295) where the fixed control
*rises* (0.306 → 0.370), and feeding rate went from 5.2% to 6.5% of timesteps. The
direction, the magnitude and the mechanism all line up with E002's diagnosis.

**It is still not significant, and the harness was wrong about that.** At n=4 the
two-tailed t threshold is 3.18, not 2. The observed t=2.50 corresponds to p≈0.09.
The previous version of `run/experiment.py` flagged "exceeds 2 SE" and would have
called this a result — a bug in the analysis, not the simulation, and the kind that
manufactures findings. Fixed: the harness now uses the correct t critical value for
the degrees of freedom available and labels anything below it "suggestive only".

So the honest position: **a clear, mechanistically explained improvement that has not
yet cleared significance at the seed count we ran.** This wants replication at 8–12
seeds, which is cheap (~20 min wall clock per condition per 4 seeds), not a longer
run.

**Structural growth does not help, and may hurt.** It is consistently the weaker
condition — t=0.77 against t=2.50, and it ends with 45,450 synapses against 24,571.
A plausible reading is that continuous rewiring destabilises what has been learned:
the no-growth hen keeps a smaller, stable connectome and does better with it. This is
a real and unanticipated finding and it inverts the naive expectation that more
structural plasticity is better. It needs its own experiment rather than a guess.

**The predator-exposure signal from E001 was noise.** Standard errors of 2469 on a
mean difference of 13. E001 flagged it as unpowered and declined to claim it, which
was correct — the apparent 43% reduction there does not survive contact with a second
run. Worth noting as a case where the discipline did its job.

## 8. Consequence

- **H2 stays `UNDER TEST`**, upgraded from "null recorded" to "positive but
  underpowered". Not `SUPPORTED` — t=2.50 against a 3.18 threshold is not support,
  however good the mechanism story is.
- **Harness bug fixed**: `run/experiment.py` used a 2-SE threshold that is far too
  lenient at small n. Now uses the two-tailed t critical value for the actual df.
  This affects how E001's result should be read as well, though its conclusion (null)
  is unchanged.
- **Next**: replicate at 12 seeds. This is the cheapest remaining step and it either
  clears the threshold or it does not.
- **New backlog item**: does structural growth hurt learning? It is the weaker
  condition in both E001 and E003. Design a contrast that varies growth rate rather
  than switching it on and off.
- **Retired**: predator exposure as a candidate primary metric. Far too noisy at this
  flock size and duration.
- **No ethics review triggered.**
