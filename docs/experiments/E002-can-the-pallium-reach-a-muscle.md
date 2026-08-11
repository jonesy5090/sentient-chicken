# E002 — can the pallium reach a muscle?

## 1. Parent hypothesis

**H2** — three-factor plasticity produces measurable behavioural improvement.

This is a diagnostic of [E001](E001-does-plasticity-help.md)'s null, not an
independent test of H2. It asks whether the null had a mechanical explanation that
made the experiment incapable of detecting learning in the first place.

## 2. Question

Motor output is `sigmoid(reflex + cortical + bias)`. The innate arc drives it with
weights of 5–8; the cortical readout is initialised at 0.05 scale so that a hatchling
is reflex-dominated. Does the cortical pathway ever gain enough influence for
anything the pallium learns to reach behaviour?

## 3. Prediction

If the readout is the bottleneck, cortical drive stays small against reflex drive
(ratio < 0.05) and `|W_out|` barely grows over a run — in which case run length is
irrelevant and E001 could not have detected learning regardless of duration.

## 4. Falsifier

A cortical/reflex ratio above ~0.25 with `|W_out|` growing would clear the readout as
the explanation and point back at the learning signal or the run length.

## 5. Design

- 20 min of chicken time, single seed, learning + growth enabled.
- **Primary metric**: ratio of mean `|cortical|` to mean `|reflex|` at the motor
  output, and the growth of `|W_out|` over the run.
- Followed by a sweep of `eta_out` at fixed `readout_scale=0.05`, 10 min per point.
- **Command**: `python -m run.diagnose --minutes 20`

## 6. Result

Baseline run, `eta_out=2e-3` (the E001 setting):

```
 t (min)   |reflex|  |cortical|    ratio   |W_out|  hunger
     1.0      1.707      0.2161    0.127    0.0403   0.321
    19.0      1.560      0.2965    0.190    0.0408   0.420

cortical drive : 0.2161 -> 0.3316  (1.53x)
|W_out|        : 0.0403 -> 0.0410  (1.02x)
```

Sweep of the readout learning rate, 10 min each, single seed:

| `eta_out` | \|W_out\| growth | cortical/reflex | hunger drift |
|---|---|---|---|
| 2e-3 *(E001 setting)* | 1.00x | 0.159 | +0.037 |
| **2e-2** | **1.17x** | **0.508** | **+0.004** |
| 2e-1 | 1.32x | 1.720 | +0.126 |

## 7. Interpretation

**The prediction was half right, and the half it got wrong is the useful part.**

Cortical drive did grow over the run — 1.53x — which looks like the pathway gaining
influence. But `|W_out|` grew only 1.02x. The readout was not learning; the growth
came from the recurrent network becoming more active and pushing more signal through
a fixed readout. That is drift, not learning, and it is exactly the kind of thing
that would have been misread as progress without separating the two measurements.

At the E001 setting the readout is effectively frozen. So E001's null was measuring a
hen whose pallium could not act on anything it learned — the experiment was not
capable of detecting the effect it was testing for.

**The sweep also finds the ceiling, which was not anticipated.** At `eta_out=2e-1`
cortical drive reaches 1.72x the reflex arc and hunger drift gets *worse* (+0.126 vs
+0.037). A hen who overrides her innate responses with an untrained pallium is worse
off than one who does not. There is a real optimum here rather than a
more-is-better direction, and 2e-2 sits near it: influence roughly half the reflex
arc, and hunger drift nearly flat.

That the pallium has to *earn* influence rather than be given it is the right design
— it is what makes "born reflexive, grows into competence" true of the model rather
than just asserted about it. The bug was that the earning rate was set ten times too
low for the mechanism to work at all.

**What this does not establish.** Single seed throughout; the hunger-drift column is
one run per setting and is not significant on its own. `|W_out|` growth and the drive
ratio are mechanical measurements rather than noisy behavioural ones, so those are
solid, but the behavioural improvement they imply still needs a powered test.

## 8. Consequence

- **`eta_out` default raised 2e-3 → 2e-2** in `hen/plasticity.py`, with the sweep
  recorded in the comment so the value is not mistaken for a guess.
- **New tool**: `run/diagnose.py`, and `brain.step` now returns the two pathway
  contributions separately (`brain.Drives`) so the ratio is measurable from any run.
  `Summary` carries `reflex_drive`, `cortical_drive` and `w_out_norm`.
- **E003 launched**: rerun the E001 contrast unchanged except for the fixed
  `eta_out`. If the null was caused by the frozen readout, it should move.
- **H2 stays `UNDER TEST`.** E002 explains the null; it does not support the
  hypothesis. That still needs E003.
- **Backlog item retired**: "does the cortical pathway ever influence behaviour" —
  answered, it does, but only once the readout can learn.
- **Backlog item added**: the ceiling at high `eta_out` suggests the balance between
  innate and learned control is itself a parameter worth studying rather than tuning
  once. It is plausibly the same trade-off real precocial birds face.
- **No ethics review triggered.** No tripwire approached.
