# E013 — the clean test of H2

> **Pre-registered.** Sections 1–5 written and committed before the run was launched.

## 1. Parent hypothesis

**H2** — three-factor plasticity produces measurable behavioural improvement.
`UNDER TEST`. No run since [E009](E009-lagged-pallial-association.md) has tested it
without a confound.

## 2. Question

Does a hen that learns regulate her drives better than a genome-matched, coop-matched
hen that cannot — measured in an environment where the metric is not being swamped by
something else?

## 3. What changed first, and why

[E012](E012-corrected-phase-1-contrast.md) found that `call_energy_cost`, added in
E005 for H3, was charged directly to **hunger**, tripling the rate it accumulates and
destroying the metric H2 is measured on. A parameter added for one hypothesis had
silently changed the measurement basis of another.

The cost has now been moved to its own budget, `vigour`:

- Calling spends vigour; silence restores it (~90 s from empty).
- Vigour **enters the reward signal**, so calling stays genuinely expensive and H3
  still has a gradient for audience-sensitivity to emerge from.
- Vigour **attenuates the call flockmates hear**, because a spent bird cannot call
  loudly. Vocal effort is now self-limiting without an arbitrary cap.
- **Hunger is untouched by calling.**

Verified before running: a fixed flock's mean hunger returns to **0.323** (it was
0.630 with the cost charged to hunger; E004's regime was ~0.33), vigour cycles in
0.68–1.0 rather than pinning, and the ethogram still passes 7/7.

## 4. Prediction

**A learning advantage returns.** Learning-without-growth beats the fixed control on
within-run hunger change by more than t=2.23 (11 df).

Confidence is moderate, not high. E004's t=3.93 was measured in the saturated regime,
and the gain is now corrected — so the effect could be smaller, or absent for reasons
that have nothing to do with the call cost.

**Secondary predictions:**
- Exploration costs something: the noise-only control is worse than the fixed control.
- Growth stays the weaker of the two learning conditions (as in E001, E003, E004).

**If this is null**, H2 is in real trouble. It would be the first clean test, and a
null with no confound left to blame moves it toward `REFUTED at this timescale`.

## 5. Design

Four conditions, exploration stated explicitly in each, 12 matched seeds, 20 min,
gain 0.70, vigour budget in place. Identical to E004 in every other respect.

- **Command**: `python -m run.experiment --minutes 20 --seeds 12`

## 6. Result

12 matched seeds, 20 min, 16 hens. Wall clock 60 min.

| condition | hunger early | hunger late | change | fed % | exposure | synapses |
|---|---|---|---|---|---|---|
| fixed (innate only) | 0.312 | 0.331 | +0.018 | 6.2 | 3413 | 36373 |
| noise only (no learning) | 0.311 | 0.326 | +0.014 | 6.2 | 4177 | 36373 |
| learning, no growth | 0.327 | 0.408 | **+0.081** | **4.7** | 4492 | **19088** |
| learning + growth | 0.329 | 0.424 | **+0.095** | 5.4 | 5668 | 40731 |

```
vs fixed:  noise only          -0.004 +/- 0.013 SE   t=0.32  noise
           learning, no growth +0.062 +/- 0.016 SE   t=3.85  SIGNIFICANT, WORSE
           learning + growth   +0.077 +/- 0.031 SE   t=2.49  SIGNIFICANT, WORSE
```

**The environment is confirmed restored**: the fixed control's +0.018 / 0.331 matches
E004's +0.027 / 0.330. The metric works again.

## 7. Interpretation

**H2 is refuted at this timescale, and not by a null — learning makes hens
significantly worse.** Both learning conditions are beaten by the control at p<0.05,
and the learning hen feeds on 4.7% of timesteps against the control's 6.2%.

**Exploration is not the culprit.** The noise-only control is statistically
indistinguishable from fixed (t=0.32). Whatever is happening is the learning rule
itself, not the variability it needs.

**The smoking gun is the synapse count.** Learning-without-growth ends with
**19,088 of 36,373 innate synapses — it has destroyed 48% of the connectome the hen
was born with**, and forages worse for it. That is not learning; it is erosion.

The likely mechanism, and it follows from things already measured: the reward
prediction error hovers near zero by construction (E001 measured mean reward ≈ −0.02
against a baseline that tracks it). Weight updates are therefore close to a random
walk. Synaptic scaling pulls row sums down, and the Dale clamp floors magnitudes at
zero, so a weight that random-walks to zero **leaves the connectome permanently** —
there is no path back. Over 20 minutes that ratchet strips half the innate wiring.
The hen loses competence she was born with and builds nothing to replace it.

**This reframes E004 rather than contradicting it.** At gain 0.9 the pallium was
saturated, so its output was near-constant and the readout could only apply a fixed
*bias* to the motor drive. The same erosion was presumably happening — but it could
not reach behaviour, and what learning adjusted was a constant offset that happened
to help. A tuned offset is not a learned policy. That is a hypothesis about E004, not
an established fact, but it is consistent with everything measured since.

**What this does not say.** It does not say plasticity cannot work here, or that the
three-factor family is wrong. It says *this* rule, at *these* settings, over *this*
horizon, is net destructive. The obvious untested variable remains time — 20 minutes
against a 3-day critical period — but that excuse is now much weaker, because the
effect is not "too small to see", it is large and in the wrong direction.

## 8. Consequence

- **H2 → `REFUTED at this timescale`.** First clean test, no confound left, and the
  result is significantly negative in both learning conditions.
- **E004 reinterpreted**, not withdrawn: its positive result is most plausibly an
  artefact of the saturated regime, where learning could only shift a constant bias.
- **H2a is answered incidentally.** Growth is the *worse* of the two learning
  conditions here too (+0.077 vs +0.062), consistent across all five runs that have
  compared them. It never helped in any regime.
- **The next question is not "retune", it is "why does the rule erode".** Concretely:
  the zero-floor on weight magnitude makes pruning irreversible, so any zero-mean
  update process monotonically destroys structure. A rule that cannot recover a
  pruned synapse is a ratchet. That is a design flaw, not a hyperparameter.
- **Candidate directions**, recorded not taken: put a floor under innate weights so
  learning modulates rather than erodes them; make the reward signal sparse and large
  rather than continuous and near-zero; or let pruned synapses recover, so drift is
  not one-way.
- **No ethics review triggered.**
