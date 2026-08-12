# E007 — does exploration let comprehension emerge?

## 1. Parent hypothesis

**H2b** — the learning rule cannot acquire behaviours outside the innate repertoire.
Feeds **H3**, which is blocked by it.

## 2. Question

[E006](E006-audibility-weighted-kin-reward.md) found comprehension flat at zero and
diagnosed the cause as a missing exploration mechanism: a reinforcement rule cannot
strengthen an action that never occurs, and the model is deterministic. Does adding
motor exploration noise let comprehension emerge?

## 3. Prediction

Comprehension — crouching in response to a played-back alarm call with no predator
visible — rises above zero in the learning condition once exploration is added, and
stays at zero without it.

## 4. Falsifier

Comprehension unchanged with exploration. That would mean E006's diagnosis was wrong
or incomplete and the blocker is somewhere else.

## 5. Design

- **Change**: Gaussian noise on the motor drive before the output nonlinearity,
  decaying with age on the critical-period schedule. Placed on the motor drive rather
  than the membrane so that `z_motor` — which traces the motor output — captures it,
  and an exploratory action that pays off is credited to the synapses that produced
  it.
- **Assays run at sigma=0.** They measure the learned policy, not the noise around it.
- **Second factor, added after the first result**: predator frequency. E005 predicted
  alarm-call learning would be starved of examples — hawks arrive on a ~900 s
  schedule, so a short run contains about one. Tested at 900 s and 30 s (30x more
  learning opportunities).
- Comprehension measured first, as the cheap mechanical readout. If it does not move,
  the audience assay is not worth running.

## 6. Result

Comprehension change over rearing, single seed, 15 min:

| exploration σ | hawk period | events/run | hatch | reared | change |
|---|---|---|---|---|---|
| 0.0 | 900 s | 1.0 | 0.0003 | −0.0004 | −0.0007 |
| 0.0 | 30 s | 30.0 | 0.0003 | −0.0005 | −0.0009 |
| 0.6 | 900 s | 1.0 | 0.0003 | −0.0007 | −0.0011 |
| 0.6 | 30 s | 30.0 | 0.0003 | −0.0002 | −0.0006 |

A sigma sweep at normal predator density was equally flat (0.0, 0.6, 1.5 → all ~0).

**The follow-up measurement that explains it.** Motor output is
`sigmoid(reflex + cortical + bias)`, and the crouch bias is −2.50:

```
drive needed for crouch > 0.5     : +2.50
innate reflex, hawk overhead      : +8.00
cortical drive to crouch, playback: +0.0017  (at hatch)
                                    -0.0149  (after rearing)
```

Theoretical cortical ceiling is +2.57, but only with all 48 motor-stub units
saturated and sign-aligned. Real motor-stub rates are 0.1–0.3 with mixed signs, so
the achievable contribution is **roughly 170x short of the threshold** — and it moved
*away* from it over rearing.

## 7. Interpretation

**Both hypotheses fail. Exploration does not rescue comprehension, and neither does
30x more predator exposure, and neither does both together.** E006's diagnosis was
correct that the rule cannot reinforce an action that never occurs, but wrong that
this was the binding constraint.

**The binding constraint is that the learned pathway can modulate behaviour but
cannot initiate it.** Crossing the crouch threshold from a standing start needs +2.50
of drive. The cortical pathway supplies 0.002. Exploration cannot help, because the
issue is not that the action is never *sampled* — it is that even when noise samples
it, the learned pathway has no way to reproduce it afterwards. There is nothing for
the credit to attach to.

That is the third finding in the same family, and together they describe a real
architectural tension rather than three separate bugs:

- **E002**: cortical influence too *high* and behaviour gets worse — an untrained
  pallium overriding good reflexes.
- **E007**: cortical influence low enough to avoid that, and new behaviours cannot be
  acquired at all.

The two pathways currently **sum into the same motor drive**, competing on a single
axis. That forces a choice between a pallium strong enough to initiate and one safe
enough not to override, and comprehension needs the first while drive regulation
needs the second.

**A resolution worth considering, not yet implemented.** In real vertebrates learned
control more often *gates or modulates* innate circuits than competes with them for
the same output. If the cortical pathway set a per-channel **gain on the reflex arc**
rather than adding its own drive, a learned signal could recruit an innate behaviour
without generating it from nothing — and could not run away, because it only scales
responses that already exist. Hearing an alarm call would raise the gain on the crouch
reflex, which is roughly how learned modulation of tectal circuits is thought to work.

This is a change to the core architecture and it should be a deliberate decision
rather than a patch applied mid-experiment. Recorded here; not done.

## 8. Consequence

- **H2b refined and upgraded.** It is not "no exploration" — it is that the learned
  pathway cannot initiate actions, measured at ~170x short of threshold. Exploration
  was a necessary addition but not a sufficient one.
- **Exploration kept** (`explore_sigma=0.6` default). It is correct biology, it costs
  almost nothing, and every future attempt at acquiring new behaviour needs it even
  though it was not sufficient here. It should not be reverted just because it failed
  to rescue this.
- **H3 remains blocked.** Three experiments, three real defects found and fixed, and
  the hypothesis has not moved. That is worth saying plainly.
- **Open architectural decision**: additive competition vs multiplicative gating
  between the innate and learned pathways. This is the fork the project is now
  standing at, and it affects everything from H3 onward.
- **Method note that keeps paying**: the cheap mechanical measurement (what drive is
  needed, what drive is available) settled in one minute what three behavioural
  experiments had been circling. Measure the mechanism, not the behaviour, when the
  behaviour will not move.
- **No ethics review triggered.**