# E020 — does H2 survive a working reward and an expressive readout?

> **Pre-registration.** Sections 1–5 written and committed before the run was launched.
> Sections 6–8 after.

## 1. Parent hypothesis

**H2** — three-factor plasticity produces measurable behavioural improvement. Currently
`REFUTED at this timescale` on the strength of
[E013](E013-clean-test-of-h2.md), which found learning significantly *worse* than not
learning (+0.062 hunger change, t=3.85).

## 2. Question

E013 was recorded as the clean test of H2. [E019](E019-three-verified-defects.md) found
it was not: 98.1% of the reward variance in that run came from the vigour term, and the
readout it was measuring could only slide a constant offset. Does the refutation survive
now that both are fixed?

## 3. What changed since E013, and why attribution is limited

**Four things differ, not one.** This is stated first because it bounds what the result
can mean.

| change | experiment | effect |
|---|---|---|
| strike penalty units | E014 | a strike was worth −100; connectome recovered 19,088 → 30,058 |
| vigour out of `reward()` | E019 | reward variance was 98.1% call cost; now hunger 54% / cold 46% |
| covariance rule (centred traces) | E019 | cortical drive variability 0.007 → 0.080 |
| audible calls | E019 | a flockmate's alarm moves the channel +0.908, was 0.0000 |

So **E020 cannot attribute any change to any single fix.** It answers "what is H2's
status now", not "which fix mattered". Attribution needs an ablation ladder, and that is
a separate experiment worth running only if the status actually moves.

The audio fix is included but should be close to inert here: H2's metric is drive
regulation, and hunger is not obviously mediated by hearing flockmates. If it turns out
to matter, that is itself informative and unpredicted.

## 4. Prediction

**Primary: the harm shrinks but does not reverse.** Learning-without-growth against the
fixed control, on within-run hunger change: E013 measured **+0.062 ± 0.016 (worse)**. I
predict **between +0.005 and +0.040, still positive (worse), and no longer significant**
at 11 df.

Reasoning: the two fixes remove a teaching signal that was pointing at the wrong thing
and a readout that could only apply one number. Both should reduce harm. But nothing
here gives the rule a *reason* to find a better foraging policy — the credit window is
still `tau_slow = 0.2 s`, and 8% drive variability is more than 0.7% without being much.

**Confidence: low, and the direction is the part I trust least.** My record on
predicting mechanism in this project is 1-for-7, and I have twice this session stated a
mechanism confidently and been wrong about which measurement mattered.

**Secondary predictions:**
- **Less connectome erosion.** The centred rule produces signed updates rather than a
  one-way drift, so fewer weights should random-walk into the Dale floor and leave
  permanently. Predict learning-without-growth ends above 30,000 synapses.
- The noise-only control stays indistinguishable from fixed (E013: t=0.32).
- Growth stays the weaker learning condition (true in all five previous comparisons).

## 5. Falsifier

**If the harm is unchanged — still ≈ +0.06 and significant — then neither the reward
composition nor the readout's expressiveness was causing it.** That would make H2's
refutation *stronger* than it is now, not weaker: two of the best available excuses would
be gone. It would also mean E019's fixes, though correct, are irrelevant to H2, and the
next place to look is the credit window (0.2 s) or the reward's near-zero prediction
error rather than anything measured so far.

**If learning becomes significantly better than fixed**, H2 moves from
`REFUTED at this timescale` to `SUPPORTED`, and E013–E016's entire interpretation —
including E015's superadditivity and E016's staging result — needs re-running, because
all of it characterised the behaviour of a rule that has since changed.

Either outcome changes a status. A result that changes nothing would mean the metric is
insensitive, which would itself need explaining.

## 6. Design

**Identical to E013**, so the comparison is as close to like-for-like as the four
changes above allow: four conditions with `explore_sigma` stated explicitly in each,
**12 matched seeds**, 20 minutes, 16 hens, gain 0.70.

- **Primary metric:** within-run change in mean hunger, learning-no-growth minus fixed,
  paired across matched seeds, two-tailed t against `_t_critical(11) = 2.201`.
- **Secondary:** fed %, live synapse count, predator exposure (retired as uninformative
  since E003/E004 — reported, not interpreted).
- **Command:** `python -m run.experiment --minutes 20 --seeds 12`

## 7. Result

*To be written after the run.*

## 8. Consequence

*To be written after the run.*
