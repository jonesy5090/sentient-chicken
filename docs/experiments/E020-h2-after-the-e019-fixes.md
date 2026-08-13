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

12 matched seeds, 20 min, 16 hens. Wall clock 54 min.

| condition | hunger early | hunger late | change | fed % | exposure | synapses |
|---|---|---|---|---|---|---|
| fixed (innate only) | 0.315 | 0.320 | +0.005 | 6.6 | 5928 | 36373 |
| noise only (no learning) | 0.308 | 0.345 | **+0.037** | 6.0 | 4631 | 36373 |
| learning, no growth | 0.307 | 0.313 | +0.006 | 6.6 | 2628 | **35480** |
| learning + growth | 0.307 | 0.305 | −0.002 | 6.7 | 2822 | 55539 |

```
vs fixed:  noise only          +0.032 +/- 0.008 SE   t=3.84  SIGNIFICANT, WORSE
           learning, no growth +0.001 +/- 0.010 SE   t=0.08  noise
           learning + growth   -0.007 +/- 0.011 SE   t=0.65  noise
```

Against E013, which is the run this replaces:

| | E013 | E020 |
|---|---|---|
| learning-no-growth vs fixed | **+0.062, t=3.85 WORSE** | **+0.001, t=0.08 noise** |
| learning-no-growth synapses | 19,088 (48% destroyed) | **35,480 (2.5% lost)** |
| noise-only vs fixed | −0.004, t=0.32 | **+0.032, t=3.84 WORSE** |

### Scoring the registered predictions

- **Primary — right in direction, marginally outside the band.** I predicted the harm
  would shrink to +0.005…+0.040 and lose significance without reversing. Measured
  **+0.001, t=0.08**: non-significant as predicted, not reversed as predicted, and
  slightly below the registered floor. Calling that a hit on direction and a miss on
  magnitude.
- **Secondary, erosion — right.** Predicted above 30,000 synapses; measured **35,480**.
  The centred rule produces signed updates instead of one-way drift, and the connectome
  the hen was born with survives.
- **Secondary, noise-only — wrong, and it is the interesting part.** Predicted it would
  stay indistinguishable from fixed. It is now **significantly worse (t=3.84)**.
- **Secondary, growth — wrong.** Predicted growth stays the weaker learning condition,
  true in all five previous comparisons. It is nominally the *better* of the two
  (−0.002 vs +0.006), though both are noise against fixed so this should not be leaned
  on.

## 8. Interpretation

**H2's refutation does not survive, and H2 is not supported either.** The headline is a
clean null: a hen that learns is statistically indistinguishable from one that cannot
(t=0.08). That is a real change — `REFUTED at this timescale` was a positive claim that
learning *harms*, and it is now withdrawn — but it must not be read as a success. The
rule does not make her better at anything measured here. It merely stops making her
worse.

**E013's two headline findings were both artefacts.** The +0.062 harm and the 48%
connectome destruction are gone, at 2.5% loss. E013's stated mechanism — a random-walk
ratchet into the Dale floor — was already withdrawn by E014; what remained of the harm
went with the reward composition and the one-dimensional readout. Four experiments
(E013–E016) characterised the behaviour of a rule that was being taught almost entirely
"did you just call" through a readout that could only slide one number.

**Which means E015 and E016 are now measuring something that no longer exists.** E015's
superadditivity and E016's muscles-first staging result were both decompositions of a
harm that is no longer present. Neither should be cited. They are not *wrong* — they
described the old rule accurately — but they are about a system that has been replaced.

**The unpredicted finding: exploration now costs something.** The noise-only control —
identical to fixed except for σ=0.6 on the motor drive, with plasticity off — is
significantly worse than fixed (+0.032, t=3.84), where in E013 it was indistinguishable
(t=0.32). I do not know why, and I am not going to invent a mechanism for it; my record
on that in this project is 2-for-9 including today. Candidates, none tested: the audio
fix means noisy call output is now genuinely audible and perturbs flockmates, where
before every channel was pinned regardless; or the effect was always there and E013's
noisier baseline (SE 0.013 vs 0.008) hid it. These are distinguishable by one run with
the audio fix reverted, and that is the right next diagnostic.

**A post-hoc reading I am flagging rather than claiming.** Learning and noise-only carry
the *same* exploration noise, and learning lands at +0.001 while noise-only lands at
+0.037. The natural story is that learning pays back the cost of the exploration it
requires — which would be a genuine, if modest, benefit and the first positive thing H2
has produced. **It is post-hoc**, the contrast was not pre-registered, and the harness
reports both only against fixed, so I do not have the paired standard error to test it
properly. Recorded as a hypothesis with its own pre-registration owed, not as a result.
This is exactly the forking path an outside review warned about, and it would be easy to
promote it to the headline.

**Exposure, as pre-registered, is reported and not interpreted.** Learning hens show
−3299 ± 2779 (t=1.19). The metric was retired as uninformative in E003/E004 and the
standard errors here are the reason.

## 9. Consequence

- **H2 moves from `REFUTED at this timescale` to `UNDER TEST` with a clean null.** The
  claim that learning harms is withdrawn. No claim that it helps replaces it.
- **E015 and E016 are marked superseded** in the tree — they decompose a harm that no
  longer exists.
- **E013 is marked superseded**, with its result kept. It was the most consequential
  wrong result in the project and the route to finding out why is the record worth
  keeping.
- **New pre-registration owed:** does learning repay the cost of its own exploration?
  Needs the learning-vs-noise contrast run as a first-class paired comparison.
- **New diagnostic owed:** why did exploration become costly? One run with the audio fix
  reverted separates the two candidates.
- **Attribution ladder now worth running**, per §3 — the status moved, so which of the
  four changes did it is a real question rather than a hypothetical one.
- **E018 is unblocked** and can be re-run as designed.
