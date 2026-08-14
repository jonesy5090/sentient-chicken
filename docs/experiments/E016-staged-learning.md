# E016 — does teaching the two pathways one at a time fix the interference?

> **Pre-registered.** Sections 1–5 committed before the run.

## 1. Parent hypothesis

**H2** — three-factor plasticity produces measurable behavioural improvement.
`REFUTED at this timescale`. This tests [E015](E015-decomposing-the-harm.md)'s
explanation of *why*.

## 2. Question

Two things learn in this model: the wiring **inside** the pallium (`W`), and the
connection **out to the muscles** (`W_out`). E015 measured their costs separately and
together:

| what learns | harm vs a hen who cannot learn |
|---|---|
| inside the pallium only | +0.010 |
| out to the muscles only | +0.021 |
| **both at once** | **+0.052** |

Both together cost far more than the two added up (+0.031). E015's reading was a
**moving target**: the outgoing connection is trying to learn what the pallium's
states *mean*, while those states are being rewritten underneath it. Neither ever
settles — like learning someone's name while they keep changing it.

**If that is right, teaching them one at a time should recover most of the loss.**

## 3. Prediction

A staged hen — pallium first with the outgoing connection held still, then the
outgoing connection with the pallium held still — lands close to the +0.021 of
learning the outgoing connection alone, rather than the +0.052 of doing both.

Stated as: **staged harm < 0.035**, i.e. clearly nearer the sum-of-parts than the
measured interaction.

**This is not a prediction that learning starts working.** Even a complete success
here leaves the hen worse than one who cannot learn at all. It would only mean the
interference is understood and removable, leaving the *representation* problem (H2d)
as the sole remaining cause.

## 4. Falsifier

Staged harm ≈ +0.052, indistinguishable from simultaneous. That kills the moving-target
reading and means the two pathways interfere for some reason that ordering does not
touch.

## 5. Design

Four conditions, 6 matched seeds, 20 min each, gain 0.70, vigour budget, exploration
stated per condition.

- **fixed** — nothing learns (anchor; must reproduce +0.036)
- **simultaneous** — both learn throughout, the current default
- **staged** — first half `eta_out=0` (pallium settles), second half `eta=0`
  (outgoing connection learns against a stable pallium)
- **staged, reversed** — the other order, as a control. If ordering matters for the
  reason claimed, pallium-first should beat muscles-first; if both stagings work
  equally, the benefit is from *not learning two things at once* rather than from
  settling the representation, which is a different explanation.

- **Command**: `python -m run.staged --minutes 20 --seeds 6`

## 6. Result

6 matched seeds, 20 min. The fixed anchor reproduced +0.036 / 5.6% exactly.

| condition | hunger change | harm vs fixed | fed % | synapses |
|---|---|---|---|---|
| fixed | +0.036 | — | 5.6 | 36,369 |
| simultaneous | +0.088 | +0.052 | 4.9 | 30,109 |
| staged, **pallium first** | +0.095 | +0.059 | 5.0 | 30,214 |
| staged, **muscles first** | **+0.052** | **+0.016** | 5.0 | 31,631 |

```
vs simultaneous:  pallium first   +0.007 +/- 0.028 SE   t=0.25   noise
                  muscles first   -0.036 +/- 0.010 SE   SIGNIFICANT
```

## 7. Interpretation

**The prediction is falsified, and the condition included as a *control* is the one
that worked.**

Pallium-first — the order the moving-target story predicted would help — is
indistinguishable from learning both at once (t=0.25). Muscles-first cuts the harm by
**69%**, from +0.052 to +0.016, and clears significance.

**So the moving-target reading is wrong as stated.** If the problem were the readout
chasing a representation that keeps changing, letting the representation settle first
would help. It does not. Something order-dependent is going on, but not that.

**A simpler reading fits all of it: the harm is dominated by whichever pathway learned
last, and the readout is the harmful one.**

- Pallium-first *ends* on readout learning — the expensive pathway (+0.021 alone in
  E015) — and lands at +0.059.
- Muscles-first *ends* on pallium learning — the cheap pathway (+0.010 alone) — and
  lands at +0.016, close to that figure.

On this account staging does not fix an interaction at all. It just determines which
pathway gets the last word, and the readout having the last word is what costs.

That is a hypothesis, and this project's record on mechanism hypotheses is now 1-for-6.
It makes a sharp, cheap prediction: **a short final stage of pallium-only learning,
appended to any schedule, should absorb most of the harm regardless of what came
before.** If that holds it is a real effect; if it does not, this reading joins the
others.

**What it does not do is rescue H2.** Muscles-first is still worse than a hen who
cannot learn at all (+0.016 against 0.000). Learning has gone from clearly harmful to
mildly harmful. That is progress in understanding, not a working learning rule.

## 8. Consequence

- **H2 unchanged**: `REFUTED at this timescale`.
- **E015's moving-target explanation is withdrawn.** The ordering effect is real and
  significant but runs opposite to what that story predicts.
- **New leading hypothesis, explicitly labelled as one**: harm is dominated by the
  pathway that learns last. Testable by appending a short pallium-only stage to an
  otherwise simultaneous schedule.
- **This is a workaround, not a fix.** Even at its best, staging leaves the hen worse
  than not learning. The underlying problem is still that the pallium cannot tell its
  inputs apart (H2d), and no scheduling trick addresses that.
- **H2d remains the critical path**, now with one more experiment pointing at it.
