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

_Pending._

## 7. Interpretation

_Pending._

## 8. Consequence

_Pending._
