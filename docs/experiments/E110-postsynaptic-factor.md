# E110 — give the update a direction of its own

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **H2**, and **H2b** directly.
The intervention [E109](E109-what-the-rule-writes.md)'s diagnosis names.

---

## 2. Question

E109 measured the obstacle. `Δcortical = m · (dz_slow · stub) · dz_motor`, so `dz_motor`
is the update's entire direction in motor space, and it sits at cosine **0.9822** to the
reflex arc's own deviation — **0.9916** in the windows where the reward fires. The rule
can only write *more of what she was already doing*.

The intervention is to change what the postsynaptic factor traces, so the update has a
direction the arc does not dictate. Two candidates:

**Node perturbation** — trace the **exploration noise** itself. `Δw ∝ m · ξ ⊗ x`, with
`ξ` the injected perturbation, is the standard unbiased estimator of a reward gradient,
and it is what `plasticity.py`'s own docstring already reaches for when it explains why
the noise is placed on the motor drive rather than the membrane: "an exploratory action
that happens to pay off gets credited to the synapses that produced it". The rule as
written does not do that — it credits the *output*, which is the arc plus a 2–5% noise
component. Crediting `ξ` alone is the version the docstring describes.

It also has the right biological anchor for this project specifically. **Birdsong
learning works this way**: LMAN injects variability into RA and reinforcement correlates
that variability with song quality. This is a model of a bird, and the one vertebrate
system where perturbation-based credit assignment is well established is birdsong.

**Cortical-only** — trace `sigmoid(cortical + b_motor)`, crediting the learned pathway's
own contribution. Weaker on theory but a direct read of "what did *I* do", and cheap to
include.

**So: does an update with its own direction produce learning that changes behaviour?**

## 3. Prediction

1. **The intervention will do what it targets.** Cosine between the postsynaptic factor
   and the reflex deviation falls below **0.3** for the noise arm (it is independent
   noise, so near zero) and below 0.6 for the cortical arm, against E109's 0.9822.
2. **Behaviour will not follow.** I predict **no** significant improvement in hunger
   against the frozen-readout control, in any arm. Four explanations have now been
   offered for this null and three are dead; the base rate for "this one is the cause" is
   poor, and node perturbation is a famously high-variance estimator being asked to work
   in 30 minutes of chicken time.
3. **If anything works, it is the noise arm**, because it is the only one that is an
   unbiased gradient estimator rather than a plausible-looking substitution.
4. Stated so a null cannot be reframed later: **I expect this experiment to close the
   line, not to open one.**

## 4. Falsifier

**Primary — and it is written to end the line.** If the cosine falls as predicted (the
intervention demonstrably removes E109's obstacle) **and** hunger is still
indistinguishable from the frozen-readout control in every arm, then removing the
obstacle does not produce learning. E109's finding would stand as a real constraint that
is **not the operative cause**, the "what the rule does" line closes alongside the three
upstream ones, and H2's null would have **no** surviving mechanistic candidate anywhere in
the rule. That is the result, and it should be recorded rather than patched with a sixth
mechanism.

**Instrument falsifier — checked and reported before the headline.** The cosine must
actually fall. If it does not, the flag is not doing what it claims and no behavioural
number in this experiment means anything.

**Magnitude-confound falsifier.** Mean `|ΔW_out|` per consolidation must be within 2× of
the baseline arm's. E109's obstacle is about *direction*; if an arm also changes the
update's *size* by an order of magnitude, any behavioural difference is confounded with
learning rate — the E089 lesson. The design controls for this (§5) and this falsifier
checks the control worked.

**Regression falsifier.** Inertness not bit-identical with the flag at its default, any
ethogram assay changing state, or the suite failing.

**Replication rule.** E021: any significant behavioural result is provisional until it
replicates on a disjoint seed block. Nothing moves the tree on block one.

## 5. Design

**`postsynaptic_factor: str = "motor"`** in `PlasticConfig`. `"motor"` is the current
rule, bit-identical. `"noise"` traces the injected exploration perturbation; `"cortical"`
traces `sigmoid(cortical + b_motor)`.

`Drives` gains a `noise` field so the perturbation is available to the caller — it is
currently added to the drive and discarded, which is why crediting it has never been
possible.

**Magnitude control, and its cost.** A raw swap changes the update's size as well as its
direction: at 30 min `explore_sigma` is ~0.4 while the motor output's deviation is far
smaller, so the noise arm would learn at a different effective rate. So the alternative
factor is **rescaled per hen per step to the norm of `dz_motor`**, leaving direction as
the only difference. `PlasticState` gains `z_post`/`z_post_bar`, allocated always and
updated only when the flag is off its default.

The cost is real and is stated rather than hidden: for a true node-perturbation estimator
the perturbation's *magnitude* carries information, and normalising it away makes this a
direction-only test rather than a faithful implementation of the estimator. That is the
right first cut — E109 identified a direction problem — but a null here is a null about
direction, not a refutation of node perturbation as such.

**Arms**, matched seeds, 8 seeds, 16 hens, 30 min rearing:

| arm | |
|---|---|
| baseline | `postsynaptic_factor="motor"` (the current rule) |
| noise | `postsynaptic_factor="noise"` |
| cortical | `postsynaptic_factor="cortical"` |
| **frozen** | `eta_out=0.0` — the readout cannot learn at all |

The frozen arm is the control the whole experiment turns on: it is what "learning changed
nothing" looks like. It does not depend on the postsynaptic factor, so one arm serves all
three.

**Measured**: cosine between the postsynaptic factor and the reflex deviation (the
instrument check, by the E109 method); mean `|ΔW_out|` per consolidation; hunger and
`caught/dive` after rearing, paired by seed against the frozen control; `|cortical|`; and
the full ethogram at any arm that moves.

### Cost

~30 minutes.

---

## 6. Result

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*
