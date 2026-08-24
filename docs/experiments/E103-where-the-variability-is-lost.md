# E103 — why does the learned pathway stop varying with situation?

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **H2**, **H2b**, **H2e**. The question left open by
[E100](E100-does-the-readout-distil-the-reflex-arc.md),
[E101](E101-top-down-suppression.md) and
[E102](E102-basal-ganglia-selective-release.md), which each found the same collapse in a
different pathway and none of which explained it.

---

## 2. Question

Three experiments have now found the same thing in three places:

| | measure | untrained | reared |
|---|---|---|---|
| E100 | `W_out` cortical drive direction | **0.62** | **0.96** |
| E101 | free reflex gate | — | near-uniform closure |
| E102 | competitive gate | — | **0.993** |

**Training makes the learned pathway *less* state-dependent.** It converges on emitting one
fixed pattern whose magnitude alone varies. E102 showed the pattern can be made *selective
across actions* by lateral competition — she suppressed pecking and turning, spared crouch
and flee — but it stayed fixed *across contexts*: "always suppress these three", never
"suppress them when a hawk is near".

Adding architecture has not touched this. Three additions, same collapse.

**Where in the chain is the situation-dependence lost?** Everything the learned pathways
read comes through one place: `motor_stub = rate(x)[:, -n_motor:]` (`brain.py:66`). Both
`W_out` and `W_str` are linear readouts of it. **If the stub is already fixed, no weight
matrix can make the output vary**, and every architectural fix is aimed at the wrong stage.

---

## 3. Prediction

Three candidate explanations, distinguishable by measuring direction stability at each
stage of the chain — observation → sensory stub → pallium → motor stub → the traces the
rule actually uses → cortical output.

**(a) Representation.** The motor stub itself becomes near-fixed during training, so the
readout has nothing varying to read. Predicts stub stability rising sharply from untrained
to reared, tracking the readout's 0.62 → 0.96.

**(b) Timescale.** The stub varies, but the eligibility traces the rule multiplies —
`z_slow` (τ 0.20) and `z_motor` (τ 0.10) — smooth the variation away before it reaches the
weight update. Predicts the stub varying while `dz_slow` / `dz_motor` are near-fixed.

**(c) Learning.** Everything upstream varies and the *weights* collapse — the update
averages over situations, so `W_out` converges on the mean-situation solution. Predicts
stub and traces both varying while the cortical output alone is fixed.

**I expect (a)**, at roughly 60% confidence. The pallium is recurrent and its own activity
dominates the motor stub's input; E017 already found the pallium collapsing under load, and
E100's untrained-to-reared jump is exactly the shape of a representation degrading rather
than a readout choosing.

**Falsifier for my own expectation:** if stub stability is below 0.85 reared, (a) is wrong
and the cause is downstream.

## 4. Falsifier

**Primary.** No stage shows a marked rise in stability between untrained and reared. The
collapse would then not be located anywhere in this chain, and the framing — that
variability is *lost* at some stage — is wrong.

**Triviality falsifier.** The *observation* itself is near-fixed (stability > 0.9). Then
nothing downstream could vary and the finding is about the world, not the brain. This is the
control that makes the rest interpretable, and it must be checked first.

**Confound falsifier.** The staged probe does not actually vary the situation — measured as
the observation's own stability being high *and* the reflex arc's output being fixed. The
reflex arc is state-dependent by construction, so it is the reference: if *it* looks fixed,
the probe is broken.

---

## 5. Design

Direction stability — mean cosine between each step's vector and that arm's own mean
direction, where 1.0 means "one fixed pattern, magnitude only" — measured at every stage:

1. `obs` (the observation itself)
2. sensory stub activity
3. pallium activity
4. **`motor_stub`** — what both readouts read
5. `dz_slow[-n_motor:]` — the presynaptic factor the rule actually multiplies
6. `dz_motor` — the postsynaptic factor
7. `cortical` — the output (E100's measure, as the anchor)
8. `reflex` — the reference, state-dependent by construction

**Untrained and reared**, 4 seeds, `hebbian_readout`, free-running with a hawk every 60 s so
situations genuinely differ. 2-minute probe with plasticity off.

### Cost

~10 minutes. This is a diagnostic, and it is cheap precisely because nothing needs building.

---

## 6. Result

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*
