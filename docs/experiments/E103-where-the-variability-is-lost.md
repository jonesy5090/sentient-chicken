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

### 6a. The variability is destroyed at the first synapse, and at hatch

Direction stability at every stage — 1.0 means one fixed pattern whose magnitude alone
varies:

| stage | untrained | reared | change |
|---|---|---|---|
| 1 observation | 0.6375 | 0.6573 | +0.02 |
| **2 sensory stub** | **0.9707** | **0.9708** | **+0.0001** |
| 3 pallium | 0.9934 | 0.9927 | −0.0006 |
| 4 motor stub | 0.9930 | 0.9925 | −0.0006 |
| 5 `dz_slow` (presyn) | 0.5890 | 0.0767 | −0.51 |
| 6 `dz_motor` (postsyn) | 0.3585 | 0.1137 | −0.24 |
| 7 cortical out | 0.6193 | **0.9587** | **+0.34** |
| 8 reflex (reference) | 0.8956 | 0.8832 | −0.01 |

**The observation genuinely varies (0.64). The sensory stub is already near-fixed (0.97).
Everything downstream inherits it, and training changes nothing** — untrained and reared are
identical at every stage from the stub onward.

Both controls clear: the observation is not near-fixed (0.66, falsifier at 0.9), and the
reflex arc varies as it must (0.88), so the probe is measuring real situational differences.

**My own hypothesis was half right and my falsifier did not catch the wrong half.** I
predicted "representation", and stated its signature as *stub stability rising sharply from
untrained to reared, tracking the readout's 0.62 → 0.96*. It does not rise. It is 0.993 at
hatch and 0.993 after 30 minutes. The stub does not *become* fixed — **it was never
otherwise.** I wrote the falsifier as "wrong if stub stability < 0.85 reared", which the
result passes while the mechanism I described is wrong. A falsifier on the *level* could not
test a claim about the *change*.

### 6b. Why — `W_in` is strictly positive

| | |
|---|---|
| `W_in` nonzero entries | **2630** |
| negative | **0** |
| positive | **2630** |
| construction | `rng.gamma(2.0, 0.5)` — strictly positive |

And the consequence, measured as the mean direction's share of a typical vector:

| | share |
|---|---|
| observation | **69.0%** |
| sensory stub | **97.8%** |

A strictly-positive projection of a strictly-positive observation gives every unit a large
common term — *"how much is in view"* — with situation differences riding on top as a small
perturbation. **The situation-specific signal falls from 31% to 2.2% in one synapse.**

**This is the same defect the project has already found twice, one stage earlier.** E070
measured a planted place association predicting 1.0000 at its own place and 0.9637 at a
different one; E071 fixed it by centring `z_lag`, whose docstring records the across-stimulus
signal at *3.7% of its DC baseline*. That is this number, downstream. **The DC domination was
patched where it was noticed and never addressed at the source.**

## 7. Interpretation

**Every learned pathway in this model reads a near-constant input, and has since hatch.**
`W_out`, `W_str` and `W_pred` are all linear readouts of activity that is 97.8% one fixed
direction. No weight matrix can extract situation-dependence that is not there, which is why
three architectural additions — a signed perceptual channel, a free gate, a competitive gate
— each produced the same collapse. They were aimed at the readout. The problem is the
representation, and it precedes them all.

**The missing mechanism is lateral inhibition**, and its absence is the more interesting way
to state the finding. Real sensory systems almost universally implement centre-surround or
divisive normalisation at the first relay, and its *function* is exactly this: discard the
common component, pass the contrast. Retinas do it, thalamic relays do it, and the avian
tectum does it. This model's sensory stage has none — just a positive random projection —
so it passes the DC and buries the signal.

**This also reframes E100, E101 and E102 rather than overturning them.** E102's hen still
learned a legible policy — suppress pecking and turning, spare crouch and flee — and did so
*despite* reading a 2.2%-contrast input. That it managed anything is more impressive in this
light, and it explains why the policy was selective across actions but fixed across
contexts: action identity is available in the readout's own weights, but context is what the
representation threw away.

**And it is a candidate explanation for the whole H2 line.** H2's clean null, H2b's
limitation, H2c and H3's comprehension and audience nulls, T2's place-avoidance null — all
require conditioning on situation, and situation is 2.2% of what the pallium receives.

## 8. Consequence

**Nothing is adopted here. This is a diagnostic and the fix is a design decision.** But the
fix is now specific rather than speculative, and it is upstream of every previous attempt.

**The candidate: lateral inhibition at the sensory stub** — subtract a local or global mean
so the projection passes contrast rather than total illumination. Biologically standard,
architecturally small, and testable by exactly the measurement above: does stub stability
fall from 0.97, and does the cortical readout regain state-dependence?

**Two cautions worth recording before anyone builds it.** First, this changes the input to
*every* recorded experiment, so it needs the strictest inertness gate yet and a full
re-baselining plan — E076/E077 cost this project two withdrawn results for a smaller change.
Second, E071 already centres `z_lag` downstream; if the source is fixed, that centring may
become redundant or actively harmful, and both should be measured together rather than
stacked.

**Recorded for the tree: H2d's "the pallium cannot separate stimuli" line of work was
looking one stage too late.** E081 measured pallial decodability at 84.6% parked; the
question this raises is what that number would be if the pallium were not receiving a
97.8%-constant input in the first place.
