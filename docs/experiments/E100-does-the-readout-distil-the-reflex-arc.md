# E100 — does `W_out` just learn to reconstruct the reflex arc?

*Sections 1–4 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **H2**, **H2b**, **H2f**, and by consequence every null beneath
them. Tests a claim recorded as **not adopted** in
[E096](E096-red-team-review-2026-08.md) §7.

---

## 2. Question

An outside review argued, from the source and without measuring it, that the rules in this
codebase can only ever distil the innate arc into the cortical pathway. The argument is
short. `hen/plasticity.py:479` updates `W_out` with

```python
dw_out = eta_out * m_out * dz_motor[:, :, None] * dz_slow[:, None, -n_motor:]
```

and `dz_motor` is the deviation of `z_motor`, which traces `motor` — **the final,
post-sigmoid motor output** (`run/simulate.py:114`). At hatch and through most of any run
that output *is* the reflex arc's: measured cortical/reflex ratios are 0.03 for a fixed hen
and 0.10 for the instrumental rule.

If true, this is not a finding about one experiment. It would make E058, E059 and E069's
conclusion — *"the rule amplifies existing innate anchors and cannot build a new
contingency from nothing"* — a **one-line consequence of the update equation** rather than
five experiments' worth of inductive work. It would predict T2's place-avoidance null
(E063 deliberately gave place channels no innate anchor), H2c's comprehension null, and
H3's audience-contingency null, with no further runs. And it would mean the question is not
*which* rule but whether this architecture can learn anything the reflex arc does not
already do.

E096 recorded it as not adopted **because it was argued and not measured**, and noted it
makes a sharp, cheap prediction. This measures it.

## 3. Prediction

If the readout is distilling the arc, then after rearing the cortical drive should be a
**scaled copy of the reflex drive**, and increasingly so with training.

1. **Cosine similarity between the cortical and reflex drive vectors rises with rearing**,
   from near zero at hatch (`W_out` is near-silent by construction) toward a high value.
   I predict **≥0.7** after 30 minutes under `hebbian_readout`.
2. **It rises further with the rule that produces the larger behavioural change.**
   `hebbian_readout` (cortical/reflex ≈ 0.90) should show a higher similarity than the
   instrumental rule (≈ 0.10).
3. **A shuffled control does not.** Correlating cortical drive against a *permuted* reflex
   vector should stay near zero, which distinguishes "distils the arc" from "both vectors
   happen to be dominated by the same few always-on channels".

I hold prediction 1 at better than even. The mechanism is plainly there in the update; what
I do not know is whether `dz_slow`'s pallial factor carries enough independent structure to
pull the readout away from a pure copy.

## 4. Falsifier

**Primary.** Cosine similarity after rearing stays below **0.4** under `hebbian_readout`.
The claim is then not supported: the readout learns something with substantial structure of
its own, E058/E059/E069's conclusion remains an empirical finding rather than a theorem,
and the review's central structural argument is **rejected on measurement** rather than
left open.

**Triviality falsifier.** The shuffled control also exceeds 0.4. The similarity would then
be an artefact of both vectors being dominated by the same always-on motor channels, and
the measurement says nothing about learning.

**Direction falsifier.** Similarity does not increase from hatch to reared. Whatever
alignment exists would then be a property of the initial connectome rather than of what the
rule learned.

---

## 5. Design

Three arms, matched seeds: **fixed** (no plasticity), **instrumental** (`enabled=True`,
defaults), **hebbian** (`hebbian_readout=True, readout_scaling_strength=0.3`). 4 seeds,
30 min rearing, 16 hens.

For each, measured over a fresh 2-minute free-running probe with plasticity off:

- **cosine(cortical, reflex)** per hen per step, over the `MOTOR_DIM` vector, averaged
- the same against a **per-step permuted** reflex vector (the triviality control)
- **at hatch** and **after rearing**, for the direction falsifier
- cortical/reflex magnitude ratio, for comparability with the numbers already recorded

`brain.step` already returns both drives separately in `Drives` — this needs no new
instrumentation, which is itself part of why the claim was cheap to test and expensive to
leave open.

---

## 6. Result

### 6a. The pre-registered channel null was the wrong null, and its falsifier caught it

| arm | at hatch | reared | channel null | **time null** | **excess** | cort/refl |
|---|---|---|---|---|---|---|
| fixed | −0.1175 | −0.1175 | −0.1348 | −0.1132 | **−0.0043** | 0.026 |
| instrumental | −0.1175 | **−0.6509** | −0.4072 | −0.6482 | **−0.0027** | 0.089 |
| hebbian | −0.1175 | **+0.5911** | +0.5677 | +0.5941 | **−0.0031** | 0.695 |

The **triviality falsifier fired**: hebbian's 0.5911 only falls to 0.5677 when the reflex
vector's channels are permuted. Without that control I would have reported "0.59 cosine
confirms the readout distils the arc", which is clean, quotable and wrong.

But the channel null is *also* wrong — permuting channels destroys the magnitude
correspondence that makes any two motor vectors superficially alike, so it cannot isolate
what the claim asserts. The right null pairs cortical drive at step *t* with reflex drive at
some *other* step: channel structure preserved exactly, only the moment-to-moment
correspondence destroyed.

**Against the time null the excess is zero in every arm** — −0.0043, −0.0027, −0.0031. The
large raw cosines survive time-shuffling untouched.

### 6b. Which has only one explanation, and it is the result

If shuffling time changes nothing, the vectors are not tracking each other moment to
moment; they simply each have a stable characteristic shape. Measured directly, as the mean
cosine between each step's drive and that arm's own mean direction — **1.0 means the pathway
emits one fixed direction and only its magnitude varies**:

| arm | cortical direction stability | reflex (reference) |
|---|---|---|
| **fixed / untrained** | **0.6193** | 0.8956 |
| instrumental, reared | **0.9133** | 0.8854 |
| hebbian, reared | **0.9587** | 0.8832 |

**Training makes the learned pathway less state-dependent, not more.** The cortical output
starts relatively variable and converges toward a single fixed direction. At hebbian's
cortical/reflex magnitude ratio of 0.695, that is a large near-constant vector being added
to the motor drive.

*(The reflex arc's own stability is 0.88 — it is not 0, because most of the time no predator
is present and its output is dominated by tonic terms. The comparison that matters is
untrained → trained cortical, 0.62 → 0.96, not cortical against reflex.)*

## 7. Interpretation

**The distillation claim is rejected on measurement.** E096 §7 recorded it as not adopted
pending a test; the test says no. The readout does not reconstruct the reflex arc — it has
no moment-to-moment relationship with it in any arm. E058/E059/E069's conclusion stays an
empirical finding rather than becoming a theorem, and the review's central structural
argument does not hold.

**What replaces it is worse for the project.** The learned pathway converges on a fixed
output direction. A fixed direction can only rescale tendencies the arc already has; it
cannot express "do X in situation A and Y in situation B", because it emits the same X
regardless of situation and varies only in how hard.

That is a mechanism for essentially every null in the tree, and unlike the distillation
claim it is measured:

- **H2b** — "the rule cannot acquire behaviours outside the innate repertoire". A fixed
  direction is exactly a rescaling of the existing repertoire.
- **H2f** — `hebbian_readout` produces a large behavioural change (cort/refl 0.695) that is
  78% a call relay. A near-constant boost on the call channels is what that looks like.
- **T2** — no place-specific avoidance at any reward magnitude. A fixed direction cannot be
  place-specific.
- **H2c, H3** — comprehension and audience contingency both require conditioning on state.

**And it vindicates a claim this project rejected for a good reason.** E013–E016 concluded
the readout "could only slide a constant". The founding red-team review overturned that
reasoning, correctly: a rank-one `ΔW = u vᵀ` contributes `u (v · x)`, which varies perfectly
well with `x`. The *reasoning* was wrong and the *conclusion* is right — measured, and for a
different reason. The readout is not constrained to a constant by rank; it **converges** to
one under training.

**Why the direction converges is not established here** and should not be guessed at. The
obvious candidate — that `dz_slow`'s pallial factor loses its state-dependence, so the
outer product's second factor becomes near-fixed — is untested.

## 8. Consequence

**Adopted: the cortical pathway's learned output is near-directionally-constant, and that
is a sufficient explanation for the project's persistent null.** This supersedes "which rule
class" as the framing: both rules tested converge to the same degeneracy, one at 0.91 and
one at 0.96.

**Not adopted: any redesign.** The measurement identifies the failure mode; it does not say
which of several architectural changes would fix it, and choosing between them is a design
decision rather than a measurement.

**The next question is why the direction collapses.** Whether it is `dz_slow` losing
state-dependence, the readout scaling, or the motor-stub representation is testable and
cheap, and it determines which fix is the right one. That is the measurement to run before
anything is redesigned.
