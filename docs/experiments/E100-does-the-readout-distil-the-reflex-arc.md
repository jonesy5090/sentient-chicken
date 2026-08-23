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

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*
