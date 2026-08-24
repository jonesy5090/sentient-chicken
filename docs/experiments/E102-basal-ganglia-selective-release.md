# E102 — a basal-ganglia gate: suppression that must choose

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **H2**, **H2b**, **H2e**. Successor to
[E101](E101-top-down-suppression.md), which established that a gate helps and that an
unselective one is not enough.

---

## 2. Question

E101 gave the forebrain its first ability to oppose the innate arc, and it worked: predation
fell **−0.0917 (t=−3.74)** and replicated at **−0.0903 (t=−3.04)** on disjoint seeds, with
the untrained-gate control null in both blocks.

**But the gate learned to close almost everything** — turning to 0.099, pecking to 0.116,
crouch to 0.476, sparing only the contact call. The benefit had a mundane mechanism:
`actuation.py` sets `mobility = 1 − crouch`, so a crouching hen cannot move and stays inside
the strike radius; a largely inert hen drifts out and is caught less. She learned that
suppressing most things is survivable, not *when* to suppress anything.

That is the predicted failure mode of a **free** gate, and it is the third appearance of the
same degeneracy — `W_out` collapsing to a fixed direction (E100), and now `W_gate` collapsing
to a near-uniform closure.

**The vertebrate answer is not a free gate.** The basal ganglia hold motor programs under
tonic inhibition and *selectively release* them: cortex → striatum → pallidum → target, with
lateral competition in striatum making release focused rather than global. Suppression
becomes a per-action decision. Birds have a well-developed medial striatum and pallidum;
this model has neither.

**Does a gate that is forced to *choose* between channels stay selective — and does the
behavioural benefit survive when the degenerate solution is unavailable?**

That second clause is the real test. If predation benefit vanishes once she cannot suppress
everything, then E101's gain was **entirely** the artefact and not a learned improvement.

---

## 3. Prediction

The mechanism is lateral competition, which is the striatal property that matters:

```
s    = W_str @ motor_stub                       # striatal drive per motor channel
gate = sigmoid(GATE_OPEN_BIAS + s - beta * mean(s))
```

Uniform changes cancel. If learning drives every `s` down together, `s − mean(s)` stays zero
and the gate does not move. **Global suppression is architecturally unavailable**; only
redistribution is. `W_str` starts at zero, so gate = sigmoid(4.0) = 0.982 and a hatchling's
reflexes arrive intact.

1. **The gate stays differentiated.** E101's gate closed 11 of 12 channels below 0.9. I
   predict **no more than 6** below 0.9, with a spread (max − min) at least as large as
   E101's.
2. **Global suppression is impossible, measured not assumed.** Mean gate stays within
   **0.05** of its hatch value of 0.982 regardless of what is learned.
3. **The predation benefit survives at reduced size.** I predict a significant reduction
   versus no gate, smaller than E101's −0.09. Held at **below even odds** — if E101's
   benefit was purely the inert-hen artefact, this should collapse to nothing, and that is
   the outcome I consider most likely.
4. **The hunger cost shrinks.** E101 cost 0.498 → 0.596 because she suppressed feeding too.
   A gate that must choose should not close pecking as readily.

## 4. Falsifier

**Selectivity falsifier.** More than 6 channels close below 0.9, or mean gate moves more than
0.05 from hatch. The competition would then not be doing its job and E102 is just E101 with
extra arithmetic.

**Primary.** The predation benefit versus no gate is not significant on **both** seed blocks.
Then E101's improvement was the degenerate solution and nothing else — **the project's only
learned behavioural gain would be an artefact of the crouch/mobility coupling**, and that
must be recorded as such rather than left standing.

**Inertness falsifier.** Any result changes with `bg_gate=False`. Asserted bit-identical.

**Instrumented, not falsifiers** (the two cheap questions E101 left open, folded in here
because C's design turns on the same issue): whether the gate's *state-dependence* rises —
E101's collapsed like `W_out`'s — and whether signed perception stays unused when a
selective gate is available alongside it.

---

## 5. Design

**`bg_gate: bool = False`** in `PlasticConfig`, with `bg_lateral: float = 1.0` as `beta`,
and `W_str` in `BrainParams` initialised to zero. When set, replaces E101's independent
per-channel sigmoids with the competitive form above. `W_str` learns on the same rule and
schedule as `W_gate` did, so a null cannot be blamed on a withheld signal.

The arcopallium is *still* not rewired. It remains the obvious descending route and remains
a separate question; E102 changes one thing.

**Measurements**, all matched-seed, 8 seeds per block, two disjoint blocks, 30 min rearing,
`hebbian_readout` throughout:

1. **Gate profile** per motor channel, against E101's, plus mean and spread.
2. **2×2** {untrained, reared} × {no gate, bg gate} on predation and hunger — the design
   E101 needed two attempts to get right, reused directly.
3. **Gate state-dependence**: does the gate vector vary with situation, or is it fixed like
   E100's direction?

### Cost

~40 minutes.

---

## 6. Result

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*
