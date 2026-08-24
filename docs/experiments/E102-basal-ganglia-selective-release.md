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

### 6a. The learned solution is selective, interpretable, and replicates almost exactly

Gate value per motor channel after 30 min, two disjoint seed blocks:

| channel | block 0–7 | block 8–15 | E101's free gate |
|---|---|---|---|
| **TURN_R** | **0.2482** | **0.2441** | 0.0992 |
| **TURN_L** | **0.2833** | **0.3070** | 0.1351 |
| **PECK** | **0.3390** | **0.3094** | 0.1163 |
| FORWARD | 0.9213 | 0.9288 | 0.5314 |
| SCRATCH | 0.9238 | 0.9217 | 0.6384 |
| CROUCH | **0.9830** | **0.9882** | 0.4756 |
| CALL_AERIAL | 0.9855 | 0.9904 | 0.4672 |
| FLEE | 0.9934 | 0.9887 | 0.6716 |
| channels closed | **3 of 12** | **3 of 12** | 11 of 12 |
| spread | 0.7506 | 0.7543 | — |

**Three channels closed against E101's eleven, and the same three both times.** What it
closed is coherent: **pecking and turning are the behaviours that put her head down**, and
`sensing.py` zeroes the aerial channel while `head_down` is high. It spared **crouch, flee,
and every call** — the entire anti-predator repertoire.

**The hen learned to suppress the behaviours that blind her, and to keep the ones that save
her.** That is the vigilance/foraging trade-off — the asymmetry this whole project is built
on — arrived at by learning rather than by wiring.

Hunger cost shrank as predicted: E101's 0.596 → **0.529 / 0.536**.

### 6b. The behavioural benefit is directionally consistent and fails its pre-registered bar

| contrast (df=7, crit 2.365) | block 0–7 | block 8–15 |
|---|---|---|
| **BG gate, reared brain** | **−0.0758, t=−2.68 ✓** | **−0.0567, t=−1.96 ✗** |
| BG gate, untrained brain | −0.0048, t=−0.13 ns | +0.0049, t=+0.28 ns |
| reared+BG vs untrained baseline | −0.0448, t=−1.42 ns | −0.0388, t=−3.54 ✓ |

**The primary falsifier FIRES.** §4 required significance on *both* blocks and the second
gives t=−1.96. The effect sizes are close (−0.076, −0.057) and both negative, and the
untrained control is null in both — but consistency of sign is not what I pre-registered,
and E021 is this project's standing warning about exactly this.

*A pooled estimate across 16 seeds would be roughly −0.066, t≈3.3. **Pooling was not
pre-registered here** — E030's was, deliberately and in advance — so that figure is recorded
as a post-hoc estimate and is not the basis of any claim.*

### 6c. Selective, but still not conditional

Gate state-dependence: **0.9927 / 0.9929** — essentially a fixed vector, unchanged from
E100's collapse. She learned *"always suppress these three"*, not *"suppress them when a
hawk is near"*.

### 6d. A falsifier of mine fired for a bad reason

§4 required mean gate to stay within 0.05 of its hatch value of 0.982; it is 0.805. **The
mechanism is not at fault.** The competition constrains mean *striatal drive* — a uniform
shift cancels in `s − mean(s)`, verified directly. It does not constrain mean *sigmoid of*
striatal drive, and sigmoid is nonlinear, so closing three channels hard while nine sit near
1.0 moves the mean with no uniform shift at all. The test asserted a property the mechanism
never had. Recorded rather than quietly dropped: this is the fifth mis-specified falsifier
in this arc.

## 7. Interpretation

**Selective release does what a free gate could not, and the difference is exactly the one
the basal ganglia exist to make.** E101's free gate closed nearly everything and won by
making the hen too inert to linger in the strike radius — a degenerate solution. Forcing the
gate to *choose* produced a different answer entirely, and a legible one: close the
head-down behaviours, keep the anti-predator ones.

**That the same three channels emerge on disjoint seeds is the strongest part of this
result.** It is not an effect size that might be noise; it is a specific, repeatable,
mechanistically sensible policy, and this project has not previously produced one.

**But the behavioural gain is not established at the bar I set**, and the honest reading is
that the mechanism replicates while the effect does not — which is an unusual and
uncomfortable combination. It suggests the policy is real and its survival benefit is
smaller or noisier than 8 seeds can resolve, which is a statement about statistical power
rather than about the hen.

**And it is still not conditional.** A selective-but-fixed gate is a better failure than a
uniform-and-fixed one, but E100's question stands untouched: the learned pathway does not
vary with situation. The basal ganglia gave selectivity across *actions*; nothing here gave
selectivity across *contexts*.

## 8. Consequence

**Adopted: `bg_gate`, off by default.** It is the first mechanism in this project to produce
a learned policy that can be read and makes sense.

**Not adopted: the predation claim.** It fails its own falsifier and is recorded as
directionally consistent, not established. Anyone wanting it must pre-register pooling or
run more seeds.

**The tree moves on the mechanism, not the effect.** H2b — "the rule cannot acquire
behaviours outside the innate repertoire" — needs qualifying: the rule can now acquire a
*selective suppression policy* over innate behaviours, which is not a new behaviour but is
more than a rescaling.

**Next, and it is E100's question in its third guise:** the gate is selective across actions
and fixed across contexts. Making it conditional is the remaining gap, and nothing in the
last three experiments has touched why the learned pathway stops varying with situation.
That is where the next work belongs — not in another architectural addition.
