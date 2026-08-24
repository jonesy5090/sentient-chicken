# E105 — stop the readout collapsing: decorrelation, and temporal adaptation

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **H2**, **H2b**, **H2e**. Fifth attempt at the collapse
[E100](E100-does-the-readout-distil-the-reflex-arc.md) found.

---

## 2. Question

Five experiments have circled one fact: the learned readout converges on a **fixed output
direction**, so it can rescale innate tendencies but never say *"do X here, Y there"*.
E101 (signed perception, a free gate), E102 (a competitive gate) and E104 (a better input
representation) all failed to move it — E104's fix made it slightly **worse**.

E103 located a real but separate defect: `W_in` is strictly positive, so the sensory relay
passes 97.8% common component. E104 fixed the spatial half of that and the readout did not
respond. **Two independent problems**, and only one has been addressed.

The remaining candidate is the learning rule itself, and it now has a measurement:

| arm | top-1 singular share | **effective rank** (12×48) |
|---|---|---|
| untrained | 68.9% | **3.45** |
| instrumental | 69.6% | 3.40 |
| **hebbian** | 73.2% | **2.07** |

**Rearing nearly halves `W_out`'s effective rank.** A Hebbian outer-product rule accumulates
toward `E[dz_motor ⊗ dz_slow]`, a cross-covariance, and a covariance dominated by its leading
direction gives `cortical = Σ σᵢ uᵢ (vᵢ·x)`. If the mixture weights barely move, the *output
direction* is fixed and only its *magnitude* varies. That is exactly "direction stability
0.96".

**And it vindicates E013–E016 a second time.** They claimed the readout "could only slide a
constant". The founding red-team overturned the *reasoning* — a rank-one `ΔW = u vᵀ`
contributes `u (v·x)`, which does vary with `x`. It varies in **magnitude**. Direction is
what a policy needs, and the conclusion was right for a reason nobody had measured.

**Does decorrelating the readout stop the collapse — and does the input have to vary for
that to help?**

---

## 3. Prediction

**Factor 1 — a decorrelating rule.** Anti-Hebbian lateral inhibition among the readout's
outputs: subtract from each output channel's update the component it shares with the others,
so two channels cannot converge on the same direction. This is the standard fix for a Hebbian
population collapsing onto its leading component, and the same mechanism family as E102's
competitive gate and E104's sensory pooling.

**Factor 2 — temporal adaptation.** E104 built the spatial half of the sensory fix (pooled
inhibition across units) and not the temporal half (each unit subtracting its own running
mean). E104 §6b measured why that matters: the stub's resting bias is **−2.000 on every
unit** and the rate nonlinearity sits on top, so the rate re-introduces what the current
lost — current DC 75.3% against rate DC 87.6%.

1. **Decorrelation raises effective rank** from 2.07 toward the untrained 3.45, and
   **cortical direction stability falls below 0.90** (from 0.9587).
2. **Temporal adaptation alone does not move the readout.** E104's spatial half did not, and
   nothing suggests the temporal half differs. Predicted explicitly so that a null on this
   arm is a confirmation rather than a surprise.
3. **The interaction is where I am least confident.** Decorrelation may only pay off if the
   input actually varies, in which case adaptation is necessary-but-not-sufficient and only
   the 2×2 shows it. **I hold this at genuinely even odds and it is the reason for the
   design.**
4. **No behavioural claim.** If direction stability falls, whether that reaches behaviour is
   a separate question and a separate experiment.

## 4. Falsifier

**Primary.** Cortical direction stability stays **at or above 0.90** in every arm. Five
interventions would then have failed, the collapse would not be attributable to the rule
either, and the honest conclusion is that it is a property of this architecture as a whole
rather than of any component — which is a real finding and should be recorded as one rather
than pursued into a sixth mechanism.

**Rank falsifier.** Effective rank does not rise above 2.5 under decorrelation. The
mechanism would then not be doing what it is for, and any stability result from it is
uninterpretable.

**Regression falsifier.** Any ethogram assay changes state, or the suite fails. Inertness
asserted bit-identical with both factors off.

**Degeneracy falsifier.** Decorrelation raises rank but the cortical *magnitude* collapses
instead — measured as mean |cortical| falling more than 50% against the control. Forcing
channels apart is not useful if it silences them.

## 5. Design

**`readout_decorrelate: float = 0.0`** in `PlasticConfig`. In `consolidate`, before applying
`dw_out`, subtract from each output row the component it shares with the mean of the others,
scaled by this factor. Off by default.

**`sensory_adapt_tau_s: float | None = None`** in `CoopConfig`. Each pool unit subtracts its
own running mean of the input current, at this time constant. `None` disables it. This is
E104's missing temporal half.

**2×2**, matched seeds: `{rule off, rule on} × {adapt off, adapt on}`. 4 seeds, 30 min
rearing, `hebbian_readout` throughout, `sensory_lateral=1.0` in the adapt arms so the spatial
and temporal halves are tested together as the complete mechanism.

Measured: effective rank and top-1 share of `W_out`; cortical direction stability; mean
|cortical|; stub stability and DC share; the full ethogram at the best arm.

### Cost

~30 minutes.

---

## 6. Result

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*
