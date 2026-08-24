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

### 5a. Implementation note — written after §5, before any measurement

The rule described above is **not what shipped**, and the reason is worth recording
because a guard test caught it rather than a result.

§5 says "subtract from each output row the component it shares with the others". I built
exactly that, and the guard test measured the shared component getting **4.7× worse**.
The cause is structural and I had missed it: the presynaptic factor is *one vector shared
by every output row*, so `dw_out = dz_motor ⊗ dz_slow` is **rank one by construction**.
Every channel already moves along the same pallial direction on every consolidation.
Projecting each row off the others cannot separate rows that are all parallel — it
over-subtracts, because the directions being projected out are themselves nearly
collinear.

This is a sharper statement of the diagnosis than §2 had. The collapse is not something
the rule *drifts into*; each individual update is rank one, and the only thing that ever
made `W_out` full-rank was the *variety* of `dz_slow` directions across consolidations.
E103 measured that variety being destroyed at the relay, one synapse in, at hatch.

What shipped is **Sanger's generalized Hebbian algorithm**: channel *m* learns on what
is left of the presynaptic vector after channels 0…*m* have accounted for it.

```python
presyn = dz_slow[:, None, -n_motor:]
if pc.readout_decorrelate > 0.0:
    recon = jnp.cumsum(dz_motor[:, :, None] * p.W_out, axis=1)
    presyn = presyn - pc.readout_decorrelate * recon
dw_out = eta_out * m_out[:, None, None] * dz_motor[:, :, None] * presyn
```

The first channel is unconstrained, the second sees only the residual, and so on; its
fixed point has orthogonal rows. Measured on one consolidation with a grown readout, row
alignment goes **0.9997 → 0.911**, and the update magnitude *rises* rather than falls, so
the degeneracy falsifier is not being satisfied by silencing anything.

**One honest departure from the textbook rule.** GHA assumes the output is the readout's
own projection `W_out · x`. Here `dz_motor` traces the motor output, which includes the
reflex arc, so the deflation subtracts a reconstruction the readout is only partly
responsible for. It is the quantity this rule already uses as its postsynaptic factor,
and inventing a second one would change two things at once. If the arm produces
something, this is the first thing to vary.

**Nothing in §§1–4 changes.** The predictions and all four falsifiers were written
against "a decorrelating rule" and stand as written.

### Cost

~30 minutes.

---

## 6. Result

### 6a. The 2×2 — the primary falsifier fires

4 seeds, 30 min rearing, `hebbian_readout`, `readout_scaling_strength=0.3`.
`scratchpad/e105_decorrelate.py`.

| rule | adapt | top-1 | **eff rank** | **cort stab** | \|cort\| | stub stab | stub DC% | obs stab |
|---|---|---|---|---|---|---|---|---|
| current | off | 78.9% | 1.94 | **0.9587** | 1.61 | 0.9708 | 97.3% | 0.6573 |
| current | on | 67.6% | 3.37 | **0.9893** | 0.46 | 0.8729 | 86.7% | 0.6827 |
| decorrelating | off | 74.0% | 2.52 | **0.9635** | 1.38 | 0.9713 | 97.3% | 0.6744 |
| decorrelating | on | 63.9% | 3.60 | **0.9822** | 0.47 | 0.8769 | 87.0% | 0.6869 |

Untrained reference on this seed block: top-1 68.4%, effective rank 3.53.

**The primary falsifier fires.** Cortical direction stability stays at or above 0.90 in
every arm — 0.9587, 0.9893, 0.9635, 0.9822 — against a requirement of below 0.90.

The rank falsifier clears, barely: decorrelation moved effective rank 1.94 → 2.52, just
past the 2.5 bar. The **degeneracy falsifier fires for both adaptation arms**: mean
|cortical| falls from 1.61 to 0.46, a 71% drop, against a 50% bar.

Prediction 1 (decorrelation raises rank *and* drops stability below 0.90) is half right
and the half that mattered is wrong. **Prediction 2 is wrong**: temporal adaptation *did*
move the readout — it nearly restored the untrained effective rank, 3.37 against 3.53 —
and made direction stability **worse**, 0.9587 → 0.9893, while cutting the cortical
pathway's magnitude by 71%. Prediction 3's interaction did not appear: the best rank
(3.60) came with a stability of 0.9822.

**Rank and direction stability came apart.** That is not a small surprise; the whole
experiment was built on rank being the mechanism behind the fixed direction.

### 6b. Where the bottleneck actually is

`W_out` reads the **motor stub**, whose own direction stability E103 measured at 0.9930
untrained and 0.9925 reared — never moved by learning, never touched by E104's sensory
fix, and never touched by anything in this experiment. If the *input* has a fixed
direction, `cortical = W_out @ stub` has a fixed direction for **any** `W_out`, at any
rank. `scratchpad/e105b_where_the_bottleneck_is.py`, same 4 seeds:

| arm | stub stab | cortical |
|---|---|---|
| current | 0.9925 | 0.9587 |
| current + adapt | 0.9934 | 0.9893 |
| decorrelating | 0.9925 | 0.9635 |
| decorr + adapt | 0.9934 | 0.9822 |

**Nothing tested moved the readout's input.** And note the adaptation arms make it
*slightly worse* — which is why their cortical stability is *higher*, exactly inverting
the intent.

E105b also carried a positive control, and **the control did not work.** It planted a
"varied" input by giving the stub's deviations random directions at the same energy, and
the planted input's stability came out at 0.9882 against a real 0.9925 — it barely varied
either, so it could not have detected a working readout. Its cortical numbers matched the
real ones to four decimal places, which looked like confirmation and was an artefact.
Recorded rather than quietly replaced: this is the failure this project's own rule names,
a control argued to work rather than measured working.

### 6c. The corrected control — the readout was never the problem

`scratchpad/e105c_input_gain_sweep.py`. Scale the stub's own deviations by a gain, hold
the mean, and read off what the reared `W_out` does with it.

**The motor stub's per-step deviation is 7.18% of its own mean. Its DC share is 99.98%.**

| dev gain | input stab | **cortical stab** | | dev gain | input stab | **cortical stab** |
|---|---|---|---|---|---|---|
| 1× (real) | 0.9925 | 0.9587 | | 10× | 0.7474 | **0.7354** |
| 2× | 0.9887 | 0.9561 | | 30× | 0.4421 | **0.4381** |
| 5× | 0.9044 | 0.8807 | | 100× | 0.2561 | **0.2596** |

**The readout is a faithful map.** Cortical stability tracks its input's stability at
every point on the sweep, to within 0.01. The reared `W_out` at effective rank **1.94** —
the most collapsed matrix in the whole experiment — produces a direction stability of
0.4381 when handed an input that varies. It could always express a state-dependent
output. It was never given a state-dependent input.

(One correction inside this measurement: the first run reported a "DC share" of 400%,
which is impossible. The mean was normed over hens *and* units while the deviations were
normed over units alone, inflating the ratio by √16. Corrected to 99.98%, and the
deviation fraction from 1.78% to 7.18%. The stability sweep was unaffected — it norms
each vector correctly — and is unchanged.)

## 7. Interpretation

**Six interventions have failed to move E100's collapse, and this experiment says why:
all six were applied somewhere other than where the problem is.**

Signed perception (E101-A), a free descending gate (E101-B), a competitive basal-ganglia
gate (E102), lateral inhibition at the sensory relay (E104), a decorrelating learning
rule and temporal adaptation (E105) — every one of them acted on the readout, on the
pathway beside it, or on the *sensory* end of the brain. The thing `W_out` reads is the
**motor stub**, and its representation is **99.98% a constant vector**, with 7% wiggle
around it, **at hatch, before any learning**.

The full chain, from E103 and this experiment together:

| stage | direction stability |
|---|---|
| observation | 0.6375 |
| sensory stub | 0.9707 |
| pallium | 0.9934 |
| **motor stub** (what `W_out` reads) | **0.9930** |

The observation genuinely varies. One strictly-positive synapse takes it to 0.97
(E103's finding). Two more stages of recurrent, overwhelmingly excitatory, strictly
positive rate dynamics take it to 0.993 and hold it there. **Every internal
representation in this brain is a near-constant vector**, and the reason is structural
and uniform: rates are sigmoids, so they cannot go negative; the recurrent weights are
excitatory-dominant; and each stage therefore adds common-mode drive rather than
cancelling it. E070/E071 found this in `z_lag`, E103 found it at the sensory relay, and
it is the same defect at every stage in between.

This reframes a great deal:

- **E100's "training makes the readout converge on a fixed direction" was reading a
  symptom, not a cause.** Its input was 99.98% fixed the whole time. Training changes the
  cortical *magnitude*, and the direction was never available to change.
- **E104 is now fully explained.** Its lateral inhibition genuinely fixed the sensory
  stub (97.3% → 86.4% DC) and cortical stability went *up*. Because the readout does not
  read the sensory stub. Two recurrent stages downstream re-imposed the common mode, and
  the adaptation arms here show the same thing more sharply: the motor stub's deviation
  fraction *shrank* (7.18% → 3.89%) when the sensory input improved.
- **E102's legible policy is exactly as far as this architecture can go.** "Always
  suppress turning and pecking" is a fixed vector. "Suppress them when a hawk is near" is
  a direction that varies with state, and no readout of a 99.98%-constant input can
  produce one. Its state-dependence measurement of 0.9927 was the same quantity.
- **H2b, H2c, H2f's relay, T2's place null and H3 all reduce to this.** Not to the
  learning rule, not to the reward, not to the credit window, not to additivity — to
  the fact that the pallium's state at time *t* is nearly the same vector as its state at
  time *t'*, whatever is happening in the world.
- **E013–E016 were right a third time, and for a third reason.** "The readout can only
  slide a constant" — not because of rank (the founding red-team's correction stands, and
  E105c demonstrates it directly: rank 1.94 produces stability 0.44 on a varying input),
  and not because of convergence (E100's account), but because the input it slides is a
  constant.

**What is not claimed.** That fixing the representation would produce comprehension, an
audience effect, or anything else in the tree. Only that the current architecture makes
those unreachable through the learned pathway, and that six mechanisms aimed at the
readout could not have worked. This is a diagnosis, not a route.

**The primary falsifier's instruction is the one to follow.** §4 said, before any of this
ran, that if no arm moved the collapse the honest conclusion is that it is a property of
the architecture rather than of any component, to be recorded as a finding rather than
pursued into a sixth mechanism. It was written to end the line, and it should.

## 8. Consequence

**Adopted, both off by default.** `readout_decorrelate=0.0` and
`sensory_adapt_tau_s=None`. Neither is recommended: decorrelation moves rank without
moving behaviour, and adaptation is actively harmful at these settings — it cuts the
cortical pathway's magnitude by 71% and makes its direction *more* fixed. They stay in
the tree because the next attempt needs to be able to hold them fixed while varying
something else, and because taking them out would make this experiment unreproducible.

**`docs/hypothesis.md`.** E100's mechanism is superseded in place: the collapse is not
something training does, it is a fixed input. H2b, H2c, H2f and H3's shared explanation
is updated to name the representation rather than the readout.

**The open question this leaves.** The chain observation 0.64 → sensory 0.97 → pallium
0.993 has two distinct losses in it, and only the first has ever been attacked. E104's
interneuron works at the relay and does not survive two recurrent stages. Whether a brain
whose rate code is strictly positive and whose recurrence is excitatory-dominant can hold
a varying representation at all is the question the whole tree now rests on, and it is an
architectural question, not an experimental one — which is why it goes to a design
decision rather than to E106.

**Not adopted.** Any further readout-side mechanism. Six is enough.

### Follow-ups

1. **The E071 interaction check** (deferred from E104) is now clearly uninterpretable and
   should be closed rather than kept open: `z_lag` centring is a downstream patch for the
   same defect, and there is no point asking whether it is redundant while the source is
   unfixed at two of three stages.
2. **The trained-flock mute** (backlog §5, open since E032) is unaffected by any of this
   and remains the highest-value untouched experiment.
3. **A red-team review is due** — five PRs merged since the last commission, which itself
   died without taking a measurement.
