# E115 — build the subpallium: a striatum and a pallidum, not a gate

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **H2**, and **H2b** directly. The first structural addition to the
brain since the model was built, rather than another mechanism attached to the readout.

---

## 2. Question

Documenting the model's anatomy against a real chicken's (README, "The chicken brain, and
how much of it we have actually built") produced one gap that lines up exactly with the
measured deficit: **the subpallium is absent entirely.** No striatum, no pallidum, no
dopaminergic midbrain.

The basal ganglia are the structure a vertebrate uses to **select among competing
actions**. And what this project has measured, repeatedly, is a model that cannot select:

- [E109](E109-what-the-rule-writes.md): the readout's update direction is the reflex arc's
  own, at cosine 0.98. Learning can amplify what she already does and cannot redirect it.
- [E112](E112-repair-the-peck-reflex.md): the residual gap to a hand-written forager is
  **staying put** — persisting in a chosen action against a drive that keeps her moving.
- [E114](E114-does-the-gate-work-through-vigilance.md): the learned gate underperforms a
  hand-lesioned version of itself, because the rule cannot stop at the useful channels.

Three descriptions of a missing action-selection circuit.

**And there is a positive clue.** E102's "basal-ganglia gate" is the *only* thing in this
project that ever produced a structured, behaviourally effective learned policy — E113
showed its channel assignment beats a scrambled one at z=−4.40 and a flat one at z=−7.00.
But it is a **weight matrix hung off the motor stub**: a striatal function with no
striatal population, no pallidum, no tonic inhibition, and no loop. The one place the
right structure was half-built is the one place learning half-worked.

**So: does a real subpallium — populations, Dale-correct signs, tonic pallidal inhibition,
selective release — let this model select actions?**

### What a real basal ganglia does, and what makes it different from a gate

Pallium projects **excitatory** onto striatum. Striatum is **GABAergic** and normally
near-silent, firing in bursts. It inhibits the **pallidum**, which is **GABAergic and
tonically active**, holding its targets under constant suppression. So a striatal burst
*removes* inhibition from one target while the rest stay suppressed: **selection by
disinhibition**, not by excitation. Striatal collaterals are themselves inhibitory, giving
competition among candidate actions for free.

None of that is expressible as a multiplicative gate on an output. It needs populations
with the right signs and the right resting states.

## 3. Prediction

1. **The circuit behaves like a basal ganglia at hatch.** Pallidal mean rate above 0.7
   (tonically active), striatal mean rate below 0.15 (near-silent). This is a manipulation
   check and it must pass before anything else is read.
2. **The motor stub is more suppressed at hatch** with the loop present, because tonic
   pallidal inhibition is arriving and nothing has yet learned to release it.
3. **Action selection sharpens.** Motor output across the 12 channels becomes less uniform
   — I expect normalised entropy to fall — and the motor stub's per-hen direction
   stability to fall below the 0.9998 [E107](E107-red-team-review-2026-08-24.md) measured,
   because the pallidum's inhibition varies with state where nothing else in this brain
   does.
4. **I do not predict a behavioural win at 30 minutes.** One hundred and fourteen
   experiments say otherwise and this one has no special claim. Stated in advance so a
   null is the expected result and a positive one has to survive replication before I
   believe it.

## 4. Falsifier

**Manipulation falsifier — reported before anything else.** If the pallidum is not
tonically active or the striatum is not near-silent, the circuit is not a basal ganglia
whatever it is labelled, and no downstream number counts.

**Degeneracy falsifier.** If the loop simply silences the motor stub — mean rate below
0.10, or mean motor output below half the baseline's — it is suppressing rather than
selecting, and that is a failure however the representation metrics move.

**Primary.** If the manipulation check passes and **neither** the selection metrics
(entropy, per-hen direction stability) **nor** behaviour (hunger, `caught/dive`) moves,
then a structurally faithful subpallium adds nothing in this model. That closes the
"missing structure" explanation the way E110 closed the "wrong postsynaptic factor" one,
and it should be recorded as the answer rather than tuned until something moves.

**Regression falsifier.** Bit-identical inertness at the default (both regions size 0);
no ethogram assay changes state; suite passes.

**Replication rule.** E021. Nothing moves the tree on one seed block.

## 5. Design

**Two new regions in `hen/regions.py`, both size 0 by default**, so `N` stays 512 and
every existing result is untouched. At the tested size: **striatum 64, pallidum 32** —
striatum much the larger, as in a real brain.

**Dale's law, deliberately region-specific.** Both populations are **100% inhibitory**,
because both are GABAergic. This needs a per-region override of `EXCITATORY_FRACTION`,
and it must be flagged loudly: E022 found a *bug* in which regions came out 100%
inhibitory by accident, from E/I being assigned by flat array index. **This is the same
shape as that bug and the opposite in kind** — there it was an artefact that made the
pallium a 256-unit pool with no inhibition; here it is the anatomy.

**Resting states, per region.** `b` is currently a uniform −2.0 everywhere. A per-region
bias vector is introduced with all six existing regions at −2.0, so the default is
unchanged, and:

- **striatum −4.0** → `sigmoid(−4.0) = 0.018` at rest. Near-silent, fires in bursts.
- **pallidum +1.5** → `sigmoid(1.5) = 0.82` at rest. Tonically active.

**The loop**, added to `REGION_CONNECTIVITY`:

| projection | density | why |
|---|---|---|
| pallium → striatum | 0.30 | corticostriatal, the plastic input |
| sensory → striatum | 0.15 | sensory drive to striatum is real |
| striatum → striatum | 0.20 | inhibitory collaterals — competition, for free |
| striatum → pallidum | 0.40 | the direct pathway |
| pallidum → pallidum | 0.10 | pallidal collaterals |
| **pallidum → motor** | 0.40 | the tonic inhibition that gets released |

Net: pallium excites striatum, striatum inhibits pallidum, pallidum's tonic inhibition of
the motor stub is lifted. Two inhibitory steps, so selection is **disinhibition**.

**Plasticity is not new.** Corticostriatal synapses are entries of `W`, so the existing
three-factor rule already updates them under `m` — which is the biologically correct site
for dopamine-gated learning. No new rule, no new flag on the learning side. That is the
point: the claim is that the *structure* was missing, not the rule.

**Departures, stated rather than discovered later.**

- **No thalamus.** In a real loop the pallidum inhibits thalamus, which excites cortex.
  Here it inhibits the motor stub directly, which is the thalamocortical target the readout
  already reads from. The loop is shortened by one relay.
- **No D1/D2 split.** One pathway, not a direct and an indirect one with opposite dopamine
  sensitivity.
- **No subthalamic nucleus**, so no hyperdirect "stop" route.
- **`m` is still a scalar**, not a dopaminergic population. That gap is real and is
  explicitly *not* addressed here; adding two things at once is E089's lesson.

**Arms**, matched seeds, 8 seeds, 16 hens, 30 min rearing: `{no subpallium, subpallium}` ×
`{frozen readout, learning}`, with the frozen arm the control E110 established as the one
that matters.

**Measured**: striatal and pallidal mean rates (the manipulation check); motor-stub mean
rate and per-hen direction stability via `run/metrics`; normalised entropy of the motor
output across channels; hunger and `caught/dive`; the full ethogram.

### Cost

~30 minutes. `N` rises 512 → 608 when enabled, about 19%, which the envelope absorbs.

---

## 6. Result

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*
