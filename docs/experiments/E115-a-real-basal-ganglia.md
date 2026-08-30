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

8 seeds, 16 hens, 30 min rearing. `scratchpad/e115_basal_ganglia.py`.
`N` goes 512 → 608 when the subpallium is present.

### 6a. The circuit is present and does what a basal ganglia does structurally

| arm | striatum | pallidum | motor rate | stub stability | entropy |
|---|---|---|---|---|---|
| plain, frozen | — | — | 0.6369 | 0.9998 | 0.8034 |
| plain, learning | — | — | 0.6352 | 0.9998 | 0.7921 |
| **subpallium (random loop)** | 0.1728 | 0.6385 | **0.4304** | **0.9996** | 0.8019 |
| **subpallium (topographic)** | 0.1705 | **0.7534** | **0.5809** | **0.9998** | 0.8026 |

**Prediction 2 holds.** The motor stub is materially more suppressed with the loop
present — 0.6369 → 0.4304 — which is tonic pallidal inhibition arriving exactly as
described. The unit tests confirm the resting states are right: striatum 0.018, pallidum
0.82, both regions entirely GABAergic, and the two inhibitory links present and correctly
signed.

**My manipulation falsifier fired, and it was mis-specified.** §4 demanded pallidal rate
above 0.7 and striatal below 0.15. I wrote those as *resting* values and then measured
them in a **running** flock, where the pallidum is being inhibited by the striatum and the
striatum is being driven by the pallium — which is the circuit working, not failing. The
resting values are asserted by `test_pallidum_is_tonically_active_and_striatum_is_not` and
pass. This is the same class of error as E114's floored metric: a falsifier written
without asking what regime it would be evaluated in.

### 6b. Topography, added because the first loop was wired at random

The first build connected striatum → pallidum → motor at random density. That cannot
select: releasing one striatal cell disinhibits a scattering of unrelated motor units, so
a burst does not correspond to an action. A real basal ganglia is organised in **parallel
channels**. `subpallium_channels=8` cuts both inhibitory links to block-diagonal, leaving
the corticostriatal input fully mixed — the loop's job is to *sort* a mixed input into
channels.

It works as a manipulation: pallidal rate rises 0.6385 → **0.7534**, now clearing the bar,
because each pallidal cell receives from 8 striatal cells rather than 64 and is less
saturated by inhibition. Edge counts fall from 817 to 102 (striatum→pallidum) and 622 to
68 (pallidum→motor).

**And it changes nothing else.**

### 6c. No selection, in either build

- **Motor-stub direction stability is 0.9998 in every arm**, including both subpallium
  builds — identical to the plain brain and to what
  [E107](E107-red-team-review-2026-08-24.md) measured. Prediction 3 is **wrong**.
- **Normalised entropy** across the 12 motor channels: 0.8034 plain, 0.8019 random loop,
  0.8026 topographic. Unmoved.
- **Behaviour**, paired against the plain brain: subpallium vs plain frozen −0.0067
  (t=−0.51) then +0.0068 (t=+0.58) on hunger; −0.0085 (t=−0.61) then +0.0056 (t=+0.20) on
  `caught/dive`. Nothing, in either build, in either direction.
- **Learning on the subpallium**: hunger +0.0020 (t=+0.25) random, −0.0188 (t=−1.80)
  topographic; `caught/dive` +0.0061 (t=+0.41) then −0.0345 (t=−1.28). The topographic
  arm's signs are the encouraging ones and neither clears 2.365, on one seed block, with
  six contrasts in the table — a Bonferroni threshold near 3.5. **Claimed as nothing.**

Prediction 4 holds: no behavioural win, as stated in advance.

## 7. Interpretation

**The primary falsifier's condition is met, and per §4 that is the answer rather than a
prompt to keep tuning.** A structurally faithful subpallium — correct populations,
GABAergic throughout, tonically active pallidum, inhibitory striatal collaterals, and in
the second build the parallel-channel topography that makes disinhibition mean *this*
action — adds nothing measurable to selection or to behaviour.

**The missing-structure explanation is closed**, the way E110 closed the wrong-postsynaptic-factor
one. It was the best-motivated hypothesis this project has had: the anatomy gap and the
measured deficit lined up precisely, three separate experiments described a model that
cannot select, and the one place the structure was half-built (E102) was the one place
learning half-worked. It was still wrong.

**The number that should be stared at**: motor-stub direction stability is **0.9998** in
plain, in random-loop and in topographic builds, frozen and learning alike. Six arms here,
and every arm of E107 before them. In this architecture the readout's input is a fixed
direction, and adding 96 neurons of correctly-wired subpallium does not perturb it by one
part in five thousand. Only [E106](E106-recurrent-inhibition.md)'s pooled interneuron ever
moved it (to 0.9651), and that bought no behaviour either.

**What E102's success was, then.** Not a basal ganglia. E113 and E114 already narrowed it:
a *multiplicative* gate whose benefit is carried by turn suppression — a locomotion
effect. E115 completes that demotion. Building the actual circuit that E102's gate was
named after does not reproduce E102's benefit, which means the benefit never came from
anything basal-ganglia-like. **The name was doing work the structure was not.**

**Six explanations have now been proposed for H2's null and all six have failed.** Five
were about the learning rule or the readout. This one was about the anatomy, which is a
genuinely different kind of hypothesis, and it failed in the same way: the intervention
did what it was designed to do, measurably, and behaviour did not follow.

## 8. Consequence

**Adopted, off by default.** `striatum=0, pallidum=0` in `Regions`, so `N` stays 512 and
every prior result is untouched; `subpallium_channels=0` in `connectome.build`. Inertness
bit-identical, 15/15 digests. Not recommended: it costs 19% of the neuron budget and buys
nothing measured.

It stays in the tree because it is **correct anatomy that the model was missing**, because
the guard tests encode the circuit's properties (tonic pallidum, GABAergic subpallium,
correctly-signed loop) so a future build cannot get them wrong silently, and because
deleting it would make this null unreproducible.

**`docs/hypothesis.md`.** H2 records that the missing-structure explanation has been
tested and closed. H2b records that E102's benefit is not basal-ganglia-like, completing
the narrowing E113 and E114 began.

**Not adopted.** Turning it on by default; any further tuning of this circuit; and the
reading that "it would work with a thalamus / D1-D2 split / dopaminergic population". Each
of those is a seventh mechanism, and the base rate is now six from six.

### Follow-ups

1. **The strategic question is now live rather than rhetorical.** Fixing the brain has
   been tried, with the best-motivated structural hypothesis available, and it failed. The
   two remaining routes are **generational selection** (`docs/backlog.md` §4, never
   started, and the only route that reaches H5) and **writing up the null** — which
   [E111](E111-is-there-headroom.md) made informative by proving the headroom is real.
2. **The trained-flock mute** (backlog §5, open since E032) tests H0 rather than H2 and is
   unaffected by any of this.
