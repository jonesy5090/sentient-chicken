# E011 — re-tuning the readout for a non-saturated network

> **Pre-registered.** Sections 1–5 written and committed before the sweep was run.

## 1. Parent hypothesis

**H2** — three-factor plasticity produces measurable behavioural improvement.
Currently `UNDER TEST` after [E010](E010-rebaseline-at-corrected-gain.md).

## 2. Question

E010 collapsed H2 (t=3.93 → 0.08) after the gain was corrected, and every hen —
control included — got substantially worse at foraging. The diagnosis was that
`readout_scale` and `eta_out` were tuned by [E002](E002-can-the-pallium-reach-a-muscle.md)
against a **saturated** network, where the pallium's near-constant output made the
cortical readout a harmless fixed bias. In a responsive network the same readout
injects real variability that an untrained bird cannot use.

**What readout does a responsive pallium need?**

## 3. Prediction

**Primary: a smaller initial readout is better, and the fixed control improves too.**
If the diagnosis is right, most of E010's damage is untrained cortical noise rather
than anything about learning. Reducing `readout_scale` should lift *both* conditions,
and the fixed control lifting is the specific signature — a control has no learning to
be helped, so if it improves, the problem was noise.

Quantitatively: at `readout_scale` ≤ 0.01 the fixed control's final hunger returns
toward E004's ~0.33 rather than E010's 0.65.

**Secondary: the learning advantage returns at some smaller readout.** Less confident.
It is possible the advantage was always an artefact of saturation, in which case no
readout setting recovers it.

**These come apart, and the distinction matters.** If the control improves but the
learning advantage does not return, the noise diagnosis is right *and* H2 was
regime-dependent — two separate findings, and the second would be the more important.

## 4. Falsifier

- **For the noise diagnosis**: the fixed control not improving as `readout_scale`
  falls. That would mean E010's degradation came from something other than untrained
  cortical drive, and the diagnosis needs redoing.
- **For H2**: no readout setting restoring a learning advantage. H2 would then move
  toward `REFUTED at this timescale` rather than staying open — it would mean the
  E004 result required saturation, and saturation is a defect.

## 5. Design

- **Swept**: `readout_scale` ∈ {0.05 (current), 0.02, 0.01, 0.005}, crossed with
  condition ∈ {fixed, learning-no-growth}.
- **Why the fixed control is run at every readout scale**: it is the only way to
  separate "less noise" from "more learning". The control's own dependence on
  `readout_scale` is the measurement, not a nuisance.
- **Seeds**: 3, and this is thin on purpose — E009 found separability varies 3.5%–25.5%
  between genomes, so single seeds are meaningless and this sweep is for *locating a
  region*, not for calling an effect. Whatever wins gets confirmed at 12 seeds.
- **Duration**: 10 min of chicken time per cell, against E010's 20. Also a
  region-finding compromise.
- **Primary metric**: final hunger (absolute), plus the learning-minus-fixed
  difference at each scale.
- `eta_out` held at 2e-2 for this pass; tuned only if the readout scale alone does not
  resolve it.

## 6. Result

_Pending._

## 7. Interpretation

_Pending._

## 8. Consequence

_Pending._
