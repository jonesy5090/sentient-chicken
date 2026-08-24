# E104 — lateral inhibition at the sensory relay

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **H2** and everything beneath it. Acts on
[E103](E103-where-the-variability-is-lost.md)'s diagnosis.

---

## 2. Question

E103 found that situation-dependence is destroyed at the first synapse, at hatch, and
identified the cause: `W_in` is strictly positive (2630 nonzero entries, **0 negative**,
`rng.gamma(2.0, 0.5)`), so every stub unit computes a positive weighted sum of a positive
observation. The mean direction's share of a typical vector goes from **69.0% in the
observation to 97.8% in the stub** — situation-specific signal falls from 31% to **2.2%** in
one projection, and everything downstream reads that.

Real sensory systems almost universally solve this at the first relay with centre-surround
or divisive normalisation: pooled inhibition subtracts the common component so the
projection passes *contrast* rather than total drive. This model has none.

**Does adding it restore the situation-dependence the learned pathways need?**

**The change is safer than it looks, and that is worth stating before the falsifiers.** The
reflex arc reads `obs` directly (`brain.py:96`), never the stub. Lateral inhibition on
`current = obs @ W_in.T` (`brain.py:67`) therefore touches only the recurrent network and the
learned pathway. **The innate ethogram is structurally out of its path**, which is why this
is a smaller blast radius than E103's caution implied.

---

## 3. Prediction

A pooled inhibitory interneuron over the afferent-receiving units:

```
pooled  = mean(current over pool units)
current = current − sensory_lateral · pooled   (pool members only)
```

At `sensory_lateral = 1.0` the common component is fully removed; the mechanism is the same
family as E102's competitive gate, which worked.

1. **Stub direction stability falls** from 0.9707 toward the observation's 0.6573. I predict
   **below 0.90**.
2. **The mean-direction share falls** from 97.8% toward the observation's 69.0%.
3. **The cortical readout regains state-dependence** — E100's reared 0.9587 falls **below
   0.90**. This is the one that matters; 1 and 2 are mechanism checks.
4. **The ethogram is unaffected at hatch**, because the reflex arc bypasses the stub. Any
   change must come through the cortical pathway, which is near-silent at hatch by design.

I hold 1 and 2 at high confidence — they are close to arithmetic. **I hold 3 at roughly even
odds**, because E103 showed the readout also *amplifies* what little variation exists at
hatch (cortical 0.6193 against a stub at 0.9930), so the readout's collapse under training
may be partly independent of its input.

## 4. Falsifier

**Primary.** Cortical direction stability after rearing stays **at or above 0.90**. The
representation would then not be the binding constraint on the readout's collapse, and
E103's diagnosis — while correct about the representation — would not explain E100.

**Mechanism falsifier.** Stub stability does not fall below 0.90, or the mean-direction share
does not fall below 90%. The lateral term would then not be doing what it is for.

**Regression falsifier.** Any ethogram assay changes state, or the suite fails. This changes
the input to the whole recurrent network, so the bar is the strictest so far: **inertness
asserted bit-identical at `sensory_lateral = 0.0`**, and the full ethogram re-run at the
adopted value.

**Interaction falsifier — recorded because E103 flagged it.** E071 centres `z_lag`
downstream for exactly this DC problem. If the source is fixed, that centring may be
redundant or harmful. Measured here as: does `pred_centred` still help, hurt, or do nothing
once lateral inhibition is on? Not a pass/fail — a measurement that must be taken rather than
assumed, because stacking two fixes for one defect is how E087 wasted an experiment.

## 5. Design

**`sensory_lateral: float = 0.0`** in `CoopConfig`, and `lateral_pool` in `BrainParams` — a
mask over units receiving afferent input, so the pooling matches the biology (an interneuron
pools the relay it sits in) rather than averaging over the whole brain, most of which
receives no afferents at all.

Sweep `sensory_lateral ∈ {0.0, 0.5, 1.0}`, 4 seeds, 30 min rearing, `hebbian_readout`.
Measured: E103's full stage-by-stage stability table, the mean-direction share, the ethogram,
and the `pred_centred` interaction.

### Cost

~20 minutes.

---

## 6. Result

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*
