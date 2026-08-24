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

### 6a. The mechanism works, and the primary falsifier fires the wrong way

4 seeds, 30 min rearing. Inertness clean at 0.0: **89 passed, 1 xfailed**, and the row
reproduces E103 exactly.

| `sensory_lateral` | stub stability | stub DC share | **cortical stability** | obs | reflex |
|---|---|---|---|---|---|
| 0.0 | 0.9708 | 97.3% | **0.9587** | 0.6573 | 0.8832 |
| 0.5 | 0.9300 | 93.2% | 0.9594 | 0.6807 | 0.8855 |
| **1.0** | **0.8632** | **86.4%** | **0.9846** | 0.6861 | 0.8952 |

**Predictions 1 and 2 held.** The relay now passes contrast: stub stability falls 0.971 →
0.863 and the DC share 97.3% → 86.4%.

**The primary falsifier FIRES, and not by a near miss.** I required cortical stability below
0.90; it *rose*, 0.9587 → 0.9846. Feeding the readout a more varied representation made its
output **more** fixed, not less.

### 6b. Why the fix is only partial — I built half the mechanism

| | input current DC | → stub **rate** DC |
|---|---|---|
| lateral 0.0 | 93.7% | 97.8% |
| lateral 1.0 | **75.3%** | **87.6%** |

Two losses, and both are mine.

**The rate re-introduces what the current removed.** The resting bias over the stub is
**exactly −2.000 on every unit, sd 0.000** — a uniform offset — and the rate nonlinearity
sits on top of it. So even where the input current's common component is subtracted, the
*rate* downstream reads keeps one.

**And the subtraction is the wrong kind.** My term removes the **instantaneous across-unit**
mean: at each step it subtracts one scalar shared by all pool units. It does not remove each
unit's **own systematic offset** — a unit with heavier afferent weights still sits higher
than its neighbours at every timestep, and that survives untouched. That is why the current's
DC only falls to 75.3% rather than toward zero.

Real sensory systems do **both**: lateral inhibition across units *and* adaptation within
each unit over time. I built the spatial half and not the temporal one.

## 7. Interpretation

**E103's diagnosis was right and is not what was holding the readout back.** The
representation genuinely was DC-dominated, lateral inhibition genuinely improves it, and the
cortical pathway's collapse is **independent of both**. Four experiments have now aimed at
that collapse — signed perception, a free gate, a competitive gate, and now the input
representation — and none has moved it.

**So there are two separate problems and this experiment separates them**, which is worth
more than the fix it failed to be:

1. **The representation is impoverished** (E103). Partially addressed here; the remaining
   path is temporal adaptation, and it is specified rather than speculative.
2. **The readout collapses to a fixed direction regardless of what it is fed** (E100).
   Untouched by any intervention so far, and now demonstrably not a consequence of (1).

**That the collapse got slightly worse is a clue, not noise.** A readout receiving more
varied input converged harder on a single direction. That is the signature of something in
the *learning rule* selecting a fixed solution — the outer product averaging over situations
— rather than of a readout faithfully reflecting a fixed input. E103 already hinted at this:
at hatch the readout *amplified* the stub's small variation (cortical 0.6193 against a stub
at 0.9930), so the readout was never a passive mirror of its input.

**I recorded prediction 3 at even odds for exactly this reason and it still surprised me.**
The direction of the effect is the informative part.

## 8. Consequence

**Adopted: `sensory_lateral`, default 0.0.** It does what it is for, the inertness gate is
clean, and the mechanism is correct as far as it goes. It is not adopted as a *default*,
because on its own it buys a better representation and no behavioural change, and changing
the input to every recorded experiment for that is not a trade worth making yet.

**The interaction falsifier is not evaluated.** E104 was to measure whether E071's
downstream `z_lag` centring becomes redundant once the source is fixed. Since the source is
only half-fixed and the readout did not respond, that measurement would not have been
interpretable, and it is deferred rather than reported. Recorded so it is not lost.

**Next, and it is the same question for the fifth time**: why does the learned readout
converge on one direction? E100 asked it, E101 and E102 failed to move it with architecture,
E103 located a different defect, and E104 has now shown that fixing that defect does not
help. The remaining candidate is the one nothing has tested — **the learning rule itself
averaging over situations**, which is where E103's three-way split pointed and where the
next work belongs. The other half of this fix, per-unit temporal adaptation, is cheap and
worth doing alongside it, but on this evidence it should not be expected to change behaviour
on its own.
