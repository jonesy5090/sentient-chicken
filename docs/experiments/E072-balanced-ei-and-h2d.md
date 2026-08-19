# E072 — H2d: the sensory→pallium projection has an unbalanced DC, and E027 already fixed this once

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**H2d** — the pallium does not form separable representations of distinct stimuli.
`SUPPORTED as a limitation`, and by [E071](E071-pred-centring.md) now the
evidence-backed critical path: the proximate blocker for T2-revised, H2c and H3
simultaneously.

## 2. Question

Every recurrent projection in this model carries a large positive DC by construction.
`EXCITATORY_FRACTION` is 0.8 and both signs draw magnitudes from the same `|normal|`,
so excitation outweighs inhibition roughly 4:1 in total. Measured on the default
connectome, the **sensory→pallium block nets +0.9339 per pallial unit**.

That is precisely the quantity E017 and E034 localised H2d's 14.5–17× loss to — *"a
clean difference lands as a small perturbation on a large common-mode drive"* — and
precisely the defect **E027 already fixed for `W_out`**, whose own comment reads: *"The
stub is 80% excitatory, so signing a rectified draw leaves every motor channel with a
large positive DC bias where the old zero-mean draw was balanced."* `W_out` now nets
−0.000000. `W` never received the same treatment.

Does balancing E and I in `W`, exactly as `w_out` already is, improve pallial
separability?

## 3. Prediction

**Balanced > baseline, substantially.** Removing a common-mode term that dominates the
projection is the direct mechanism E017/E034 measured. Magnitude uncertain — stated
before running as "materially more than 1.5×", with low confidence in the precise
figure.

**Balanced > gain-matched control, and this is the real prediction.** Reducing gain
scales common-mode *and* differential down together, so it should not improve the
*ratio* much (consistent with E009's finding that relative separability moves with gain
while absolute separability barely does). Balancing removes the common mode while
preserving the differential. If balancing only matches the gain-matched control, the
effect is "less drive" rather than "less common-mode", and the mechanism claim is
wrong.

## 4. Falsifier

If balanced separability is indistinguishable from the gain-matched control on a paired
12-genome test, this is not a fix for H2d — it is a rescaling, and the DC framing is
wrong. Given E035's history on exactly this kind of structural claim, the paired test
against a matched control *is* the experiment; an unpaired improvement over baseline
alone would not be evidence.

## 5. Design

**Metric**: E041's `pallial_sep` verbatim — settle on hand-injected `o_hawk`, `o_call`,
`o_rest`; report `RMS(hawk − call) / mean|rest|` over pallial units. Same
settle-and-separate probe as E009/E017/E023/E034/E035/E041, so numbers are directly
comparable to that series.

**Paired, 12 genomes**, per E035's finding that unpaired ratio-of-means on 6 genomes
produced a false positive *and*, on one pass, a false negative for a structural question
of this shape. Genome-to-genome spread on this quantity is ~6×.

**Confound already removed in the implementation**: balancing scales inhibition up,
which would raise `sum|W|` ~60% and confound "balanced E/I" with "more total weight" —
the identical confound that invalidated both modality-segregation figures. The flag
renormalises to hold `sum|W|` constant (verified: 3153 both ways), so it changes the
E/I *ratio* and nothing else.

**Conditions**, all paired per genome:

| condition | purpose |
|---|---|
| baseline | current default, DC +0.93/unit |
| `balanced_ei=True` | DC ≈ 0, magnitude matched |
| gain-matched control | unbalanced, `gain` reduced to match balanced's mean pallial rate |

The gain-matched control is chosen *after* measuring balanced's mean rate, by search —
which is legitimate because it is a control being matched to a condition, not a
treatment being tuned to an outcome.

**Secondary, reported but not the primary claim**: place-to-place pallial correlation
(E070's 0.94–0.96), the target E071 set for T2-revised. Separability and place
correlation are related but not identical, and this experiment is powered for the first.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e072_balanced_ei.py
```

## 6. Result

**Primary — separability, paired 12 genomes, threshold t=2.201:**

| condition | separability | mean pallial rate |
|---|---|---|
| baseline | 0.0961 | 0.2724 |
| `balanced_ei` | 0.0867 | 0.1191 |
| gain-matched (gain=0.05) | 0.0033 | 0.1220 |

| contrast | Δ ± SE | t | |
|---|---|---|---|
| **balanced vs baseline** | −0.0094 ± 0.0127 | **0.74** | **not significant (0.90×)** |
| gain-matched vs baseline | −0.0928 ± 0.0124 | 7.46 | significant *decrease* (0.03×) |
| balanced vs gain-matched | +0.0834 ± 0.0046 | 18.29 | significant (26.28×) |

**That 26.28× must not be quoted as a result.** The gain-matched control turned out
degenerate: matching balanced's mean pallial rate required `gain=0.05` against a default
of 0.95, which collapses separability to 0.0033 — a dead network. "Balanced beats a dead
network by 26×" is true and worthless. The control failed as a control, and since the
primary comparison showed no effect there is nothing for it to attribute anyway.

**Secondary — place-to-place pallial correlation** (6 genomes, all pairs among 5 grid
cells, naturalistic full observations via `sensing.observe`):

| condition | correlation |
|---|---|
| baseline | 0.9807 |
| `balanced_ei` | **0.7520** |

## 7. Interpretation

**The primary prediction is falsified. Balancing E/I does not improve H2d's
separability** — 0.90×, t=0.74, on a properly paired 12-genome test. The DC is
genuinely there (+0.9339 per pallial unit, verified) and removing it genuinely does not
help this metric. Removing the common mode dropped mean pallial rate from 0.27 to 0.12
and took the differential signal down with it, proportionally.

**But the two probes disagree, and that is the finding worth keeping.** The same
intervention leaves the classic separability metric untouched while taking place-to-place
correlation from 0.98 to 0.75. They differ in what they feed the network:

- The settle-and-separate probe used since E009 injects a **single channel at amplitude
  1.0 into an otherwise-zero observation**. There is almost no common mode to remove, so
  balancing has almost nothing to do.
- The place measurement uses a **full naturalistic observation** where dozens of channels
  are active and the place block is graded. Here the common mode is large, and removing
  it helps substantially.

If that reading holds, H2d's entire measurement history — E009, E017, E023, E034, E035,
E041 — rests on a probe whose input statistics are unrepresentative of live operation,
and which is specifically blind to common-mode effects. **E019 raised a version of this
concern already** ("separability had been measured on hand-injected observations
describing a situation the hen never experienced"); it was answered by showing the
*contrast occurs* in live operation, not by changing the probe.

**Confidence is asymmetric here and should stay that way.** The null is a properly
paired 12-genome test and is solid. The place result is a 6-genome descriptive
comparison with no paired statistic — 0.98 vs 0.75 is a large gap, but E035 is the
standing warning about exactly this kind of number on exactly this kind of structural
question. It is a lead, not a result.

## 8. Consequence

**H2d is not fixed, and `balanced_ei` is not adopted as a default.** It stays in the
codebase, off, as the tested implementation of a hypothesis that did not pay off on the
metric it was aimed at.

**The next experiment is a probe question, not a mechanism question**, and it is cheap:
re-measure H2d's separability under naturalistic observations rather than hand-injected
sparse ones, paired across genomes, with and without `balanced_ei`. That either
reproduces the place result — in which case a decade of H2d numbers need re-reading and
`balanced_ei` becomes a live candidate again — or it does not, in which case the place
finding is about place coding specifically and H2d's history stands.

Do not skip to re-running T2-revised on `balanced_ei` because 0.98 → 0.75 looks
encouraging. E071's 0.180 looked encouraging too and meant 0.565 once `relu` was
applied. The place number needs a paired test and a naturalistic-probe replication
before anything is built on it.

**Standing candidate unchanged**: E041's density finding (~2× at full connectivity,
monotonic, no optimum found) remains the only intervention with a measured positive
effect on H2d's own metric, and is still not adopted.
