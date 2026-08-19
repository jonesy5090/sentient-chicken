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

_Not yet run._

## 7. Interpretation

_Pending §6._

## 8. Consequence

_Pending §6._
