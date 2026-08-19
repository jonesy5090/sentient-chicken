# E078 — does E041's density result survive a naturalistic probe?

> **Pre-registered.** Sections 1–5 written and committed before the run.

## 1. Parent hypothesis

**H2d**. After [E077](E077-reread-balanced-ei.md) closed the balanced-E/I branch,
**E041's density finding is the only intervention with a measured positive effect on
H2d's own metric** — and it has never been measured under anything but the sparse
E009-series probe.

## 2. Question

E041 swept `sensory_pallium_density` on a paired 12-genome design and found separability
falling as density falls (t=4.08–5.37 across four reductions) and *rising* monotonically
as it rises, with **no optimum found up to full connectivity (~2× the default)**. It was
recorded as "promising, checked, not adopted".

Its stated mechanism was: at random-sparse density, **too few pallial units get any
connection to the one or two informative channels at all**.

That mechanism is specific to sparse input. The E009-series probe injects a *single*
channel at amplitude 1.0; a naturalistic observation has 14 nonzero channels. If the
mechanism is right, spreading information across many more channels should make random
sparse sampling far less costly — because a unit that misses one informative channel is
likely to catch another.

Does the density effect survive naturalistic input, and at what size?

## 3. Prediction

**The effect should weaken substantially under naturalistic input**, for the reason
above: E041's mechanism depends on informative channels being rare enough to miss, and
naturalistically they are not. Stated numerically before running: the density ratio
(1.00 vs 0.30) should be **materially below E041's ~2×** on the naturalistic probe, while
reproducing near ~2× on the sparse probe.

**Direction should stay positive** in both. Nothing in the mechanism predicts denser
being actively harmful.

If instead the effect is *undiminished* naturalistically, E041's mechanism is wrong
about *why* density helps, even though the effect itself is real — which would matter,
because the mechanism is what makes "no optimum found up to 1.0" sound safe to act on.

## 4. Falsifier

If density's effect is null or reversed on the naturalistic probe, E041's finding does
not transfer to the regime a hen actually inhabits, and H2d has **no** remaining
intervention with a positive effect — which is a materially worse position than the tree
currently records and should be stated plainly.

## 5. Design

Both probes, same genomes, paired — the design E073 used, so probe-to-probe comparison
is itself paired and cannot be attributed to genome sampling. **Run on E076's corrected
baseline** (`place_cells_enabled=False`, `contamination_enabled=False`), which is the
whole reason this is worth redoing.

- **Densities**: 0.15, 0.30 (default), 0.60, 1.00 — brackets the default both ways and
  reaches the full connectivity E041 found best.
- **Metric**: `pallial_sep` verbatim from E041 (`RMS(hawk − call) / mean|rest|` over
  pallial units), so numbers stay comparable to the E009/E017/E023/E034/E035/E041 series.
- **12 genomes**, paired, per E035's finding that unpaired ratio-of-means on this
  quantity produced both a false positive and a false negative.
- **Reported**: separability and mean pallial rate per density per probe, plus the paired
  contrast of each density against the 0.30 default within each probe.

Mean rate is reported because density changes total input drive — if separability tracks
drive rather than density as such, that shows up here rather than being discovered two
experiments later.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e078_density_naturalistic.py
```

## 6. Result

_Not yet run._

## 7. Interpretation

_Pending §6._

## 8. Consequence

_Pending §6._
