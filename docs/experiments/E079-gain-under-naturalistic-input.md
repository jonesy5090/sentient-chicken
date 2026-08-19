# E079 — the recurrent gain default was set on the sparse probe too

> **Pre-registered.** Sections 1–5 written and committed before the run.

## 1. Parent hypothesis

**H2d**. [E078](E078-density-under-naturalistic-input.md) left saturation as the only
live lead, and the falsifier fired: no intervention has a positive effect on H2d under
naturalistic input.

## 2. Question

`hen/connectome.py`'s `gain` default is **0.95**, set by [E023](E023-ei-fix-and-rebaseline.md)'s
sweep:

| gain | mean pallial rate | separability |
|---|---|---|
| 0.70 | 0.189 | 4.5% |
| **0.95 (default)** | **0.276** | **7.4%** |
| 1.00 | 0.320 | 9.4% (peak) |

**That sweep ran on the sparse E009-series probe**, where the network sits at ~0.27 and
is not saturated at all. Under naturalistic input the same connectome sits at 0.4602,
and live operation at **0.6907** (E073/E076). E023 chose a gain by optimising in a
regime the hen is never in — the same error E078 just found in E041's density
recommendation, and E073 found in E023's own saturation claim.

Where does separability peak under naturalistic input, and is 0.95 anywhere near it?

## 3. Prediction

**Gains above 0.95 should hurt.** E078 established that naturalistic separability falls
as drive rises past ~0.46 (0.0814 at rate 0.4602 → 0.0581 at 0.6745). Raising gain
raises drive. This part I hold with reasonable confidence.

**Below 0.95 I genuinely do not know**, and say so rather than pick the flattering
guess. Two mechanisms pull opposite ways: E023's sparse sweep found separability falling
monotonically with gain (0.70 → 4.5%), suggesting lower is worse; but E078's naturalistic
data found the *lowest*-drive condition also worse than the middle (rate 0.3786 → 0.0698
vs 0.4602 → 0.0814). If both hold, there is an interior optimum and 0.95 may already be
close to it — in which case this experiment reproduces E078's density outcome, and
validates a second default rather than moving it.

**The outcome that would matter**: a gain that both improves naturalistic separability
*and* pulls live rate out of saturation. That would be the first positive H2d
intervention in the tree.

## 4. Falsifier

If naturalistic separability peaks at or near 0.95, the gain default is validated and
saturation cannot be addressed by gain alone — leaving H2d with no lead at all, which
should be stated as plainly as E078's falsifier was.

## 5. Design

Both probes, same 12 genomes, paired — the E073/E078 design, so probe-to-probe
comparison is itself paired. On E076's corrected baseline.

- **Gains**: 0.40, 0.60, 0.80, 0.95 (default), 1.10.
- **Metric**: `pallial_sep` verbatim from E041, keeping the E009-series comparable.
- **Also reported**: mean pallial rate per cell, since gain acts *through* drive and the
  whole question is where drive should sit.
- **Then, for the default and the naturalistic best**: **live** mean pallial rate from an
  actual rollout (16 hens, 5 min, 3 seeds), not a settle. Separability improvements that
  leave the network saturated in live operation are not worth having, and the settle
  probes cannot see that.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e079_gain_naturalistic.py
```

## 6. Result

_Not yet run._

## 7. Interpretation

_Pending §6._

## 8. Consequence

_Pending §6._
