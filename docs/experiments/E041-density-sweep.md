# E041 — does sparser sensory→pallium connectivity fix H2d's representational bottleneck?

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**H2d** — the pallium does not form separable representations of distinct stimuli.
`SUPPORTED as a limitation`, reprioritised by E034 (the contrast it depends on occurs on
11.9% of hen-steps, not rare). Two candidate fixes have been tried and both failed on
proper (paired) measurement: the auditory scaffold (E018/E036) supplies comprehension by
construction but doesn't help learning build on it; modality segregation (E035) does not
improve separability at all. This is the third candidate, going after the mechanism
E017/E034 actually localised rather than a structural workaround.

## 2. Question

E017 and E034 localised H2d's loss to fan-in dilution at the single sensory→pallium
projection: each pallial unit sums ~19 stub inputs (density 0.30 over a 64-unit stub),
of which one or two carry any given distinction, so a real difference lands as a small
perturbation on a large common-mode drive. **If each pallial unit summed fewer inputs,
would the same informative channels be diluted less, and separability improve?**

## 3. Prediction

Separability rises as density falls, because fewer inputs per pallial unit means less
common-mode noise diluting whatever signal is present — this is the direct mechanical
prediction of "fan-in dilution" as the stated cause. No specific magnitude predicted;
E035 showed this project's intuitions about magnitude on this exact metric have been
wrong before. **The prediction is directional, not quantitative, and registered as such
on purpose.**

**Named risk, stated before running so it can't be explained away afterward:** at very
low density, `fan_in` for some pallial units could approach the ~1-2 informative channels
themselves, at which point gain normalisation (`gain / sqrt(fan_in)`) could push weights
large enough to destabilise the recurrent dynamics rather than sharpen them — a
non-monotonic relationship (separability rises then falls) is a real possible outcome,
not just noise, and should be reported as such if seen.

## 4. Falsifier

**H2d's fan-in-dilution mechanism is wrong, or at least incomplete, if separability does
not rise as density falls** (flat or falling relationship, outside genome noise). That
would mean the loss is not simply "informative inputs get outvoted by uninformative
ones," and the mechanism needs revisiting.

**Not a falsifier on its own:** a rise that is real but insufficient to close the ~14×
gap — E017/E034's own numbers already show partial fixes (Field-L segregation, before
its correction) are the norm, not the exception, for this problem.

## 5. Design

**Paired per genome, from the first run — not an unpaired sweep.** Same settle-and-
separate probe as E009/E017/E023/E034/E035 (hawk overhead vs. flockmate's aerial call,
presented alone, matched amplitude, 2 s to settle, read at the pallium). Density values
{0.30 (default), 0.15, 0.08, 0.04, 0.02}, each measured on the **same 12 genome seeds**,
so every comparison is a matched pair against the 0.30 baseline — the exact discipline
E035 was built to enforce after finding an unpaired 6-genome comparison gave a false
signal (twice, in both directions, across E017/E034's original measurement and this
session's own first attempt at reproducing it).

`sensory_pallium_density` added to `connectome.build()` for this — overrides
`REGION_CONNECTIVITY`'s sensory→pallium entry only, leaving every other region pair and
`aud_fraction`/`modality_segregated` untouched. Fan-in-based gain normalisation (the
existing `gain / sqrt(fan_in)` mechanism) automatically re-scales surviving synapses for
whatever density results, same as E035's structural implementation relied on.

**Command:** `python -m scratchpad.e041_density --genomes 12`

**Not yet in scope, deliberately:** whether a density that improves separability also
preserves or harms H2's drive-regulation contrast (fed %/hunger) — that is a regression
check on whatever density this run identifies as promising, run separately once there is
a specific value worth checking, not swept blind alongside this one.

## 6. Result

*Pending — filled in after the run, not before.*

## 7. Interpretation

*Pending §6.*

## 8. Consequence

*Pending §6.*
