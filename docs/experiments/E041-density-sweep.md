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

12 genomes, paired against the density=0.30 baseline on the same seeds:

```
density      mean rate    separability   vs 0.30
0.30            0.2724    0.0872+-0.0492     1.00x
0.15            0.2648    0.0576+-0.0266     0.66x
0.08            0.2618    0.0419+-0.0188     0.48x
0.04            0.2602    0.0313+-0.0138     0.36x
0.02            0.2595    0.0240+-0.0116     0.28x

paired contrast vs density=0.30, threshold t=2.201 (df=11):
  0.15 - 0.30: -0.0296 +/- 0.0073  t=4.08  SIGNIFICANT
  0.08 - 0.30: -0.0453 +/- 0.0096  t=4.72  SIGNIFICANT
  0.04 - 0.30: -0.0559 +/- 0.0112  t=4.98  SIGNIFICANT
  0.02 - 0.30: -0.0631 +/- 0.0118  t=5.37  SIGNIFICANT
```

**The falsifier fires, clearly and monotonically.** Every density reduction makes
separability significantly *worse*, not better — the opposite of the registered
prediction, confirmed at full statistical power after a 2-genome smoke test already
showed the same direction. Mean pallial rate drops only slightly across the sweep
(0.272 → 0.260), so this is not a saturation or dead-network artefact; the network stays
in a normal operating regime throughout.

## 7. Interpretation

**"Fan-in dilution" as stated — informative channels get outvoted by noisy ones, so
fewer total inputs should mean less dilution — is wrong, or at least incomplete.** The
mechanism that actually appears to dominate is different: with only 1–2 informative
channels among ~53 exteroceptive channels, and connections drawn independently at
random, *lowering density lowers the probability that any given pallial unit is
connected to an informative channel at all.* At density 0.30, essentially every pallial
unit samples enough of the stub to catch at least a fraction of the informative signal
somewhere in its input, diluted but present. At density 0.02, most pallial units likely
sample *none* of the 1–2 informative source units, and carry no information about the
distinction whatsoever — not diluted, simply absent. The population-level RMS
separability metric averages over all pallial units, so a growing fraction of
completely uninformative units drags the whole measurement down, dominating whatever
cleaner signal the shrinking fraction of still-connected units might carry.

**This reframes the mechanism rather than closing the question.** The problem is not
"too many inputs dilute the signal" — it is closer to "random sparse sampling gives too
few units a chance to see the signal at all, at any density tested." That points away
from *reducing* random connectivity and toward *guaranteeing* informative channels reach
pallial units reliably — structured, not merely sparser, wiring. This is a different
proposal from `modality_segregated` (E035), which guaranteed audio channels their own
*disjoint* pallial slice but did not increase how reliably any individual unit in that
slice connects to the specific 1–2 informative channels within audio — worth checking
directly as a follow-up rather than assumed.

## Addendum — the other direction, run after §6-7 were written, not pre-registered

**This was not in the original design.** §5 registered only densities at or below the
default (0.30), because the prediction was that *lowering* density would help. Once that
was falsified, the natural next question — does the relationship have an optimum near
0.30, or does it keep moving the other way — was checked directly rather than assumed.
Recorded here as a post-hoc addendum, per this project's practice of logging what was
explored outside the registered design rather than presenting it as predicted.

Same 12 genomes, same paired method, densities {0.30, 0.50, 0.70, 1.00}:

```
0.30  0.0872 +- 0.0492
0.50  0.1170 +- 0.0783
0.70  0.1443 +- 0.1060
1.00  0.1783 +- 0.1381

0.50 - 0.30: +0.0299 +/- 0.0090  t=3.31  SIGNIFICANT
0.70 - 0.30: +0.0571 +/- 0.0174  t=3.29  SIGNIFICANT
1.00 - 0.30: +0.0911 +/- 0.0271  t=3.36  SIGNIFICANT
```

**No optimum in the tested range — separability rises monotonically all the way to full
connectivity**, roughly 2× the default at density 1.0. Combined with §6's result, the
full curve (0.02 → 1.00) is monotonically increasing throughout: more connections is
simply better, everywhere tested. This confirms the §7 reinterpretation directly rather
than leaving it as an inference from the low-density side alone — separability here
tracks *how reliably the informative channels reach the pallium*, and full connectivity
is the most reliable a random projection can be.

**Not free, and not verified against anything else yet.** Full connectivity on this one
block is a real architecture change — more synapses, more compute for that block
specifically (though it is one off-diagonal block of the full connectome, not the whole
matrix) — and separability is not the only thing that matters: `CLAUDE.md`'s design
invariants treat throughput as a correctness constraint, and nothing here has checked
whether density 1.0 changes H2's drive-regulation contrast, breaks Dale's law balance,
or costs real wall-clock time. **This is a promising direction, not yet a decision** —
see Consequence.

## 8. Consequence

- **This specific fix is ruled out.** Do not lower `sensory_pallium_density` below its
  default expecting better separability; it makes H2d's problem worse, and the effect is
  large and consistent across a 15× range.
- **`sensory_pallium_density` stays in `connectome.build()`** as a tested, working
  parameter — the negative result required it to exist and be measured correctly, the
  same reasoning that kept `modality_segregated` after E035.
- **H2d's mechanism is refined, not resolved.** Fan-in dilution is real (E017's original
  stage-by-stage measurement stands — separability collapses specifically at the
  sensory→pallium projection), but the fix implied by "dilution" — fewer inputs — is
  backwards. The actual lever is *reliability of connection to the informative source*,
  not *total input count*. A natural next step: does increasing the *number of
  informative channels each pallial unit is guaranteed to reach* (rather than reducing
  how many total inputs it has) help — e.g. a connectivity prior that biases specifically
  toward the aerial-visual and alarm-call channels, not merely toward audio generally.
- **The addendum sharpens this rather than replacing it.** Separability rises
  monotonically across the *entire* tested range (0.02 → 1.00) with no optimum found —
  more connections is better everywhere checked, and full connectivity (1.0) gives
  roughly 2× the default's separability. The lever really is reliability of connection,
  not a sparsity trade-off with a sweet spot.
- ~~**Next, before adopting anything**: a regression check at
  `sensory_pallium_density=1.0`...~~ **Done, same session, both clean.**

  **Throughput: no effect, and there is a structural reason it couldn't have one.**
  `hen/connectome.py`'s weights are stored dense with a boolean mask, not sparse
  (`CLAUDE.md`'s own design invariant) — the full N×N matrix is multiplied every step
  regardless of how many mask entries are true. Measured directly, reversed order,
  20,000-step rollouts: 58.4x and 58.0x real-time at density 1.0 and 0.30 respectively,
  within run-to-run noise.

  **H2's contrast: not broken, 8 seeds, clean world (`food_deplete_rate=0`, matching
  E037):**

  ```
  density        fixed      learn    learn-fixed
  0.30 (default) +0.0441   -0.0040   -0.0481 +/- 0.0211  t=2.28  thr=2.37  not sig
  1.00 (full)     +0.0133  -0.0201   -0.0334 +/- 0.0116  t=2.87  thr=2.37  SIGNIFICANT
  ```

  Both densities show the same sign (learning lowers hunger relative to fixed); full
  density's is tighter and clears threshold at n=8. **Not being treated as a finding
  that full density helps H2** — E037 already demonstrated this exact contrast can swing
  from t=2.96 to t=2.19 in *opposite* directions across two 12-seed blocks of the same
  experiment, so an 8-seed result in either density condition is exactly the kind of
  single-block evidence this project's own standing rule says cannot move anything. What
  it does establish: **full density does not obviously break foraging regulation**, which
  was the actual question this check existed to answer, and the direction is at least not
  a red flag.
- **Targeted connectivity (biasing toward specific informative channels rather than
  raising density generally) remains untested and may be a cheaper way to the same
  effect** — worth comparing against brute-force full connectivity once the regression
  check above is done, not instead of it.
