# E012 — the corrected phase 1 contrast, and what was actually wrong

## 1. Parent hypothesis

**H2** — three-factor plasticity produces measurable behavioural improvement.
Supersedes the confounded [E010](E010-rebaseline-at-corrected-gain.md).

## 2. What E010 got wrong, and what I then got wrong about that

E010 concluded that correcting the gain collapsed H2. Investigating that produced
**two successive wrong attributions before the right one**, which is the most useful
part of this record.

1. **E010 blamed the gain.** It had also, silently, given every condition the
   exploration noise added in E007 — the control included, where E004 had none.
2. **I then blamed the noise.** Measured: the noise-only control differs from a truly
   fixed one by **+0.018**, against a gap from E004 of ~+0.19. It accounts for under a
   tenth of the degradation.
3. **The actual cause is `call_energy_cost`**, added in E005 for a completely
   different hypothesis (H3). It triples the rate at which hunger accumulates, and it
   dominates the metric E004 was measured on.

## 3. The isolation

Fixed, deterministic hens only, 20 min, 3 seeds — nothing to do with learning:

| gain | call cost | final hunger | change | |
|---|---|---|---|---|
| 0.90 | 0 | 0.371 | +0.060 | ← the E004 configuration |
| 0.90 | 8e-4 | 0.671 | +0.245 | |
| 0.70 | 0 | 0.365 | **+0.033** | |
| 0.70 | 8e-4 | 0.638 | +0.224 | ← current |

**The gain is nearly neutral, and slightly favourable.** At zero call cost, 0.90 gives
+0.060 and 0.70 gives +0.033 — the corrected gain is *better*. E010's headline was
wrong.

**The call energy cost explains essentially all of it**, at either gain (+0.185 at
0.90, +0.191 at 0.70).

## 4. The corrected contrast

12 matched seeds, 20 min, gain 0.70, exploration stated explicitly per condition:

| condition | hunger change | fed % | exposure | synapses |
|---|---|---|---|---|
| fixed (innate only) | +0.215 | 5.7 | 5241 | 36373 |
| noise only (no learning) | +0.234 | 5.6 | 4533 | 36373 |
| learning, no growth | +0.231 | 6.1 | 3597 | 21797 |
| learning + growth | +0.219 | 6.6 | 5316 | 40344 |

```
vs fixed:  noise only          +0.018 +/- 0.016 SE   t=1.12  worse (suggestive)
           learning, no growth +0.016 +/- 0.040 SE   t=0.40  worse (noise)
           learning + growth   +0.004 +/- 0.044 SE   t=0.09  worse (noise)
```

**H2 is not supported here.** Learning does not beat the fixed control. It roughly
cancels the cost of its own exploration (+0.016 against noise-only's +0.018) and no
more.

## 5. Interpretation

**The metric was destroyed by a feature added for a different hypothesis.** Calling
now costs energy — a deliberate, justified addition from E005, needed so that
audience-sensitive calling has anything to be sensitive *about*. But it triples
hunger accumulation, so every hen is dominated by what her voice costs rather than by
how well she forages. Hunger regulation can no longer discriminate between conditions
because it is measuring something else.

So this run does not cleanly test H2 either. It tests H2 through a metric that the
call cost has swamped.

**Three experiments — E010, E011, E012 — were spent on a question that was never
about the gain.** The gain correction is fine and mildly beneficial. E009's saturation
finding stands on its own direct measurement and is untouched by any of this.

**What I should have done, and it is the rule already written in four other
experiment files:** isolate one variable at a time and measure the mechanism. Instead
I changed the gain, observed a collapse, and reasoned about causes twice before
running the 2x2 that settled it in six minutes.

## 6. Consequence

- **H2 stays `UNDER TEST`.** Not supported by E012, but E012's metric is compromised,
  so this is not evidence against it either. Three runs now sit in that category.
- **E013**: re-run the contrast with `call_energy_cost=0`, which reproduces E004's
  environment while keeping the corrected gain. That is the clean test of whether
  learning helps, and it is the one that should have followed E009.
- **The call cost needs a home that is not the hunger drive.** It exists for H3, and
  it makes H2's metric unusable. Options: a separate energy budget, a much smaller
  coefficient, or a foraging-efficiency metric that is not the drive itself. This is a
  design decision, recorded not taken.
- **E010's invalidity notice needs amending** — it correctly says the run was
  confounded, and it names the wrong confound as primary.
- **Guards added** (`tests/test_plasticity.py`): a condition named "fixed" must not
  carry exploration, and every condition must state `explore_sigma` explicitly rather
  than inherit it. Both would have caught E010.
- **Standing lesson, now paid for twice in one sitting**: a parameter added for one
  hypothesis silently changed the measurement basis of another. Nothing flagged it,
  because nothing was watching for it. A cheap fixed-hen baseline re-run after any
  environment change would have.