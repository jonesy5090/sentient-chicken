# E118 — the search consumes its own fuel

> Sections 1–5 written before anything ran. Written after E117's result was known, and
> E117 changes one of the arms — see §5.

## 1. Parent hypothesis

**H0**, via [E116](E116-generational-selection.md) follow-up 2 and
[E117](E117-where-standing-variation-comes-from.md).

E116 found that selection across generations works, and then stops working:

> Population diversity — mean pairwise `|W_out|` distance — collapses from 0.132 to
> **0.0155–0.0176**, about **12–13% of its founding value** […] essentially all of the
> improvement happens by generation 3 and the curve is flat after that. **Selection
> consumes the standing variation it needs, and mutation at σ=0.05 does not replace it
> fast enough.**

E117 then established *why* variation is the binding constraint: what differs between
hens at hatch is real but inaudible, and only becomes audible when the cortical readout
is amplified. A search that spends its founding variation and cannot regenerate it is a
search that gets one shot.

## 2. Question

Can the generational loop be made to keep improving past generation 3 — by mutating
harder, selecting more weakly, recombining parents, or by selecting on a criterion that
is not three-quarters noise?

## 3. Prediction

1. **Higher mutation will help, and then hurt.** σ=0.30 replaces variation faster than
   truncation consumes it, but `_mutate` scales noise by each matrix's own RMS, so σ=0.30
   is a 30% perturbation of every live synapse per generation — E117 measured that a
   *one-off* σ=2.0 already costs 0.03 hunger units. Predicted: σ=0.15 improves on
   baseline, σ=0.30 is no better than baseline and possibly worse.
2. **Weaker truncation (8 parents of 16, not 4) will preserve diversity** — final
   diversity above **25%** of founding against baseline's 12–13% — and will improve *less
   per generation* but for longer, ending at least as good as baseline.
3. **Recombination will beat both**, because it makes new combinations out of variation
   that already exists instead of degrading tuned weights with noise. Final diversity
   above 25%, and the largest gen-0→gen-15 improvement of any arm.
4. **`drives_only` will show the largest effect of any single change.** E117 measured
   predation repeatability at **zero in all seven arms** while `evolve.fitness` gives it
   74% of its variance, so roughly three-quarters of the selection differential is spent
   on luck. Removing it should more than double the improvement.

Predictions 1–3 are the ones I expect to be wrong about; this project's base rate on
mechanism predictions is poor and stating them is the point.

## 4. Falsifier

**For the parent claim** (that diversity collapse is what stalls the search): if no arm
that demonstrably preserves diversity — final diversity above 25% of founding, measured,
not assumed — improves on baseline's gen-0→gen-15 change, then diversity collapse is a
correlate of the plateau and not its cause, and "run the search longer/harder" stops
being a live direction. E116's plateau would then need a different explanation and I
should stop proposing search-hyperparameter fixes, exactly as the project stopped
proposing brain fixes after six failures.

**Against a spurious win:** an arm can improve mean hunger by pushing the flock toward
the saturated regime E117 found at `readout_1.00` — a bird doing everything flat out is
not hungry because she pecks constantly, and that is not adaptation. So every arm reports
**mean crouch, flee and forward drive at the final generation** alongside fitness. Any
arm whose improvement comes with motor channels above ~0.9 is recorded as degenerate.
This is the same check that caught `readout_1.00` in E117 and `hebbian_readout` in E055.

**Against the unselected control being inert:** E101/E102 were invalidated by a control
that could not move by construction. Here the control draws parents at random and carries
identical mutation load, so it *can* drift — E116 measured it drifting the wrong way
(−0.0116). If the control shows exactly zero change in every arm, it is not doing its job
and the contrast means nothing.

## 5. Design

**Held identical across arms:** founder construction (`founder_sigma=2.0`, without which
E117 measured nothing to select), 16 hens, 600 s lifetimes, 16 generations, 4 lineages per
arm, `hawk_period_s=50`, and the world key per lineage. Arms differ in one search
parameter each.

| arm | change | tests |
|---|---|---|
| `baseline` | σ=0.05, 4 parents, `caught_weight=0.10` | E116's configuration |
| `mut_0.15` | σ=0.15 | mutation supply |
| `mut_0.30` | σ=0.30 | mutation supply, past the useful point |
| `parents_8` | 8 parents of 16 | selection strength |
| `recomb` | uniform crossover of two parents, then mutate | recombination |
| `drives_only` | `caught_weight=0.0` | **E117's correction** — stop selecting on luck |

Every arm runs a matched **unselected control** (parents drawn uniformly at random,
identical mutation load), which is the only thing that separates "selection worked" from
"mutation drifted".

`recomb` needs a real change to `run/evolve.py` — `_breed` currently copies one parent per
child. Uniform crossover per synapse, with Dale's law enforced afterwards exactly as
mutation enforces it, since a recombined connectome must obey the same invariant a learned
one does.

**Primary metric.** Mean hunger at generation 15 minus at generation 0, **selected minus
its own matched control**, per lineage; 4 lineages give the error bar. Same statistic
E116 used, so the numbers are directly comparable to its −0.0359 / −0.0315.

**Secondary, pre-declared:** final diversity as a fraction of founding (this is
prediction 2 and 3's falsifier, so it is not exploratory); generation at which 90% of the
total improvement has happened (E116's "all by generation 3"); the motor-saturation
triple from §4.

**Replicates.** 4 lineages per arm per condition, matching E116. 6 arms × 2 conditions ×
4 lineages × 16 generations × 13.3 s ≈ **2 hours 15 minutes**. E116's own bar was 4
lineages and it noted that this is thin; any arm that wins here needs a second block
before it changes anything, and this experiment does not claim otherwise.

**Command.**

```bash
PYTHONPATH=. python scratchpad/e118_diversity.py
```

## 6. Result

Two disjoint lineage blocks (0–3 and 4–7). Block 1 ran all six arms; block 2 re-ran the
four load-bearing ones. `improve` = hunger at generation 0 minus at generation 15, so
positive means the flock got better.

### 6a. The falsifier fires, in both blocks

§4 said: if no arm that demonstrably preserves diversity improves on baseline, then
diversity collapse is a **correlate** of the plateau and not its cause.

| arm | diversity kept (`W_out`) | vs baseline, selected lineages, pooled 8 lineages |
|---|---|---|
| `parents_8` | 17–33% | **−0.0008 ± 0.0124, t=−0.06** |
| `mut_0.15` | 37% (both blocks) | **+0.0120 ± 0.0154, t=+0.78** |
| `mut_0.30` | 79% (block 1) | **−0.0556 ± 0.0174, t=−3.19** — *worse* |
| `recomb` | 12% (block 1) | +0.0106 ± 0.0093, t=+1.14 |
| baseline | 11–14% | — |

Three arms held diversity far above baseline's collapse — one of them at **79%**, six
times baseline — and **not one converted that into an improvement.** The arm that *did*
improve, `drives_only`, retained **11–13%**: the same collapse as baseline.

**Diversity collapse is not what stalls the search.** It is real, it is reproducible, and
it is beside the point. E116's reading of its own plateau ("selection consumes the standing
variation it needs, and mutation at σ=0.05 does not replace it fast enough") is the
natural story and it is wrong — the fuel is not what ran out.

Nothing extended the search horizon either. 90% of all improvement arrives by generation
**1.5–4.3** in every arm of both blocks, including the winners.

### 6b. The criterion was the problem — E117's correction, confirmed

`drives_only` changes one thing: `caught_weight` 0.10 → 0.0, removing from the selection
criterion the term E117 measured as carrying no information about the hen. Because a
control never consults fitness, `baseline` and `drives_only` **share their unselected
control exactly** — verified numerically, not assumed — so their selected lineages are
paired on founders, worlds, mutation draws and parent-count. It is the cleanest contrast
here.

| pooled over 8 lineages (df=7, bar 2.365) | value |
|---|---|
| `drives_only` − `baseline`, selected | **+0.0299 ± 0.0093, t=+3.21** |
| `drives_only`, absolute improvement | **+0.0381 ± 0.0079, t=+4.80** — **8/8 lineages improved** |
| `baseline`, absolute improvement | +0.0082 ± 0.0120, t=+0.68 — 4/8 lineages, a coin flip |
| `drives_only`, selected − control | **+0.0495 ± 0.0140, t=+3.54** |
| `baseline`, selected − control | +0.0196 ± 0.0183, t=+1.07 |

Per block the criterion contrast is **+0.0444 (t=4.31)** then **+0.0155 (t=1.22)** — same
sign, one clearing df=3's bar and one not, pooling to t=3.21. Thinner than the headline
number suggests, and the strongest single statement is the plainest one: **every one of
eight `drives_only` lineages improved, against four of eight for baseline.**

### 6c. `sel − ctl` is a treacherous statistic, and it nearly fooled me

The summary table's own primary metric ranked `mut_0.15` first: selected-minus-control
**+0.0526 (t=4.17)** in block 1 and **+0.0381 (t=3.95)** in block 2, the only arm clearing
df=3's threshold in either block. It replicates cleanly. It is also meaningless as a
between-arm comparison.

The control degrades **under mutation load**, and σ=0.15 is three times baseline's. So a
higher mutation rate hands selection more damage to repair and inflates
selected-minus-control without selection being any more effective. Against baseline's
*selected* lineages — the comparison that asks whether the flock actually ends up better —
`mut_0.15` is **+0.0120 ± 0.0154, t=0.78**, in both blocks (+0.0126, +0.0115). Nothing.

Pooled `sel − ctl` ranks `mut_0.15` (t=5.78) above `drives_only` (t=3.54); pooled absolute
improvement ranks them +0.0202 against +0.0381. **The two statistics disagree about the
winner, and the one E116 used and I pre-registered is the misleading one.**

### 6d. Mutation buys diversity by damaging the bird

`mut_0.30` retained 79% of founding diversity and **0 of 4 lineages improved at all**;
selected hunger got *worse* by 0.059. Its motor profile at the final generation —
forward 0.98, peck 0.93, crouch 0.91, flee 0.66 — trips §4's saturation bar, the same
everything-at-once regime E117 found at `readout_scale=1.00`. `mut_0.15` is partway there
too (forward 0.95 in both blocks) and is flagged.

So mutation does replace standing variation, exactly as predicted, by randomising the
connectome toward a state where hens differ because they are all differently broken.

### 6e. E116's own result does not reproduce at this configuration

`baseline` is E116's configuration, and its selected-minus-control pooled over 8 lineages
is **+0.0196 ± 0.0183, t=1.07** — not significant, with 4 of 8 lineages improving. E116
reported −0.0359 ± 0.0119 and −0.0315 ± 0.0145 on two blocks, pooling to t=−3.71.

I cannot attribute the difference cleanly. E116's script was never committed, so its coop
is unrecoverable; this runs at `hawk_period_s=50` and E116's predator rate is unknown, and
E117 already showed the predation term dominating fitness variance at every rate tested.
What can be said is that **the one result E116 rests on is, at the nearest configuration I
can reconstruct, a coin flip** — and that `drives_only`, which differs from it by deleting
the noisiest term of its criterion, is not.

### 6f. Predictions scored

1. σ=0.15 improves on baseline, σ=0.30 does not — **wrong on the first half.** σ=0.15 is
   +0.0120, t=0.78 against baseline. Right that σ=0.30 is worse.
2. Weaker truncation preserves diversity (**right**, 17–33%) and ends at least as good as
   baseline (**wrong** — −0.0008, t=−0.06; it ends exactly level, having achieved nothing).
3. Recombination preserves diversity above 25% and gives the largest improvement —
   **wrong on both**: 12.3% diversity, +0.0106 (t=1.14).
4. `drives_only` shows the largest effect of any single change — **right**, and it is the
   only one of the four that came from a measurement (E117) rather than from reasoning
   about search dynamics.

Three mechanism predictions, three wrong. The project's base rate holds.

## 7. Interpretation

**What stalls the generational search is not a shortage of variation but a criterion that
cannot see.** Every intervention on the search itself — mutate harder, truncate less
hard, recombine — either did nothing or made the bird worse, and the two that preserved
the most diversity are exactly the two that achieved the least with it. Deleting one term
from the fitness function, which touches the search not at all, took selected lineages from
4-of-8 improving to 8-of-8.

That is the same shape as this project's oldest lesson, arriving in a new place. E019's
table lists seven conclusions about the brain that were really conclusions about a broken
instrument; CLAUDE.md's §3 says a null is only informative if the instrument could have
shown a positive. E116 looked at a plateau and proposed a mechanism inside the search.
**The plateau was mostly the scoreboard.**

**What this does not license.** It does not show that `drives_only` is the right criterion
— only that it is better than one containing a term with zero repeatability. It does not
show the search is now healthy: `drives_only` still does 90% of its work by generation 1.5
and still ends with 11–13% of its founding diversity, so something continues to stop it,
and this experiment rules out four candidate somethings without naming the survivor.
And the absolute effect remains small — 0.038 hunger units against the 0.21 of headroom
E111 measured, still under a fifth of what a hand-written forager achieves.

**A confound I cannot remove.** `drives_only` and `baseline` differ in the criterion, and
the criterion determines which hens become parents, so the two arms diverge in their
*populations* from generation 1 onward. That is not a flaw — it is the manipulation — but
it means "drives_only lineages end better" cannot be decomposed into "selection was more
efficient" versus "the drives-selected population happened to be a better place to start
generation 2 from". Separating those needs a design that switches criterion mid-run.

**The thing that should worry the next experiment.** A drives-only criterion is indifferent
to hawks, and `coop/world.py` attaches no physiological cost to being caught — hunger moves
only with `at_food_any & pecking`. So the criterion that makes selection work is one on
which *avoiding predators is worth nothing*, and plausibly worth less than nothing, since
crouching costs foraging time. An alarm channel exists to prevent predation. E119 measures
this directly before the H0 ladder is run, because if it holds, the ladder has no criterion
to run on.

## 8. Consequence

**`run/evolve.py`** gains `recombine` (uniform crossover, Dale-enforced) with a guard test
covering the three ways it could have silently done nothing, and `run_lineage` now returns
the final connectome and tracks `W` diversity alongside `W_out`. Recombination is **not**
adopted as a default: it neither preserved diversity nor improved outcomes.

**`docs/hypothesis.md`.** H0 records that the generational plateau is not a diversity
problem, and that the criterion carries it.

**`docs/backlog.md` §8.** E116's follow-up 2 (diversity collapse) is **answered and
closed** — it was the wrong diagnosis. Follow-up 3 (the H0 ladder) is marked blocked
pending E119. New item: what *does* stop `drives_only` at generation 2, given that four
search fixes have been ruled out.

**Not adopted:** any claim that the generational loop now works, or that `caught_weight=0`
is the right criterion rather than merely a less noisy one.

**Code.** `scratchpad/e118_diversity.py`, `scratchpad/e118b_analysis.py`.
