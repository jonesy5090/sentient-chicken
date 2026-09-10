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

*(written after the run)*

## 7. Interpretation

*(written after the run)*

## 8. Consequence

*(written after the run)*
