# E116 — can selection across generations do what learning within a lifetime cannot?

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **H2**, and past it. `docs/backlog.md` §4 ("Generational turnover
is not optional"), opened before this project's first experiment and never started.

---

## 2. Question

Six explanations for H2's null have now been proposed and all six have failed. Five were
about the learning rule or the readout. The sixth,
[E115](E115-a-real-basal-ganglia.md), was about the anatomy — a structurally faithful
subpallium, the best-motivated hypothesis available — and it failed identically: the
intervention did exactly what it was designed to do, measurably, and behaviour did not
follow.

**Fixing the brain has been tried. This changes the mechanism instead.**

The premise is simple and it is the one `docs/backlog.md` §4 has been making since before
E001: **H0 does not require within-lifetime plasticity.** It requires the channel to do
work. H4 is `SUPPORTED` and runs with plasticity *off*. If a connectome that forages well
cannot be *learned*, it can still be *selected* — and selection across generations is a
fundamentally different search, with no credit-assignment problem, no 0.2 s eligibility
window, and no requirement that the update direction be anything in particular.

[E111](E111-is-there-headroom.md) established that the target exists: a hand-written
forager reaches hunger **0.4223** against the reflex hen's **0.6332**, replicated at
t≈7. **The question is whether selection can find any of that 0.21.**

**And this is the only route that reaches H5 at all.** Compositional structure needs a
transmission bottleneck — each generation acquiring the code from a limited sample of the
last — which is generational turnover by definition.

**This experiment is the instrument check for the whole approach**, not the H0 ladder.
Before asking whether a channel changes what an *evolving* flock can do, the question is
whether evolution improves anything here at all. If it cannot, nothing downstream matters.

## 3. Prediction

1. **Selection produces measurable improvement.** Mean hunger falls across generations in
   the selected lineages. **This is the first time in this project I have predicted a
   positive**, and the reason is specific: evolution has no credit-assignment problem.
   It does not need to know which synapse helped, only which hen did.
2. **The unselected control does not improve.** Same mutation, same population size,
   parents chosen at random. If both arms improve, the gain is drift or mutation acting as
   annealing, not selection.
3. **It will not reach the oracle.** I expect a fraction of E111's 0.21 — perhaps a
   quarter — because the search is over synaptic weights inside a fixed anatomy, with a
   population of 16 and a few dozen generations.
4. **Fitness will be heritable but noisily so.** Predation is a flock property and hunger
   depends on where the patches fell, so a good hen can have a bad life. If the
   parent–offspring fitness correlation is near zero, selection has nothing to act on and
   §4's instrument falsifier fires.

## 4. Falsifier

**Primary — and it closes the route.** If selected lineages do not beat the unselected
control on hunger by more than 2× the paired standard error after the full run, then
generational selection does not work in this model either. With within-lifetime learning
already closed six ways, that would leave **writing up the null** as the remaining honest
option, and it should be recorded as that rather than tuned.

**Instrument falsifier — reported before the headline.** Two things must be true or
selection cannot operate and no result means anything:

- **Variation**: fitness must differ across hens at generation 0. Reported as the
  coefficient of variation; it must exceed 0.05.
- **Heritability**: offspring fitness must correlate with parent fitness. Reported as the
  parent–offspring correlation across the run; it must exceed 0.1.

**Degeneracy falsifier.** If the population collapses to identical hens — mean pairwise
`|W| ` difference falling below 10% of its generation-0 value — there is no variation left
and any plateau is an artefact of that rather than of the search.

**Regression falsifier.** No change to any existing default; the suite passes. This adds a
module and touches nothing that runs today.

**Replication rule.** E021. Four independent lineages per arm, and nothing moves the tree
on one seed block.

## 5. Design

**New module `run/evolve.py`. No change to any existing default**, and plasticity is
**off** throughout — that is the point.

**What is heritable.** Per-hen `W` (recurrent weights) and `W_out` (the readout). The
shared genome — the connectivity `mask`, `W_in`, `dale`, the resting biases, and the
reflex arc — is held **fixed**, so this is selection on synaptic weights inside a fixed
anatomy. That is a real limitation and it is stated rather than discovered: a fuller
version would evolve the mask too, but the mask is shared across the flock, so per-hen
selection cannot vary it.

**Fitness.** Not a task objective handed down from outside. Cumulative homeostatic
welfare over a lifetime, on the same basis `plasticity.reward` already uses — drive
reduction, minus the cost of being caught:

```
fitness = -(mean hunger + mean cold + mean thirst) - strike_penalty * strikes
```

A hen is selected for staying alive and comfortable, which is the only thing a real animal
optimises.

**The generational loop**, per lineage:

1. Run the flock for a **10-minute lifetime**, plasticity off.
2. Score each of the 16 hens.
3. **Truncation selection**: the top 4 become parents; each contributes 4 offspring.
4. **Mutation**: Gaussian noise on live synapses of `W` and `W_out`, σ proportional to
   each matrix's own scale, then `_enforce_dale` so an inhibitory neuron can never be
   mutated excitatory — the same invariant learning is held to.
5. Repeat for **20 generations**.

**Arms**: `{selection, no selection}` × 4 independent lineages. The control is identical
in every respect except that parents are drawn at random rather than by rank, so mutation
load and population size are matched exactly.

**Measured**: mean hunger and `caught/dive` per generation; fitness mean and spread;
generation-0 coefficient of variation and the parent–offspring correlation (the instrument
checks); mean pairwise weight distance (the diversity check).

### Cost

~40 minutes for both arms.

---

## 6. Result

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*
