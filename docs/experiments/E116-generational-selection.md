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

### 6a. The instrument falsifier fired before the experiment ran — and diagnosing it is the finding

§4 required fitness to be **heritable**: parent–offspring correlation above 0.1. Measured
first, as §4 demanded, and it is **zero**.

The cleanest form of the measurement takes the *same* connectomes through the *same*
world twice, with only exploration noise differing, and correlates the two scores. That is
the ceiling on heritability — if a hen's own score does not predict itself, it cannot
predict her offspring's:

| lifetime | repeatability, same world | across worlds |
|---|---|---|
| 300 s | **−0.063** | +0.127 |
| 600 s | **−0.216** | −0.010 |
| 1200 s | **+0.078** | −0.021 |

**A hen's fitness carries no information about her brain.** It is where she happened to be
when a hawk came, and which patch she reached first. Selecting the top 4 selects the lucky
4 — and a first smoke test showed exactly that pathology, with selected lineages doing
*worse* than random-parent controls, because selection concentrates luck while mutation
degrades a tuned connectome.

### 6b. Why — and it is not what I expected

Two explanations fit: the varying part of the brain has no behavioural consequence, or the
variation is too small. Amplifying the between-hen difference distinguishes them.

| founder σ | `W_out` spread | **fitness repeatability** | hunger repeatability |
|---|---|---|---|
| 0.00 (as built) | 0.0485 | **−0.216** | 0.249 |
| 0.05 | 0.0486 | 0.052 | 0.197 |
| 0.50 | 0.0630 | −0.284 | −0.131 |
| **2.00** | **0.1319** | **+0.473** | **+0.608** |

**The variation is too small, not inconsequential.** At 2.7× the natural spread, fitness
becomes strongly repeatable. The brain *can* determine the outcome; a flock straight out
of `connectome.build` is simply too genetically uniform for selection to see anything.

That is an ordinary population-genetics fact — **you cannot select without standing
variation** — and it has never been checked here because nothing before E116 needed it.

Also visible in that table and worth keeping: at natural variation, *hunger* is somewhat
repeatable (0.249) while *fitness* is not (−0.216). **Adding the predation term destroys
the signal the hunger term carries**, because predation is the luck-dominated half.

### 6c. With standing variation, selection works — and it replicates

`founder_sigma=2.0` at generation 0 only. 16 generations, 10-minute lifetimes, 4
independent lineages per arm, two disjoint blocks.

| | gen 0 | gen 3 | gen 9 | gen 15 |
|---|---|---|---|---|
| **block 1, selected** | 0.4959 | 0.4509 | 0.4511 | **0.4716** |
| block 1, control | 0.4959 | 0.5039 | 0.5100 | 0.5075 |
| **block 2, selected** | 0.4465 | 0.4236 | 0.4132 | **0.4206** |
| block 2, control | 0.4465 | 0.4470 | 0.4546 | 0.4521 |

| | selected improves | control changes | **selected − control** | vs the 2×SE bar |
|---|---|---|---|---|
| block 1 (lineages 0–3) | **+0.0243** | −0.0116 | **−0.0359 ± 0.0119** (t=−3.00) | 0.0238 — **clears** |
| block 2 (lineages 4–7) | **+0.0259** | −0.0056 | **−0.0315 ± 0.0145** (t=−2.18) | 0.0290 — **clears** |

**Both blocks clear the pre-registered bar**, and the two are strikingly consistent:
improvements of +0.0243 and +0.0259, differences of −0.0359 and −0.0315. Selected lineages
improve; unselected ones **degrade** under identical mutation load, which is the control
doing its job.

Neither block clears a conventional t-test — df=3 per block sets the threshold at 3.182,
and 4 lineages is a thin design. Pooled across all 8 lineages (**post-hoc**, per E030's
precedent, since pooling was not pre-registered): **−0.0341 ± 0.0092, t=−3.71, df=6**
against 2.447.

Prediction 1 holds — the first positive prediction I have made in this project, and it
held. Prediction 2 holds: the control does not improve. Prediction 3 holds: it is a
fraction of the available headroom, not the whole thing.

### 6d. The search stalls, and the degeneracy falsifier nearly fires

Population diversity — mean pairwise `|W_out|` distance — collapses from 0.132 to
**0.0155–0.0176**, about **12–13% of its founding value**. §4's degeneracy bar was 10%, so
it does not fire, but it is close, and it explains the shape of the result: essentially all
of the improvement happens by generation 3 and the curve is flat after that. **Selection
consumes the standing variation it needs, and mutation at σ=0.05 does not replace it fast
enough.**

### 6e. A comparison I should not make

E111's references — reflex hen 0.6332, camped oracle 0.4223 — are measured over **30-minute**
lifetimes. E116 runs **10-minute** ones, and hunger equilibrates upward over time, so
E116's 0.4959 at generation 0 is not "better than E111's reflex hen". The two are not
comparable and the script prints the E111 line in a way that invites exactly that error.
**The only valid comparisons here are within E116**: selected against control, and
generation 0 against generation 15.

## 7. Interpretation

**Selection across generations does something that within-lifetime learning has never
done in this project: it produces a replicated behavioural improvement over a matched
control.** It is small — about 0.025 hunger units over 16 generations — and it required
a change nobody had thought to make.

**The change is the real finding.** For 116 experiments this project has assumed that a
flock built by `connectome.build` is a population. It is not, in the sense selection
needs: the between-hen variation it produces is so small that a hen's fitness is
uncorrelated with her own brain. **Repeatability ~0 means no adaptive process — learning
or selection — can act on individual differences**, because there are no individual
differences that reach behaviour.

That connects to the rest of the tree rather than sitting apart from it. E107 measured
every hen's cortical output as a fixed direction; E109 measured the readout's update as
confined to the reflex arc's own direction; E110 found learning does not beat a frozen
readout. **All 16 hens share the reflex arc, the reflex arc determines behaviour, and the
part that differs between them barely reaches the muscles.** E116 is the same fact from
the population's side.

**What it does not show.** That evolution solves H2. The improvement captures a small
fraction of what a hand-written policy achieves, it plateaus after three generations as
diversity collapses, and it needed founder variation supplied artificially rather than
arising from the model. The honest claim is narrow: **given standing variation, selection
finds improvements that learning does not — and the binding constraint is now the
variation, not the search.**

**Why this matters for H0 rather than just H2.** `docs/backlog.md` §4 wants generational
turnover for a different reason: a transmission bottleneck is what H5's compositional
structure requires. E116 says the machinery works and names its prerequisite. Any future
ladder that evolves flocks with an intact channel against a yoked one now has a working
substrate — and a known first question, which is where the standing variation comes from.

## 8. Consequence

**New module `run/evolve.py`.** No existing default changes and nothing that runs today is
touched; plasticity is off throughout, which is the point. Three guard tests: fitness
terms commensurate, Dale's law binding on mutation exactly as it binds on learning, and
the unselected control matched in everything but selection.

**`docs/hypothesis.md`.** H2 records that the missing ingredient for any adaptive process
is standing variation that reaches behaviour, measured at repeatability ~0 for a
naturally-built flock. H0 records that generational machinery exists and works.

**Not adopted.** Any claim that evolution solves foraging here, or that E116's numbers are
comparable to E111's.

### Follow-ups

1. **Where does standing variation come from?** This is now the binding constraint, and it
   is a question about `connectome.build` rather than about search. A flock whose hens
   differ enough to be selectable is a modelling decision nobody has made deliberately.
2. **Diversity collapse.** σ=0.05 mutation does not replace what truncation selection
   consumes; improvement stops at generation 3. Higher mutation, more parents, or
   recombination — untested, and cheap.
3. **Then the H0 ladder**: evolve flocks with an intact channel against a yoked one. That
   is the experiment this module was built for, and E116 says the substrate works.
