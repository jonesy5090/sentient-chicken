# E117 — where does standing variation come from?

> Sections 1–5 written before anything ran.

## 1. Parent hypothesis

**H0**, via [E116](E116-generational-selection.md). E116 established that generational
selection produces a replicated behavioural improvement where within-lifetime learning
never has — but only after founder connectomes were mutated with σ=2.0, an amplification
with no biological justification, chosen because it worked. E116's own follow-up list
names this as the binding constraint:

> Where does standing variation come from? This is now the binding constraint, and it is
> a question about `connectome.build` rather than about search. A flock whose hens differ
> enough to be selectable is a modelling decision nobody has made deliberately.

This is a prerequisite for the H0 ladder (evolve an intact channel against a yoked one),
which is the first route this project has to H5.

## 2. Question

Which choice in `connectome.build` is responsible for a flock's fitness being
uncorrelated with its brains — the *amplitude* of the cortical pathway, the *amount of
anatomy* that is shared flock-wide, or the *exploration noise* fitness is measured
through?

## 3. Prediction

The per-hen variation that exists at hatch lives entirely in `W` and `W_out`. Both reach
the muscles only through `W_out`, which is deliberately scaled by `readout_scale=0.05`
("at hatch the cortical pathway is near-silent and behaviour is dominated by the innate
reflex arc"). Everything that dominates behaviour — the reflex matrix, `b_motor`, `W_in`,
`mask`, `dale` — is **identical across the flock by construction**.

So, numerically, before running:

1. **Amplitude is the throttle.** `readout_scale=1.00` (20×) will take fitness
   repeatability from ~0 to **above +0.30**, without adding any variation — the same
   differences, merely audible.
2. **Per-hen anatomy will do less than amplitude**, because a per-hen connectivity mask
   still speaks through the same 0.05-scaled readout. Predicted repeatability **+0.05 to
   +0.20**, i.e. detectable but well short of arm 3.
3. **Exploration noise is not the explanation.** Cutting `explore_sigma` 0.6 → 0.1 will
   move repeatability by **less than +0.15**, because E116 already found that quadrupling
   the lifetime (300 s → 1200 s) did not average the noise away — which is what a
   fitness dominated by i.i.d. evaluation noise would have done.
4. **E116's positive control reproduces**: `founder_sigma=2.0` gives repeatability
   **above +0.30**.

I have been wrong about the direction of a prediction in this project often enough that
these are recorded to be scored against, not to be right.

## 4. Falsifier

Two, and they cut in different directions.

**Against the experiment's premise:** if `founder_sigma=2.0` does *not* reproduce a
repeatability above +0.30, the instrument does not detect variation that is known to be
present, and no arm's null means anything. E116's script was never committed, so this
check is the only thing standing between this experiment and an unverifiable baseline.
It runs first and the rest is not interpreted without it.

**Against every arm, including a "successful" one:** an arm can raise repeatability by
**breaking hens rather than differentiating them**. A hen whose readout is loud enough to
override her reflex arc may crouch permanently, and "reliably catatonic" is perfectly
repeatable and completely useless — it is variation in how damaged you are, not variation
selection can climb. So **mean fitness is reported for every arm alongside
repeatability**, and any arm whose repeatability rises while its mean fitness falls
materially below the default arm's is recorded as degenerate, not as a fix. E019's
"hunger starts at the equilibrium" and E116's own "selection concentrates luck" are the
same error in two other costumes.

## 5. Design

**Repeatability, and why it is the right quantity.** Take the *same* 16 connectomes
through the *same* coop twice, changing only the run key (exploration noise, and the
world's own stochastic draws — the coop layout and starting positions are identical).
Correlate the two fitness vectors across the 16 hens. This is the ceiling on
heritability: if a hen's score does not predict her own score, it cannot predict her
offspring's. Same definition E116 §6a used.

**Fixing E116's error bar.** E116 reported single correlations on n=16 hens, where the
standard error of a Pearson r is ≈ 1/√13 ≈ 0.28. Its table of −0.216, +0.052, −0.284,
+0.473 is therefore four numbers of which **at most one is distinguishable from zero or
from each other**, and the −0.216 vs +0.052 contrast it reasons from is noise. Here every
arm is measured on **8 independent genome seeds**, averaged through a Fisher z-transform,
and reported as mean r ± SE with a t against df=7 (threshold 2.365). No arm's verdict
rests on a single correlation.

**Conditions.** Six arms. The world (`hawk_period_s=50`, see below), the coop layout, the
lifetime (600 s), the flock size (16) and both run keys are identical across arms; arms
differ only in how the founding flock is built.

| arm | construction | what it tests |
|---|---|---|
| `default` | `connectome.build` unchanged | baseline |
| `founder2` | `founder_sigma=2.0` | E116's fix — **positive control** |
| `readout_0.25` | `readout_scale` 0.05 → 0.25 (5×) | amplitude |
| `readout_1.00` | `readout_scale` 0.05 → 1.00 (20×) | amplitude |
| `per_hen_mask` | each hen gets her own connectivity draw at the same density | anatomy |
| `explore_0.1` | `explore_sigma` 0.6 → 0.1 | evaluation noise |

`per_hen_mask` needs no change to `brain.step`: `W` is already `(H, N, N)`, and `p.mask`
is read only by structural growth (off here) and by reporting, so a per-hen mask applied
to `W` is a genuine per-hen anatomy with the shared mask retained as bookkeeping.
`W_in`, `dale`, `b`, `b_motor` and the reflex matrix stay flock-wide in every arm — a
real limit on how much "anatomy" this arm can vary, stated here rather than discovered
later.

**Primary metric.** Fisher-z-averaged repeatability of **fitness** (`evolve.fitness`),
per arm, over 8 genome seeds.

**Reported alongside, not exploratory — required by §4:** mean fitness and mean hunger
per arm, and the repeatability of the two fitness components separately (`−drives` and
the predation term). E116 found hunger repeatable at +0.249 while fitness was not, and
concluded the predation term destroys the signal; with a real error bar that claim is
testable rather than asserted.

**Exploratory:** between-hen spread of `W_out`, and between-hen spread of realised
behaviour (per-hen mean motor output), to confirm each arm moves the variable it claims
to — CLAUDE.md check 1, "does the manipulated variable actually vary".

**World configuration, and a discrepancy worth recording.** E116's script was never
committed, so its exact coop is unrecoverable. It reports `n_caught_any` at mean 1.23,
sd 0.74 per hen and predation at 51% of fitness variance. Measured here on
`DEFAULT_COOP` at a 600 s lifetime: at `hawk_period_s=900` (the default) **no hen is ever
caught**, so the predation term is identically zero and `caught_weight` is inert; at 50 s
the counts are mean 1.69, sd 0.92 — the closest recoverable match — but predation is
**74%** of fitness variance, not 51%. I cannot reproduce E116's 51% at any predator rate
tested (25 s → 74%, 30 s → 85%, 40 s → 72%, 50 s → 74%). `hawk_period_s=50` is used, and
the components are reported separately precisely because the weighting between them
cannot be verified against E116.

**Replicates.** 8 genome seeds per arm. 6 arms × 8 seeds × 2 runs = 96 rollouts at ~13.3
s each ≈ 25 minutes including compiles.

**Command.**

```bash
PYTHONPATH=. python scratchpad/e117_standing_variation.py
```

Commit: this file's parent.

## 6. Result

Seven arms, two disjoint seed blocks (genomes 1000–1007 and 1008–1015), 16 correlations
per arm. Everything below is Fisher-z averaged; the bar at df=15 is 2.131.

### 6a. The headline, and it is about the metric as much as the flock

**r(−drives) — the homeostatic half of fitness:**

| arm | block 1 | block 2 | **pooled, 16 seeds** |
|---|---|---|---|
| `default` | +0.063 ± 0.079 | −0.090 ± 0.066 | **−0.014 ± 0.053, t=−0.25** |
| `founder2` | +0.501 ± 0.132 | +0.729 ± 0.087 | **+0.628 ± 0.090, t=8.17** |
| `readout_0.25` | +0.369 ± 0.148 | +0.630 ± 0.114 | **+0.511 ± 0.101, t=5.58** |
| `readout_1.00` | +0.881 ± 0.203 | — | block 1 only — **degenerate, see 6c** |
| `shared_shift` | +0.069 ± 0.057 | +0.272 ± 0.062 | **+0.172 ± 0.049, t=3.57** |
| `per_hen_mask` | +0.077 ± 0.067 | +0.034 ± 0.080 | **+0.055 ± 0.050, t=1.10** |
| `explore_0.1` | +0.085 ± 0.074 | +0.195 ± 0.083 | +0.140 ± 0.056, t=2.52 |

**r(fitness) — drives plus the predation term:**

| arm | block 1 | block 2 | **pooled** |
|---|---|---|---|
| `default` | −0.019 ± 0.070 | −0.090 ± 0.123 | −0.055 ± 0.069 |
| `founder2` | +0.105 ± 0.087 | +0.432 ± 0.059 | **+0.276 ± 0.069, t=4.13** |
| `readout_0.25` | +0.136 ± 0.099 | +0.317 ± 0.096 | **+0.228 ± 0.071, t=3.28** |
| `shared_shift` | +0.045 ± 0.118 | +0.074 ± 0.046 | +0.060 ± 0.061, t=0.98 |
| `per_hen_mask` | −0.105 ± 0.084 | +0.066 ± 0.097 | −0.020 ± 0.066 |
| `explore_0.1` | **−0.118 ± 0.033, t=−3.63** | +0.138 ± 0.114 | +0.010 ± 0.066 |

Contrasts against `default` on r(−drives), pooled (unpaired t on z, df≈30, bar 2.04):
`founder2` **+0.752 ± 0.105, t=7.16**; `readout_0.25` **+0.578 ± 0.114, t=5.05**;
`shared_shift` **+0.187 ± 0.072, t=2.59**.

### 6b. E116's number is not refuted, and my block 1 nearly said it was

Block 1 gave `founder2` a fitness repeatability of **+0.105** against E116's +0.473, which
reads like a failure to replicate. Block 2 gave **+0.432**. The pooled estimate is
**+0.276 ± 0.069**, and E116's single correlation sits inside the block-to-block spread of
the same manipulation. **E116's value is high but not wrong; what was wrong was reporting
it without an error bar**, and a one-block reading of my own data would have made the
mirror-image mistake in the other direction. This is E021's lesson arriving for the
fourteenth time, and it caught me mid-experiment rather than afterwards only because the
second block was pre-registered in §5.

The stable quantity is the homeostatic one: `founder2`'s r(−drives) is +0.501 and +0.729
across the two blocks, pooling to **+0.628, t=8.17**. E116's side observation — that
hunger is repeatable where fitness is not — is the finding, and it is much larger than
the thing E116 led with.

### 6c. Amplitude works, and past a point it works by wrecking the bird

`readout_1.00` scored the highest repeatability of any arm (+0.881) and is worthless. The
motor profile says why:

| arm | forward | peck | crouch | flee | hunger | caught/hen (144 dives) |
|---|---|---|---|---|---|---|
| `default` | 0.500 | 0.532 | 0.159 | 0.082 | 0.475 | 1.44 |
| `readout_0.25` | 0.872 | 0.767 | 0.501 | 0.386 | 0.499 | 1.94 |
| `readout_1.00` | 0.992 | 0.980 | 0.968 | 0.949 | 0.623 | **0.00** |
| `founder2` | 0.810 | 0.686 | 0.330 | 0.321 | 0.497 | 1.19 |

At `readout_1.00` every motor channel is pinned near 1.0 — she walks, pecks, crouches and
flees flat out, simultaneously and permanently. No hen is caught in 144 dives, where a
default flock loses 1.44 each. This is E027's documented failure exactly ("hens crouched
permanently and were struck 0 times in 3000 steps at a predator rate that had produced
1000") and CLAUDE.md's own saturation entry: her neurons are all firing flat out, so
nothing can stand out against the background. **Perfectly repeatable, because being
broken is reproducible.** §4's falsifier fires and the arm is recorded as degenerate.

Prediction 1 said amplitude would take repeatability above +0.30 and it did. It was right
for a reason that makes it useless, which is why §4 existed.

### 6d. The dissociation is real but only partial

`shared_shift` multiplies the readout's magnitude by 5 — the same factor as
`readout_0.25` — by adding a *constant* to every hen, so between-hen `W_out` spread stays
at 0.0445 against `readout_0.25`'s 0.2225. It reaches the same operating point (hunger
0.4928 vs 0.4819; mean fitness −1.400 vs −1.473) with **no added individual differences at
all**.

It scores **+0.172 ± 0.049, t=3.57** — small, but not zero, and block-inconsistent
(+0.069 then +0.272). So moving the flock off the reflex arc's operating point does
produce *some* repeatability by itself. But it accounts for **+0.187 of `founder2`'s
+0.752** contrast against default — roughly a quarter. **Three-quarters of the effect is
individual differences being made audible; a quarter is the operating-point shift.** My
block-1 reading, where `shared_shift` was flat at +0.069, would have claimed a clean
dissociation that the second block does not support.

### 6e. Anatomy varies and nothing happens

`per_hen_mask` gives each hen her own connectivity draw from the same region-pair
probabilities. It genuinely varies: **22.65% of synapses differ between hen 0 and hen 1**,
against **0.00%** for every other arm. Repeatability pooled: **+0.055 ± 0.050, t=1.10** —
null, in both blocks.

The manipulated variable moved a long way and the dependent variable did not move at all.
Different wiring does not make a hen's outcome hers, because the wiring still speaks
through the same near-silent readout into the same shared reflex arc.

### 6f. Getting caught is not a property of the hen

> **Corrected by [E119](E119-can-an-alarm-channel-be-selected-for.md): true at
> `hawk_period_s=50`, false at 10.** Everything in this section was measured at one
> predator density, and the conclusion drawn from it — that the H0 ladder is blocked —
> was drawn one density too early. At 10 s, r(caught) is **+0.478 ± 0.079 (t=6.57)**
> against the +0.049 measured here. Catches are rare events sampled around a hen-specific
> rate; at 1.75 catches per life the sampling noise buries the rate, at 4.67 it does not.
> **Eleven measurements of the same under-powered configuration are one measurement.**
> The numbers below stand as measured; the generalisation does not.

Predation repeatability was measured in all seven arms across both blocks — **eleven
arm-by-block measurements, and not one reaches significance.** Block 1 scatters slightly
negative (−0.110 to +0.099), block 2 slightly positive (−0.008 to +0.174, largest t=1.98).

At `hawk_period_s=50` predation supplies ~74% of `evolve.fitness`'s variance. So roughly
three-quarters of the criterion E116 selects on carries no information about the hen being
selected. That is not a subtle statistical point: **selection on `evolve.fitness` spends
most of its selection differential on luck**, which is precisely the pathology E116's own
first smoke test hit ("selection concentrates luck while mutation degrades a tuned
connectome").

### 6g. A single-block result I withdrew

Block 1 gave `explore_0.1` a **significantly negative** fitness repeatability, −0.118 ±
0.033, t=−3.63: a hen who did well once did *worse* on the replicate. I built a diagnostic
for it, hypothesising competition — `food_deplete_rate=2e-2` makes food a shared
exhaustible resource — and the diagnostic looked confirmatory: switching depletion off
moved that arm from −0.118 (t=−3.63) to −0.015 (t=−0.13), while `founder2`'s drives
repeatability was untouched (+0.501 → +0.519).

**Block 2 gives the same arm +0.138 ± 0.114, and pooled it is +0.010 ± 0.066.** The
negative correlation did not replicate, so there was nothing for the competition story to
explain. **The mechanism is withdrawn**, and the depletion measurement is retained only as
what it is: an unremarkable observation that removing depletion does not change
repeatability much either way. E021 is the precedent and this is the same shape — a t of
−3.63 on one block of seeds, gone on the next.

I am recording this because the diagnostic *worked*, in the sense of producing a clean
confirmatory-looking number for a hypothesis about noise.

### 6h. An inconsistency found while smoke-testing, unrelated to the result

`connectome.build` draws recurrent weights up to **1.387** in magnitude while
`plasticity.w_max` is **0.5**, so ~0.025% of synapses (about 1,000 per flock) are clipped
the first time `_enforce_dale` runs — which for an evolutionary run is generation 0's
mutation, and for a plastic run is the first consolidation. It is not new and not caused by
anything here; it applies identically to every arm and every condition, so it confounds
nothing in this experiment. Recorded because it means "mutation with σ=0" is not the
identity operation, which is how it was found.

## 7. Interpretation

**What a naturally built flock is.** `connectome.build` gives every hen the same mask, the
same `W_in`, the same `dale`, the same biases and the same reflex arc, and differs them
only in the magnitudes of `W` and `W_out`. `W_out` is then scaled by `readout_scale=0.05`
so the cortical pathway "starts nearly silent and has to *earn* influence". The result is
a flock whose members are individuals on paper and interchangeable in behaviour: fitness
repeatability **−0.014 ± 0.053** over 16 genomes. **This is not a population in the sense
any adaptive process needs.**

**Where the variation has to come from, then.** Not from anatomy — `per_hen_mask` varied
22.65% of synapses and produced nothing. It comes from *how loudly the individual part of
the brain speaks*, and the two arms that worked (`founder2`, `readout_0.25`) both work by
turning that volume up. `founder2` does it by perturbing the weights themselves,
`readout_0.25` by scaling the channel they speak through; they land within one standard
error of each other on the drives metric (+0.628 and +0.511).

**And the volume knob has a ceiling that is close.** `readout_1.00` is only four times
`readout_0.25` and it destroys the bird. E002's original calibration of `eta_out` found
the same wall from the other side — "at 2e-1 cortical drive overwhelms the innate arc and
behaviour gets worse; a hen who overrides her reflexes with an untrained pallium is worse
off than one who does not". **The usable band between "inaudible" and "saturated" is
roughly one order of magnitude**, and both E116's founder mutation and E117's readout
scaling are sitting inside it, near enough to the top that mean fitness already degrades
(−1.126 → −1.41/−1.47).

That is the honest shape of the finding, and it is uncomfortable: **selectability and
competence trade off against each other here.** Every arm that made hens distinguishable
also made them worse. Nothing tested made a flock both varied and healthy.

**What I cannot rule out.** The operating-point shift is a real contributor (+0.172,
t=3.57) and is confounded with individual differences in both successful arms, because
raising the volume moves the mean as well as the spread. Separating them properly needs an
arm that raises between-hen spread while holding the *mean* readout magnitude fixed —
scaling deviations from the flock mean rather than adding to them. That is a clean
follow-up and it was not run here.

Also unaddressed: `W_in`, `dale`, `b` and the reflex arc stay flock-wide in every arm,
because per-hen afferents would need `brain.step`'s `obs @ p.W_in.T` to become a batched
einsum. Given that per-hen *recurrent* anatomy did nothing, per-hen afferents are not an
obvious next lever, but they are untested and they are the input side rather than the
recurrent side.

**For the H0 ladder specifically.** Predation repeatability is zero, and predation is what
an alarm channel is *for*. A generational contrast of an intact channel against a yoked one
is scored on a fitness whose predation term carries no information about the hen. Either
the criterion changes, or the predator statistics change enough that catches become a
property of the bird, before that ladder is worth running. This is exactly CLAUDE.md's
"before the run, not after: does the instrument work?" and the answer today is that it
does not.

## 8. Consequence

**`docs/hypothesis.md`.** H0 records that generational selection's prerequisite is
quantified: a flock from `connectome.build` has fitness repeatability −0.014 ± 0.053 over
16 genomes, the variation that fixes it must be made *audible* rather than merely present,
and the band between inaudible and saturated is about one order of magnitude wide.

**`docs/backlog.md` §8.** E116's follow-up 1 is answered and closed. Three new items: the
mean-preserving variance arm that separates spread from operating point; the fitness
criterion's predation term carrying no signal; and a note that per-hen anatomy is a
measured dead end rather than an untried idea.

**Nothing changes in `hen/connectome.py`.** No arm here earned adoption as a default:
`readout_0.25` buys repeatability at 0.04 hunger units, and this experiment was designed to
locate the constraint, not to pick a value. Picking one is E118/E119's problem, with a
criterion that is not three-quarters noise.

**Two corrections to earlier work**, both recorded above rather than quietly fixed: E116's
+0.473 stands as high-but-not-refuted (6b), and this experiment's own block-1 competition
mechanism is withdrawn (6g).

**Code.** `scratchpad/e117_standing_variation.py`, `scratchpad/e117b_diagnostics.py`,
`scratchpad/e117c_pool.py`. No change to any module under `hen/`, `coop/` or `run/` from
this experiment.

