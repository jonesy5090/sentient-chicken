# E084 — is *where she is* linearly decodable from pallial state while she is moving?

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **T2** → **T2-revised**, mechanism 2 (the shared allocentric
place population, E063). Direct successor to
[E083](E083-gakel-anchor-produces-leaving.md).

---

## 2. Question

E083 found that the plant E082 and E083 both relied on is **anti-selective in the live
run** — 0.656 at the planted feeder against 1.244 elsewhere, with the innermost distance
bin the lowest of seven and the peak in a ring 5–7 m away. The cause it identified is a
regime mismatch: the discriminant is fitted on a hen **parked** at a grid centre and read
back on a hen **moving**.

That diagnosis has an obvious remedy — fit on live states — and an obvious risk, which is
the actual question here. E081's 84.6% decodability, the number that unblocked this whole
route, was itself measured on parked hens at five cell centres under 0.35 m of jitter.
**Nobody has ever measured whether position is linearly decodable from pallial state
during free movement.** T2-revised needs exactly that and has been assuming it for four
experiments.

So, in two parts:

**Part A.** Fit the discriminant on live trajectory states labelled by position. On a
**held-out run** — same connectome, different trajectory — can it tell *at the feeder*
from *elsewhere*? And does the parked-fit plant, evaluated on the same held-out data,
fail as E083's diagnosis predicts?

**Part B.** Only if Part A clears its gate: plant the live-fitted discriminant and re-run
E083's behavioural contrast.

Part A is not a preliminary. If it fails, the finding moves from the instrument to the
representation, and it is the first result in this arc that is about the hen rather than
about our measurement of her.

---

## 3. Prediction

### Part A

1. **The live-fitted discriminant clears the gate**: held-out balanced accuracy ≥ **70%**
   at the grid-spacing radius (3.33 m), selectivity ratio (mean `relu(pred)` at the
   feeder ÷ elsewhere) ≥ **2.0**, and a distance profile that **decreases** across the
   first three bins.
2. **The parked-fit plant fails the same gate on the same data** — balanced accuracy near
   chance (50%) and selectivity ratio below 1.0. This is E083's diagnosis stated as a
   prediction, tested on data that did not generate it.
3. Accuracy is **higher at the tight radius (1.5 m)** than at 3.33 m, because
   `place_sigma` is 2.0 and a 3.33 m disc mixes in places whose place code differs
   substantially.

I hold prediction 1 with much less confidence than 2. The pallial state during free
movement is dominated by hunger, pecking, flockmates and calls; position is one input
among many, and the place cells were given no innate anchors by construction (E063).

### Part B

Occupancy at the planted feeder falls monotonically with `pred_gain`, by ≥15% relative at
gain 2.0 — E083's prediction unchanged, now with an instrument that has been shown to
work.

---

## 4. Falsifier

**Part A gate (primary).** Held-out balanced accuracy < 70% at 3.33 m, **or** selectivity
ratio < 2.0, **or** the distance profile not decreasing across the first three bins.

If this fires, **T2-revised's mechanism 2 is insufficient as built.** The shared
allocentric population would be present and measurable when the hen is parked, and not
linearly readable while she moves — which is the regime the whole hypothesis needs. Part B
does not run, and the consequence is a claim about the representation, not the plumbing.

**Diagnosis falsifier.** The parked-fit plant scores ≥70% balanced accuracy on held-out
live data. E083's account of *why* the plant inverted would then be wrong, and the
inversion would need a different explanation before anything is built on it.

**Leakage falsifier.** Train-run accuracy exceeds held-out accuracy by more than 15
points. The estimator is memorising a trajectory rather than reading a position, and no
threshold on the held-out number means anything.

**Part B falsifiers.** Unchanged from E083 §4 — primary, agitation, starvation, reflex.

---

## 5. Design

### Part A

`scratchpad/e084_live_place_decoding.py`. Per seed *s* ∈ {0,1,2,3}, one connectome
(`gakel_scaffold=True`, `shared_place_map=True`), and **two runs that differ only in the
world key**:

- **train run** — world key `fold_in(key(s), 2)`, the key E082/E083 used.
- **held-out run** — world key `fold_in(key(s), 7)`. Same connectome, same food layout,
  different trajectory.

Both at `pred_gain=0` with no plant installed, so the collection is not driven by its own
output. 20 simulated minutes each, states sampled every 100 steps (1 s) → 1200 samples ×
16 hens = 19 200 vectors per run.

The sampled state is `z_lag − z_lag_bar` masked by `pred_src` — **exactly what the runtime
readout consumes**, not raw `rate(x)`. This is the correction E082's first run needed and
E081 never applied.

**Estimator, deliberately unchanged from E082's:** `w = mean(at feeder) − mean(elsewhere)`,
masked and normalised. Only the *data* changes, so any difference is attributable to
regime and not to a better fitting method.

**Evaluated on the held-out run**, with the decision threshold taken from the **train**
run's class means so nothing is fitted on test data:

- balanced accuracy, at radius 3.33 m and 1.5 m;
- selectivity ratio, mean `relu(pred)` at the feeder ÷ elsewhere;
- the 7-bin distance profile, per-seed normalised — the three diagnostics that caught the
  inversion in E083, so the comparison is like-for-like;
- the same three for the **parked-fit plant** on the same held-out data;
- train-run accuracy alongside held-out, for the leakage falsifier.

### Part B

`scratchpad/e083_leaving_anchor.py` with `plant()` swapped for the live-fitted
discriminant, and the pre-flight replaced by a **selectivity** assertion — fires at the
feeder **and** near-silent elsewhere, profile decreasing — rather than the amplitude-only
assertion that has now passed twice while the plant was useless. Everything else matched
to E083: 4 seeds, same gain ladder, same arena, same anchor.

### Cost

Part A ~8 minutes (8 runs, no gain ladder). Part B ~15 minutes, and only if the gate
clears.

---

## 6. Result

**Part A did not complete, and the reason it did not complete is the result.**

The run crashed on seed 1 with a degenerate split: in the held-out run, **0 of 19 200
samples** fell within 3.33 m of the planted feeder. Sixteen hens, twenty simulated
minutes, and not one of them went near it — while the training run of the same
connectome had a base rate of 0.424. The guard that raised this was added in the same
session, replacing a silent `NaN` that the first attempt had propagated into every
aggregate.

Seed 0 completed before the crash, and both plants performed the same:

| radius | fit | held-out acc | train acc | ratio |
|---|---|---|---|---|
| 3.33 m | live | 44.4% | 48.2% | 2.40 |
| 3.33 m | parked | 44.8% | — | 2.43 |
| 1.50 m | live | 47.6% | 45.1% | 2.53 |
| 1.50 m | parked | 49.6% | — | 2.53 |

One seed is not a result, and **the pre-registered Part A gate cannot be evaluated** — the
run never reached four. But chance is 50%, every cell is at or below it, and the live-fit
and parked-fit numbers are within a point of each other, which is not what E084 was built
expecting to see.

### 6b. Diagnostic — the flock aggregates, and occupancy is a property of the world key

*Post-hoc, not pre-registered.* `scratchpad/e084_coverage.py`, free-running, no plant:

| seed | world key | cells visited (of 25) | occ P | occ P′ | flock spread (m) |
|---|---|---|---|---|---|
| 0 | fold 0 | 24 | 0.481 | 0.004 |3.30 |
| 0 | fold 6 | 20 | 0.043 | 0.485 | 3.17 |
| 1 | fold 0 | 18 | 0.144 | 0.193 | 5.18 |
| 1 | fold 6 | 14 | **0.000** | 0.750 | 1.66 |
| 2 | fold 0 | 23 | 0.198 | 0.322 | 7.21 |
| 2 | fold 6 | 15 | 0.030 | 0.651 | 2.91 |
| 3 | fold 0 | 22 | 0.338 | 0.068 | 3.67 |
| 3 | fold 6 | 17 | 0.381 | 0.414 | 4.06 |

Coverage is not the problem — the flock reaches 14–24 of 25 cells. **Aggregation is.**
Spread runs 1.66–7.21 m in a 20 m arena: the hens clump, the clump settles where it
starts, and occupancy of any named cell follows from that. Changing only the world key
moves occupancy at P from 0.000 to 0.481.

### 6c. Diagnostic — the metric cannot resolve the effect E082 and E083 predicted

*Post-hoc, not pre-registered.* `scratchpad/e084_power.py`. E082 and E083 pair across
gains on shared seeds, so between-seed variance partly cancels and the raw spread above
is not the number that matters. The number that matters is the sd of the **within-seed**
difference. Run at E083's exact metric, gains 0.0 and 2.0, **8 seeds** — double E083's
block:

| seed | occ @ gain 0 | occ @ gain 2 | diff |
|---|---|---|---|
| 0 | 0.6193 | 0.6309 | +0.0116 |
| 1 | 0.2579 | 0.2201 | −0.0378 |
| 2 | 0.2760 | 0.3410 | +0.0650 |
| 3 | 0.5450 | 0.6588 | +0.1138 |
| 4 | 0.0948 | 0.0948 | +0.0000 |
| 5 | 0.9651 | 0.9586 | −0.0065 |
| 6 | 0.2727 | 0.3323 | +0.0596 |
| 7 | 0.3644 | 0.3752 | +0.0108 |

Baseline mean **0.4244**, sd **0.2751** — sd/mean **0.65**, individual seeds spanning
0.0948 to 0.9651, a tenfold range. Pairing helps a great deal: the paired difference has
sd **0.0487**.

| n | t crit | min detectable diff | as % of baseline |
|---|---|---|---|
| 4 | 3.182 | 0.0775 | **18.3%** |
| 8 | 2.365 | 0.0407 | 9.6% |

**E082 and E083 both pre-registered a 15% effect at n=4, against a minimum detectable
effect of 18.3%.**

Observed difference at n=8: **+0.0271 (+6.4%), t=+1.57** — not significant, and positive.

## 7. Interpretation

**E082's and E083's primary falsifiers were guaranteed to fire before either experiment
ran.** Their pre-registered effect (15%) sits below the resolution of the metric at the
sample size they used (18.3%). A real, exactly-as-predicted 15% avoidance effect would
have been reported as a null both times. This is `CLAUDE.md` check 6 — *would a positive
result be detectable at all?* — and the answer was no, on arithmetic available before
either run.

**So the T2 arc now has two independent and individually sufficient explanations for its
nulls, and neither is about the hen.** E083 found the plant anti-selective, so there was
no valid treatment. E084 finds the metric under-resolved, so there was no valid
measurement. Either alone accounts for everything E082 and E083 reported.

**The mechanism behind the variance is the flock's own cohesion.** `approach_flockmates`
is innate and works; the consequence is that sixteen hens behave as roughly one clump,
so the effective sample size for a spatial metric is far closer to *one animal per seed*
than to sixteen. Where that clump settles is set by initial conditions. Occupancy at a
fixed, pre-chosen cell therefore inherits the full between-run variance of clump
location.

**Two seeds are also at the boundary**, which is `CLAUDE.md` check 5 — *does the dependent
variable have room to move in both directions?* Seed 5 sits at 0.9651, where a treatment
that raised occupancy could barely register; seed 4 at 0.0948, where one that lowered it
could barely register. Choosing the target cell in advance guarantees that some seeds
will have almost no room in the direction of interest.

**One anomaly recorded rather than chased.** Seed 4 returns occupancy 0.0948 at both
gains — bit-identical, so `pred_gain` changed that trajectory not at all. The runs are
deterministic, so this means `relu(pred)` on the gakel channel was zero throughout that
run. It is consistent with the anti-selective plant, but it is not explained, and it is
the same signature as E070's `enabled=False` defect, which was a real bug. Worth a check
before the next behavioural contrast.

**Part A's own question is still open.** One seed at chance, with live-fit and parked-fit
indistinguishable, is a hint and nothing more. It is *not* evidence that fitting on live
states fails — and note it also fails to reproduce E083's parked-fit selectivity ratio of
0.53 (seed 0 held-out gives 2.43), which is itself a sign that these quantities swing with
the world key as much as everything else here does.

## 8. Consequence

**Nothing about T2-revised's mechanisms has been tested yet.** Not mechanism 1, not
mechanism 2. Four behavioural experiments (E070, E082, E083, and E084's Part B, unrun)
have measured instrument properties.

**The metric must be fixed before any further behavioural contrast, and the fix follows
from the diagnosis.** Choose the target feeder **per seed, from an independent baseline
run** — the cell that flock actually occupies — rather than fixing it in advance. Then
every seed starts with high occupancy at the target and has room to fall, which removes
the ceiling/floor problem and most of the clump-location variance at once.

Selecting the target from run key A and running **both** arms on run key B keeps
regression-to-the-mean identical in the two arms, so it cancels in the paired difference
rather than masquerading as an effect. **And n=8 minimum**, where the metric resolves
9.6%.

**Then, and only then, Part A and Part B.** Part A needs the degenerate-split case handled
rather than crashed on — with a per-seed target cell it largely disappears, since the
target is by construction somewhere the flock goes.

**Standing correction to how this arc pre-registers effect sizes.** E082 and E083 both
named a threshold without checking the metric could resolve it. That check is arithmetic
on a variance estimate and costs one run of the baseline condition. It should precede any
future falsifier that names a percentage.
