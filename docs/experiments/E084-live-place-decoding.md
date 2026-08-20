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

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*
