# E085 — repairing T2's instrument, and establishing what it can resolve

*Sections 1–5 written and committed before anything was run. A design-informing
diagnostic preceded this and is reported in §5 as an input, explicitly not as a result.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **T2** → **T2-revised**. Direct successor to
[E084](E084-live-place-decoding.md).

**This experiment makes no behavioural claim.** Its entire output is an instrument and a
number saying what that instrument can resolve. E086 will use both.

---

## 2. Question

T2-revised has two independent, individually sufficient reasons for producing nulls, and
neither is about the hen:

- **No valid treatment.** The plant is anti-selective in the live run — 0.656 at the
  planted feeder against 1.244 elsewhere (E083).
- **No valid measurement.** Occupancy at a fixed cell has a minimum detectable effect of
  18.3% of baseline at n=4, against the 15% both E082 and E083 pre-registered (E084).

Four behavioural experiments have measured instrument properties. This one fixes the
instrument deliberately instead of discovering it by accident, and answers three
questions:

**A. Does the repaired target-selection procedure work?** Choose the target feeder per
seed from an independent baseline run, so every seed starts where the flock actually is
and has room to fall.

**B. What can the metric resolve?** Establish the minimum detectable effect for occupancy
and for a new candidate, **mean dwell per visit**, from a null–null contrast that does not
consume the treatment.

**C. Is position linearly decodable during free movement?** E084's Part A, unanswered —
its run crashed on a degenerate split at seed 1, and one seed completed at chance.

---

## 3. Prediction

1. **Target selection holds.** Occupancy at the per-seed selected cell stays above 0.15 in
   the test run on every seed, and the drift from selection run to test run is symmetric
   rather than systematically downward.
2. **Dwell per visit resolves better than occupancy.** MDE for mean dwell at n=8 is below
   **25%** of baseline, and below occupancy's MDE on the same runs.
3. **Decodability fails its gate.** Held-out balanced accuracy below 70%. I expect this
   and am recording it as the prediction rather than the hoped-for outcome: E084's one
   completed seed gave 44.4% live-fit against 44.8% parked-fit, and the pallial state
   during free movement is dominated by hunger, pecking, flockmates and calls, with the
   place cells given no innate anchors by construction (E063).
4. **The parked-fit plant also fails**, at or below chance, reproducing E083's diagnosis
   on data that did not generate it.
5. **`pred_gain` changes every seed's trajectory.** E084 seed 4 returned bit-identical
   occupancy at gains 0.0 and 2.0, which is the signature of E070's `enabled=False`
   defect. Either it reproduces and is a bug, or it does not and was a property of that
   plant.

## 4. Falsifier

**Selection falsifier.** Occupancy at the selected cell drops below 0.15 on any seed in
the test run, or the mean drift across seeds is worse than −20% relative. E084's repair
recommendation would then not work, and the target would need to be *created* — food
placed where the flock already is — rather than found.

**Resolution falsifier.** MDE for mean dwell at n=8 exceeds 25% of baseline. Dwell is then
no better than occupancy and the instrument problem is not solved by changing the
dependent variable. This is the one I most want to know about, because it decides whether
E086 can exist.

**Decodability gate.** Held-out balanced accuracy < 70% at the per-seed target radius,
**or** selectivity ratio < 2.0, **or** the distance profile not decreasing across the
first three bins. If it fires, **mechanism 2 is insufficient as built**: the shared
allocentric population is readable when the hen is parked and not while she moves, which
is the regime the hypothesis needs. That would be the first result in this arc genuinely
about the hen.

**Diagnosis falsifier.** Parked-fit held-out accuracy ≥ 70%, which would overturn E083's
account of why the plant inverted.

**Leakage falsifier.** Train accuracy exceeds held-out by more than 15 points.

**Determinism falsifier.** Any seed returns bit-identical behaviour at two different
`pred_gain` values. That is a bug, not a result, and blocks everything downstream until
explained.

---

## 5. Design

### Input from the design diagnostic (`scratchpad/e085_metric_design.py`)

Reported here as a design input, **not** as a result of this experiment. It established
two facts the design depends on:

**Clump location is a property of the world key, not the dynamics RNG.** Occupancy-vector
correlation (25 cells) between two runs sharing a world key but differing in run key:
**0.963, 0.996, 0.992, 0.994**. Between runs differing in both: −0.252, 0.550, 0.649,
0.512. So a target chosen in one run is still where the flock is in another run of the
same world.

**Dwell per visit is well-sampled**: 142–510 visits per run at 17–75 s mean.

### Structure

Per seed *s* ∈ 0…7 (n=8, where E084 measured occupancy resolves 9.6%), one connectome
(`gakel_scaffold=True`, `shared_place_map=True`), one world key *k*, three run keys:

- **selection run** — `fold_in(k, 2)`, `pred_gain=0`, no plant.
- **test run** — `fold_in(k, 9)`.
- **null-twin run** — `fold_in(k, 11)`.

**Target selection.** Target = the cell with highest occupancy in the *selection* run.
Control = the second-highest, so the agitation falsifier has a cell with real traffic to
test against rather than an arbitrary one. Both fixed before any treatment is applied.

**Part A — selection validity.** Occupancy at the target in the test run, against the
selection run, per seed.

**Part B — resolution.** MDE from a **null–null contrast**: the same null condition run on
the test key and the null-twin key. The paired difference between two runs differing only
in dynamics RNG *is* the run-to-run noise any treatment must exceed. This deliberately
does **not** use the treatment, so E086's threshold is not derived from E086's own data —
which is the error the standing correction in E084 §8 was written to prevent. Reported for
occupancy and for mean dwell per visit, with MDE at n=4 and n=8.

**Part C — decodability.** As E084's Part A, with two changes. The label radius is
centred on the **per-seed target**, so the degenerate split that crashed E084 cannot
occur by construction. And states are sampled from the selection run and evaluated on the
test run — same world, different trajectory. Estimator unchanged from E082's
difference-of-means, so any difference is attributable to regime rather than to a better
fitting method; state is `z_lag − z_lag_bar` masked by `pred_src`, what the runtime
consumes. Parked-fit plant evaluated on the same held-out data for comparison.

**Determinism check.** For one seed, assert the test run at `pred_gain=0` and
`pred_gain=2` differ. Runs are deterministic, so bit-identity means the gain reached
nothing.

### Cost

~20 minutes. 8 seeds × 3 runs, no gain ladder.

---

## 6. Result

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*
