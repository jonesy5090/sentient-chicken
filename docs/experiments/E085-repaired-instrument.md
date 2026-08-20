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

8 seeds, 24 runs, 1064 s. Per seed the target and control cells are chosen from the
selection run, so they differ between seeds by design.

| seed | target | ctrl | occ sel | occ test | dwell test (s) | visits | live acc | parked acc |
|---|---|---|---|---|---|---|---|---|
| 0 | 2 | 7 | 0.619 | 0.552 | 74.67 | 142 | 50.9% | 50.9% |
| 1 | 5 | 6 | 0.449 | 0.405 | 24.86 | 313 | 50.6% | 50.2% |
| 2 | 11 | 10 | 0.404 | 0.466 | 17.38 | 515 | 57.4% | 57.2% |
| 3 | 3 | 2 | 0.639 | 0.717 | 34.58 | 398 | 53.7% | 46.6% |
| 4 | 24 | 23 | 0.832 | 0.827 | 278.71 | 57 | 66.4% | 66.4% |
| 5 | 2 | 7 | 0.965 | 0.971 | 716.78 | 26 | 84.9% | 15.1% |
| 6 | 6 | 10 | 0.498 | 0.458 | 23.52 | 374 | 53.6% | 46.4% |
| 7 | 6 | 2 | 0.389 | 0.419 | 27.86 | 289 | 59.6% | 56.4% |

**Part A — selection validity: clear.** Occupancy at the target is 0.600 in the selection
run and **0.602** in the test run, minimum 0.405. Drift is **+0.8%** on average and
symmetric per seed (−11%, −10%, +15%, +12%, −1%, +1%, −8%, +8%) — scatter, not regression.
**The repair E084 recommended works.**

**Part B — resolution.** Null–null contrast, no treatment involved:

| metric | baseline | diff sd | MDE n=4 | MDE n=8 |
|---|---|---|---|---|
| occupancy at per-seed target | 0.6020 | 0.0365 | **9.7%** | **5.1%** |
| mean dwell per visit | 149.80 s | 133.05 | 141.3% | 74.3% |

**The resolution falsifier fires, and prediction 2 was wrong.** Dwell is far *worse* than
occupancy, not better. But the headline is the other column: **occupancy at a per-seed
target resolves 9.7% at n=4 and 5.1% at n=8**, against E084's **18.3% at n=4** for a fixed
cell. Target selection alone roughly halved the minimum detectable effect.

**Part C — decodability: the gate fires.**

| | held-out acc | vs chance | ratio | profile decreasing |
|---|---|---|---|---|
| live-fit | 59.6% | +9.6 ± 4.1, t=+2.38 | 1.70 | no |
| parked-fit | 48.7% | −1.3 ± 5.3, t=−0.25 | 0.60 | no |

Gate requires ≥70%, ratio ≥2.0, decreasing profile. All three fail. Leakage clear (train
57.6% vs held-out 59.6%, −2.1 points). Diagnosis falsifier clear.

Distance profile on held-out data:

| bin (m) | 0–1 | 1–2 | 2–3.3 | 3.3–5 | 5–7 | 7–10 | 10+ |
|---|---|---|---|---|---|---|---|
| live-fit | **0.653** | 1.050 | 1.077 | 0.838 | 1.011 | 1.282 | 1.089 |
| parked-fit | **0.486** | 0.671 | 0.672 | 0.941 | 1.319 | 1.522 | 1.390 |

### 6b. Post-hoc: the headline accuracy is mostly class imbalance

*Not pre-registered.* The 59.6% clears chance by a hair (t=+2.38 against t(7)=2.365), and
per-seed accuracy correlates with how skewed that seed's split is at **r=+0.870**. Seeds
4 and 5 have the hen at the target 83% and 97% of the time, where balanced accuracy on a
97/3 split is unstable; they contribute 66.4% and 84.9%.

On the **six seeds with balanced splits** (occupancy within 0.25 of 0.5): **54.3%**,
+4.3 ± 1.5, **t=+2.95** against t(5)=2.571.

So the effect is real and it is *small*: about four percentage points above chance. The
larger number is an artefact of measuring balanced accuracy on degenerate splits. This
subset was chosen on a methodological argument rather than by looking at the outcomes, but
it is still post-hoc and wants fresh seeds before it is leaned on.

### 6c. Two engineering faults, recorded because they share a failure mode

Both presented as **exit code 0 with an empty output file**. Python block-buffers when
piped, so a `SIGKILL` discards everything unflushed, and a crash that reports success and
produces no output is indistinguishable from a legitimate empty result.

1. **Unguarded module driver.** Importing `e083_leaving_anchor` re-ran its full 15-minute
   gain ladder as an import side effect. The first fix — `raise SystemExit` at module
   scope — was worse, killing the importing process with exit code 0. Both drivers are now
   behind `if __name__ == "__main__"`.
2. **OOM.** Emitting `z_lag − z_lag_bar` from every one of 120 000 scan steps allocates
   120000 × 16 × 512 × 4 B = **3.9 GB per run**, three runs per seed, against 9 GB
   available. Replaced with a nested scan: the inner loop emits only the occupancy bitmap
   every step (dwell needs every step), the outer loop emits the 512-wide pallial state
   once per 100-step block. ~90 MB per run.

Everything is now run with `python -u`.

## 7. Interpretation

**The instrument is fixed, and it was fixed by target selection rather than by changing
the dependent variable.** Choosing the target per seed from an independent run removes the
ceiling/floor problem and most of the clump-location variance, and it does so without
introducing regression, because both arms share a test run key and the selection run is
not one of them. Occupancy at a per-seed target resolves **5.1% at n=8**. E086 can be run.

**Mean dwell per visit is a bad statistic here, for a reason worth keeping.** It is
heavy-tailed: seeds 4 and 5 give 279 s and 717 s from 57 and 26 visits, because a hen who
essentially never leaves produces one enormous visit that dominates the mean. Conditioning
on arrival sounded like it would remove clump-location dependence; instead it concentrated
it. I predicted the opposite and was wrong.

**Position is linearly decodable during free movement, and far too weakly to matter.**
54.3% on balanced splits — about four points above chance, reliable (t=2.95) and tiny.
E081 measured 84.6% on parked states. The two are not the same task (E081 discriminated
five parked point-locations; this discriminates within-3.33 m-of-target from not, during
free movement), so the numbers should not be subtracted. But the direction and the size of
the gap are the point: **the place code is legible when she is standing still and nearly
illegible while she is moving.**

**Even the live-fitted discriminant fires least when she is closest to the target** —
innermost bin 0.653, the lowest of seven. Fitting on live states removed the *inversion*
seen in the parked-fit profile but did not produce selectivity. Whatever the difference-
of-means direction captures, it is not proximity.

**E083's diagnosis is confirmed on data that did not generate it.** The parked-fit plant
scores 48.7% held-out — chance — with selectivity ratio **0.60** and an increasing distance
profile, reproducing E083's 0.53. Prediction 4 correct, diagnosis falsifier clear.

**Why this is a fact about the representation and not another instrument problem.**
`W_pred` is a linear readout; a linear decoder is therefore the relevant class, and the
same estimator recovers 84.6% from the same population when the hen is parked. The place
cells are present and working. What defeats them is that pallial state under free movement
is dominated by hunger, pecking, flockmates and calls, and E063 deliberately gave the place
channels no innate anchors — so nothing amplifies them into the variance a linear readout
can find. That is the prediction §3 recorded, and it held.

**One limitation, stated plainly.** The determinism check ran on **seed 0** and cleared —
`pred_gain` does change the trajectory. E084's anomaly was on **seed 4**, under a different
target and a fixed-cell design. So that anomaly is **not resolved**, only shown not to be
universal.

## 8. Consequence

**T2-revised's mechanism 2 is insufficient as built.** This is the first result in this arc
that is about the model rather than about our measurement of it. The shared allocentric
population is readable when the hen is parked and carries ~4 points of linearly decodable
signal while she moves — nowhere near enough for `W_pred` to bind a place to a call.

**E086 is unblocked on the metric and blocked on the representation.** The instrument now
resolves 5.1% at n=8, so a behavioural contrast is finally worth running — but there is no
point running one until there is a place signal for the association to attach to.

**Three routes, in the order I would try them.**

1. **Give the place channels an innate anchor.** The one structural asymmetry that made
   H2f work was that the rule amplifies what innate wiring already emphasises (E058/E059,
   E069). Place cells were deliberately given none. A weak innate place→pallium projection
   would raise position's share of pallial variance without wiring *which* place is
   aversive — the production-innate / usage-learned split used everywhere else here.
2. **Widen the place population.** 25 cells at `place_sigma` 2.0 in a 20 m arena is coarse,
   and `OBS_DIM` growth is cheap.
3. **Accept the null and re-scope T2.** If position cannot be made legible under movement
   without distorting the model, then T2 as literally stated — durable avoidance of *this
   specific feeder* — is not reachable in this architecture, and that is a publishable
   finding rather than a failure.

**Do not run E086 against the current representation.** It would produce a fifth null with
a known cause, which is the category of experiment this arc has already run four times.

**Adopted for future work:** the null–null noise floor. Estimating a metric's resolution
from two null runs costs one extra run, does not consume the treatment, and would have
caught E082's and E083's under-powering before either ran.
