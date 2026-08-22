# E088 — a frozen centring baseline: both selectivity and signal, or neither

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **T2** → **T2-revised**, the `W_pred` readout. Direct successor to
[E087](E087-prediction-centring-timescale.md).

---

## 2. Question

E087 established that the prediction pathway's centring costs ~20 points of place
decodability, that the cost is the baseline **tracking** rather than its timescale or its
scope, and that a **constant** baseline sits in a better place on the trade-off:

| baseline | selectivity (E070's measure) | decodability under movement |
|---|---|---|
| none (raw `z_lag`) | 1.04 — E070's failure, replicated | 90.0% |
| **constant** | **5.00** | **89.8%** |
| EMA, τ 20 s (the runtime) | 23.28 | 73.7% |

But the constant used in that measurement was **the mean across settled place states**,
which requires knowing the places in advance. It is a diagnostic, not a mechanism — nothing
in a running simulation can compute it.

**Does a baseline that tracks for a calibration period and then freezes deliver the same
thing?** It is implementable causally, and it is biologically unembarrassing: a
developmental calibration of what "average activity" means, fixed once rather than
re-estimated forever.

---

## 3. Prediction

1. **Decodability recovers**: ≥85% on balanced-split seeds, against the runtime's 73.7%
   and the uncentred ceiling of 90.0%.
2. **Selectivity survives**: ratio ≥2.0, the same bar E087 used. I expect it near the
   constant baseline's 5.00 rather than the EMA's 23.28, and I am recording in advance that
   **a fall from 23.28 to ~5 is an accepted cost, not a failure** — 5× separation against
   E070's 1.04 is a working readout, and the decodability it buys is what the association
   needs.
3. **Both at once, on one configuration.** E087's whole lesson is that these trade, so
   neither number counts unless the other is measured on the same setting.
4. **Freeze time matters and has an optimum.** Too early and the baseline freezes before it
   represents anything; too late and it has already tracked the signal away. I predict the
   best freeze time is **1–3× `baseline_tau_s`** (20–60 s).
5. **Default inert.** `pred_bar_freeze_s=None` means never freeze; every prior result
   bit-identical.

I hold prediction 1 less firmly than prediction 2. E087's constant was computed from
*settled, parked* states spanning five places deliberately; a baseline frozen from a
free-running hen sees whatever she happened to do in the calibration window, which on an
aggregating flock may be one corner of the arena.

## 4. Falsifier

**Primary.** Decodability at the best freeze time stays below **85%**. The frozen baseline
is then not a practical substitute for E087's idealised constant, and the gap between them
— what a causal calibration can see versus what an oracle can compute — becomes the thing
to explain.

**Selectivity falsifier.** Ratio below **2.0** at the freeze time that maximises
decodability. Both axes must clear on the same configuration or the result does not count.

**Calibration-window falsifier.** Decodability varies by more than 15 points across the
freeze times tested with no interpretable trend. That would mean the frozen baseline is
capturing whatever the hen happened to be doing rather than a stable property, and it is
not a mechanism, it is a lottery.

**Default falsifier.** Any behaviour changes with `pred_bar_freeze_s=None`. Asserted
bit-identical.

---

## 5. Design

### The change

`pred_bar_freeze_s: float | None = None` on `PlasticConfig`. `z_lag_bar` updates as now
while the world clock is below it, and is held constant thereafter. `None` never freezes,
which is current behaviour exactly.

The world clock (`w.t`) is already the age source used for the exploration schedule, so
this needs no new state and stays inside the compiled scan.

### Part A — freeze-time sweep

E086/E087's harness unchanged: `place_to_hippocampus=True`, 8 seeds, per-seed target
selection, held-out protocol (fit on the selection run, evaluate on the test run of the
same world), hippocampus units, radius 3.33 m.

Sweep `pred_bar_freeze_s ∈ {10, 20, 40, 60, 120}` plus `None` as the control. Reference
lines already measured: runtime EMA 73.7%, idealised constant 89.8%, raw 90.0%.

Each condition also reports the **frozen baseline's convergence** at the moment of
freezing — `mean|z_lag_bar| / mean|z_lag|` — so a baseline frozen before it represented
anything is visible as such rather than reported as a result. This is the same guard E087
carried, moved to the quantity that now matters.

### Part B — selectivity at every freeze time

E070's measurement, as E087 ran it: plant a place association, read its prediction at the
planted place and at the others, report the ratio. Run at **every** freeze time, not only
the best one, so prediction 3's "both at once" is checkable rather than asserted.

### Guard

A test asserting `z_lag_bar` stops changing after the freeze time and does not before it,
at `n_hens=16`.

### Cost

~35 minutes. 6 conditions × 8 seeds for Part A; Part B reuses the planting machinery.

---

## 5b. Pre-registered replication rule — written after Part A/B, before the replication ran

*Sequencing stated plainly: §6 below reports Parts A and B. This subsection was written
and committed **after** seeing them and **before** the replication in §6c was run. It is
therefore a pre-registration for the replication only, and Parts A/B remain the
data that generated the hypothesis it tests.*

Parts A and B fired the selectivity falsifier as written, and the reason is a defect in
my decision rule rather than in the mechanism. The rule selected the freeze time that
maximises **decodability** and then tested **selectivity** there — selecting on one axis
and testing the other, which is exactly what E087's lesson said not to do, and which I
restated in §3 prediction 3 of this document before contradicting it in §4.

The freeze time it picked (20 s) is one where the baseline had reached only **0.600** of
the trace's magnitude — it had not yet converged, so it is a poor estimate of the DC term,
which is the one thing the centring exists to remove. That is not a discovered excuse:
§5's design reports convergence at the freeze moment *precisely* so "a baseline frozen
before it represented anything is visible rather than reported as a result".

**The corrected rule, fixed now for the replication:**

1. **Admissible** freeze times are those with convergence ≥ **0.80** at the freeze moment.
   A baseline frozen before it represents the trace is not a candidate, whatever it scores.
2. Among admissible times, the operating point is the one maximising decodability.
3. Both bars must clear **at that same point**: decodability ≥85% and selectivity ≥2.0.

**Replication:** seeds **8–15**, disjoint from Parts A/B's 0–7, at the two admissible
candidates (40 s and 60 s) plus the never-freeze control. `CLAUDE.md`'s standing rule —
no status changes on one seed block, and test a post-hoc observation on data that did not
generate it — applies here in full. **If 40 s does not clear both bars on fresh seeds, it
is not adopted.**

## 6. Result

### Parts A and B — 8 seeds (0–7), 3771 s

| freeze (s) | held-out | balanced-split | convergence @ freeze | pred @ P | pred elsewhere | selectivity |
|---|---|---|---|---|---|---|
| 10 | 90.2% | 90.4% | 0.336 | 1.0000 | 1.0683 | **0.94** |
| 20 | 86.8% | **90.9%** | 0.600 | 1.0000 | 1.5003 | **0.67** |
| **40** | 82.9% | **90.5%** | **0.862** | 1.0000 | 0.1365 | **7.33** |
| 60 | 80.3% | 88.3% | 0.952 | 1.0000 | 0.1982 | 5.04 |
| 120 | 77.2% | 79.4% | 1.013 | 1.0000 | −0.0347 | 28.80 |
| None (control) | 69.2% | **73.7%** | 0.994 | 1.0000 | 0.1145 | 8.73 |

**The control reproduces E087 exactly** — 73.7% balanced, against E087's 73.7%. Matched.

**Primary falsifier: clear.** Freezing recovers the whole loss. At 10–40 s decodability is
90.4–90.9%, against the runtime's 73.7%, E087's idealised constant at 89.8% and the raw
ceiling at 90.0%. **A causally computable frozen baseline matches the oracle constant.**

**Selectivity falsifier: FIRES as written.** At the argmax-decodability freeze (20 s) the
ratio is 0.67 — the prediction at a control place *exceeds* the prediction at the target,
which is E070's failure returning in a worse form.

**Calibration-window falsifier: clear.** Spread across freeze times is 11.5 points against
a 15-point threshold, and it has a clear trend rather than being a lottery.

**And the trend is the whole story: selectivity tracks convergence, decodability opposes
it.** Frozen before the baseline represents the trace (conv 0.34, 0.60) selectivity
collapses to 0.94 and 0.67. Frozen once it does (conv 0.86, 0.95) selectivity is 7.33 and
5.04 while decodability is still 90.5% and 88.3%. Frozen long after (conv 1.01) selectivity
reaches 28.80 but decodability has fallen to 79.4%.

**So both bars clear simultaneously at 40 s: 90.5% decodability and 7.33 selectivity.**

### 6c. Replication on fresh seeds

*To be filled after the replication runs, under §5b's rule.*

## 7. Interpretation

*To be filled after the replication.*

## 8. Consequence

*To be filled after the replication.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*
