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

## 6. Result

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*
