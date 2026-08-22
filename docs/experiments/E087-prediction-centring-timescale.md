# E087 — decoupling the prediction-centring timescale from the reward baseline

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **T2** → **T2-revised**, the `W_pred` readout. Direct successor to
[E086](E086-place-to-hippocampus.md).

---

## 2. Question

E086 fixed the place representation — parked decodability 99.5%, and **90.0% under
movement when the hippocampus is read directly**. It also showed `W_pred` cannot get at
that, and split the loss:

| readout (held-out, balanced-split seeds) | |
|---|---|
| hippocampus, raw rate | 90.7% |
| hippocampus, `z_lag` only (low-pass, τ 1.5 s) | 90.0% |
| **hippocampus, `z_lag − z_lag_bar`** — what the runtime reads | **73.7%** |

The lag costs nothing. **The centring costs ~20 points.** `z_lag_bar` is a 20-second
running mean; E085 measured dwell times of **17–75 s**. The baseline tracks position and
subtracts it — a high-pass whose corner sits on the signal's own timescale.

**But the centring is load-bearing.** E070 measured a planted place association predicting
**1.0000 at its own place and 0.9637 at a different one**, because `z_lag` is strictly
positive with mean ~0.23 while the across-stimulus signal is 3.7% of that DC. Projected
raw, DC dominates and nothing is selective. E071 added the centring and it worked.

So the pathway needs the DC removed **and** slow signals preserved, and cannot currently do
both. Note also that the 20 s is not a considered choice: `z_lag_bar` updates with
`a_b = cfg.dt / pc.baseline_tau_s` (`plasticity.py:355`), **the same constant as the reward
baseline**, with nothing in the source stating that as a decision. The reward baseline
wants to track on the timescale of reinforcement; this one wants to be slow compared to
whatever the prediction is about.

**Does giving `z_lag_bar` its own time constant recover decodability without bringing back
E070's selectivity failure?**

---

## 3. Prediction

The two requirements are separable in principle — remove true DC, do not track a 17–75 s
dwell — so a corner well above the dwell timescale should satisfy both.

1. **Decodability climbs monotonically with `pred_bar_tau_s`**, from 73.7% at 20 s toward
   the uncentred ceiling of 90.0%. I predict **≥85%** at 300 s.
2. **E070's selectivity failure does not return.** Measured as E070 measured it: a planted
   association's prediction at its own place against a different place. E070's failure was
   1.0000 vs 0.9637 — a ratio of 1.04. I predict the ratio stays **above 2.0** at every
   tau tested, because the DC is genuinely constant and a 300 s mean still removes it.
3. **The default is unchanged.** `pred_bar_tau_s=None` inherits `baseline_tau_s`, so every
   prior experiment's behaviour is bit-identical.

I hold prediction 2 more firmly than prediction 1. Removing DC needs only that the
baseline be slower than nothing; preserving position needs it slower than dwell, and dwell
varies tenfold across seeds (E085), so a single tau may not suit every seed.

## 4. Falsifier

**Primary.** Decodability at the best tau stays below **80%** — less than half the distance
from 73.7% to the 90.0% uncentred ceiling. The centring would then not be the recoverable
part of the loss, and the remaining ~16 points would need a different explanation before
anything is built on it.

**Selectivity falsifier — the important one.** The planted-association ratio falls below
**2.0** at the tau that maximises decodability. That is E070's failure returning, and it
would mean the two requirements are *not* separable by timescale alone: the pathway would
need a different mechanism for DC removal (per-unit normalisation, say) rather than a
slower mean. **This is why E087 measures selectivity rather than assuming it** — it is the
whole reason the centring exists, and E086's diagnostic did not test it.

**Default falsifier.** Any behaviour changes when `pred_bar_tau_s` is left at its default.
Checked by asserting bit-identical trajectories.

**Convergence falsifier.** At long tau the baseline may not converge within the run, in
which case "centred" silently equals "raw" — the exact error E071 documented and E082
repeated. Every condition must report its convergence ratio, and any tau whose baseline
has not converged is excluded from interpretation rather than reported as a result.

---

## 5. Design

### The change

Add `pred_bar_tau_s: float | None = None` to `PlasticConfig`. `None` inherits
`baseline_tau_s`, so the default is bit-identical and no prior result moves. `z_lag_bar`'s
update rate becomes `cfg.dt / (pc.pred_bar_tau_s or pc.baseline_tau_s)`.

This is a decoupling, not a retuning: the reward baseline keeps its 20 s, and nothing is
recommended as a new default until this experiment says what the value should be.

### Part A — decodability sweep

E086's harness unchanged. `place_to_hippocampus=True`, 8 seeds, same per-seed target
selection, same held-out protocol, hippocampus units. Sweep
`pred_bar_tau_s ∈ {20 (current), 60, 150, 300, 600}`, plus the uncentred ceiling and raw
rate as reference lines already measured.

Each condition reports its **convergence ratio** — `mean|z_lag_bar| / mean|z_lag|` at the
end of the run. A ratio near 1 means the baseline has tracked the trace and centring is
doing its job; a ratio near 0 means it never converged and the condition is uninterpretable.

### Part B — selectivity, measured the way E070 measured it

For each tau, plant a place association and read its prediction at the planted place and
at a different place, exactly as E070 did. Report the ratio. E070's failure was 1.04.

Both parts run on the same connectomes and the same seeds, so Part B's answer applies to
the configuration Part A recommends.

### Cost

~25 minutes. 5 taus × 8 seeds for Part A; Part B reuses the planting machinery and is
cheap.

---

## 6. Result

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*
