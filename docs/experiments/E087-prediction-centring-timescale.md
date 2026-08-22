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

### Part A — the sweep. Primary falsifier fires.

8 seeds, 3164 s. Decodability from the hippocampus, held-out.

| `pred_bar_tau_s` | held-out acc | balanced-split | convergence |
|---|---|---|---|
| 20 (current) | 69.2% | **73.7%** | 0.994 |
| 60 | 63.6% | 67.2% | 0.992 |
| 150 | 65.1% | 58.1% | 0.993 |
| 300 | 63.3% | 58.5% | 0.974 |
| 600 | 64.9% | 64.3% | 0.855 |

**Decodability does not climb with tau. It is best at the current value and worse at every
longer one**, non-monotonically, with no trend toward the 90.0% uncentred ceiling. The
convergence falsifier is clear at every tau (0.855–0.994), so none of these is silently
uncentred — the E071/E082 error is ruled out rather than assumed away.

### Part B — selectivity. Prediction 2 held, and running it mattered.

| `pred_bar_tau_s` | pred @ P | pred @ P′ | ratio |
|---|---|---|---|
| 20 | 1.0000 | −0.0312 | **32.04** |
| 60 | 1.0000 | −0.0914 | 10.94 |
| 150 | 1.0000 | 0.1044 | 9.58 |
| 300 | 1.0000 | 1.8747 | **0.53** |
| 600 | 1.0000 | 1.0432 | 0.96 |

Selectivity is excellent at the current 20 s and **degrades monotonically as tau
lengthens**. At 300 s the prediction at the *control* place (1.87) exceeds the prediction
at the target — worse than E070's original failure. Had Part B not been pre-registered,
300 s would have looked merely mediocre on decodability instead of catastrophic on the
thing the term exists for.

### 6b. Diagnostic — the between-hen hypothesis, also falsified

*Post-hoc.* `z_lag_bar` is a **per-hen** baseline, and the flock aggregates (E084: spread
1.66–7.21 m in a 20 m arena; E085: single-cell occupancy 0.09–0.97 across seeds), so
position is partly a between-hen variable and a per-hen baseline might be removing it.
Tested by removing the between-hen component by hand, at zero timescale:

| variant | all seeds | balanced |
|---|---|---|
| raw `z_lag` | 89.3% | 90.0% |
| minus the across-hen (global) mean | 88.4% | 89.2% |
| minus **each hen's own full-run mean** | 89.1% | **89.8%** |

**Removing each hen's own constant mean costs nothing** — 89.8% against 90.0% raw. The
between-hen hypothesis is wrong too.

### 6c. Diagnostic — it is the *form* of the baseline, not its timescale or its scope

*Post-hoc.* What separates the 89.8% of a constant per-hen mean from the 73.7% of the
runtime is that one is **constant** and the other **tracks**. Selectivity under all three,
on identical settled states:

| baseline | pred @ P | pred elsewhere | ratio | decodability (moving) |
|---|---|---|---|---|
| none (raw) | 1.0000 | 1.0101 | **1.04** | 90.0% |
| **constant** | 1.0000 | −0.2000 | **5.00** | **89.8%** |
| EMA, τ 20 s (runtime) | 1.0000 | 0.1145 | 23.28 | 73.7% |

The `none` row **reproduces E070's failure exactly** — E070 measured 1.0000 vs 0.9637, a
ratio of 1.04, and this returns 1.04 on an independent implementation. That is a clean
replication of the finding that motivated the centring in the first place.

And **a constant baseline clears both bars at once**: 89.8% decodability (against the
runtime's 73.7%) with selectivity 5.00, comfortably above the pre-registered threshold of
2.0.

## 7. Interpretation

**Prediction 1 was wrong, and so were both mechanisms I proposed for it.** The timescale
story — a 20 s baseline tracking 17–75 s dwells and subtracting position — predicted that
longer tau would help. It hurts. The between-hen story predicted that removing a per-hen
constant would cost what the per-hen EMA costs. It costs nothing. Both were tested before
being written anywhere, and both are recorded here because the sequence is the useful part:
a plausible mechanism sitting next to a real number is not evidence, which is this
project's oldest lesson and one I have now re-learned twice in two experiments.

**What survives is narrower and better supported: the damage is done by the baseline
*tracking*, not by its timescale, its scope, or the lag.** A constant DC removal keeps
89.8%; an exponential moving average of the same quantity keeps 73.7%. A tracking baseline
is a high-pass, and the hen's pallial state during free movement evidently carries position
in components that a high-pass removes regardless of where its corner sits.

**Why lengthening tau made things worse rather than better is not explained**, and I am not
going to propose a third mechanism for it. The candidate worth noting is that
`z_lag_bar` starts at zero and needs roughly 3τ to settle, so at 600 s it is still rising
through a 1200 s run — a large, systematically decaying term that the convergence ratio
(which compares magnitudes, not stationarity) does not detect. That is a hypothesis, it is
untested, and it is written here as a lead rather than a finding.

**The centring is vindicated on its own terms.** At the current 20 s it delivers a
selectivity ratio of 32.0 against E070's failure at 1.04. Nobody should read this
experiment as saying the centring was a mistake. It says the centring buys selectivity at a
price in signal, and that **a different form of the same operation buys nearly all of the
selectivity at almost none of the price**.

**The trade-off is real but lopsided.** Constant gives 5.00 selectivity and 89.8%
decodability; the EMA gives 23.28 and 73.7%. Whether 5.00 is behaviourally sufficient is
genuinely open — it is 5× separation and well above the bar I set in advance, but no
behaviour has been run on it, and the honest statement is that this is a measurement about
readouts, not about hens.

## 8. Consequence

**Kept: `pred_bar_tau_s`, defaulting to `None`.** It inherits `baseline_tau_s`, so every
prior result is bit-identical (87/87). Its value is now documented as *not* a tuning knob:
Part A shows no setting of it helps, and settings above ~150 s actively destroy
selectivity. It stays because the coupling it removed — the prediction baseline sharing a
constant with the *reward* baseline, with nothing in the source stating that as a choice —
was worth breaking on its own.

**Recommended next, and the design follows from 6c: a *frozen* baseline.** Let `z_lag_bar`
track for a calibration period and then hold constant — `pred_bar_freeze_s`, default
`None` meaning never freeze, so the default stays inert. That is implementable in the
runtime, unlike the across-place mean used in 6c's measurement, which requires knowing the
places in advance. It is also biologically unembarrassing: a developmental calibration of
what "average activity" means, fixed once rather than continuously re-estimated.

**Its pre-registration must carry both axes**, because 6c establishes they trade against
each other: decodability ≥85% *and* selectivity ≥2.0, measured together, on the same
configuration. Either alone is meaningless here.

**Not recommended: sourcing `W_pred` from the hippocampus alone.** E086 recorded it as
available (worth ~10 points) rather than advised, and that stands — it is a real effect,
but a pallium that reads only its hippocampus and ignores everything else is an
architectural claim nobody has argued for.

**Still standing: any learning run on this needs an autoencoder control** (E086 §5), and
the **`strike_penalty` audit** remains the largest unexamined risk in the tree.
