# E071 — centring the prediction pathway: necessary, not sufficient

> **Diagnostic and fix.** Sections 1–5 written before running; §6 corrects a claim made
> in E070's own interpretation, which is the main reason this file exists.

## 1. Parent hypothesis

**T2-revised**, and behind it H2d/E017 — the pallial separability problem the backlog
has carried as "the critical path" since E017. [E070](E070-t2-revised-chain-positive-control.md)
found a hand-planted place→gakel association predicting 1.0000 at its own place and
0.9637 elsewhere, and attributed it to `W_pred` never having received E019's centring
fix. This tests that attribution.

## 2. Question

Does centring the prediction pathway's source — subtracting a running mean of `z_lag`,
the direct analogue of what `z_slow_bar` does for the recurrent rule — restore place
selectivity in a planted association?

## 3. Prediction

Selectivity should improve substantially. The across-place signal is 3.7% of the DC
baseline (E070), and removing a DC term that dominates a projection is exactly what
centring is for. Whether it improves *enough* to be behaviourally useful was not
predicted — stated as open before running.

## 4. Falsifier

If centring leaves the prediction as unselective as before, E070's attribution is wrong
and the defect lies elsewhere.

## 5. Design

`pred_centred`, a new `PlasticConfig` flag, **off by default** so it is a contrast
rather than a change — matching `pred_enabled`'s own precedent, and because E042/E043
ran H2c against the uncentred rule and their recorded numbers describe that rule.
Applied to both halves that had the defect: the readout in `run/simulate.py` (which
passes `pred_from` into `brain.step`) and `W_pred`'s own update in `consolidate`.

Measurement repeats E070's planted-association test: plant place P → gakel channel,
read the prediction back at P and at three other grid cells.

**Timescales, which E070's first follow-up got wrong.** `z_lag` has `tau_lag=1.5 s` and
`z_lag_bar` follows `baseline_tau_s=20 s`. An early version of this test settled for
4 seconds, leaving `|z_lag_bar|/|z_lag| ≈ 0` — so "centred" was arithmetically
indistinguishable from raw and the flag appeared to do nothing. The run below tours
five places for 300 s (15× `baseline_tau_s`) before parking, and **reports the
convergence ratio rather than assuming it**.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e071_pred_centring.py
```

## 6. Result

Convergence confirmed: `|z_lag| = 18.044`, `|z_lag_bar| = 18.493`, ratio 1.025.

| | at P | elsewhere | mean ratio |
|---|---|---|---|
| `pred_centred=False` | +1.0000 | +1.1357, +0.9650, +1.0239 | **1.042** |
| `pred_centred=True` | +1.0000 | −1.1554, +1.3027, +0.3921 | **0.180** |

**The 0.180 is misleading and must not be quoted on its own.** It is produced by
cancellation — a large negative offsetting a large positive — not by the other places
predicting weakly. What reaches behaviour is `relu(predicted)` (`brain.py`), which
clips the negative to zero: **0, 1.3027, 0.3921**. One distractor place therefore drives
*stronger* withdrawal than the planted place, and the behaviourally effective ratio is
**0.565**, not 0.180.

## 7. Interpretation

**Centring is a real fix for a real defect, and it does not solve the problem.**

Uncentred, the prediction was flat across places (ratio 1.042 — marginally *higher*
elsewhere than at the planted place). Centred, it varies substantially by place. So
E070's diagnosis was correct as far as it went: the DC term genuinely dominated the
projection, `W_pred` genuinely never received E019's fix, and that fix genuinely
belongs in the codebase on its own merits.

**But E070's framing was too optimistic and is corrected here.** It concluded "for
place information the projection is fine, the readout is not." That is not what this
shows. With the DC removed, the residual pallial representations of different places
remain highly correlated — E070's own measurement put place-to-place correlation at
0.94–0.96 — so a linear readout keyed to P still responds strongly to other places.
The prediction becomes *variable* across places rather than *selective* for one.

The honest diagnosis is that both problems are present: an uncentred readout (now
fixable, and fixed here behind a flag) **and** genuinely poor pallial separability —
which is H2d/E017 exactly, the item the backlog has carried as the critical path since
E017 and which no experiment has yet moved.

**Three comparison places is a small sample** and the individual values are noisy; the
qualitative conclusion (variable, not selective) is safe, the precise ratio is not.

## 8. Consequence

**`pred_centred` is kept, off by default.** It corrects a defect of the same class and
shape as E019's, in the last pathway that never received that fix, and E070's numbers
would have been read wrongly without it. It is not switched on by default because
E042/E043's recorded H2c results describe the uncentred rule, and changing the
comparison basis silently is what this project's conventions exist to prevent. Turning
it on should be its own decision, with the affected results re-run.

**T2-revised stays paused, and the reason is now correctly identified.** It is not
blocked by a missing mechanism (both are built and correct) nor solely by an uncentred
readout (now addressable). It is blocked by pallial separability. Any hypothesis that
needs the pallium to distinguish *which* stimulus — T2-revised, H2c, H3 — is blocked by
the same thing.

**That makes H2d/E017 the highest-value open item in the project**, on evidence rather
than assertion: it is now the proximate blocker for three separate hypotheses, and this
experiment shows the cheaper candidate fix does not substitute for it. E017 relocated
the problem to fan-in dilution at sensory→pallium and E041 reframed it around density;
neither resolved it. Whatever is attempted next should be measured against a concrete
target — place-to-place pallial correlation, currently 0.94–0.96, needs to come down
before a linear readout can be referential.

**Do not re-run E070 on the strength of the 0.180 figure.** That is the specific trap
this write-up exists to close: the headline number looks like success and the
behaviourally relevant one (0.565, with a distractor exceeding the target) does not.
