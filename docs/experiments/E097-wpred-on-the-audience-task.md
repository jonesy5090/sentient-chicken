# E097 — `W_pred` on the audience task: the rule H2f's falsifier actually named

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **H2f**, and specifically its registered falsifier, which has never
been tested. Successor to [E096](E096-red-team-review-2026-08.md).

---

## 2. Question

H2f's falsifier reads:

> *a rule closer to Pavlovian association (**e.g. sourced from `W_pred`**, already
> architecturally positioned for this per H2c) succeeds where the instrumental rule failed,
> on the same task and the same scaffold.*

What was implemented and scored against it was `hebbian_readout` — a constant modulator on
`W_out`, updated from `dz_motor × dz_slow`, both same-time traces (`tau_motor` 0.10,
`tau_slow` 0.20). It shares one property with the falsifier's description, *not
reward-gated*, and lacks the distinguishing one: there is no cue→outcome direction in it at
all.

**`W_pred` is the pathway the falsifier names and it has never been run on this task.** It
is a masked delta rule (`pred_err = observability × (obs − predicted)`), non-reward-gated,
sourced from `z_lag` at `tau_lag` 1.5 — the only genuine lag in the codebase — and it
writes onto the observation the *reflex arc* reads (`brain.py:85`), so a learned cue does
not have to recreate a behaviour, only the percept that already drives it.

**There is a specific mechanism by which it could produce an audience effect**, and stating
it in advance is what makes the falsifier meaningful: if "flockmates in view" reliably
precedes "aerial alarm audio" during rearing, `W_pred` can learn to *hallucinate the alarm
channel when flockmates are present*. `innate.py` wires `IDX_AERIAL → M_CALL_AERIAL` at
7.0, so that hallucination drives alarm calling. Cue → predicted outcome → innate response
is precisely Pavlovian, and it is available to `W_pred` and unavailable to
`hebbian_readout`.

**And E096 changes how this must be measured.** The audience assay is confounded: the
staged hawk sits 5–9 m from the audience, so they alarm-call, and 79% of H2f's
audience-specific effect is carried by that audio rather than by their presence. **The
primary measure here is therefore the *muted* contrast** — audience present but silent —
which is the quantity the audience effect was always supposed to be.

That is also the sharp test of the mechanism above. A prediction learned from "flockmates
present" is *internal*: it should survive muting, because she is generating the percept
herself rather than hearing it.

---

## 3. Prediction

1. **Muted audience-specific DiD ≥ +0.10.** With the audience present but silent, a
   `W_pred`-reared flock calls more than an alone flock, beyond the food-channel control.
   E096 measured `hebbian_readout` at **+0.0577** muted; the bar is roughly double that.
2. **Muted DiD is a larger fraction of intact DiD than `hebbian_readout`'s 21%.** If the
   effect is internally generated it should depend less on the audio.
3. **The food-channel control stays below +0.05 muted.** Specificity: the effect must be
   about calling in company, not about calling more in general.
4. **`pred_gain` matters.** At `pred_gain=0` the pathway is wired but cannot reach
   perception, so the DiD should fall back toward the fixed-hen baseline. This is the
   internal control that the effect is `W_pred`'s and not the scaffold's.

I hold prediction 1 at roughly even odds. The mechanism is available and the pathway is
built for it, but no learning rule in this project has yet produced a targeted behavioural
change, and `W_pred`'s own arc (E087/E088) showed it needs a well-conditioned readout to
carry anything.

## 4. Falsifier

**Primary.** Muted audience-specific DiD < **+0.10**. H2f's falsifier then has been
attempted and not met: the Pavlovian route does not produce an audience effect either, and
H2f should be read as `NOT SUPPORTED` on its own terms rather than `SUPPORTED, CONFOUNDED`
— because the confounded result would be the only thing holding it up and the rule it
actually named would have failed.

**Specificity falsifier.** Food-channel muted DiD ≥ +0.05. Indiscriminate elevation, which
is exactly what E058 used to reject H2c's apparent result.

**Gain falsifier.** The muted DiD at `pred_gain=0` is within 0.03 of the value at
`pred_gain>0`. The effect would then not be `W_pred`'s at all.

**Reproduction falsifier.** The `hebbian_readout` arm does not reproduce E096's muted
+0.0577 within 0.03. That arm is run alongside as a positive control on the harness; if it
does not reproduce, nothing else in the run is interpretable.

---

## 5. Design

Four arms, matched seeds, 8 seeds, 30 minutes' rearing at E057's configuration
(`food_deplete_rate=0.0`, auditory scaffold on):

| arm | rule | purpose |
|---|---|---|
| **S** | plasticity off | baseline |
| **H** | `hebbian_readout`, `readout_scaling_strength=0.3` | E096 reproduction / harness control |
| **P0** | `pred_enabled`, `pred_gain=0.0` | `W_pred` wired but unable to reach perception |
| **P** | `pred_enabled`, `pred_gain=1.0`, `pred_centred`, `pred_bar_freeze_s=60` | the falsifier's rule |

E088's adopted centring configuration is used for the `W_pred` arms, because E087 measured
the tracking baseline removing ~20 points of the signal `W_pred` reads, and running the
falsifier's rule through a readout known to be degraded would not be a fair test of it.

**Every arm is assayed twice on the same reared brain** — audio intact and audio muted at
test — and **the muted contrast is the primary measure**. Reporting both keeps
comparability with E057 while making the confound visible rather than inherited.

Reported per arm: `alarm_alone`, `alarm_audience`, `food_alone`, `food_audience`, each
intact and muted; the DiD of each; and the fraction surviving muting.

### Cost

~25 minutes.

---

## 6. Result

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*
