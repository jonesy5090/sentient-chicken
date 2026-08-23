# E098 — repair the audience assay, then give `W_pred` a fair test

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **H2f** and **H3**, both of which are measured through
`run/audience.py`. Successor to [E097](E097-wpred-on-the-audience-task.md).

---

## 2. Question

E097 left three defects, all in the instrument rather than the hypothesis.

**(a) The assay cannot read what the rule learned.** Two independent routes:
`run/audience.py:101-112`'s `assay()` calls `simulate.rollout` **without passing the reared
`ps`**, so a fresh zero-filled trace state is built; and `run/simulate.py:102`'s
`if not pc.enabled: return` sits **above** `update_traces`, so `z_lag` could not update even
if it were passed. A `W_pred` projection trained on a centred lagged trace is therefore read
at test either from zeros or from instantaneous `rate(x)`, depending on `pred_enabled`.
**E071's error for the fourth time**, and this one silently nullifies the pathway.

**(b) The staging manufactures an audience effect in an unlearned hen.** E097's
plasticity-off arm shows intact DiD **+0.0650** collapsing to **+0.0020** when the audience
is muted. The staged hawk sits 5–9 m from the audience, inside `vision_range`, so they
alarm-call and "audience present" is also "aerial channel driven". Every audience number
this project has reported inherits it.

**(c) A `W_pred` null may be about the world, not the rule.** At `hawk_period_s=900`, 30
minutes of rearing contains roughly **two** hawk events. The mechanism E097 pre-registered
needs "flockmates in view precedes aerial alarm audio" to be learnable, and the pairing rate
has never been measured.

**Does a repaired assay, on a world that contains the contingency, let `W_pred` show an
audience effect?**

---

## 3. Prediction

1. **The trace fix is behaviourally inert where it should be.** Plasticity-off runs without
   `pred_enabled` are **bit-identical** before and after. Traces are state, not learning;
   updating them must change nothing that does not read them.
2. **The fix is not inert for `W_pred`.** With the reared `ps` carried and traces
   maintained, arm P's assay differs measurably from E097's — the projection now reads the
   signal it was trained on. No direction predicted.
3. **The pairing rate at `hawk_period_s=900` is too low to learn from**: fewer than 5
   flockmate-call events per rearing run. At 60 s it should exceed 20.
4. **Muted DiD ≥ +0.10 for `W_pred` on the repaired assay at the higher event rate** — the
   bar E097 set and could not fairly test. Held at **below** even odds: E097's arm P
   *suppressed* alarm calling by 61%, and while that is consistent with reading an untrained
   signal, it is not evidence the trained one would help.

## 4. Falsifier

**Inertness falsifier (a hard gate).** Any plasticity-off, non-`pred` result changes after
the trace fix. Then the fix is not a fix, it is a change to every recorded result, and it
must be reverted and redesigned. Checked by asserting bit-identity on the full suite and on
arm S's four assay numbers.

**Reachability falsifier.** Arm P's numbers are unchanged after the fix. The projection
would then still not be reading the trained signal, and (a) is not the defect it appears to
be.

**Primary.** Muted audience-specific DiD < +0.10 for `W_pred` on the repaired assay at a
rearing world where the pairing occurs >20 times. H2f's falsifier would then have been
fairly attempted and not met, and **that reading is licensed this time** because both §2(a)
and §2(c) will have been removed.

**Specificity falsifier.** Food-channel muted DiD ≥ +0.05.

---

## 5. Design

### The three repairs

1. **Move `update_traces` above the `pc.enabled` early return** in `run/simulate.py`, so
   trace state is maintained whenever the pathway that reads it is enabled, and only
   *weight* updates stay gated on `enabled`. Traces are state; learning is the weight
   change. Guarded by prediction 1's bit-identity assertion.
2. **`assay()` accepts and carries the reared `ps`.** Without it the fix above is moot.
3. **Report the muted contrast as the assay's primary number.** `AudienceResult` gains the
   muted quartet and `alarm_effect` becomes the muted one, with the intact figure retained
   as `alarm_effect_audible` for comparability with E057/E074/E096. This is a reporting
   change, not a new measurement — the muted contrast is the one that means what the name
   says.

### The measurement

**Part A — pairing rate.** Count, per rearing run, how often a flockmate's aerial call is
audible to the focal hen, at `hawk_period_s ∈ {900, 300, 60}`. This decides Part B's world
and is the check §2(c) says was never done.

**Part B — `W_pred`, fairly.** E097's four arms unchanged, on the repaired assay, at the
`hawk_period_s` Part A selects. 8 seeds, 30 min rearing, each brain assayed twice.

### Cost

Part A ~5 min. Part B ~25 min.

---

## 6. Result

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*
