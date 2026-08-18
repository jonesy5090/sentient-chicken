# E047 — H3 re-run: does the audience effect emerge on the corrected connectome and world?

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**H3** — learned usage reproduces the audience effect without being programmed.
`UNDER TEST`, two nulls (E005, E006), formally "blocked by H2b" though E036/E040
already found that the *specific* precondition H2b's story required (an innate
response to hearing an alarm call) doesn't rescue audience-sensitive calling even when
supplied directly. This re-runs E005/E006's *original* design — audible-kin vs
flat-kin reward, both calling types (alarm and food) — which has never been re-tested
since E019 (calls made audible), E023 (E/I fix), or E037–E040 (`food_deplete_rate`
audit). `run/audience.py`'s default (non-`--scaffold-2x2`) path already implements this
design unchanged; no new code is needed, only a re-run.

## 2. Question

E006 found comprehension exactly zero (~0.0005) on a connectome where the auditory
channel itself was later found to be inaudible (E019) — so E005/E006's nulls on both
`alarm_effect` and `food_effect` were measured on a broken instrument for the alarm case
at minimum. `food_effect` in particular showed a *promising, unresolved* trend in E005
(+0.032, t=0.64 — numerically the right sign, not powered to confirm) that was never
retested once the channel was fixed. **Does either audience effect — alarm or food —
emerge, or clarify, on the fully corrected connectome and world?**

## 3. Prediction

**No confident directional prediction — registered as genuinely uncertain**, consistent
with this session's read on H2b/H2d/H2f: the representational and rule-kind blockers
diagnosed for the alarm-call case likely apply here too, since the underlying learning
machinery (three-factor `W_out`, or `W_pred`'s associative rule) hasn't changed. If
E036/E040's null (learned audience-sensitivity does not emerge even with comprehension
supplied by construction) generalises, both effects should stay near zero here too.

**What would be genuinely informative either way**: whether `food_effect` — the
prediction that trended positive in E005 — now clarifies into a real, measurable effect
on a working channel, since food calls never had the "no innate response to hearing"
problem alarm calls did (food-call comprehension was never the blocker H2b/H2c
diagnosed) and were never specifically re-tested since the E019 audio fix.

## 4. Falsifier

**For H3 broadly**: if both `alarm_effect` and `food_effect` remain indistinguishable
from the fixed control after rearing, H3's null is confirmed on the corrected system,
narrowing "blocked by H2b" (an incomplete story, per E036/E040) to something closer to
"blocked by the same rule-kind/representational limits E042–E044 diagnosed for
comprehension generally" — H3 and H2c would then be understood as symptoms of the same
underlying limitation, not separate open questions.

**A positive result on either channel** — audience-sensitive calling emerging where it
didn't before — would be the first positive result this project's learning rule has
ever produced on a behaviourally meaningful metric, and would need immediate replication
on a fresh seed block before being trusted (this project's own standing rule).

## 5. Design

`run/audience.py`'s existing, unmodified default path: three conditions (`audible kin`
— E006's audibility-weighted fix, `flat kin` — E005's original, `fixed` — the required
control, which must stay flat regardless of any learning). Both `alarm_effect` and
`food_effect` measured before and after rearing, matching the harness's own built-in
manipulation check (comprehension, must-be-zero fixed control).

**World**: 16 hens, `spec.DEFAULT_COOP` defaults otherwise (hawk every 900 s — H3 was
never about predator density, unlike H4/T1), `food_deplete_rate=0` (this session's
standing discipline; not previously an issue for E005/E006 since depletion didn't exist
yet, but the current codebase's default would introduce it if not controlled for).

**Duration**: 30 minutes (the harness's own default, matching E005/E006's scale).

**Replicates**: 8 seeds — this session's established first-pass count.

**Command:**
```bash
python -m run.audience --minutes 30 --seeds 8 --food-deplete-rate 0.0
```

## 6. Result

*Pending — filled in after the run, not before.*

## 7. Interpretation

*Pending §6.*

## 8. Consequence

*Pending §6.*
