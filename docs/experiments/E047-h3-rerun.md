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

8 seeds, 30 min rearing, 16 hens, `food_deplete_rate=0`. Wall clock 899 s.

```
condition          when      alarm alone  alarm aud.   effect  food alone  food aud.   effect
audible kin        hatch           0.320       0.323   +0.003       0.363      0.362   -0.001
audible kin        reared          0.318       0.320   +0.001       0.365      0.365   +0.000
flat kin (E005)    hatch           0.320       0.323   +0.003       0.363      0.362   -0.001
flat kin (E005)    reared          0.319       0.321   +0.002       0.362      0.360   -0.002
fixed (control)    hatch           0.320       0.323   +0.003       0.363      0.362   -0.001
fixed (control)    reared          0.320       0.323   +0.003       0.363      0.362   -0.001

comprehension (manipulation check): -0.0001 in every condition, hatch and reared alike.

effect change over rearing:
  audible kin  alarm  -0.002 +/- 0.001  t=1.22  need 2.37  suggestive (wrong sign)
  audible kin  food   +0.001 +/- 0.009  t=0.13  noise
  flat kin     alarm  -0.001 +/- 0.001  t=0.97  noise
  flat kin     food   -0.001 +/- 0.007  t=0.22  noise
  fixed        alarm  +0.000            t=0.00  noise (control holds, as required)
  fixed        food   +0.000            t=0.00  noise (control holds, as required)
```

**Comprehension is exactly zero, confirming this is now a genuine architectural
finding rather than a broken-instrument artefact.** E006's original ~0 measurement was
taken on a connectome where the auditory channel later turned out to be inaudible
(E019). Here, with audibility fixed, comprehension without the scaffold is *still*
exactly zero — H2b's diagnosis (no innate response to hearing a call, so nothing for the
rule to reinforce) is confirmed as a real architectural fact on the corrected system,
not an artefact of the defect that motivated it.

**Neither audience effect emerges, and the one promising lead does not replicate.**
Both `alarm_effect` and `food_effect` stay at or near zero across every condition,
before and after rearing. The one point of live interest registered in §3 — E005's
`food_effect` trending positive (+0.032, t=0.64, never confirmed) — does not replicate
on 8 fresh, properly-instrumented seeds; if anything the point estimates here trend
slightly negative, well within noise. The required fixed control holds flat throughout
(exactly 0.000 in both channels), confirming the assay itself is sound.

## 7. Interpretation

**H3's null is now confirmed on the corrected system, closing the "was it ever a
measurement problem" question for good.** E005/E006 ran on a connectome with an
inaudible channel and a saturated call-cost reward; both are fixed, and the same null
persists, cleanly, on both audience-effect channels.

**Combined with E036/E040, H3 has now been tested both ways this project could imagine
it working, and both fail.** Without a comprehension scaffold (this file): comprehension
stays at zero, so there's nothing for audience-conditional learning to be built on — a
foothold problem, exactly as H2b originally diagnosed. With a comprehension scaffold
supplying that foothold directly, by construction (E036/E040): comprehension is real
(0.19) but learned audience-sensitivity still doesn't emerge on top of it. The
"blocked by H2b" story was correct as far as it went and incomplete: fixing the
precondition it named did not fix the outcome, the same pattern E042–E044 already
found for H2c specifically.

**The food-call channel is not a distinct, unexplored angle after all.** It was
genuinely untested since the audio fix and worth checking (§2), but the result is the
same null as the alarm-call channel, not a different story — no evidence that food
calls, despite lacking the "no innate response" problem in the same form, offer an
easier path to audience-sensitive learning.

## 8. Consequence

- **H3 stays `UNDER TEST`, but the blocking explanation is corrected.** Not "blocked by
  H2b" as a standalone, sufficient explanation — H2b's precondition has now been
  supplied directly (E036/E040) and the outcome didn't change. H3's null traces to the
  same underlying limitation E042–E044 diagnosed for H2c: the learning rule, whichever
  pathway it runs through, does not build the specific new contingency these tasks need,
  even when every named precondition is met.
- **E005's `food_effect` citation should not be carried forward as a live, unresolved
  lead.** It was a real, correctly-flagged uncertainty at the time (t=0.64, genuinely
  underpowered); on a proper re-test it does not hold up. Recorded as non-replicating,
  not deleted.
- **This closes the last open "maybe it's just measurement" explanation for H3.** Every
  instrument-level defect this project has found and fixed (inaudible channel, saturated
  reward, E/I bug, food depletion) has now been controlled for in at least one H3 test.
  What remains is the same architectural question already open for H2c and H2's own
  clean null: whether the rule, in its current form, can ever build the kind of new,
  context-conditional behaviour these hypotheses need — not a defect still waiting to
  be found.
