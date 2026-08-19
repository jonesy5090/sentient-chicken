# E065 — T2 Stage 2: does the flock learn to avoid the poisoned feeder?

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**T2** — the rotating poisoned feeder. `NOT STARTED`. Stages 1/1b/1c
(E060–E062) validated the scaffold; E063/E064 filled two representational
prerequisites (self-location via place cells, and a location cue for hens who only
hear the gakel call). This is the actual hypothesis test `docs/backlog.md` §3
originally specified: **a flock with a working communication channel converges
toward roughly one hen's mistake per contamination rotation; a flock without one
pays roughly N times that.**

## 2. Question

With the validated scaffold, the two location prerequisites, and H2f's own validated
learning rule (`hebbian_readout=True`, `readout_scaling_strength=0.3` — the only rule
this project has ever measured building a genuine, replicated positive learning
result, E057), does a flock hearing the real, intact gakel channel (**L**) reduce its
sickness rate over developmental time more than a flock hearing a decorrelated,
yoked version of the same channel (**C?**) — and does either differ from a flock that
cannot learn at all (**S**, a fixed connectome)?

## 3. Prediction

**No confident prediction that this succeeds.** E058/E059 already found this
project's only validated learning rule amplifies an *existing innate anchor*; it does
not build a new stimulus-response association from nothing. T2's anchor (turn away
from `CLS_SICK`) already exists and already produces real dispersal (E061), so the
precondition E058/E059's finding requires is satisfied — closer to H2f's own task
shape than H2c's. But H2f's anchor was audience-conditioned calling, a comparatively
short-latency association; T2 asks for something new even by H2f's standard: a
durable bias that outlasts the visible cue and generalises across a changing set of
egocentric encounters with the same allocentric place. Whether the rule can do that is
genuinely open, which is the entire reason this stage exists rather than being assumed
from E057.

**If it works**, expect: L's sickness-per-rotation rate falls further from early to
late in the run than C?'s does (a difference-in-differences, not a flat average — the
same design E057 used to separate a real effect from ambient drift), and S (no
learning at all) shows no comparable within-run trend in either direction.

## 4. Falsifier

If L's early-to-late reduction in sickness-per-rotation is not clearly larger than
C?'s, the anchor-amplification mechanism validated for H2f does not transfer to T2's
specific durable, place-based contingency — consistent with, not contradicting,
E058/E059's own finding about this rule's limits. Not a falsifier of H2f, which
already stands on independent evidence, nor of E063/E064's prerequisites, which are
validated separately on their own geometric/correctness grounds regardless of what
learning does with them.

**Mandatory diagnostics, run before trusting any positive result** (CLAUDE.md's own
standing rule):
1. **Is the rule actually active?** Report mean `|W_out|` drift for both L and C? over
   the run. If it's ~0, nothing was learned regardless of the metric.
2. **Matched control metric.** Alongside sickness-per-rotation, measure an unrelated
   outcome with no mechanistic route to the gakel channel — total water intake. If
   *this* also shows an L > C? early-to-late difference, the result is general
   excitability or maturation, not a targeted effect (E058's own diagnostic, reused).
3. **S baseline.** If S also shows a within-run sickness-rate trend comparable to L's,
   the effect is population-level habituation/exploration-reduction over developmental
   time, unrelated to either channel or to learning at all.

## 5. Design

**World**: 16 hens, `spec.DEFAULT_COOP` defaults (`contamination_period_s=300s`,
confirmed via E062's sweep as needing no change; `food_deplete_rate=0`, this session's
standing discipline for foraging-adjacent metrics).

**Conditions**, isolating channel content exactly as H4's own ladder does (same
bandwidth, same cost, differing only in whose calls carry real information):

| condition | `PlasticConfig` | `channel_mode` |
|---|---|---|
| **S** | `enabled=False` | `intact` (moot — no learning to bias) |
| **C?** | H2f's validated rule (below) | `yoked` |
| **L** | H2f's validated rule (below) | `intact` |

H2f's validated rule, unchanged from E057: `PlasticConfig(enabled=True,
growth_enabled=False, kin_audible=True, explore_sigma=0.6, hebbian_readout=True,
readout_scaling_strength=0.3)`. `channel_mode='yoked'` requires
`call_log_steps=spec.YOKE_LOG_STEPS`, per `coop/sensing.py`'s own guard.

**Duration**: 90 minutes (5400s) — at the 300s rotation period this gives exactly 18
completed rotations per run, enough to define a clean early window (rotations 1–4)
and late window (rotations 15–18) with ten rotations of runway between them for
learning to accumulate, while keeping compute bounded (540,000 steps/run).

**Metric**: sickness-onset events, summed across the whole flock, bucketed by
`contamination_epoch` (exact rotation boundaries, no detection heuristic needed —
epochs are deterministic from `t` and `contamination_period_s`). Early rate = total
onsets in epochs 0–3 ÷ 4; late rate = total onsets in epochs 14–17 ÷ 4. Primary
statistic: `(late − early)` for L vs. the same quantity for C?, one-sample and
paired t-tests against `run.experiment._t_critical`, matching this project's
established statistical apparatus.

**Replicates**: 8 seeds per condition, matching H2f's own final, validated sample
size (E057) — the standard this project already treats as sufficient to trust a
headline positive result, rather than the 3-seed register used for mechanism-only
diagnostics (E025/E048/E061/E062).

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e065_t2_stage2_contrast.py --seeds 8 --minutes 90
```

## 6. Result

8 seeds/condition, 90 min (18 rotations of 300s), `threshold(df=7)=2.365`.

**Primary — sickness-per-rotation, early (rotations 1–4) vs. late (rotations 15–18):**

| condition | early | late |
|---|---|---|
| S | 5.906 | 7.625 |
| C? | 9.312 | 9.062 |
| L | 8.562 | 9.438 |

| contrast | mean ± SE | t | |
|---|---|---|---|
| L: late − early | +0.875 ± 0.693 | 1.26 | not significant |
| C?: late − early | −0.250 ± 0.620 | 0.40 | not significant |
| S: late − early | +1.719 ± 1.209 | 1.42 | not significant |
| **PRIMARY: (L late−early) − (C? late−early)** | **+1.125 ± 0.715** | **1.57** | **not significant** |

**Diagnostic 1 (is the rule active?)**: final mean `|W_out|` — C? 0.0632, L 0.0632,
identical to four decimal places.

**Diagnostic 2 (matched control, water intake)**: (L late−early) − (C? late−early) =
−266.5 ± 361.4, t=0.74, not significant.

Wall clock: 3238s (~54 min) for all 24 runs.

## 7. Interpretation

**The falsifier fires. Not just non-significant — the sign points the wrong way.**
L's early-to-late change is *more positive* (sickness got slightly worse) than C?'s
(which got slightly better), the opposite of the prediction, well short of
significance either way. Nothing here supports T2's headline claim at this
configuration.

**Diagnostic 1 needs a careful read, not a dismissal.** C? and L landing on the exact
same aggregate `|W_out|` looks at first like "the rule wasn't active," but that is not
the right conclusion: this is a mean over the *entire* readout matrix, and E057's own
finding was that channel content changes *which* weights move (a targeted,
audience-specific component riding alongside a general one), not necessarily the
aggregate magnitude. A rule that is genuinely active but produces localized,
channel-specific changes too small to move the whole-matrix mean would look exactly
like this. What this diagnostic rules out is a *bigger* failure mode (the rule
silently disabled, e.g. `pc.enabled` not taking effect) — it does not, on its own,
distinguish "no differential learning happened" from "differential learning happened
but stayed local." That distinction was not checked here and would need a more
targeted weight-level diagnostic to resolve, the same escalation E057 needed after
E055/E056's own aggregate measures were ambiguous.

**S's own early-to-late change is worth noticing.** S (fixed connectome, no learning
at all) shows the *largest* nominal increase (+1.719) of the three conditions, though
also not significant. This is consistent with the sickness-rate trend across a run
being dominated by within-run population dynamics common to every condition
(exploration patterns, spatial spread, or some other developmental drift unrelated to
learning) rather than by anything channel- or learning-specific — exactly the kind of
confound the S baseline exists to catch, and it did its job here even though nothing
reached significance.

**Diagnostic 2 is a clean, uninformative null**, which is the right outcome for a
diagnostic guarding against a false positive that never arose — nothing to defend
here since the primary result itself was null.

**Is this a trustworthy null, per CLAUDE.md's own rule ("a null is only informative if
the instrument could have shown a positive")?** The metric has real room to move in
both directions (values ranged 5.9–9.4 across conditions and windows, nowhere near a
ceiling or floor), and the underlying learning rule, statistical apparatus and
seed-matched paired design are the exact ones E057 used to detect a real, replicated
effect in a different context — so the apparatus is known-capable of finding signal
when it exists. What was *not* run here is a positive control specific to this exact
metric (sickness-per-rotation) — a planted, guaranteed-detectable effect to confirm
this particular pipeline can see it. Flagged as a real gap, not glossed over: if this
null is ever treated as more than a first-pass result (e.g. before writing it into
`README.md`'s summary of what works), that control is the next thing to run, not an
afterthought.

## 8. Consequence

**T2's headline prediction is not supported at this configuration** — the H2f rule
that produced this project's only genuine, replicated positive learning result
(audience-sensitive calling, E057) does not carry over to building the durable,
place-based avoidance T2's claim requires, even with both representational
prerequisites (E063, E064) in place. Consistent with, not contradicted by, E058/E059's
own finding that this rule amplifies existing anchors rather than building new
stimulus-response associations from nothing — T2 asked for something harder than
either H2f or H2c did (a bias that must generalise across a changing set of egocentric
encounters with the same allocentric place, not just re-time an existing response),
and it did not clear that bar.

Update `docs/hypothesis.md`'s T2 node: status moves from `NOT STARTED` to a tested,
clean null — the first time T2 has actually been checked against data rather than
designed. Update `docs/backlog.md`'s T2 section to record the result and close out the
open "Stage 2" line item.

**Not necessarily the end of T2.** Genuinely open, not pursued here: (1) the local
vs. aggregate weight-change question flagged in §7 — a targeted diagnostic could still
find a real but small, swamped-by-noise effect; (2) whether a longer run (more than 18
rotations) or a different learning rule entirely would fare better, though nothing in
this project's history suggests a better rule is close at hand; (3) whether the S
baseline's own unexplained upward drift deserves its own look before trusting *any*
future within-run comparison on this metric. None of these are scheduled — this
result is reported as what it is, a clean null on the configuration actually tested,
not a claim that the broader question is closed.
