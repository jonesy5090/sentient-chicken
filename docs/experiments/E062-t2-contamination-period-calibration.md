# E062 — T2 Stage 1c: calibrate `contamination_period_s`

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**T2** — the rotating poisoned feeder. `NOT STARTED`, Stage 1 (E060) and Stage 1b
(E061) both complete. This is Stage 1c: `docs/backlog.md` §2 names the constraint
directly — "the answer must change faster than an individual can learn it, but slower
than the flock can propagate it... finding that band is a sweep, and finding it is
itself a result." `contamination_period_s=300.0` has been a first-pass placeholder
since E060, explicitly flagged there and in `coop/spec.py`'s own comment as not yet
calibrated.

## 2. Question

Three sub-questions, each independently checkable:

1. **Does the acoustic/visual "audience" a sickness event reaches actually grow with a
   longer period, or does it saturate quickly?** The arena is 20×20 m and
   `hear_range=15.0` — already noted in `coop/spec.py`'s own comment that "every hen
   already hears every other." If that holds, the *call* side of "slow enough to
   propagate" is nearly free at this flock size, and the real constraint is the
   *visual* `CLS_SICK` cue (gated by `vision_range=10.0` and actual proximity, not a
   flock-wide broadcast) — a hen needs to actually pass near the sick hen while she is
   still sick.
2. **Do rotations overlap with a still-recovering sickness window?**
   `sickness_duration_s=60s` is fixed (already validated, not swept here). If
   `contamination_period_s` is not comfortably larger than that, a new rotation can
   fire while a hen from the *previous* rotation is still visibly sick — which would
   let the location signal Stage 2 depends on get attributed to the wrong (now-safe)
   feeder. This is a correctness precondition for Stage 2, not just an efficiency one.
3. **How many rotations does a given period buy in a realistic run length?** Too long
   a period starves Stage 2 of punishment events to learn from, regardless of how well
   checks 1–2 look.

## 3. Prediction

**Check 1 (audience size)**: expected to saturate quickly, at or near the shortest
candidate period — the same "every hen hears every other" argument already documented
for the auditory channel at this arena size, extended to the (slower, proximity-gated)
visual channel over a 60-second sickness window in a 20×20 m arena. A period much
longer than `sickness_duration_s` should not buy materially more audience.

**Check 2 (overlap)**: expected to be near zero once `contamination_period_s`
comfortably exceeds `sickness_duration_s` (60s) — a straightforward arithmetic
consequence, not a subtle result — but this project's own rule is to measure the
"straightforward" claim rather than assume it, so it is checked at every candidate
including ones deliberately too short, to see the failure mode rather than infer it.

**Check 3 (rotations per run)**: mechanically determined by duration/period, included
for completeness so the tradeoff is visible in one place rather than requiring a
separate arithmetic note in the writeup.

## 4. Falsifier

**Check 1**: if audience size keeps growing materially across the candidate range
(not saturating near the short end), the "propagation is nearly free here" story is
wrong, and a longer period is genuinely buying more reachable audience — a real
finding that would revise the design note in `coop/spec.py`.

**Check 2**: if overlap fraction is non-negligible (>5%, an arbitrary but stated
threshold) at any candidate being considered as the eventual default, that candidate
is ruled out regardless of how Check 1 looks — this is a correctness gate, not a
tradeoff to optimise around.

**Check 3**: informational, not falsifiable on its own — read alongside Checks 1–2 to
pick a value with enough rotations for Stage 2's eventual statistical power.

## 5. Design

**World**: 16 hens, `spec.DEFAULT_COOP` defaults otherwise, `food_deplete_rate=0`
(standing discipline for foraging-adjacent metrics not specifically about depletion).
`PlasticConfig(enabled=False)` — Stage 1c is still about the instrument/parameter, not
the hypothesis.

**Sweep**: `contamination_period_s` in `{100, 200, 300, 450, 600}` seconds — brackets
the current 300s default on both sides, plus one candidate (100s) deliberately close
to `sickness_duration_s=60s` to observe the overlap failure mode directly rather than
only from a safe distance.

**Duration**: a fixed 40 minutes (240,000 steps) for every candidate, rather than a
duration scaled to each period — keeps the sweep a single consistent instrument and
still gives the longest candidate (600s) at least 4 rotations and the shortest (100s)
24, enough for both ends of the tradeoff to be visible.

**Check 1 (audience)**: per sick-timestep, count *other* hens within `vision_range`
(10 m) of any currently-sick hen; report the mean, matching E061 Check 3's
distance-tracing methodology.

**Check 2 (overlap)**: at each contamination-epoch transition (rotation), record
whether any hen is still `sick_on` at that instant; report the fraction of rotations
where this is true.

**Check 3 (rotations)**: count contamination-epoch transitions per run; report mean
across seeds.

**Replicates**: 3 seeds per period, matching this project's own register for this
class of mechanism-calibration diagnostic (E025, E048, E061 all used 3).

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e062_t2_period_sweep.py
```

## 6. Result

16 hens, `PlasticConfig(enabled=False)`, `food_deplete_rate=0`, 40-minute runs
(240,000 steps), 3 seeds per period.

| period (s) | mean audience | rotations | overlap frac |
|---|---|---|---|
| 100 | 13.673 | 23.0 | 0.609 |
| 200 | 14.076 | 11.0 | 0.485 |
| 300 | 13.941 | 7.0 | 0.429 |
| 450 | 14.160 | 5.0 | 0.533 |
| 600 | 13.862 | 3.0 | 0.556 |

## 7. Interpretation

**Check 1 confirmed exactly as predicted.** Mean audience is flat at ~14 (out of a
possible 15 — the whole rest of the flock) across every candidate period, including
the shortest (100s). The "propagation is nearly free at this arena size" story holds:
the gregariousness reflex already keeps the flock clumped tightly enough (E025's own
finding) that a sick hen is visible to nearly everyone almost as soon as she is sick,
regardless of period. A longer period buys no additional reachable audience.

**Check 2 did not confirm the prediction, and the reason is itself the finding.**
Overlap fraction was expected to fall toward zero once `contamination_period_s`
comfortably exceeded `sickness_duration_s=60s`. Instead it stayed in the 43–61% band
across the *entire* sweep, including 600s — ten times the sickness duration — with no
clear monotonic trend (450s measured higher than 300s, most likely seed noise given
only 3–5 rotations to average over at the longer periods).

The arithmetic explains why. E061 measured ~22.3 sickness-onset events per 20 minutes
at the 300s default — about 5.6 discovery events per rotation. At 60s of sickness
each, that is **~335 cumulative sick-seconds per 300-second rotation** — more sick-time
than the rotation itself contains. In this uncalibrated, *no-learning* baseline, the
flock does not discover the bad feeder once per rotation and then fall quiet; multiple
hens independently rediscover it throughout the whole window, so someone is sick
almost continuously regardless of how long the period is. Overlap, as measured here,
is not really testing "does the previous rotation's straggler linger too long" — it is
picking up this much larger, rotation-independent background rate of *repeated*
mistakes, which is exactly the "C? pays roughly N times" baseline T2's own prediction
(§2 of the backlog) describes, now directly measured rather than assumed.

One real limitation of this check as built: it asks "is *any* hen sick right now,"
not "is a hen still sick specifically from the feeder that was *just* rotated away
from." Those are different risks — the second is the one the design concern in §4 was
actually about (a stale cue misattributed to a now-safe feeder) — and distinguishing
them needs per-feeder sickness attribution that `World` does not currently track
(`at_bad_food_prev` is a flat boolean, not indexed by which of the `n_food` feeders
caused it). This check cannot rule the specific misattribution risk in or out; it
measured a real and larger effect instead.

**Check 3** behaves exactly as arithmetic predicts: rotations fall roughly as
duration/period (23, 11, 7, 5, 3), the direct tradeoff against Stage 2's eventual
statistical power.

## 8. Consequence

Checks 1 and 3 give no reason to move off the 300s default: audience is already
saturated at every candidate tested, so a longer period buys nothing on that axis,
while a shorter period would only help Check 3's rotation count. Check 2 does not
discriminate between candidates either, once its actual mechanism is understood — the
background repeated-mistake rate this measured is independent of period at this flock
size and discovery rate. **`contamination_period_s` stays at 300.0`** — Stage 1c found
no evidence to recalibrate it, which is itself the result this stage was designed to
produce (a checked "no change needed," not a skipped check).

The real, useful finding is the reframing of overlap: at 300s and no learning, the
flock is essentially never free of at least one sick hen (~43% overlap even measured
at the coarse, any-feeder granularity) — a direct, measured picture of the "C? pays
roughly N times" baseline, not the "one mistake per rotation" state T2 predicts *L*
should approach. That gap between baseline and prediction is Stage 2's actual claim to
test, not a defect to fix here.

Flagged for Stage 2 (or a future addendum, only if it turns out to matter
empirically): per-feeder sickness attribution does not exist yet, so the specific
stale-cue-misattribution risk named in this experiment's own falsifier (§4) remains
unmeasured. Worth building only if Stage 2's results show a pattern consistent with
misattribution (e.g., L failing to converge toward one mistake per rotation in a way
inconsistent with the learning-rule limits already known from E058/E059) — not worth
building speculatively now.

No change to `docs/hypothesis.md`'s T2 claim or `README.md`. `docs/backlog.md`'s T2
staging section: mark Stage 1c done, recommendation "no change," and note the
flagged-but-not-built per-feeder attribution gap.
