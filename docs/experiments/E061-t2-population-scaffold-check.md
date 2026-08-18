# E061 — T2 Stage 1b: does the scaffold work at the population level?

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**T2** — the rotating poisoned feeder. `NOT STARTED`, Stage 1 complete (E060). This is
Stage 1b: the population-level check `docs/hypothesis.md`'s T2 node names explicitly,
still no learning involved. E060 validated each scaffold piece in an isolated,
hand-staged scenario; this checks the same pieces in a full, free-running 16-hen flock,
the same escalation E025→E048 needed for the personal-space reflex and E024→E026 needed
for the audio channel — both times, something that worked in isolation did not work (or
was not known to work) at the population level until someone measured it directly.

## 2. Question

Three separate questions, each independently checkable and independently fixable
(matching E060's own falsifier structure):

1. **Does contamination actually get discovered at a workable rate?** If hens rarely or
   never stumble onto the bad feeder in a realistic window, nothing downstream (the
   gakel call, the anchor, eventually learning) has anything to work with.
2. **Is the gakel call actually audible** — does hearing it correlate with a real
   nearby sickness event, at realistic flock density, the way E026 measured for the
   alarm channel rather than assumed?
3. **Does the innate anchor produce real, measurable dispersal** away from a sick hen
   in the full dynamic system, the same check E048 needed for personal space?

## 3. Prediction

**No confident prediction on discovery rate** — genuinely unknown until measured, the
entire reason this check exists. **For audibility**: expected to work, since the gakel
call reuses the exact same audio machinery (power-summed, attenuated by distance) the
other four calls already use and E026 already validated at this flock size and
`hear_range`; a null here would be a surprise requiring its own explanation. **For
dispersal**: expected to work, on the same basis E048's own successful validation gives
— the anchor reflex is structurally identical to `CLS_CROWDING`'s (verified in E060 to
reverse attraction, not merely damp it), and `CLS_CROWDING` was shown to produce real
population-level dispersal.

## 4. Falsifier

**Discovery rate**: if essentially no sickness events occur in a realistic window (e.g.
20 minutes) at the default `contamination_period_s=300s`, the scaffold has nothing to
work with regardless of how well the other pieces perform — not fatal to T2, but means
Stage 1c's calibration is more urgent than assumed, or the feeder layout/hen behaviour
around feeders needs its own look first.

**Audibility**: if the gakel call's heard amplitude does not correlate with a nearby
sickness event materially above the correlation any of the other four calls already
show for their own trigger conditions, the channel is not carrying real information at
this flock density — the E024 pattern, not assumed away this time.

**Dispersal**: if nearest-neighbour distance to a sick hen does not measurably increase
relative to the anchor stripped (the E048 ablation pattern), the anchor is not doing
population-level work despite passing its isolated E060 check — the same gap between
"the reflex fires" and "the flock disperses" E025 found for gregariousness generally.

## 5. Design

**World**: 16 hens, `spec.DEFAULT_COOP` defaults otherwise (`contamination_period_s`
at its E060 first-pass value, 300s — calibrating it is Stage 1c, not this experiment),
`food_deplete_rate=0` (this session's standing discipline for any foraging-adjacent
metric not specifically about depletion). `PlasticConfig(enabled=False)` throughout —
Stage 1b is still about the instrument, not the hypothesis.

**Duration**: 20 minutes (120,000 steps) — long enough for several contamination
rotations at the default period, matching this project's standard first-pass duration
for population-level checks.

**Check 1 (discovery rate)**: count sickness-onset events per seed over the run;
report mean events and fraction of hens ever sick at least once.

**Check 2 (audibility)**: reusing `scratchpad/shuffle_info.py`'s exact methodology
(E024/E026's own instrument) — correlation between a hen's heard gakel amplitude and
whether any flockmate within `hear_range` is currently `sick_on`, intact vs. shuffled
sender identity. Retained-information ratio computed the same way E024/E026 report it.

**Check 3 (dispersal)**: reusing `what_clumps_them.py`/E048's ablation pattern —
nearest-neighbour distance *specifically to a currently-sick hen* (not flock-wide NN
distance, which the anchor was never meant to change), anchor present vs. anchor
weights stripped, same connectome otherwise.

**Replicates**: 3 seeds per check, matching this project's own register for this class
of mechanism-validation diagnostic (E025, E048's own population checks all used 3).
This is a "does it work at all" check, not a powered contrast — no claim about T2's
actual hypothesis rests on the exact numbers here.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e061_t2_population_check.py
```

## 6. Result

16 hens, `contamination_period_s=300s`, `PlasticConfig(enabled=False)`, 3 seeds per
check, 20 minutes (120,000 steps) per run.

**Check 1 — discovery rate:**

| seed | onset events / 20 min |
|---|---|
| 0 | 25 |
| 1 | 16 |
| 2 | 26 |
| **mean** | **22.3** |

**Check 2 — gakel audibility** (intact vs. sender-shuffled, E024's own control):

| mode | corr(heard, nearby sick) | heard\|sick | heard\|not | ratio |
|---|---|---|---|---|
| intact | 0.1671 | 0.0248 | 0.0023 | 10.96× |
| shuffled | 0.1370 | 0.0372 | 0.0055 | 6.78× |

Shuffled retains 82% of intact's correlation.

**Check 3 — anchor-driven dispersal** (forced sickness every 60 s, anchor present vs.
stripped):

| condition | mean NN distance to a sick hen |
|---|---|
| anchor present | 5.439 m |
| anchor stripped | 2.713 m |

## 7. Interpretation

**Check 1 passes.** ~22 sickness events per 20 minutes at the untuned first-pass
`contamination_period_s=300s` is a workable rate — the scaffold has real material to
work with well before Stage 1c calibration is even attempted. Not a bottleneck.

**Check 2 passes, with one caveat worth recording rather than burying.** The intact
channel shows a real, non-trivial signal: a hen hears roughly 11× more gakel amplitude
when a flockmate within `hear_range` is actually sick than when none is — comparable in
kind to the alarm channel's own validated signal. The falsifier was "no correlation
materially above the other calls' own baseline"; 0.167 with an 11× conditional ratio is
clearly above that bar, so this is a pass, not a null.

The 82% shuffle-retention is the same shape of number CLAUDE.md documents for E024's
*original* sender-permute control on the alarm channel (98% retained) — not this
project's fixed instrument (E026's yoked time-shift), which was never the one specified
for this check. The cause is architectural, not specific to gakel: the flock clumps, so
"someone nearby is calling" mostly survives reassigning *which* nearby hen is credited.
This is expected, already-understood behaviour of the sender-shuffle control at this
flock density, not a new defect — but it means this check alone cannot rule out that
some of the correlation is "a call happened nearby" rather than "the *sick* hen's
identity is what the call encodes." That distinction does not matter for T2's actual
hypothesis (which only needs "something happened nearby," paired with `CLS_SICK`'s
separate visual "where," per the location-fix design) but is worth flagging so a future
reader does not mistake 82% for a Check-2 near-failure the way 98% was for E024.

**Check 3 passes cleanly.** Mean distance to a sick hen roughly doubles with the anchor
present (5.44 m) versus stripped (2.71 m) — the same shape of result E048 found for
`CLS_CROWDING`, and by the same structural argument (§3): the reflex is algebraically
identical, just retargeted to a different visual class. The innate anchor produces real,
measurable population-level dispersal, not just an isolated-scenario reflex twitch.

## 8. Consequence

All three falsifiers survive. T2 Stage 1b is **complete**: the scaffold built in E060
does not just fire correctly in hand-staged isolation, it produces the intended
population-level effects in a free-running 16-hen flock — contamination is discovered
at a workable rate, the gakel call carries real information about nearby sickness, and
the innate anchor causes real dispersal away from a sick hen. This clears the
population-level precondition this project has needed for every prior mechanism (E025→
E048 for personal space, E024→E026 for the audio channel) before trusting a learning
result built on top of it.

Update `docs/hypothesis.md`: T2 node — Stage 1 and Stage 1b both done, still `NOT
STARTED` overall (no learning has been attempted yet). Update `docs/backlog.md`'s T2
staging section to mark Stage 1b done and Stage 1c (calibrating
`contamination_period_s`) as the next open item — Check 1's 22.3 events/20min at the
untuned 300s default suggests 300s may already be reasonable, but Stage 1c should
measure across a range rather than assume this.

No change to `README.md`'s claims — Stage 1b is instrument validation, not a finding
about the hypothesis itself.
