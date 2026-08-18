# E051 — wall avoidance: IDX_WALL was sensed but never wired to anything

> **Written after implementation, not before** — the third experiment in a row this
> session with that deviation (E048, E050). Flagged again rather than silently
> continuing the pattern; this needs to stop being normal.

## 1. Parent hypothesis

Not part of the H-tree. This is a reflex-arc gap discovered via the offline-replay
viewer (`viz/`, PR #16): a hen was observed reliably wandering to and sitting at the
edge of the arena. Nearest existing hypothesis: none directly, though a chronically
wall-pinned hen would confound any spatial-distribution metric (E025, E048, T1, T2)
the same way clumping did — worth recording for that reason even without a named
parent node.

## 2. Question

`coop/spec.py`'s `IDX_WALL` is sensed (`coop/sensing.py` computes proximity to the
nearest wall every step) but `hen/innate.py` never reads it. Does adding a reflex that
turns a hen away from the wall she's nearest to reduce how long she spends pinned
against the boundary?

## 3. Prediction

A hen carried to a wall by some other drive (foraging, cold, personal-space repulsion)
currently has nothing to turn her back — `actuation.py`'s position clip stops her
physically but not her heading, so she can sit there indefinitely. A working reflex
should cut both the fraction of hen-steps spent near a wall and, especially, the
longest single continuous dwell — the quantity that would actually be visible as "one
chicken always at the edge" in a several-minute recording.

## 4. Falsifier

If dwell time and longest-dwell are unchanged (or worse) with the reflex wired in, the
fix is not doing what it's meant to — most likely a sign error in which way it turns
her, which a working reflex must get right per-wall and per-heading, not just on
average.

## 5. Design

**The fix**: two new somatic channels, `IDX_WALL_ESCAPE_L`/`IDX_WALL_ESCAPE_R`
(`coop/spec.py`, `OBS_DIM` 71 → 73), computed in `coop/sensing.py` from the existing
`IDX_WALL` proximity plus the nearest wall's outward normal, expressed relative to the
hen's heading (the same frame `_bin_proximity` uses everywhere else) and split into two
non-negative "turn this way" magnitudes. `hen/innate.py` wires each straight to its
matching turn motor (weight 3.0, matching the existing ground-threat turn-away reflex —
the closest existing analogue).

**Deliberately no forward suppression.** `IDX_WALL` alone has no direction and doesn't
change when a hen turns, only when she moves — a reflex suppressing forward drive on it
directly would deadlock (turning away doesn't lower `IDX_WALL`, so the suppression would
never release, and she'd spin in place rather than walk out). The kinematics make this
avoidable: `heading += turn * turn_rate * dt` has no `speed` term, so the turn reflex
rotates her independent of forward motion; whatever forward drive is already active
(tonic, hunger, thirst) carries her out once she's facing clear.

**Diagnostic** (`scratchpad/e051d_wall_dwell_check.py`): ablation against the same
connectome (E025/E048 pattern) — reflex weights on the two new channels zeroed vs. left
as-shipped. 16 hens, 10-minute rollout (60,000 steps, matching a typical `run.record`
viz session's length), 3 seeds. **Primary metrics**: mean fraction of hen-steps with
`IDX_WALL > 0`, and the longest single continuous run any hen spends there.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e051d_wall_dwell_check.py
```

Also checked: an isolated single-hen unit test (`scratchpad/e051_wall_escape_check.py`)
placing a hen 0.1 m from a wall facing directly into it, confirming the reflex alone
(no other drives active) rotates her from 180° to 0° over ~3.6 seconds and she moves
back out to 0.43 m.

## 6. Result

```
pre-fix (no wall escape)     mean hen-steps near a wall: 0.50%  longest single dwell: 2261 steps (22.6s)
fixed                        mean hen-steps near a wall: 0.11%  longest single dwell: 234 steps (2.3s)
```

**The falsifier does not fire.** Both metrics move in the predicted direction and by a
large margin: overall wall-dwelling time drops ~4.5×, and the worst-case single dwell —
the number that determines whether a viewer notices a hen "stuck" at the edge — drops
from 22.6 seconds to 2.3 seconds, roughly a 10× reduction. 22.6 continuous seconds is
easily long enough to be conspicuous in a several-minute recording at normal playback
speed, matching the reported observation directly.

The isolated single-hen check confirms the mechanism is doing what it's supposed to,
not just moving an aggregate statistic: a hen facing directly into a wall smoothly
rotates to face away and walks back out, with no other drives present to help her.

## 7. Interpretation

**This was a straightforward reflex-arc gap, not a subtle emergent effect.** `IDX_WALL`
being sensed but unwired is exactly the kind of omission the offset-based observation
layout makes easy to miss — nothing in the type system or the test suite flags an input
channel with zero reflex weight, and the position clip in `actuation.py` was enough to
keep the simulation numerically well-behaved (no hen ever left the arena), which is
probably why this went unnoticed until someone watched a recording rather than read
aggregate statistics.

**2.3 seconds of worst-case dwell is not zero.** The reflex reduces but does not
eliminate wall contact — a hen can still touch a wall briefly while her heading catches
up, particularly right after being carried there fast (e.g. fleeing, or freshly
repelled by crowding). This is expected and not a defect: continuous real-time turning
takes some nonzero time, and the fix's job was to stop indefinite pinning, not to make
walls unreachable.

## 8. Consequence

- **New reflex shipped**: `hen/innate.py` wires `IDX_WALL_ESCAPE_L/R` at weight 3.0.
  `OBS_DIM` moves 71 → 73.
- **New unit tests** (`tests/test_phase0.py`):
  `test_wall_escape_channels_point_away_from_the_nearest_wall` (channel gating and
  sign correctness, per wall and per heading) and
  `test_wall_escape_reflex_turns_a_cornered_hen_away` (end-to-end, isolated).
- **A pre-existing test's seed sensitivity was exposed, not caused, by this change**:
  `test_being_caught_does_not_dominate_the_reward_where_hawks_are_common` uses a single
  hardcoded seed pair and, per its own docstring, was already running at "the shortest
  window measured to contain one" strike — already marginal. `OBS_DIM` growing shifted
  what that fixed seed draws for the connectome's random init (expected: any structural
  change to the observation layout does this), and seed 0's closest approach moved from
  a hit to a 1.549 m miss against a 1.5 m strike radius. An 8-seed sweep found strikes in
  6 of 8 seeds, confirming this is seed sensitivity rather than a behavioural regression
  (wall proximity was 0% active throughout that test's specific run, so the new reflex
  never even fired). Fixed by trying a short, fixed list of seeds rather than trusting
  one — the test's own logic is otherwise unchanged.
- **No hypothesis-tree status changes.** This doesn't ladder up to a named H-node; it's
  filed here for the record and because it touches `OBS_DIM`, the same re-baselining
  concern E048 raised.
