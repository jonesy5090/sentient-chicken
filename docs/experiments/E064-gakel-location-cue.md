# E064 — a gakel-call location cue for listeners beyond visual range

> **Pre-registered.** Sections 1–5 written before validation; the design itself was
> fully settled in discussion before any code was written — see §5's reasoning.

## 1. Parent hypothesis

**T2** — the rotating poisoned feeder. `NOT STARTED`, Stages 1/1b/1c (E060–E062)
complete. E063 added an allocentric place-cell channel as a Stage 2 prerequisite, but
review of E063 itself surfaced a second, distinct gap: it only solves durable location
memory for a hen who directly *witnesses* a sickness event. This experiment fills the
second half — a hen who only *hears* the gakel call from beyond visual range.

## 2. Question

Trace the actual mechanism precisely, the way CLAUDE.md's "do the arithmetic" check
demands, rather than assume E063 already closed the loop. Two distinct cases:

1. **A hen close enough to see the incident.** `peck_radius=0.30` means a hen falls
   sick within 0.3 m of the feeder; `BIN_WIDTH_DEG=25°` means that separation stays
   inside one egocentric bin at any normal viewing distance. A nearby healthy hen gets
   `CLS_SICK` and `CLS_FOOD` active in the *same bin* simultaneously, and her own
   E063 place cells tag her current location at that moment. This pathway is real and
   needs nothing further.
2. **A hen who only hears the gakel call from beyond visual range** — precisely the
   case the call was built for ("broadcasts past visual range", E060's own design
   comment). Audio in this model has never carried direction: every call is pure
   distance-attenuated amplitude, no bearing. Her own place cells report *her*
   location, not the caller's. Nothing connects the two. For this hen, E063 alone
   changes nothing.

Does closing this gap need the pallium to learn to reconstruct a bearing from
scratch (which would also need her own allocentric heading exposed as a channel —
not currently available, and a further scope increase), or can a simpler, more
directly-grounded signal do the job?

## 3. Prediction

A signal computed directly from the caller's *own* place-cell pattern (reusing E063's
existing grid), weighted by how audible she currently is, should let listeners beyond
visual range receive a genuinely useful — if coarse — location estimate: strong and
sharp for a loud, nearby caller, faint and blurred for a distant one, exactly zero when
nobody is calling. This is arithmetic and geometry, not a hypothesis, so the value is
in verifying the implementation matches the design rather than assuming it does — the
same discipline E063 applied to itself.

The one property that is *not* just arithmetic and needed direct verification: under
`channel_mode='yoked'` (H4's headline control, which T2's own falsifier already names
as its planned C? condition), a listener must be cued to where the caller *was* when
she called, never her current position. Handing over the caller's current position
alongside a time-shifted call would leak exactly the real-time contingency the yoked
control exists to destroy — the same class of leak E024's shuffled control had for
plain audibility (retained 98% of the information it was meant to destroy). This is
the one place a real implementation bug was plausible rather than merely typo-level,
and is checked directly, not assumed correct by construction.

## 4. Falsifier

If the cue does not peak at the caller's location, does not fade with distance, or is
nonzero when nobody is gakel-calling, the geometry is wrong and must be fixed. If,
under `channel_mode='yoked'`, the cue reflects the caller's *current* position rather
than her position at call time, the control is broken and no C? contrast built on top
of it could be trusted — this is the single highest-priority check in this experiment,
given how directly it bears on Stage 2's eventual validity. Separately: if adding this
channel changes any previously-validated result (full test suite, full ethogram), that
is a regression this experiment must catch.

## 5. Design

**Why this design, not a bearing channel.** A first design draft considered exposing
an egocentric angle-of-arrival per call (closer to how real directional hearing
actually works) and letting the pallium combine it with self-location and heading to
triangulate a source. Rejected for two reasons: this model does not currently expose a
hen's own allocentric heading as an observation channel at all (only ever the relative
angle to something), so triangulation would need a *third* new prerequisite; and
asking a ~256-unit pallium to learn trigonometry from scratch is a much larger ask than
this project needs to make to test T2's actual question, which is about *whether
communication transfers useful information*, not about whether this network can
reinvent sound localisation unaided. The channel built instead hands over a
ready-made, already-blurred location estimate — computed the same way `CLS_SICK`
already hands over ground-truth-but-range-limited location rather than asking vision
to triangulate depth from raw disparity. What is still genuinely left to learn: whether
a hen attends to this channel at all, and whether she can bind it to a lasting
avoidance response — the actual open question, unchanged.

**Implementation**: `coop/spec.py` adds `GAKEL_PLACE_LO:GAKEL_PLACE_HI` (25 more
channels, reusing `N_PLACE`), appended after `PLACE_HI` — again the highest offset, no
existing index moves. `OBS_DIM` 113 → 138. `coop/sensing.py`'s `_place_cells` is
generalised to accept any leading batch shape (`(H, 2) → (H, N_PLACE)` for
self-location, `(H, H, 2) → (H, H, N_PLACE)` batched over listener and caller for this
channel). A new `_gakel_location_cue(weight, caller_pos, cfg)` computes, per listener,
an *un*normalised loudness-weighted sum of callers' place-cell patterns — deliberately
un-normalised so a faint call contributes a faint, low-confidence pattern rather than a
full-strength one, which is the "coarse" part of coarse directional hearing, achieved
without injecting artificial noise.

**The yoked correctness requirement** needed a real code change, not just a new
formula: `World` gains `pos_log`, a ring buffer matching `call_log`'s exact
shape/lifecycle (`(cfg.call_log_steps, H, 2)`, written every step in `world.step`),
so that under `channel_mode='yoked'` the caller's position can be read back at the same
lag as her call, via the same per-listener `idx` the existing yoked branch already
computes for `call_log`.

**Weighting respects the existing H4 condition ladder with no special-casing**: the
cue is built from `atten` (already routed through `_channel()`) and either
`w.calls`/`w.pos` directly (intact/shuffled/self/none/severed) or the lagged
`call_log`/`pos_log` slices (yoked) — the same branch structure `observe()` already
uses for the ordinary audio channel, extended rather than duplicated.

**Scope**: gakel-specific, not a general "where is every call coming from" upgrade —
matching the narrow, need-driven precedent every prior channel in this file follows
(`CLS_CROWDING` for personal space, `CLS_SICK` for sickness, wall-escape for walls). A
future hypothesis needing directional hearing for a different call is its own addition.

**Checks** (unit tests, `tests/test_phase0.py`):
1. `test_gakel_cue_is_zero_when_nobody_is_calling`
2. `test_gakel_cue_points_at_the_caller_not_the_listener`
3. `test_gakel_cue_fades_with_distance`
4. `test_yoked_gakel_cue_uses_the_callers_position_when_she_called_not_now` — the
   highest-priority check, per §3/§4 above.

**Regression check**: full `pytest tests/ -q` must still pass, including the unchanged
ethogram.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
```

## 6. Result

`pytest tests/ -q`: **81/81 passed** (77 pre-existing + 4 new gakel-location-cue
tests). All four checks passed after one fix (below):

- Zero when nobody is calling: exact zero.
- Cue peaks at the caller's location, not the listener's own — confirmed directly
  against `_place_cells` computed on the caller's true position.
- Fades with distance: a farther caller produces a strictly weaker cue.
- **The priority check**: under `channel_mode='yoked'`, a listener's cue peaks at the
  caller's position *at the moment she called*, and reads effectively zero
  (`< 1e-6`) at her later, current position — verified directly with a caller who
  moved 22+ m between calling and observation, staying within `hear_range` the whole
  time so the check isolates the position leak from the (separate, expected) "did she
  move out of range" effect.

**One real bug caught during validation, in the test rather than the implementation**:
`test_gakel_cue_points_at_the_caller_not_the_listener`'s first draft placed the two
hens 18.4 m apart — beyond `hear_range=15.0` — so `atten=0` correctly zeroed
everything and the test failed for the right reason applied to the wrong claim.
Fixed by moving the pair to 14.1 m apart (still clearly distinct place cells, now
actually audible). Caught by running the test rather than reasoning about it — the
same "measure it, don't just write the prose" discipline this project applies to
itself throughout.

## 7. Interpretation

The channel does what it claims, including the one property that actually mattered
enough to need direct verification rather than "it's arithmetic, it's fine" — the
yoked-mode position leak. That this required a real code change (`World.pos_log`, a
second ring buffer alongside `call_log`) rather than only a new formula is itself
informative: had this channel been added without checking the interaction, a listener
under `channel_mode='yoked'` would have received a caller's *current* real-time
position stapled to a decorrelated call, silently reintroducing exactly the
contingency the control exists to destroy. Every future L vs. C? contrast in T2 Stage
2 depends on this control being sound, so finding and fixing this now — before it
could quietly bias a result — is the same value E024's own correction eventually
delivered for H4, just paid earlier this time.

## 8. Consequence

The second Stage 2 prerequisite is validated. Between E063 and E064, the model now
has: (a) a hen who witnesses a sickness event directly can tag her own location via
place cells, and (b) a hen who only hears the gakel call from beyond visual range can
receive a coarse, honestly-degraded location estimate of the caller, correctly
decorrelated from real-time position under the yoked control. Update
`docs/hypothesis.md`'s T2 node to record both prerequisites and their validated
status; T2 remains `NOT STARTED` — this is scaffolding, not a hypothesis result.
`docs/backlog.md` already updated with this experiment's summary.

Stage 2 (the actual L vs. C? learning contrast) can now be designed. Both halves of
the location problem — direct witness and indirect testimony — have a real channel to
learn on, and both correctly respect the H4-style condition ladder T2's own falsifier
already commits to reusing.
