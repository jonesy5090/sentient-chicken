# E060 — T2 Stage 1: build and validate the contamination/sickness/gakel scaffold

> **Pre-registered. Design only — nothing in this file is built yet.** Sections 1–5
> specify what will be built and how it will be validated; sections 6–8 are written
> after implementation and the validation run, per this project's standing discipline.
> This is Stage 1 of two: it builds and validates the innate scaffold with no learning
> involved. Stage 2 (the actual L vs. C? contrast) is a separate, later pre-registration
> that depends on this one passing.

## 1. Parent hypothesis

**T2** — the rotating poisoned feeder. `NOT STARTED`. This experiment does not test
T2's claim; it builds and validates the four pieces of scaffolding T2's claim depends
on, the same way E025/E048 built and validated the personal-space reflex before
anything used it, and E051 did the same for wall avoidance. "Test the instrument before
the hypothesis" (`CLAUDE.md`) applies with particular force here because the scaffold
is unusually large (a new world mechanic, a new call, a new vision class, and a new
reflex, built together) — validating each piece in isolation, and the whole thing
together, before Stage 2 spends any compute on a learning contrast is the entire point
of splitting this into two stages.

## 2. Question

Does the proposed scaffold — contamination, sickness with a physiological slowdown,
gakel-call production, the `CLS_SICK` visual channel, and an innate turn-away reflex —
actually behave as designed: a hen who eats contaminated food gets visibly slowed for a
bounded duration then recovers, calls once near the onset, and flockmates who see her
turn away, all without any learning switched on?

## 3. Prediction

Each piece should show up cleanly in a staged ethogram assay, matching the style
`run/probes.py` already uses for every other innate behaviour:
- A hen fed from a contaminated patch enters the sick state on the next step (rising
  edge, no delay) and her speed drops sharply relative to an unsick, otherwise
  identical hen.
- The gakel call peaks near sickness onset and returns to baseline well before
  `sick_t` reaches zero (the discovery-pulse shape, matching `IDX_FOOD_ARRIVAL`'s
  validated behaviour in E053).
- A flockmate placed near a staged sick hen shows a measurable turning bias away from
  her, with no such bias when the same flockmate is healthy.
- None of the above requires any change in `PlasticConfig` — the whole scaffold works
  identically with plasticity fully disabled, since none of it is learned.

## 4. Falsifier

Any of the four pieces failing its own staged check falsifies that piece specifically
(matching this project's practice of localising failures rather than reporting a
blanket "scaffold works/doesn't"): no rising-edge sickness onset, no measurable
mobility drop, gakel call not shaped like a discovery pulse (e.g. continuous instead,
repeating E053's original defect for a different channel), or no turning bias away
from a visible sick flockmate. Each is independently checkable and independently
fixable before Stage 2 begins.

## 5. Design

**New `World` fields** (mirroring existing patterns exactly, not inventing new ones):
- `food_contaminated: (F,) bool` — which patches are currently bad. Rotates on
  `contamination_period_s` (new `CoopConfig` field); invisible in `food_amount` or any
  existing observation, exactly as `docs/backlog.md`'s original design specifies.
- `sick_t: (H,) float`, `sick_on: (H,) bool` — same idiom as `hawk_t`/`hawk_on`. Set to
  `sickness_duration_s` on the rising edge of `fed & food_contaminated[patch]`
  (reusing the existing `fed`/`at_food` computation in `world.step`, gated by
  contamination the same way `feeders`/`food_amount` already reads `at_food`);
  otherwise decays via `max(sick_t - dt, 0)`.
- `sick_call_drive: (H,) float`, `at_sick_prev`-style edge tracking — same rising-edge
  idiom `food_call_drive`/`at_food_prev` already use (E053), so the gakel call gets the
  identical discovery-pulse shape by construction, not by a second implementation of
  the same idea.

**Physiological effect** (`coop/actuation.py`): a mobility multiplier applied
alongside the existing `crouch`-derived one — `sickness_mobility_scale` (proposed
first-pass value 0.15: markedly slowed, not frozen, matching "visibly slow / still"
rather than a hard freeze) — applied mechanically, reading `w.sick_on` directly, the
same way `apply_motor` already reads other `World` state. Not mediated by the reflex
arc or any learned weight: this is a physiological fact about being sick, not a
behaviour she is choosing or that could be trained away.

**New motor/observation channels** (`coop/spec.py`, cascading via the existing
offset-based layout):
- `M_CALL_GAKEL`, extending `MOTOR_DIM` 11 → 12 and `N_CALLS` 4 → 5.
- `CLS_SICK`, extending `N_VIS_CLASSES` 5 → 6 (the personal-space/wall-escape pattern:
  `OBS_DIM` grows automatically, no existing channel's index or meaning changes).
- `IDX_SICKNESS_ONSET` (interoceptive, the `IDX_FOOD_ARRIVAL` pattern) driving the
  gakel call.

**New innate reflex** (`hen/innate.py`): `M_CALL_GAKEL` wired to
`IDX_SICKNESS_ONSET` (weight chosen the same way `M_CALL_FOOD`'s was — comparable
magnitude, 4.0, since both are discovery-pulse-driven single calls); a turn-away
reflex from `CLS_SICK`, wired exactly like `CLS_CROWDING`'s (opposite turn channel,
weight derived the same algebraic way — must exceed whatever attraction pull is active
in the same bins, i.e. `CLS_FLOCKMATE`'s 1.2, for genuine avoidance rather than damped
attraction; propose matching `CLS_CROWDING`'s own 4.0 for consistency, re-derived
properly at implementation time rather than assumed).

**Not built in this stage, deliberately**: any reward cost for being sick beyond the
opportunity cost the mobility slowdown already imposes (matching how `crouch`'s
opportunity cost is what makes hiding costly, without a separate explicit penalty);
any interoceptive "I am sick" self-channel for the discoverer's own individual
learning (a real, separate, secondary question — does *instrumental* learning let the
discoverer avoid her own mistake — noted in `docs/backlog.md` as a design option, not
required for Stage 1 or for T2's primary, social-learning claim); audience-sensitivity
on the gakel call (left for learning, exactly as every other call's audience-sensitivity
already is).

**Validation instrument**: new probes in `run/probes.py`, matching the existing
ethogram style exactly (staged scenarios, hand-placed positions, `_staged`/`_run`
helpers) — one assay per falsifier condition in §4. `PlasticConfig(enabled=False)`
throughout; this stage makes no claim about learning.

**Command (once built):**
```bash
PYTHONPATH=. .venv/bin/python -m run.probes
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
```

## 6. Result

_Not yet built._

## 7. Interpretation

_Pending §6._

## 8. Consequence

_Pending §6._
