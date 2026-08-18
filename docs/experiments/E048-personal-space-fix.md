# E048 — the E025 flock-clumping fix: a personal-space vision channel

> **Written after implementation began, not before.** Sections 1–5 below describe the
> design as it was actually reasoned through and built, but code was written and the
> test suite run before this file existed — a deviation from the project's own
> discipline (CLAUDE.md, "Sections 1–5 ... written **before** running anything"). Flagged
> here rather than silently backdated, in the spirit of E025's own retrospective
> write-up. The diagnostic in §6 was run only after this file's sections 1–5 were
> committed, so the *result* is honestly pre-registered even though the implementation
> was not.

## 1. Parent hypothesis

The E025-adjacent backlog item (`docs/backlog.md`): "the flock clumps... gregariousness's
attraction-only wiring is the actual cause... needs a crowding/individual-distance
channel." Feeds T1/T2 indirectly — both assume hens are spatially distinct from each
other, which E025 found the current wiring defeats — and H4's original sender-scrambling
control, which E024/E026 found is weakened when the flock shares most of its neighbours.

## 2. Question

Does adding a dedicated personal-space observation channel — so the reflex arc can turn
a hen *away* from a flockmate at close range, not just toward one at any range — disperse
the flock, without breaking the gregariousness/thermotaxis behaviour (huddling) that
channel also serves?

## 3. Prediction

Nearest-neighbour distance and strike-radius overlap improve (increase / decrease
respectively) relative to the pre-fix reflex arc, measured on the current,
E023-corrected connectome. The improvement will be smaller than E025's own ablation
(zeroing gregariousness entirely: nn 0.39 → 1.62, strike 21.9% → 6.8%), because this fix
is a targeted correction to the wiring, not a removal of the behaviour. Cold and fed/hen
should stay close to the pre-fix values — a fix that disperses the flock by breaking
huddling or feeding would not be a fix.

## 4. Falsifier

If nearest-neighbour distance and strike-radius overlap do not move from the pre-fix
condition (matched on the same, current connectome), the wiring defect E025 diagnosed
was not, in fact, the binding constraint — something else would need to be found.
If cold or fed/hen move substantially, the fix has traded one problem for another and is
not usable as-is.

## 5. Design

**The fix** (`coop/spec.py`, `coop/sensing.py`, `hen/innate.py`): a fifth vision class,
`CLS_CROWDING`, derived from the existing `CLS_FLOCKMATE` proximity value — zero until a
flockmate is closer than `PERSONAL_SPACE_THRESHOLD` (0.95, i.e. within 0.5 m at the
default 10 m vision range), ramping linearly to 1 at contact. The reflex arc wires this
to the opposite turn channel from `CLS_FLOCKMATE`'s attraction, at weight 4.0 against
attraction's 1.2 — algebraically required to exceed 1.2 for the net drive to actually
reverse sign at close range rather than merely damp toward zero (a linear reflex map
cannot produce attract-then-repel from one channel; this is why a second channel was
needed rather than a retuned existing weight, the same conclusion E025 reached).
`OBS_DIM` moves 59 → 71 as a mechanical consequence of the offset-based observation
layout (`coop/spec.py`).

`PERSONAL_SPACE_THRESHOLD` was originally set to 0.9 (the 1 m boundary matching
`CoopConfig.huddle_radius` exactly), on the reasoning that repulsion should engage
"roughly at the boundary of huddling distance." That was wrong: it put repulsion
exactly where huddling needs to happen, and `test_reward_is_not_dominated_by_one_component`
caught the consequence (hunger's share of reward variance crossed the test's 80%
ceiling, traced to suppressed huddling reducing cold's contribution). Retuned to 0.95
(0.5 m), leaving a 0.5–1.0 m band where hens can huddle uncontested.

**Diagnostic**: reuses `scratchpad/what_clumps_them.py`'s exact methodology
(`scratchpad/e048_personal_space_fix.py`) — 16 hens, 6-minute rollout (36,000 steps), no
plasticity, `hawk_period_s=60`, `pallium_scale=1.5`, `auditory_scaffold=True` — for
direct comparability with E025's own numbers in shape, though not in absolute value
(this runs on the E023-corrected connectome; E025 predates that fix, so raw numbers are
not expected to match, only the within-run contrast is valid, the same caveat E025's own
file records for its comparison against E024).

**Conditions**, all on the current codebase so only the reflex differs:
- **pre-fix**: `CLS_CROWDING` wired but its reflex weights zeroed — exactly what the arc
  did before this session.
- **fixed**: the arc as it ships.
- **no gregariousness** (reference only, not a comparison condition): `CLS_FLOCKMATE`
  zeroed entirely, reproducing E025's original ablation for scale.

**Primary metric**: nearest-neighbour distance, strike-radius overlap (both from E025).
**Secondary, exploratory**: cold, fed/hen (checking the fix does not break huddling or
feeding).
**Replicates**: 3 seeds, matching E025's own diagnostic register — this is a mechanism
check, not a powered contrast, the same standard E025 itself used for this question.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e048_personal_space_fix.py
```

## 6. Result

```
condition                     nn dist   spread  in strike radius    cold  fed/hen
pre-fix (E025 baseline)          0.14     2.57             26.8%   0.146     1034
fixed (E025 personal space)      0.38     5.46             21.8%   0.164      984
no gregariousness (ref.)         1.70    18.15              8.2%   0.278     1247
```

**The fix disperses the flock.** Nearest-neighbour distance nearly triples (0.14 → 0.38
m), strike-radius overlap drops five points (26.8% → 21.8%), matching the falsifier's
"no movement" condition failing to occur. Cold moves modestly (0.146 → 0.164, still far
below the no-gregariousness row's 0.278) and fed/hen is essentially flat (1034 → 984,
nowhere near the no-gregariousness row's 1247, which reflects hens no longer competing
for the same patches at all). The prediction in §3 — real dispersal, smaller than full
removal, cold and feeding held close to baseline — holds on all four counts.

**Pre-fix numbers do not match E025's own baseline row** (0.14/26.8% here vs. 0.39/21.9%
recorded there) — expected and flagged in §5: E025 predates the E023 E/I fix, so the two
runs are on different connectomes. Only the within-this-run contrast (pre-fix vs. fixed,
both on the current codebase) is a valid comparison.

## 7. Interpretation

**The wiring defect E025 diagnosed was real and the targeted fix addresses it,
partially and safely.** A single new observation channel, wired with an
algebraically-derived weight, produces measurable dispersal without the collateral
damage (broken huddling, distorted feeding) that fully removing gregariousness causes.
This is consistent with E025's own reading: gregariousness is not the wrong behaviour,
it is missing the repulsive half real fowl have (documented individual distance), and
supplying just that half is enough to move the numbers without removing the behaviour's
function.

**The dispersal is partial, not complete**, and this experiment does not establish
whether it is *enough* — enough for what H4's control needed, or what T1/T2 will need,
was not measured here (E025 itself made this decision against H4's shuffled-channel
information-retention metric, not the strike-radius proxy; that metric has not been
re-run against this fix). That is the natural next check before either task assumes the
dispersal problem is closed.

**Confound not fully ruled out**: `PERSONAL_SPACE_THRESHOLD`'s retuning (0.9 → 0.95) was
driven by a test failure, not by this diagnostic — the diagnostic above was only run
once, at the final (0.95) value. Whether 0.9 would also have shown comparable dispersal,
just with worse cold/huddling numbers, was not measured; the two changes (does it
disperse, does it preserve huddling) were not independently varied.

## 8. Consequence

- **The E025-adjacent backlog item is closed**, `docs/backlog.md` updated in place with
  a forward pointer to this file, not deleted.
- **`docs/hypothesis.md` gets a new, narrower re-baselining banner** (alongside E023's):
  `OBS_DIM` 59 → 71. Unlike E023, no existing channel's value or index changes — this is
  flagged as a scoped caveat, not a second full re-baseline.
- **New unit tests** (`tests/test_phase0.py`): `test_crowding_channel_activates_only_inside_personal_space`
  and `test_personal_space_reflex_dominates_attraction_at_contact`, guarding the two
  properties this fix depends on (the channel is gated correctly; the repulsion weight
  genuinely exceeds the attraction weight, not just damps it).
- **Stale `OBS_DIM`/"59-dim" references corrected**: `coop/spec.py`'s module docstring,
  `CLAUDE.md` and `README.md`'s layout diagrams, `coop/world.py`'s module docstring —
  none were load-bearing (the codebase reads `spec.OBS_DIM` symbolically throughout,
  confirmed before editing), all were stale comments.
- **Not yet done, flagged rather than silently skipped**: re-running H4's own
  information-retention metric (E026's measure) against this fix, to check whether the
  dispersal achieved here is enough to matter for the shuffled-channel control it was
  originally motivated by.
