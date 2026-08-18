# E053 — food call fires on finding food, not on continuously seeing it

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

None of the existing H-tree nodes directly, per the same reasoning as E051: this is a
scaffolding-fidelity fix to the innate reflex arc (call *production*, which the project's
biology already treats as hardwired), not a claim about learning. Directly closes a
long-standing `docs/backlog.md` item: "The innate food call fires on sight, out to 10 m.
Twelve of sixteen hens food-call continuously... Real cockerels food-call on *finding*
food... This is a reflex-arc change and needs its own hypothesis node — it is not a bug
fix and should not be done as one." Also the necessary first step before
[E054](E054-food-call-saturation-and-pallium-capacity.md), which asks whether this
saturation was crowding out pallium capacity that could otherwise represent the rarer
alarm channel (H2c/H2d) — per CLAUDE.md's "test the instrument before the hypothesis,"
that question can't be asked until the instrument (the food-call channel) is fixed.

## 2. Question

The current reflex fires `M_CALL_FOOD` continuously from raw `CLS_FOOD` sight, with no
memory of whether she was already looking at it a moment ago. A user watching the
offline-replay viewer observed this directly ("the number of food calls seems
significant, almost constant") and asked whether it could be masking or crowding out
the (separate-channel, but rarer) alarm calls. Measured before any change (informal
check, 16 hens, 1 minute, no predators): 4/16 hens calling FOOD on over half of all
steps, 42.8% of all hen-steps with an active food call. **Does replacing continuous
sight-gating with a discovery pulse — firing once on arrival, decaying over a few
seconds — reduce this to something closer to a real cockerel's discovery-call pattern
without breaking the underlying behaviour (finding food still gets announced)?**

## 3. Prediction

The discovery-pulse design should cut both measures sharply: no hen should sit above
50% food-calling (since the pulse only lasts `food_call_decay_s` = 4 s regardless of how
long she stays at a patch), and the flock-wide mean fraction of hen-steps calling should
fall by roughly the ratio of pulse duration to typical patch dwell time — patches are
tuned (`food_deplete_rate`'s own comment) to "support a couple of birds for roughly a
minute," so a fix working as intended should land in the low single-digit percent, not
merely reduced.

## 4. Falsifier

If the mean calling fraction stays above ~20%, or any hen still sits above 50%, the
pulse isn't functioning as a genuine edge-trigger — most likely because
`food_call_decay_s` is too long, or the edge-detection (`at_food_prev`) isn't correctly
resetting between visits to different or the same patch.

## 5. Design

**The fix**: two new per-hen `World` fields, `food_call_drive` and `at_food_prev`
(`coop/world.py`), following the exact rising-edge idiom `strike_event` already uses
(`struck * (1.0 - w.struck_prev)`): `food_arrival = at_food_any & (w.at_food_prev < 0.5)`,
then `food_call_drive` jumps to 1.0 on that edge and otherwise decays linearly over
`cfg.food_call_decay_s` (new `CoopConfig` field, default 4.0 s). Exposed as a new
interoceptive channel, `IDX_FOOD_ARRIVAL` (`coop/spec.py`, `OBS_DIM` 71 → 72).
`hen/innate.py`'s `M_CALL_FOOD` reflex now reads this channel instead of raw `CLS_FOOD`
sight (same weight, 4.0, so peak calling strength is unchanged — only the temporal
pattern changes). Not gated on hunger or pecking, matching the existing pecking reflex's
own reasoning (indiscriminate discovery response) and leaving audience-sensitivity
untouched — that modulation stays deliberately unwired, left for plasticity, per
`hen/innate.py`'s existing documented design.

**New probe** (`run/probes.py`): `food_call_on_arrival_not_continuous` — a hen staged
already at a patch must peak-call in the first 50 steps and fall to a low mean by step
400–500 (well past the 4 s / 400-step decay window) despite never leaving the food.

**New unit test** (`tests/test_phase0.py`):
`test_food_call_drive_spikes_on_arrival_and_decays` — checks the world-state mechanism
directly (arrival edge fires exactly at 1.0, decays below 0.05 within 500 steps of
continuous presence) and that `sensing.observe` correctly surfaces it.

**Population re-check**: same informal instrument as §2 (16 hens, 1 minute, no
predators), re-run unmodified on the fixed codebase for a direct before/after.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python -m run.probes
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
PYTHONPATH=. .venv/bin/python scratchpad/e053_food_call_rate_recheck.py
```

## 6. Result

```
                                    pre-fix    fixed
hens calling FOOD on >50% of steps:   4/16      0/16
mean fraction of hen-steps calling:  0.428     0.042
```

**The falsifier does not fire.** No hen sits above 50% (down from 4), and the flock-wide
mean drops from 42.8% to 4.2% — a ~10× reduction, comfortably past the "low single
digits" bar set in §3. The new probe passes (early peak 0.82, well above the 0.5
threshold; late mean 0.08, below the 0.1 ceiling) and the new unit test confirms the
underlying mechanism directly: the drive hits exactly 1.0 on arrival and decays to
under 0.05 within 500 steps of continuous presence at the same patch. Full suite: 62/62
tests pass, 8/8 ethogram assays pass (peck-at-food and every other existing assay
unaffected).

## 7. Interpretation

**This is a straightforward fix to a genuine fidelity gap, not a subtle behavioural
trade-off.** The reflex now does what the biology says it should — announce discovery,
not narrate continued presence — and every other documented innate behaviour is
unaffected (verified by the full ethogram, not just the new probe). The residual 4.2%
is expected, not noise to chase further: sixteen hens repeatedly discovering patches
(their own or a neighbour's freshly-regrown one) over a full minute will produce some
baseline rate of genuine discovery pulses, and that is exactly what should remain.

**What this does not yet establish**: whether the previous near-constant food-calling
was actually consequential for anything downstream — the representational-bottleneck
question that motivated fixing this in the first place. That is
[E054](E054-food-call-saturation-and-pallium-capacity.md), run next, now that the
instrument is fixed.

## 8. Consequence

- **Closes the long-standing backlog item** ("food call channel saturated... needs its
  own hypothesis node"), `docs/backlog.md` updated with a forward pointer.
- **`OBS_DIM` moves 71 → 72.** `docs/hypothesis.md`'s re-baselining banner extended
  (same scoped-caveat treatment as E048/E051 — no existing channel's index or meaning
  changes).
- **New world-state pattern established for future reflex-arc discovery-style fixes**:
  the rising-edge-plus-decay idiom, reusing `strike_event`'s existing pattern rather
  than inventing a new one.
- **Enables E054**, the actual question this was raised to answer.
