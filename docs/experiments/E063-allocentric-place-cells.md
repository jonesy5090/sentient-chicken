# E063 — an innate allocentric place-cell channel (T2 Stage 2 prerequisite)

> **Pre-registered.** Sections 1–5 written and committed before validation runs.

## 1. Parent hypothesis

**T2** — the rotating poisoned feeder. `NOT STARTED`, Stages 1/1b/1c (E060–E062) all
complete. Before Stage 2 (the actual L vs. C? learning contrast) could be designed, a
review found that T2's literal claim — durable avoidance of *this specific feeder*
that outlasts the visible/audible cue, recognised on a later approach from any
direction — is not representable by anything currently in this model. This is not a
Stage 2 diagnostic; it is a prerequisite scaffold, the same kind of gap E060 found and
filled for the bad-food call itself before Stage 1 could run.

## 2. Question

Does anything in this model give a hen a sense of *where she is*, independent of
which way she is facing? Checked directly against `coop/sensing.py`: every existing
channel (`vis`'s food/water/flock/threat/crowding/sick bins, the aerial channel, the
wall-escape channels) is computed relative to the hen's current heading — genuinely by
design, per that file's own module docstring, and correct for everything built so
far. But it means a hen who turns away from a location loses any representation of
having been there. `hen/regions.py` already names a `hippocampus` region (E016's
connectome) with 80 units and no function distinct from the other four recurrent
pools — real place cells (O'Keefe & Dostrovsky, 1971, documented in birds too) are the
natural, minimal thing to give it, rather than inventing an ungrounded "location"
signal.

## 3. Prediction

A fixed grid of Gaussian-tuned place cells, computed directly from `w.pos`
(allocentric by construction, no learning involved in the *sensing* itself — matching
this project's standing split, where every sense is innate and only usage is learned),
should: (a) peak at the cell nearest a hen's actual position, (b) stay exactly
constant under a change of heading alone, and (c) produce visibly different
population patterns for hens in different parts of the arena. None of this is in
doubt geometrically — it is arithmetic, not a hypothesis — so the value of checking it
is precisely CLAUDE.md's own point about prose vs. measurement: verify the geometry
directly rather than assume the implementation matches the docstring.

The open, genuinely uncertain question this scaffold exists to eventually let Stage 2
ask is whether plasticity can turn a place-cell pattern seen while a flockmate falls
sick into a durable avoidance response — not tested here.

## 4. Falsifier

If the peak-activation, heading-independence, or location-discrimination checks fail,
the channel does not do what its geometry claims and must be fixed before any Stage 2
work depends on it. Separately: if adding this channel changes any existing,
already-validated result — the full ethogram (`run/probes.py`) or the full test suite
— that is a regression this experiment must catch, not one Stage 2 discovers by
producing confusing numbers later. (No new ethogram probe is added for this channel
specifically: `run/probes.py`'s own stated premise is that every probe checks an
*innate behaviour*, and there is deliberately no innate reflex wired to place cells —
raw location carries no innate meaning on its own, only a learned association could
make it matter. The three new unit tests are the correct check for a sensing-only
channel with no behaviour attached, the same split E060 used between
`test_sick_channel_only_fires_for_sick_flockmates` (sensing) and
`test_avoid_sick_reflex_dominates_attraction` (behaviour) — only the sensing half
applies here.)

## 5. Design

**Implementation**: `coop/spec.py` adds a `PLACE_GRID=5` (5×5=25 cells) block,
appended after `AUDIO_HI` — the highest offset in the observation vector, so no
existing channel's index moves. `coop/sensing.py` adds `_place_cells()`: a Gaussian
bump per grid cell, `exp(-d²/(2·place_sigma²))`, centred on a fixed grid tiling
`cfg.size`, computed from `w.pos` alone (no heading term). `place_sigma=2.0` (new
`CoopConfig` field) is chosen so adjacent cells (≈3.3 m apart at the default 20 m
arena) sit at ≈32% of each other's peak — enough overlap for smooth generalisation,
peaked enough that a population pattern should typically resolve `n_food=4`
randomly-placed feeders as distinct. First-pass, not calibrated, flagged the same way
`contamination_period_s` was until E062 — a candidate for its own sweep later if
Stage 2's results motivate one.

No changes needed to `hen/connectome.py` (place cells are exteroceptive and are
picked up automatically by the existing `w_in[s_lo:s_hi, extero]` routing that every
prior channel addition already used) or `hen/innate.py` (no reflex reads them, by
design).

**Checks** (all unit tests, `tests/test_phase0.py`):
1. `test_place_cells_peak_near_the_nearest_grid_center` — a hen standing exactly on a
   grid centre reads ≈1.0 on that cell and <0.5 on every other.
2. `test_place_cells_are_independent_of_heading` — rotating a stationary hen leaves
   her place-cell pattern unchanged (`allclose`, not just "similar").
3. `test_place_cells_discriminate_distinct_locations` — two hens in opposite corners
   of the arena show population-pattern correlation <0.1.

**Regression check**: full `pytest tests/ -q` (which runs the full `run/probes.py`
ethogram as part of the suite) must still pass at 74+3=77/77, unchanged in substance
from pre-E063, confirming the new channel does not silently perturb any previously
validated innate behaviour.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
```

## 6. Result

`pytest tests/ -q`: **77/77 passed** (74 pre-existing + 3 new place-cell unit tests),
including the full `run/probes.py` ethogram unchanged. All three geometry checks
passed on the first run with no fixes needed:

- Peak-at-centre: `place[nearest_cell] ≈ 1.00000`, next-highest cell `< 0.5`.
- Heading-independence: identical arrays (`allclose`, not approximate) under a full
  heading rotation with position fixed.
- Discrimination: population-pattern correlation between opposite arena corners
  `< 0.1`.

Manual spot-check (not asserted, just inspected): a hen at `(2.86, 2.86)` — close to
but not exactly on the nearest grid centre (2.857, 2.857) — read 0.9455 on that cell
and 0.1589 on each of its two nearest neighbours, a clean single-bump pattern falling
off smoothly, confirming the Gaussian tuning behaves as intended off-centre too, not
just at the exact grid points the unit tests target directly.

## 7. Interpretation

The channel does what its geometry claims, and nothing else moved. 77/77 unchanged
(in the sense that all 74 prior tests, including the full ethogram, still pass) is the
right evidence for the "doesn't leak into other hypotheses" concern this experiment
was checking for: no innate reflex reads the new channel (by design — `hen/innate.py`
was not touched), so every previously-validated *behaviour* is bit-for-bit driven by
the same fixed reflex weights as before. What does change, unavoidably, is the
connectome's random sensory-afferent initialisation (`W_in`) for the *plastic*
pathway — the same consequence every prior `OBS_DIM` extension in this project has
already had (E048, E051, E053, E060), since `hen/connectome.py`'s afferent draw
consumes a differently-shaped random stream once the exteroceptive channel list grows.
This is expected and already-normalised in this codebase, not a new risk E063
introduces.

## 8. Consequence

The scaffold is validated. `hen/regions.py`'s `hippocampus` region now has a real,
biologically-grounded input it did not have before (`OBS_DIM` 88 → 113), and nothing
downstream needs a code change to receive it — `hen/connectome.py`'s existing
exteroceptive routing already reaches it, matching every prior channel addition's
precedent.

T2 Stage 2 (the L vs. C? learning contrast) can now be designed against a model that
actually has somewhere to put a location-specific association, rather than one that
architecturally cannot represent the claim being tested. Update `docs/hypothesis.md`'s
T2 node to note this prerequisite and its validated status; no change to T2's own
`NOT STARTED` status — this is scaffolding, not a result about the hypothesis. Update
`docs/backlog.md`'s T2 section to record that Stage 2 was blocked on an architectural
gap, now filled.
