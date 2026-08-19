"""Shared specification: the sensory and motor interface between world and hen.

This module is the contract. `coop/sensing.py` writes the observation vector,
`hen/innate.py` reads it to wire reflexes, and `coop/actuation.py` reads the motor
vector. Nothing else should hardcode these offsets.

The dimensions here are the whole thesis of the project made literal. A real chicken
spends ~42M neurons on an optic tectum and ~182M on a cerebellum: 78% of its brain on
vision and motor control. We replace those with a compact observation (`OBS_DIM`) and
motor vector (`MOTOR_DIM`), see below for both dimensions' history, and spend the
savings elsewhere.
"""

from typing import NamedTuple

# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------
# Chickens have near-panoramic vision with a narrow binocular overlap. We model the
# retina as angular bins over a 300 degree field centred on the heading, rather than
# ray-casting geometry: for behaviour at this scale, "what is roughly over there and
# how close is it" is the whole of the signal.

N_BINS = 12
FIELD_OF_VIEW_DEG = 300.0
BIN_WIDTH_DEG = FIELD_OF_VIEW_DEG / N_BINS

# Object classes resolved per bin. Deliberately tiny: a chicken in a coop needs to
# tell food from water from a flockmate from something that will eat her.
CLS_FOOD = 0
CLS_WATER = 1
CLS_FLOCKMATE = 2
CLS_GROUND_THREAT = 3
# Personal space, added in the E025 fix. CLS_FLOCKMATE is proximity-graded with no
# ceiling -- attraction strengthens monotonically all the way to contact, which is
# the wiring defect E025 traced the flock's clumping to: "a cohesion force should
# weaken as birds converge; this model has attraction only, with the gain pointing
# the wrong way." A single linear reflex weight cannot turn an attractive response
# into a repulsive one at close range (that needs a sign change as a function of
# distance, which a linear map of one channel cannot produce) -- hence a second,
# separate channel rather than a retuned existing one. Zero until a flockmate is
# well inside `PERSONAL_SPACE_THRESHOLD` of the vision range, then ramps to 1 at
# contact; the reflex arc wires it to turn *away*, opposing CLS_FLOCKMATE's
# turn-toward with a larger weight so repulsion wins once triggered.
CLS_CROWDING = 4
# Sick-flockmate visibility (T2, E060). Same `_bin_proximity` mechanism as
# CLS_FLOCKMATE, gated to fire only for a hen currently in the sickness state
# (`world.py`'s `sick_on`) -- a directly located visual referent for "something bad
# happened here", the fix for the acoustic call's inability to carry *which* feeder
# on its own (the same problem flagged for the ordinary food call). Paired with the
# gakel call (below) the same way the aerial threat is paired with seeing a hawk:
# the call broadcasts that something happened past visual range and gives learning a
# stimulus to condition on, the visual channel supplies where.
CLS_SICK = 5
N_VIS_CLASSES = 6

# Raw CLS_FLOCKMATE proximity above which CLS_CROWDING starts activating (both are
# `1 - d/vision_range`, so 0.95 is `d < vision_range * 0.05` = 0.5 m at the default
# 10 m range. Deliberately well *inside* `CoopConfig.huddle_radius` (1.00 m), not at
# its edge: an earlier version set this to 0.9 (the 1 m boundary itself), which put
# repulsion exactly where huddling needs to happen and measurably suppressed it --
# `test_reward_is_not_dominated_by_one_component` caught the consequence, cold's
# share of reward variance dropping enough to push hunger over the dominance
# threshold. 0.5 m leaves a band (0.5-1.0 m) where hens can huddle uncontested, and
# only fires repulsion for genuine crowding closer than that.
PERSONAL_SPACE_THRESHOLD = 0.95

# ---------------------------------------------------------------------------
# Observation layout
# ---------------------------------------------------------------------------
VIS_LO = 0
VIS_HI = VIS_LO + N_BINS * N_VIS_CLASSES          # 48 lateral visual channels

IDX_AERIAL = VIS_HI                                # 1 overhead channel

INTERO_LO = IDX_AERIAL + 1                         # 6 interoceptive drives
IDX_HUNGER = INTERO_LO + 0
IDX_THIRST = INTERO_LO + 1
IDX_COLD = INTERO_LO + 2
IDX_ISOLATION = INTERO_LO + 3
# Food-discovery pulse (E053). Spikes to 1 on the rising edge of arriving at a food
# patch, then decays over `CoopConfig.food_call_decay_s` -- not a continuous read of
# "food is visible" the way `CLS_FOOD` is. Real cockerels food-call on *finding* food,
# not continuously while standing at it; the previous reflex read raw sight, which
# stays high for as long as a hen remains at a patch, and roughly a quarter of the
# flock was found calling on more than half of all steps (E052 measurement, pre-fix).
# A feedforward reflex cannot tell "just arrived" from "still here" using the current
# observation alone -- that needs memory of the previous step, which is why this is
# world state computed in `world.py` (the same pattern `hunger`/`cold`/`vigour` already
# use) rather than something `sensing.py` derives fresh each step.
IDX_FOOD_ARRIVAL = INTERO_LO + 4
# Sickness-onset pulse (T2, E060). Same rising-edge-plus-decay idiom as
# IDX_FOOD_ARRIVAL, sourced from `world.py`'s `sick_call_drive`: spikes on newly
# falling sick (eating from a contaminated patch), decays over
# `CoopConfig.gakel_call_decay_s`. Drives the gakel call -- a discovery pulse, not
# continuous distress calling for the whole sick duration, deliberately matching
# E053's fix rather than repeating the defect it corrected for a different channel.
IDX_SICKNESS_ONSET = INTERO_LO + 5
INTERO_HI = INTERO_LO + 6

IDX_WALL = INTERO_HI + 0                           # 4 somatic channels
IDX_SPEED = INTERO_HI + 1
# Directional wall-escape signal. IDX_WALL alone has no direction -- it's the distance
# to the nearest wall, a scalar that doesn't change when a hen turns, only when she
# moves. A reflex that suppressed forward drive on IDX_WALL directly would deadlock:
# turning away from the wall wouldn't lower IDX_WALL (she hasn't moved yet), so forward
# would stay suppressed and she could never actually walk out. These two channels carry
# which way to turn instead -- proximity-gated, and gated to fire on only one side, so
# a linear reflex can wire each straight to its matching turn motor with no sign
# reversal needed, the same design as CLS_CROWDING above but for a boundary rather than
# a flockmate.
IDX_WALL_ESCAPE_L = INTERO_HI + 2
IDX_WALL_ESCAPE_R = INTERO_HI + 3

AUDIO_LO = INTERO_HI + 4                           # 5 auditory call channels
AUDIO_HI = AUDIO_LO + 5

# ---------------------------------------------------------------------------
# Allocentric place cells (T2 Stage 2 prerequisite)
# ---------------------------------------------------------------------------
# Every channel above is egocentric -- "what is roughly over there", relative to the
# hen's own heading. That is deliberate (see `coop/sensing.py`'s module docstring) and
# it is also why nothing in this model can represent "feeder #2, specifically" once a
# hen turns away from it: an egocentric bin index describes a direction, not a place.
# T2's actual claim -- durable avoidance of a particular contaminated feeder that
# outlasts the visible/audible cue -- needs a location signal that survives a change of
# heading, which nothing built before E063 provides.
#
# Real place cells (O'Keefe & Dostrovsky, 1971; found in birds too, not just mammals)
# fire by an animal's location in the environment, independent of orientation. That is
# exactly the missing piece, and `hen/regions.py` already names a `hippocampus` region
# (E016's connectome) that has had no distinct function of its own until now -- adding
# an allocentric input finally gives it one, rather than leaving it a generic recurrent
# pool with a biological name it hasn't earned.
#
# Innate, like every other sense in this model: a fixed receptive field tuned to a
# patch of the arena is a sensory fact (the same simplification `CLS_FOOD`'s fixed
# proximity gating already makes), not a learned skill -- real place fields do refine
# with experience, but this project's standing rule is that *production/reception* of
# every sense is wired and only *usage* is left to learn (Konishi's finding, applied
# uniformly). What plasticity has to discover -- the entire open question -- is
# whether a place-cell pattern seen while a flockmate falls sick can be turned into a
# lasting avoidance response, the same "durable, referential contingency" bar H2f
# already cleared for a different anchor (E057) and H2c already failed to clear from
# nothing (E058/E059).
PLACE_GRID = 5                                     # 5x5 = 25 cells tiling the arena
N_PLACE = PLACE_GRID * PLACE_GRID

PLACE_LO = AUDIO_HI
PLACE_HI = PLACE_LO + N_PLACE

# ---------------------------------------------------------------------------
# Gakel-call location cue (T2 Stage 2 prerequisite, E064)
# ---------------------------------------------------------------------------
# Self-location alone does not close the loop. A hen close enough to *see* a sick
# flockmate at a feeder gets `CLS_SICK` and `CLS_FOOD` in the same egocentric bin and
# her own place cells tag the moment -- a real, learnable pathway. A hen who only
# *hears* the gakel call from beyond visual range gets nothing spatial at all: audio in
# this model has never carried direction (every call is pure distance-attenuated
# amplitude, the same simplification real avian hearing's approximate directionality
# is not modelled here), and her own place cells report *her* location, not the
# caller's. There is no channel connecting the two -- exactly the gap the gakel call
# was supposed to close ("broadcasts past visual range", E060) but, on inspection,
# doesn't.
#
# This channel closes it directly rather than asking the pallium to reconstruct a
# bearing from scratch: for each listener, the *loudness-weighted mixture of gakel
# callers' own place-cell patterns* she currently hears -- reusing the exact same grid
# and tuning `PLACE_LO:PLACE_HI` already computes for self-location, just evaluated at
# the caller's position and scaled by how audible she is. A faint, distant call gives a
# faint, blurry location estimate; a loud, nearby one gives a sharp one; nobody calling
# gives exactly zero. No new machinery, no requirement that a small pallium learn
# trigonometry from an egocentric bearing plus its own heading (which this model does
# not even expose) -- the "coarse" localisation real animals get from hearing is
# supplied directly, the same way `CLS_SICK` supplies its own ground-truth-but
# range-limited location rather than asking vision to triangulate depth from scratch.
#
# Gakel-specific, not a general "where is every call coming from" upgrade -- the same
# narrow, need-driven scope every prior channel in this file used (`CLS_CROWDING` for
# personal space, `CLS_SICK` for sickness, wall-escape for walls). If a future
# hypothesis needs directional hearing for a different call, that is its own addition.
GAKEL_PLACE_LO = PLACE_HI
GAKEL_PLACE_HI = GAKEL_PLACE_LO + N_PLACE

OBS_DIM = GAKEL_PLACE_HI                           # 138 (59 pre-E025, 71 pre-E051, 73 pre-E053, 74 pre-E060, 88 pre-E063, 113 pre-E064)


def vis_index(bin_idx: int, cls: int) -> int:
    """Flat observation index of one (bin, class) visual channel."""
    return VIS_LO + bin_idx * N_VIS_CLASSES + cls


# ---------------------------------------------------------------------------
# Motor layout
# ---------------------------------------------------------------------------
M_FORWARD = 0
M_TURN_L = 1
M_TURN_R = 2
M_PECK = 3
M_SCRATCH = 4
M_CROUCH = 5
M_FLEE = 6
M_CALL_CONTACT = 7
M_CALL_FOOD = 8
M_CALL_AERIAL = 9
M_CALL_GROUND = 10
# Gakel call (T2, E060): a real, documented frustration / negative-expectation
# vocalisation in Gallus gallus, not an invented "danger: bad food" call. Innate in
# production like every other call here, fired on the sickness-onset pulse
# (IDX_SICKNESS_ONSET). See README.md's call repertoire table for the biological
# grounding and why this was chosen over inventing a bespoke signal.
M_CALL_GAKEL = 11
MOTOR_DIM = 12                                     # 11 pre-E060, first motor-dim change

# The five call channels, in the order they appear in both the motor vector and the
# auditory slice of the observation. Konishi (1963) deafened day-old chicks and they
# developed the normal repertoire regardless, so production of all five is innate and
# hardwired in `hen/innate.py`. Only *usage* and *comprehension* are left to learn.
CALL_MOTOR_IDX = (M_CALL_CONTACT, M_CALL_FOOD, M_CALL_AERIAL, M_CALL_GROUND,
                  M_CALL_GAKEL)
N_CALLS = 5

# Position of the gakel call within any (..., N_CALLS)-shaped array (`w.calls`,
# `call_log`, the audio slice of an observation) -- used by the E064 location cue.
GAKEL_CALL_IDX = CALL_MOTOR_IDX.index(M_CALL_GAKEL)

# A silent hen must emit silence. Motor channels are sigmoids, so a bird at rest sits
# at sigmoid(REST_BIAS) = 0.076 on *every* channel including the four call ones -- a
# floor that is an artefact of the nonlinearity, not a vocalisation. Until E019 that
# floor was emitted as a real call: with 16 hens summing into one clipped channel, the
# aerial-alarm input read 0.999 at rest and a full-amplitude call from an adjacent bird
# moved it by 0.0000. Every communication experiment before E019 ran on that constant.
#
# Subtracting the floor and rescaling removes exactly the artefact and nothing else.
# It is deliberately not a 0.5 threshold like peck and crouch use: alarm calls are
# *graded* by urgency in real fowl, and the audience assay stages its hawk at 7 m
# precisely to read a mid-range call, which a half-threshold would silence.
CALL_FLOOR = 0.0759   # sigmoid(REST_BIAS); asserted against innate.py in the tests

# Depth the yoked control needs: must exceed `yoke_min_lag_s / dt` plus the spread of
# per-hen lags, or it reads a slot that has not been written yet. 4096 = 41 s at 10 ms.
# Set via `CoopConfig.call_log_steps`, which defaults to 1 -- see there.
YOKE_LOG_STEPS = 4096

# Actions that put the head down. This single fact is why the project can have
# language at all: a hen with her beak in the dirt cannot scan for hawks, so a
# flockmate's alarm call carries information she does not otherwise have. Without
# this asymmetry there is no pressure for communication to emerge, no matter how
# many neurons the bird is given.
HEAD_DOWN_ACTIONS = (M_PECK, M_SCRATCH)


class CoopConfig(NamedTuple):
    """Physical parameters of the world. SI units, seconds and metres."""

    # Integration
    dt: float = 0.01                  # 10 ms; safe for tau >= 50 ms at dt/tau <= 0.2

    # Arena
    size: float = 20.0                # 20 x 20 m run
    n_hens: int = 16
    n_food: int = 4
    n_water: int = 2

    # Locomotion (chickens walk ~0.3 m/s and can run over 1 m/s)
    walk_speed: float = 0.30
    flee_speed: float = 1.20
    turn_rate: float = 2.0            # rad/s

    # Perception
    vision_range: float = 10.0
    peck_radius: float = 0.30
    drink_radius: float = 0.40
    hear_range: float = 15.0
    huddle_radius: float = 1.00

    # Drives: seconds to saturate from empty.
    # Intake is a *rate*, not a per-step gain. Motor channels are continuous
    # activations sampled every dt, so a per-step gain would let a hen "peck" a
    # hundred times a second and pin every drive at zero -- which leaves phase 1 with
    # no motivational gradient to learn from. These are chosen so a hen must forage
    # roughly a third of her time to stay fed, which is in the right ballpark for a
    # real bird.
    hunger_fill_s: float = 1800.0
    thirst_fill_s: float = 2400.0
    cold_fill_s: float = 1200.0
    # Intake scales with the drive itself: a satiated bird stops swallowing. Without
    # that, nothing regulates intake at all -- a hatchling pecks at the sight of food
    # rather than because she is hungry, so hunger would simply pin at whichever end
    # the rate favoured. Learning to forage *because* you are hungry is a phase 1
    # target, not something to wire in here.
    peck_food_rate: float = 3.0e-2    # per second of feeding, scaled by hunger
    drink_rate: float = 4.0e-2

    # Patches deplete as they are worked and recover when left alone. Until E025 they
    # did neither: `food_amount` was initialised to 1.0 and never changed, so a patch
    # was an infinite resource and there was no reason for any hen ever to leave one.
    #
    # Added on the theory that this would disperse the flock and rescue E024's control.
    # **It does neither** (E025): patches genuinely draw down to 0.70 and the flock does
    # not move -- strike-radius overlap went 23.0% -> 21.9%. The binding force is the
    # gregariousness reflex, not foraging (E025 ablation), and the control was never a
    # dispersal problem at all (E026). Kept because finite patches are biologically
    # right, are a precondition for T2, and raise feeding 34% when hens do spread.
    #
    # Real hens work a patch and move on, and depletion-with-recovery is the standard
    # way to make a foraging environment produce dispersal. Rates chosen so a patch
    # supports a couple of birds for roughly a minute and recovers over about five --
    # long enough that leaving is worthwhile, short enough that the coop does not run
    # out of food over a 20-minute run.
    food_deplete_rate: float = 2.0e-2   # per second per hen feeding at the patch
    food_regrow_s: float = 300.0        # seconds for an empty patch to refill
    huddle_warm_rate: float = 4.0e-4  # per second, per flockmate within huddle_radius
    huddle_max: int = 3               # a hen needs 2-3 close neighbours to stay warm

    # Calling costs energy, and that cost lives in its own budget rather than in
    # hunger. It was charged to hunger from E005 until E012, which is how it came to
    # triple the rate hunger accumulates and destroy the metric H2 is measured on --
    # a parameter added for one hypothesis silently changing the measurement basis of
    # another. `vigour` keeps the cost real without touching the foraging drive.
    #
    # Two things make the cost bite. It enters the reward signal, so calling is
    # genuinely expensive and audience-sensitivity has a gradient to emerge from; and
    # it attenuates the call that flockmates actually hear, because a tired bird
    # cannot call loudly. The second makes vocal effort self-limiting without any
    # arbitrary cap.
    call_vigour_drain: float = 1.5e-2   # per second per unit of call amplitude
    vigour_recovery_s: float = 90.0     # seconds to recover fully from empty

    # Food-discovery pulse decay (E053, IDX_FOOD_ARRIVAL). A call bout, not a level --
    # much shorter than typical patch dwell time (`food_deplete_rate`'s own comment: "a
    # patch supports a couple of birds for roughly a minute"), so a hen who stays at a
    # feeder spends most of that minute silent on this channel, not continuously calling.
    food_call_decay_s: float = 4.0

    # --- T2 contamination/sickness scaffold (E060) ------------------------------
    # Which feeder is contaminated rotates on this period. First-pass placeholder,
    # not yet calibrated -- `docs/backlog.md`'s own design note names this "a sweep,
    # and finding it is itself a result" (Stage 1c, not yet run): the period needs to
    # be fast enough that private information stays valuable, slow enough that it can
    # actually propagate through the flock before rotating again.
    # Master switch for the allocentric spatial channels -- E063's place cells and
    # E064's gakel location cue (E076). Added for the same reason as
    # `contamination_enabled` below and found the same way: E063 shipped without an
    # opt-in, and its block turns out to be **25.1% of all observation drive, active
    # 100% of the time** (E075) -- 25 permanently-on channels added to what had been an
    # 88-channel observation. Every experiment since E063 has carried that, whether or
    # not it has anything to do with T2.
    #
    # **Default flipped to `False` by E076**, which closed the bisect: with this block
    # and contamination both off, E057's contrast reproduces to within noise (general
    # +0.132 vs +0.123, audience-specific +0.241 vs +0.232, food control null at
    # t=0.98). Left on, it breaks that control. T2's own experiments opt in.
    place_cells_enabled: bool = False

    # Master switch for T2's contamination mechanic (E075). **Added because E060 did
    # not have one**, so from E060 onward every experiment in the project silently ran
    # with hens being poisoned, gakel-calling and moving at `sickness_mobility_scale`
    # -- measured at 32 sickness onsets in a 30-minute 16-hen run. That is a change to
    # the comparison basis for every hypothesis, made without an opt-in, which is
    # exactly what this file's other additions are careful to avoid (`legacy_audio`,
    # `auditory_scaffold`, `pred_enabled`, `readout_scaling_strength`, `gakel_scaffold`,
    # `balanced_ei` are all off by default for this reason).
    #
    # **Default flipped to `False` by E076.** It accounted for roughly half the damage
    # to H2f's control on its own (t=10.04 -> t=2.70). T2's own experiments opt in.
    contamination_enabled: bool = False
    contamination_period_s: float = 300.0
    # How long a hen who eats contaminated food stays in the sickness state -- long
    # enough to be a reliable, observable cue to flockmates (the whole point of
    # CLS_SICK), short enough not to permanently cripple her. First-pass, matching the
    # order of magnitude `hawk_dive_s`/`ground_pred_dwell_s` already use for a bounded
    # event duration in this file.
    sickness_duration_s: float = 60.0
    # Fraction of normal mobility retained while sick. "Visibly slow / still", not a
    # hard freeze like crouching -- 0.15 leaves a clearly-reduced but nonzero drift,
    # applied mechanically in `actuation.py`, not mediated by the reflex arc or any
    # learned weight: this is a physiological fact about being sick, the same way
    # crouching mechanically zeroes locomotion regardless of what drove the crouch.
    sickness_mobility_scale: float = 0.15
    # Gakel-call discovery-pulse decay (IDX_SICKNESS_ONSET), matching
    # food_call_decay_s's own reasoning and default: a bout, not continuous distress
    # calling for the whole sickness duration.
    gakel_call_decay_s: float = 4.0

    # --- Allocentric place cells (T2 Stage 2 prerequisite, E063) ----------------
    # Gaussian receptive-field width for the `N_PLACE` grid `coop/sensing.py` computes
    # from `size` and `spec.PLACE_GRID`. At the default 20 m arena and a 5x5 grid, cell
    # centres sit ~3.3 m apart; this sigma puts a cell's nearest neighbour at ~32% of
    # its own peak response (exp(-3.3^2 / (2*2.0^2))) -- overlapping enough for smooth
    # generalisation between adjacent locations, peaked enough that the population
    # pattern should still resolve `n_food=4` randomly-placed feeders as distinct in
    # most draws. First-pass, not yet calibrated -- flagged the same way
    # `contamination_period_s` was until E062 swept it.
    place_sigma: float = 2.0

    # Restores the pre-E019 audio path: emit the raw sigmoid (so a resting hen calls
    # at 0.076 on every channel) and sum voices linearly into a clip. It exists purely
    # so E021 can ask what the audio fix changed, and it is the only route by which a
    # non-learning flock differed between E013 and E020 -- every other fix since E013
    # touches plasticity, which those conditions have switched off. Never a default.
    legacy_audio: bool = False

    # --- The H4 condition ladder (docs/backlog.md §1) --------------------------
    # How the auditory channel is wired between hens. Every condition delivers the
    # same bandwidth and the same energetic cost; they differ only in *whose* calls
    # reach whom, which is what isolates information transfer from everything else
    # that carrying a channel involves.
    #
    #   "intact"   L  -- she hears her flockmates. The hypothesis.
    #   "yoked"    C? -- she hears the flock's REAL call stream, shifted in time by a
    #                    per-hen lag longer than a hawk dive. Identical rate, amplitude
    #                    distribution, burst structure and cost; zero contingency with
    #                    her current world. THE HEADLINE CONTROL.
    #   "shuffled" -- RETAINED AND NOT A CONTROL. Permutes who-hears-whom within a
    #                    timestep. E024 used it as the control and E026 showed it keeps
    #                    98% of the intact channel's correlation with "a hawk is on me",
    #                    because every hen already hears every other (hear_range 15 m in
    #                    a 20 m arena) and any within-timestep permutation preserves
    #                    "someone is calling right now" -- which in this coop is almost
    #                    the whole signal. Kept only so E024 can be reproduced.
    #   "severed"  C0 -- she emits; nobody hears. Isolates the motor cost of calling.
    #   "self"     Cs -- she hears only herself. A channel used as private memory is
    #                    not communication, and without this the difference is
    #                    invisible.
    #   "none"     N/C- -- no auditory input at all.
    channel_mode: str = "intact"
    # Re-drawn every `shuffle_period_s` so a permutation cannot be defeated by a fixed
    # mapping the flock could, in principle, learn around.
    shuffle_period_s: float = 10.0
    # Minimum per-hen lag for the yoked channel, in seconds. Must exceed `hawk_dive_s`
    # by a clear margin or the shifted stream still overlaps the event it is supposed
    # to be decorrelated from.
    yoke_min_lag_s: float = 20.0
    # Depth of the call-history ring buffer. **1 by default, i.e. off.**
    #
    # It exists only for the yoked control, and it is not free: a 4096x16x4 array in
    # the scan carry is 512 KB, and this simulation is memory-bandwidth bound. I wrote
    # in the source that "one slot written and one gathered per step does not threaten
    # the bandwidth budget", did not measure it, and the throughput guard caught the
    # regression at 4.9x real time against a 5.0x floor. Reasoning about a bandwidth
    # cost is not measuring one.
    #
    # So the default coop pays nothing and the yoked condition sets
    # `call_log_steps=spec.YOKE_LOG_STEPS` for itself.
    call_log_steps: int = 1

    # Predation
    hawk_period_s: float = 900.0      # a hawk passes over roughly every 15 min
    hawk_dive_s: float = 12.0
    # How long the hawk is overhead and visible before it can strike.
    #
    # Without this there is no warning interval at all: `_step_predators` used to put
    # the hawk at its final position the instant a dive began, so a hen was in danger
    # from the same step she could first have been told about it. Measured consequence
    # (E026) -- P(caught | blind and at risk at onset) was 1.000 deaf, 0.984 intact,
    # 0.981 yoked. A ceiling in every condition, because no signal arriving at the
    # instant of arrival can help anybody. That is why H4 was untestable, and why six
    # attempts at a staged assay each had to invent a stoop phase by hand.
    #
    # A real raptor stoops from height and is visible during the descent; the whole
    # point of an alarm call is that it reaches you during that window. 2 s against a
    # 12 s dive, so most of the encounter is unchanged.
    hawk_approach_s: float = 2.0
    hawk_strike_radius: float = 1.50
    # How far off a targeted hen the hawk comes down. A hawk hunts rather than landing
    # at random, so it aims at a bird; the spread keeps the encounter uncertain instead
    # of scripted. At 3 m against a 1.5 m strike radius, the targeted hen is often but
    # not always at risk, and who else is depends on where the flock is standing.
    hawk_aim_spread: float = 3.0
    ground_pred_period_s: float = 1800.0
    ground_pred_dwell_s: float = 30.0
    ground_pred_speed: float = 0.60


DEFAULT_COOP = CoopConfig()
