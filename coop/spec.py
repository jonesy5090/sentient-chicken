"""Shared specification: the sensory and motor interface between world and hen.

This module is the contract. `coop/sensing.py` writes the observation vector,
`hen/innate.py` reads it to wire reflexes, and `coop/actuation.py` reads the motor
vector. Nothing else should hardcode these offsets.

The dimensions here are the whole thesis of the project made literal. A real chicken
spends ~42M neurons on an optic tectum and ~182M on a cerebellum: 78% of its brain on
vision and motor control. We replace those with a 59-dimensional observation and an
11-dimensional motor vector, and spend the savings elsewhere.
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
N_VIS_CLASSES = 4

# ---------------------------------------------------------------------------
# Observation layout
# ---------------------------------------------------------------------------
VIS_LO = 0
VIS_HI = VIS_LO + N_BINS * N_VIS_CLASSES          # 48 lateral visual channels

IDX_AERIAL = VIS_HI                                # 1 overhead channel

INTERO_LO = IDX_AERIAL + 1                         # 4 interoceptive drives
IDX_HUNGER = INTERO_LO + 0
IDX_THIRST = INTERO_LO + 1
IDX_COLD = INTERO_LO + 2
IDX_ISOLATION = INTERO_LO + 3
INTERO_HI = INTERO_LO + 4

IDX_WALL = INTERO_HI + 0                           # 2 somatic channels
IDX_SPEED = INTERO_HI + 1

AUDIO_LO = IDX_SPEED + 1                           # 4 auditory call channels
AUDIO_HI = AUDIO_LO + 4

OBS_DIM = AUDIO_HI                                 # 59


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
MOTOR_DIM = 11

# The four call channels, in the order they appear in both the motor vector and the
# auditory slice of the observation. Konishi (1963) deafened day-old chicks and they
# developed the normal repertoire regardless, so production of all four is innate and
# hardwired in `hen/innate.py`. Only *usage* and *comprehension* are left to learn.
CALL_MOTOR_IDX = (M_CALL_CONTACT, M_CALL_FOOD, M_CALL_AERIAL, M_CALL_GROUND)
N_CALLS = 4

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

    # Predation
    hawk_period_s: float = 900.0      # a hawk passes over roughly every 15 min
    hawk_dive_s: float = 12.0
    hawk_strike_radius: float = 1.50
    ground_pred_period_s: float = 1800.0
    ground_pred_dwell_s: float = 30.0
    ground_pred_speed: float = 0.60


DEFAULT_COOP = CoopConfig()
