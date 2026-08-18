"""What a hen is born knowing: a fixed sensory -> motor reflex arc.

Biologically this is the brainstem and tectal reflex pathway, which bypasses the
pallium entirely. Nothing here is learned and nothing here changes; phase 1 adds
plasticity to the *cortical* pathway alongside it, never to this.

The call reflexes deserve a note. Konishi (1963) deafened day-old chicks and they
went on to develop the normal repertoire and normal call forms; isolation-reared
birds do the same. Chickens are vocal non-learners, so call *production* belongs
here, hardwired, alongside pecking and crouching.

What is deliberately absent is the audience effect. Real cockerels alarm- and
food-call far more readily with a hen present than when alone, graded by audience
type. That is *usage*, which is learned and socially contingent, so it is not wired
in -- it is a prediction for phase 2/3 to reproduce, and a way for the model to be
wrong.

The auditory scaffold below is optional and off by default; see `AUDITORY_SCAFFOLD`.
"""

import numpy as np

from coop import spec

# Bins 0..5 lie to the hen's right, 6..11 to her left; 5 and 6 straddle the beak.
_RIGHT = range(0, spec.N_BINS // 2)
_LEFT = range(spec.N_BINS // 2, spec.N_BINS)
_FRONT = (spec.N_BINS // 2 - 1, spec.N_BINS // 2)

# Resting drive. Motor output is sigmoid(reflex + cortical + bias), and actions with a
# 0.5 threshold (peck, crouch, flee) must stay off until a reflex actually fires.
REST_BIAS = -2.5
TONIC_FORWARD = 1.4     # a hatchling that never moves learns nothing


# Weight of every auditory scaffold connection (E018). One number, not four, so that
# it cannot be quietly tuned per-channel into the shape of a result.
#
# Chosen a priori against REST_BIAS: sigmoid(1.5 - 2.5) = 0.27, a partial graded
# response rather than a full one. That is what the biology shows -- parentally naive
# chicks stay in tonic immobility *longer* on hearing a conspecific fear squawk, they
# do not perform the whole anti-predator sequence. It is 19% of the visual crouch
# weight of 8.0, so seeing a hawk always dominates being told about one: first-hand
# information beats second-hand, which is the correct ordering.
SCAFFOLD_WEIGHT = 1.5


def reflex_matrix(auditory_scaffold: bool = False,
                  scaffold_gain: float = 1.0) -> np.ndarray:
    """The innate arc as a fixed (MOTOR_DIM, OBS_DIM) matrix.

    `auditory_scaffold` adds an innate response to *hearing* an alarm call. It is off
    by default and switched on only by E018's scaffolded conditions, because turning
    it on silently would change the comparison basis for every experiment before it.
    See `_add_auditory_scaffold` for what it wires and, more importantly, what it does
    not.

    `scaffold_gain` scales that response and exists for **positive controls**. E028's
    ladder returned a contrast that did not clear (-0.029, t=1.42), and a null is only
    informative if the instrument could have shown a positive -- so the way to read that
    number is to plant an effect of known size and check the metric finds it. Gain 1.0
    is the real hen; anything above it is a deliberately exaggerated bird used to test
    the harness, never to make a claim about biology.
    """
    r = np.zeros((spec.MOTOR_DIM, spec.OBS_DIM), dtype=np.float32)

    def w(motor, obs_idx, weight):
        r[motor, obs_idx] += weight

    # --- Pecking at contrast. Neonatal pecking is famously indiscriminate: chicks
    # peck at small objects whether or not they are hungry, and refine by experience.
    # So this is driven by the sight of food alone, with no hunger gate.
    for b in _FRONT:
        w(spec.M_PECK, spec.vis_index(b, spec.CLS_FOOD), 7.0)
        w(spec.M_PECK, spec.vis_index(b, spec.CLS_WATER), 5.0)

    # --- Orienting toward food and water ---
    for b in _LEFT:
        w(spec.M_TURN_L, spec.vis_index(b, spec.CLS_FOOD), 2.5)
        w(spec.M_TURN_L, spec.vis_index(b, spec.CLS_WATER), 1.5)
    for b in _RIGHT:
        w(spec.M_TURN_R, spec.vis_index(b, spec.CLS_FOOD), 2.5)
        w(spec.M_TURN_R, spec.vis_index(b, spec.CLS_WATER), 1.5)

    # --- Aerial predator: crouch and freeze, and call.
    # The aerial channel is already zeroed when the head is down, so a foraging hen
    # simply does not get this reflex. She has to be looking up, or be told.
    w(spec.M_CROUCH, spec.IDX_AERIAL, 8.0)
    w(spec.M_CALL_AERIAL, spec.IDX_AERIAL, 7.0)
    w(spec.M_FORWARD, spec.IDX_AERIAL, -6.0)

    # --- Ground predator: run, and call. A different call, because the appropriate
    # response differs -- this is the aerial/terrestrial distinction that makes
    # chicken alarm calls functionally referential.
    for b in range(spec.N_BINS):
        w(spec.M_FLEE, spec.vis_index(b, spec.CLS_GROUND_THREAT), 6.0)
        w(spec.M_CALL_GROUND, spec.vis_index(b, spec.CLS_GROUND_THREAT), 5.0)
    for b in _LEFT:   # turn away from it
        w(spec.M_TURN_R, spec.vis_index(b, spec.CLS_GROUND_THREAT), 3.0)
    for b in _RIGHT:
        w(spec.M_TURN_L, spec.vis_index(b, spec.CLS_GROUND_THREAT), 3.0)

    # --- Foraging drive: hunger and thirst make her walk and scratch ---
    w(spec.M_FORWARD, spec.IDX_HUNGER, 2.0)
    w(spec.M_FORWARD, spec.IDX_THIRST, 1.5)
    w(spec.M_SCRATCH, spec.IDX_HUNGER, 3.0)

    # --- Gregariousness and thermotaxis. A cold chick moves toward flockmates and
    # huddles; huddling is the only heat source in the coop.
    for b in _LEFT:
        w(spec.M_TURN_L, spec.vis_index(b, spec.CLS_FLOCKMATE), 1.2)
    for b in _RIGHT:
        w(spec.M_TURN_R, spec.vis_index(b, spec.CLS_FLOCKMATE), 1.2)
    w(spec.M_FORWARD, spec.IDX_COLD, 2.5)

    # --- Personal space (E025). Turn *away* from a flockmate once she is well inside
    # PERSONAL_SPACE_THRESHOLD -- the mirror of the attraction wiring above, on the
    # opposite turn channel. Weight must exceed the attraction weight (1.2) for
    # repulsion to actually win at close range rather than merely damping attraction;
    # 4.0 gives a clear margin, not a marginal one, matching how every other reflex
    # weight in this file is chosen. Below PERSONAL_SPACE_THRESHOLD, CLS_CROWDING is
    # exactly zero, so this adds nothing to ordinary flocking or huddling distances.
    for b in _LEFT:
        w(spec.M_TURN_R, spec.vis_index(b, spec.CLS_CROWDING), 4.0)
    for b in _RIGHT:
        w(spec.M_TURN_L, spec.vis_index(b, spec.CLS_CROWDING), 4.0)

    # --- Distress: an isolated chick calls until someone answers ---
    w(spec.M_CALL_CONTACT, spec.IDX_ISOLATION, 5.0)
    w(spec.M_CALL_CONTACT, spec.IDX_COLD, 2.0)

    # --- Food call on finding food. Innate in production; the audience modulation
    # that a real cockerel shows is left for plasticity to discover.
    for b in _FRONT:
        w(spec.M_CALL_FOOD, spec.vis_index(b, spec.CLS_FOOD), 4.0)

    if auditory_scaffold:
        _add_auditory_scaffold(w, scaffold_gain)

    return r


def _add_auditory_scaffold(w, gain: float = 1.0) -> None:
    """An innate response to *hearing* an alarm call. E018.

    Until E018 the arc had no auditory entry at all -- every weight from the four call
    channels was exactly zero, against 8.0 for crouch on seeing a hawk. That came from
    reading "comprehension is learned" as "comprehension is learned from nothing",
    which over-reads the biology. Parentally naive chicks already respond
    differentially to conspecific fear calls, and what develops with experience is
    *what the call is about* -- accurate predator identification forms gradually, and
    chickens even come to respond to tit alarm calls, which cannot be innate. The
    learned part is association off a stimulus that already arouses (Curio's mobbing
    chain, transmitted through six naive birds with no reward anywhere in it), not
    discovery by trial and error.

    Two channels, matched to the two responses, keeping the aerial/terrestrial
    distinction that makes chicken alarm calls functionally referential.
    """
    aerial_call = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)
    ground_call = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_GROUND)

    w(spec.M_CROUCH, aerial_call, SCAFFOLD_WEIGHT * gain)
    w(spec.M_FLEE, ground_call, SCAFFOLD_WEIGHT * gain)

    # Raising the head. This is not decoration and it is not symmetry with the visual
    # arc -- it is required, for a reason specific to how posture is computed here.
    # `actuation.py` derives head_down from peck and scratch alone; crouching only
    # zeroes locomotion. So a hen who crouches at a call while still pecking stays
    # blind to the sky, and the call has restored nothing. Suppressing the head-down
    # actions is what converts a heard call back into the information the caller had.
    # It is also simply what birds do on hearing an alarm.
    #
    # The visual arc needs no equivalent, because a hen who can *see* a hawk already
    # has her head up by definition.
    #
    # This is the part of the scaffold most likely to do E018's work for it: it opens
    # a route to a survival benefit with no learning anywhere -- call, head up, she
    # sees the hawk herself, and the visual reflex fires at weight 8.0. That route is
    # real and is how it works in nature. It is pre-registered as *innate*, and the
    # scaffold-without-learning condition exists to measure exactly how much of any
    # benefit it accounts for.
    for call in (aerial_call, ground_call):
        w(spec.M_PECK, call, -SCAFFOLD_WEIGHT * gain)
        w(spec.M_SCRATCH, call, -SCAFFOLD_WEIGHT * gain)

    # Deliberately NOT wired, each for its own reason:
    #
    # *No relay.* Hearing an alarm does not trigger producing one, though real
    # chickens do chain alarm calls. A relay makes the acoustic environment
    # self-driving and changes it for every hen at once, which would confound the
    # audience assay -- the quantity being measured is precisely whether a hen calls
    # more when others are present.
    #
    # *No posture, audience or context dependence.* The scaffold fires identically
    # whether she is head-down or head-up, alone or in company, hungry or fed. That
    # conditioning is what learning has to add; wiring any of it in would be wiring in
    # the answer.
    #
    # *Nothing on the food or contact channels.* They stay neutral with respect to
    # predators, which keeps a second-order conditioning test available later -- the
    # Curio design, where a neutral cue paired with an alarm becomes alarming itself.


def reflex_bias() -> np.ndarray:
    b = np.full((spec.MOTOR_DIM,), REST_BIAS, dtype=np.float32)
    b[spec.M_FORWARD] += TONIC_FORWARD
    return b
