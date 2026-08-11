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


def reflex_matrix() -> np.ndarray:
    """The innate arc as a fixed (MOTOR_DIM, OBS_DIM) matrix."""
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

    # --- Distress: an isolated chick calls until someone answers ---
    w(spec.M_CALL_CONTACT, spec.IDX_ISOLATION, 5.0)
    w(spec.M_CALL_CONTACT, spec.IDX_COLD, 2.0)

    # --- Food call on finding food. Innate in production; the audience modulation
    # that a real cockerel shows is left for plasticity to discover.
    for b in _FRONT:
        w(spec.M_CALL_FOOD, spec.vis_index(b, spec.CLS_FOOD), 4.0)

    return r


def reflex_bias() -> np.ndarray:
    b = np.full((spec.MOTOR_DIM,), REST_BIAS, dtype=np.float32)
    b[spec.M_FORWARD] += TONIC_FORWARD
    return b
