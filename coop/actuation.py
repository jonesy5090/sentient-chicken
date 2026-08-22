"""Motor vector -> movement. The cerebellum analogue's entire output.

A real chicken spends 182 million neurons (63% of its brain) on the cerebellum,
coordinating a body with a neck, two wings, two legs and a great many feathers.
Here, eleven scalars in [0, 1] move a point around a plane. Everything the project
saves, it saves here and in `sensing.py`.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from coop import spec
from coop.spec import CoopConfig


class Kinematics(NamedTuple):
    pos: jax.Array
    heading: jax.Array
    speed: jax.Array
    head_down: jax.Array


def apply_motor(w, motor: jax.Array, cfg: CoopConfig) -> Kinematics:
    """Integrate one step of locomotion from motor activations (H, MOTOR_DIM)."""
    fwd = motor[:, spec.M_FORWARD]
    turn = motor[:, spec.M_TURN_L] - motor[:, spec.M_TURN_R]
    crouch = motor[:, spec.M_CROUCH]
    flee = motor[:, spec.M_FLEE]

    # Head-down posture, the gate that makes alarm calls informative. A hen pecking
    # or scratching has her beak in the litter and cannot scan the sky; `sensing.py`
    # multiplies the aerial channel by (1 - head_down).
    head_down = jnp.clip(
        jnp.max(motor[:, list(spec.HEAD_DOWN_ACTIONS)], axis=-1), 0.0, 1.0)

    # Crouching is freezing: it suppresses locomotion entirely. That is the trade the
    # hen makes -- invisible to the hawk, but not foraging and not going anywhere.
    mobility = jnp.clip(1.0 - crouch, 0.0, 1.0)
    # Sickness (T2, E060): a physiological constraint, not a decision -- applied
    # directly here the same way crouch's own mechanical effect is, not mediated by
    # the reflex arc or any learned weight. "Visibly slow", not frozen solid like
    # crouching: sickness_mobility_scale leaves a clearly-reduced but nonzero residual.
    mobility = jnp.where(w.sick_on, mobility * cfg.sickness_mobility_scale, mobility)
    # Stop to eat (E092). Keyed on `M_PECK` alone, NOT on `head_down`.
    #
    # The first version used `head_down`, which is `max(M_PECK, M_SCRATCH)`, and that is
    # wrong for a specific reason: `M_SCRATCH` is driven by hunger and is therefore on
    # everywhere, at sigmoid(hunger*3.0 + REST_BIAS) -- 0.269 at hunger 0.5. Gating on it
    # throttles locomotion across the whole arena instead of at food, which produces a
    # uniformly slower hen rather than an intermittent one. Measured: at gain 1.0 she
    # never reached a feeder at all, food_amount stayed at 1.000 and hunger rose to 0.63.
    #
    # `M_PECK` alone is not enough either, and the reason is E090: it added a hunger
    # term to pecking, so at hunger 0.43 a hen pecks at sigmoid(4.0*0.43 - 2.5) = 0.31
    # *everywhere*, not just at a feeder. Measured: gating on it gave speed 0.0864 at
    # food against 0.0794 away -- no localisation at all, just a uniformly slower hen.
    #
    # So the gate needs both terms: pecking AND standing at a patch. `at_food_prev` is
    # already computed by the world as an edge detector and carries exactly that.
    mobility = mobility * (1.0 - cfg.peck_stops_walking
                           * motor[:, spec.M_PECK] * w.at_food_prev)
    speed = mobility * (fwd * cfg.walk_speed + flee * cfg.flee_speed)

    heading = w.heading + turn * cfg.turn_rate * cfg.dt * mobility
    heading = (heading + jnp.pi) % (2 * jnp.pi) - jnp.pi

    step_vec = jnp.stack([jnp.cos(heading), jnp.sin(heading)], axis=-1)
    pos = w.pos + step_vec * (speed * cfg.dt)[:, None]
    pos = jnp.clip(pos, 0.0, cfg.size)   # the run is fenced

    return Kinematics(pos=pos, heading=heading, speed=speed, head_down=head_down)
