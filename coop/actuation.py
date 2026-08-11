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
    speed = mobility * (fwd * cfg.walk_speed + flee * cfg.flee_speed)

    heading = w.heading + turn * cfg.turn_rate * cfg.dt * mobility
    heading = (heading + jnp.pi) % (2 * jnp.pi) - jnp.pi

    step_vec = jnp.stack([jnp.cos(heading), jnp.sin(heading)], axis=-1)
    pos = w.pos + step_vec * (speed * cfg.dt)[:, None]
    pos = jnp.clip(pos, 0.0, cfg.size)   # the run is fenced

    return Kinematics(pos=pos, heading=heading, speed=speed, head_down=head_down)
