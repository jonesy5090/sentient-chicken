"""Assembling the hen: one step of brain, from observation to motor vector.

Two pathways run in parallel to the motor output, which is the point of the design:

  reflex arc   obs -> motor          fixed, innate, subcortical (`innate.py`)
  cortical     obs -> pallium -> ... -> motor stub -> motor      plastic, from phase 1

At hatch the cortical readout is scaled down to near-silence, so behaviour is almost
entirely reflexive. The pallium is wired and running -- it simply has nothing useful
to contribute yet. Learning, when it arrives in phase 1, will act only on the
cortical pathway; the reflex arc stays fixed for life, as it does in a real bird.
"""

import jax
import jax.numpy as jnp

from hen import neurons
from hen.connectome import BrainParams


def initial_state(p: BrainParams, n_hens: int) -> jax.Array:
    """Membrane state at hatch: at rest, not silent."""
    return jnp.broadcast_to(p.b[None, :], (n_hens, p.b.shape[0])).copy()


def step(x: jax.Array, obs: jax.Array, p: BrainParams, dt: float):
    """Returns (x_next, motor) with motor in [0, 1], shape (H, MOTOR_DIM)."""
    current = obs @ p.W_in.T
    x, _ = neurons.ctrnn_step(x, p.W, current, p.b, p.tau, dt)

    # The motor stub is the last region, so its width is fixed by the readout shape.
    n_motor = p.W_out.shape[-1]
    motor_stub = neurons.rate(x)[:, -n_motor:]
    cortical = jnp.einsum("hmn,hn->hm", p.W_out, motor_stub)

    reflex = obs @ p.reflex.T
    motor = jax.nn.sigmoid(reflex + cortical + p.b_motor[None, :])
    return x, motor
