"""Assembling the hen: one step of brain, from observation to motor vector.

Two pathways run in parallel to the motor output, which is the point of the design:

  reflex arc   obs -> motor          fixed, innate, subcortical (`innate.py`)
  cortical     obs -> pallium -> ... -> motor stub -> motor      plastic, from phase 1

At hatch the cortical readout is scaled down to near-silence, so behaviour is almost
entirely reflexive. The pallium is wired and running -- it simply has nothing useful
to contribute yet. Learning, when it arrives in phase 1, will act only on the
cortical pathway; the reflex arc stays fixed for life, as it does in a real bird.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hen import neurons
from hen.connectome import BrainParams


class Drives(NamedTuple):
    """The two pathways' contributions to the motor output, kept separable.

    Their ratio is the diagnostic that says whether the pallium can influence
    behaviour at all. If `cortical` stays negligible against `reflex`, then whatever
    the pallium learns cannot reach a muscle, and no amount of rearing time will
    change that.
    """
    reflex: jax.Array      # (H, MOTOR_DIM)
    cortical: jax.Array    # (H, MOTOR_DIM)
    predicted: jax.Array   # (H, OBS_DIM) top-down contribution to perception


def initial_state(p: BrainParams, n_hens: int) -> jax.Array:
    """Membrane state at hatch: at rest, not silent."""
    return jnp.broadcast_to(p.b[None, :], (n_hens, p.b.shape[0])).copy()


def step(x: jax.Array, obs: jax.Array, p: BrainParams, dt: float,
         key: jax.Array = None, sigma: float = 0.0, pred_gain: float = 0.0):
    """Returns (x_next, motor, drives); motor in [0, 1], shape (H, MOTOR_DIM).

    `sigma` adds Gaussian noise to the motor drive before the output nonlinearity.
    Without it the hen is deterministic — same state, same action, forever — and a
    reinforcement rule cannot strengthen an action that never occurs. E006 found this
    is what stops comprehension emerging: to learn "crouch when you hear an alarm",
    she has to crouch on hearing one at least once, and crouching is only ever driven
    by *seeing* a hawk.

    The noise goes on the motor drive rather than on the membrane state deliberately.
    `z_motor` traces the motor output, so noise placed here is captured by the
    eligibility trace, and an exploratory action that happens to pay off gets credited
    to the synapses that produced it. Noise on the membrane would explore the
    dynamics but blur that credit.

    Assays pass sigma=0: they measure the learned policy, not the noise around it.
    """
    current = obs @ p.W_in.T
    x, _ = neurons.ctrnn_step(x, p.W, current, p.b, p.tau, dt)

    # The motor stub is the last region, so its width is fixed by the readout shape.
    n_motor = p.W_out.shape[-1]
    motor_stub = neurons.rate(x)[:, -n_motor:]
    cortical = jnp.einsum("hmn,hn->hm", p.W_out, motor_stub)

    # Top-down association. The pallium writes onto the observation the *reflex arc*
    # reads -- not onto the motor output, and not back into its own afferents. So a
    # learned cue does not have to recreate a behaviour, only the percept that already
    # drives it, and the innate arc supplies the rest. E007 measured why that matters:
    # driving crouch directly needs +2.50 against a cortical capacity of 0.002, while
    # driving it through the aerial channel needs ~0.3, because the reflex weight is 8.
    #
    # relu, so association can only add percepts, never suppress real ones; clipped to
    # the observation's own range so a hen cannot perceive more vividly than reality.
    predicted = jnp.einsum("hon,hn->ho", p.W_pred, neurons.rate(x))
    reflex_in = jnp.clip(obs + pred_gain * jax.nn.relu(predicted), 0.0, 1.0)
    reflex = reflex_in @ p.reflex.T
    drive = reflex + cortical + p.b_motor[None, :]
    # `sigma` is traced under jit (it decays with the world clock), so this cannot be
    # a Python conditional. A caller that wants no noise at all passes key=None.
    if key is not None:
        drive = drive + sigma * jax.random.normal(key, drive.shape)
    return x, jax.nn.sigmoid(drive), Drives(
        reflex=reflex, cortical=cortical, predicted=predicted)
