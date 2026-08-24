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

# Bias holding E101-B's reflex gate open at hatch: sigmoid(4.0) = 0.982, so an untrained
# hen's reflex arc reaches her muscles essentially intact and only learning can close it.
GATE_OPEN_BIAS = 4.0


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
    # The raw afferent current, before any relay processing. Carried out so the caller
    # can advance the adaptive baseline (E105) without recomputing the projection.
    current: jax.Array     # (H, N)


def initial_state(p: BrainParams, n_hens: int) -> jax.Array:
    """Membrane state at hatch: at rest, not silent."""
    return jnp.broadcast_to(p.b[None, :], (n_hens, p.b.shape[0])).copy()


def step(x: jax.Array, obs: jax.Array, p: BrainParams, dt: float,
         key: jax.Array = None, sigma: float = 0.0, pred_gain: float = 0.0,
         pred_from: jax.Array = None, pred_signed: bool = False,
         reflex_gate: bool = False, bg_gate: bool = False,
         bg_lateral: float = 1.0, sensory_lateral: float = 0.0,
         adapt_bar: jax.Array = None, recurrent_lateral: float = 0.0):
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
    raw_current = current
    # Temporal adaptation at the relay (E105), applied before the spatial pooling below
    # so each unit is first referred to its own recent history and the interneuron then
    # pools what is left. The two halves remove different things: this one removes what
    # a unit always sees, the pooling removes what every unit sees right now.
    if adapt_bar is not None:
        current = current - adapt_bar * p.lateral_pool[None, :]
    # Lateral inhibition at the sensory relay (E104). A pooled inhibitory interneuron
    # subtracts the component common to every afferent-receiving unit, so the projection
    # passes CONTRAST rather than total drive.
    #
    # E103 measured why this is needed. `W_in` is strictly positive -- 2630 nonzero
    # entries, none negative -- so a positive observation through positive weights gives
    # every unit a large shared term, "how much is in view". The mean direction's share
    # rises from 69.0% of the observation to 97.8% of the stub: situation-specific signal
    # drops from 31% to 2.2% in ONE synapse, at hatch, before any learning. Every learned
    # pathway in this model has been a linear readout of that near-constant since.
    #
    # Real relays almost universally do this -- retina, thalamus, avian tectum -- and its
    # function is exactly to discard the common component. Pooling is over `lateral_pool`
    # rather than all N units, because an interneuron pools the relay it sits in.
    if sensory_lateral:
        pool = p.lateral_pool[None, :]
        pooled = (jnp.sum(current * pool, axis=-1, keepdims=True)
                  / (jnp.sum(pool) + 1e-9))
        current = current - sensory_lateral * pooled * pool
    # E106: pooled inhibitory interneurons in the recurrent regions -- pallium and motor
    # stub, the two stages E105 found unaddressed. `sensory_lateral` above acts on the
    # afferent *current*, one stage earlier; this acts on the *rate*, which is where the
    # common mode survives. E104 6b measured why that distinction matters: subtracting
    # the mean current left the rate's DC share HIGHER than the current's (87.6% against
    # 75.3%), because every unit rests at the same bias and so sits at the same point on
    # the same sigmoid.
    #
    # Applied once, here, so that everything the population projects to sees the same
    # thing -- the recurrent weights, `W_out`, `W_pred` and `W_str`. An interneuron sits
    # between a population and everything downstream, not between it and a chosen one.
    r_proj = None
    if recurrent_lateral:
        r_proj = neurons.pooled(neurons.rate(x), p.region_pools, recurrent_lateral)
    x, _ = neurons.ctrnn_step(x, p.W, current, p.b, p.tau, dt, r_proj)

    # The motor stub is the last region, so its width is fixed by the readout shape.
    n_motor = p.W_out.shape[-1]
    rates = neurons.rate(x)
    if recurrent_lateral:
        rates = neurons.pooled(rates, p.region_pools, recurrent_lateral)
    motor_stub = rates[:, -n_motor:]
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
    # Sourced from a lagged pallial trace when one is supplied. E008 found the
    # instantaneous version was an autoencoder -- it mapped the current state to the
    # current observation, so during a hawk event it learned to predict the hawk from
    # the hawk. A lag makes it map what the brain was doing *before* to what is
    # observed *now*, which is the cue-to-outcome direction association needs.
    src = rates if pred_from is None else pred_from
    predicted = jnp.einsum("hon,hn->ho", p.W_pred, src * p.pred_src[None, :])
    # `pred_signed` (E101-A) removes the relu, so a negative prediction MASKS a real
    # percept rather than being discarded. The clip still holds both ends: she cannot
    # perceive more vividly than reality, and now cannot perceive below nothing either.
    top_down = predicted if pred_signed else jax.nn.relu(predicted)
    reflex_in = jnp.clip(obs + pred_gain * top_down, 0.0, 1.0)
    reflex = reflex_in @ p.reflex.T

    # `reflex_gate` (E101-B) lets the forebrain decide how much of the arc reaches the
    # muscles. Without it the two pathways meet by pure addition and the learned one --
    # measured 98.4% excitatory, peak opposition ~46x too small against the 8.0 crouch
    # reflex -- can only ever shout louder. `W_gate` is initialised so the gate sits at
    # ~1.0 at hatch: a newly hatched hen suppresses nothing and her arc arrives intact,
    # which is the correct developmental starting point.
    if bg_gate:
        # E102: basal-ganglia-style competitive release. `s - mean(s)` means a uniform
        # shift in striatal drive cancels exactly, so the gate cannot close everything --
        # it can only decide which channels to close *relative to the others*. That is
        # the striatal lateral-competition property, and it is the one thing E101-B's
        # free gate lacked.
        s = jnp.einsum("hmn,hn->hm", p.W_str, motor_stub)
        s = s - bg_lateral * jnp.mean(s, axis=-1, keepdims=True)
        reflex = reflex * jax.nn.sigmoid(GATE_OPEN_BIAS + s)
    elif reflex_gate:
        reflex = reflex * jax.nn.sigmoid(
            jnp.einsum("hmn,hn->hm", p.W_gate, motor_stub) + GATE_OPEN_BIAS)

    drive = reflex + cortical + p.b_motor[None, :]
    # `sigma` is traced under jit (it decays with the world clock), so this cannot be
    # a Python conditional. A caller that wants no noise at all passes key=None.
    if key is not None:
        drive = drive + sigma * jax.random.normal(key, drive.shape)
    return x, jax.nn.sigmoid(drive), Drives(
        reflex=reflex, cortical=cortical, predicted=predicted,
        current=raw_current)
