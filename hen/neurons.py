"""Continuous-time rate neurons.

    tau dx/dt = -x + W r(x) + W_in u + b,     r = sigmoid(x)

Rates rather than spikes, which is what makes a chicken's development affordable on a
laptop: the units are cheaper, and time constants of 50 ms and up permit a 10 ms
timestep instead of the 1 ms a spiking model would need. That second point is a free
factor of ten and it is the reason a 30-day rearing run fits in a night.

r = sigmoid keeps rates non-negative and bounded, which matters over the billions of
steps a developmental run takes -- an unbounded rate function will eventually find a
way to diverge. Inhibition comes from the sign of the weights (Dale's law), not from
negative rates.
"""

import jax
import jax.numpy as jnp


def rate(x: jax.Array) -> jax.Array:
    """Firing rate as a fraction of maximum, in [0, 1]."""
    return jax.nn.sigmoid(x)


def ctrnn_step(x: jax.Array, w: jax.Array, current: jax.Array,
               b: jax.Array, tau: jax.Array, dt: float):
    """One forward-Euler step. Returns (x_next, r).

    x       (H, N)      membrane state
    w       (H, N, N)   recurrent weights, w[h, i, j] is j -> i
    current (H, N)      afferent input already projected through W_in
    """
    r = rate(x)
    recurrent = jnp.einsum("hij,hj->hi", w, r)
    dx = (-x + recurrent + current + b[None, :]) / tau[None, :]
    return x + dt * dx, r


def stability_ratio(tau: jax.Array, dt: float) -> float:
    """max(dt / tau). Forward Euler needs this comfortably below 1; we target 0.2."""
    return float(jnp.max(dt / tau))
