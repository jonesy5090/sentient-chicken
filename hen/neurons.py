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


def pooled(r: jax.Array, pools: jax.Array, strength: float) -> jax.Array:
    """What a population's targets receive once it has an inhibitory interneuron (E106).

    `pools` is (R, N), one row per interneuron, 1.0 on the units it pools. Each target
    loses `strength` times the mean rate of its own pool.

    This is not a change to the rate code and does not make any neuron's output
    negative. `W r - lambda*mean_pool(r)*1` is `(W - (lambda/n) 1 1^T) r`: a weight
    matrix with one broad-projecting inhibitory unit added to it. The neuron's own rate
    is untouched; what changes is what its targets are handed.

    Why this and not balanced weights. `balanced_ei` (E072) makes each row's recurrent
    weights sum to zero, which removes the common component from the *current*. E104 6b
    measured the rate nonlinearity putting it straight back -- current DC 75.3% against
    rate DC 87.6% -- because every unit rests at the same bias (-2.000, sd 0.000) and so
    sits at the same point on the same sigmoid. Subtracting the pool's mean *rate* acts
    after the nonlinearity, which is where the common mode actually is.
    """
    counts = jnp.sum(pools, axis=-1)                       # (R,)
    means = (r @ pools.T) / (counts[None, :] + 1e-9)       # (H, R)
    return r - strength * (means @ pools)


def ctrnn_step(x: jax.Array, w: jax.Array, current: jax.Array,
               b: jax.Array, tau: jax.Array, dt: float,
               r_proj: jax.Array = None):
    """One forward-Euler step. Returns (x_next, r).

    x       (H, N)      membrane state
    w       (H, N, N)   recurrent weights, w[h, i, j] is j -> i
    current (H, N)      afferent input already projected through W_in
    r_proj  (H, N)      what the recurrent weights see, if an interneuron intervenes
                        (E106). Defaults to the rates themselves. `r` is still returned
                        unmodified -- the interneuron changes what targets receive, not
                        what the neuron does.
    """
    r = rate(x)
    recurrent = jnp.einsum("hij,hj->hi", w, r if r_proj is None else r_proj)
    dx = (-x + recurrent + current + b[None, :]) / tau[None, :]
    return x + dt * dx, r


def stability_ratio(tau: jax.Array, dt: float) -> float:
    """max(dt / tau). Forward Euler needs this comfortably below 1; we target 0.2."""
    return float(jnp.max(dt / tau))
