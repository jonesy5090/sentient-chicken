"""The closed loop: world -> senses -> brain -> motor -> world.

Everything runs inside a single `jax.lax.scan`, world included. That is the most
consequential performance decision in the project: a NumPy environment driving a JAX
brain would pay a host round-trip every 10 ms and forfeit one to two orders of
magnitude. The JAX RL environments (XLand-MiniGrid, Craftax, Navix) reach millions of
steps per second precisely by refusing to leave the accelerator, and the same applies
on CPU, where the win comes from XLA fusing the whole step into one kernel chain.

Long runs are chunked so that a 30-day rearing does not try to materialise a trace
with 260 million entries. Per-chunk summaries give a time series at whatever
resolution `chunk_s` sets.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from coop import sensing, spec, world
from coop.spec import CoopConfig
from hen import brain


class Trace(NamedTuple):
    """Per-step record. Only used for short assay rollouts."""
    motor: jax.Array        # (T, H, MOTOR_DIM)
    obs: jax.Array          # (T, H, OBS_DIM)


class Summary(NamedTuple):
    """Per-chunk record, for developmental runs."""
    t_s: jax.Array          # (C,) biological seconds at chunk end
    motor: jax.Array        # (C, MOTOR_DIM) flock-mean activation
    calls: jax.Array        # (C, N_CALLS)
    hunger: jax.Array       # (C,)
    thirst: jax.Array
    cold: jax.Array
    head_down: jax.Array    # (C,) fraction of time with the beak down
    struck: jax.Array       # (C,) cumulative predator contacts, flock total
    fed: jax.Array


def _one_step(carry, _, p, cfg):
    w, x, key = carry
    key, k = jax.random.split(key)
    obs = sensing.observe(w, cfg)
    x, motor = brain.step(x, obs, p, cfg.dt)
    w = world.step(w, motor, k, cfg)
    return (w, x, key), (motor, obs)


@partial(jax.jit, static_argnames=("cfg", "n_steps"))
def rollout(w, x, p, key, cfg: CoopConfig, n_steps: int):
    """Short run with a full per-step trace. Use for probes, not for lifetimes."""
    (w, x, key), (motor, obs) = jax.lax.scan(
        partial(_one_step, p=p, cfg=cfg), (w, x, key), None, length=n_steps)
    return w, x, key, Trace(motor=motor, obs=obs)


@partial(jax.jit, static_argnames=("cfg", "n_steps"))
def rollout_quiet(w, x, p, key, cfg: CoopConfig, n_steps: int):
    """Same dynamics, no trace retained. This is what the benchmark times."""
    def body(carry, _):
        carry, _out = _one_step(carry, None, p, cfg)
        return carry, None
    (w, x, key), _ = jax.lax.scan(body, (w, x, key), None, length=n_steps)
    return w, x, key


@partial(jax.jit, static_argnames=("cfg", "n_chunks", "chunk_steps"))
def _chunked(w, x, p, key, cfg: CoopConfig, n_chunks: int, chunk_steps: int):
    def chunk(carry, _):
        (w, x, key), (motor, obs) = jax.lax.scan(
            partial(_one_step, p=p, cfg=cfg), carry, None, length=chunk_steps)
        s = Summary(
            t_s=w.t.astype(jnp.float32) * cfg.dt,
            motor=jnp.mean(motor, axis=(0, 1)),
            calls=jnp.mean(motor[:, :, list(spec.CALL_MOTOR_IDX)], axis=(0, 1)),
            hunger=jnp.mean(w.hunger),
            thirst=jnp.mean(w.thirst),
            cold=jnp.mean(w.cold),
            head_down=jnp.mean(jnp.max(
                motor[:, :, list(spec.HEAD_DOWN_ACTIONS)], axis=-1)),
            struck=jnp.sum(w.n_struck),
            fed=jnp.sum(w.n_fed),
        )
        return (w, x, key), s
    return jax.lax.scan(chunk, (w, x, key), None, length=n_chunks)


def simulate(w, x, p, key, cfg: CoopConfig, seconds: float, chunk_s: float = 60.0):
    """Run for `seconds` of biological time, summarising every `chunk_s`."""
    chunk_steps = max(1, int(round(chunk_s / cfg.dt)))
    n_chunks = max(1, int(round(seconds / chunk_s)))
    (w, x, key), summary = _chunked(w, x, p, key, cfg, n_chunks, chunk_steps)
    return w, x, key, summary
