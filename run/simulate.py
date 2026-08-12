"""The closed loop: world -> senses -> brain -> motor -> world, with learning.

Everything runs inside a single `jax.lax.scan`, world included. That is the most
consequential performance decision in the project: a NumPy environment driving a JAX
brain would pay a host round-trip every 10 ms and forfeit one to two orders of
magnitude. The JAX RL environments (XLand-MiniGrid, Craftax, Navix) reach millions of
steps per second precisely by refusing to leave the accelerator, and the same applies
on CPU, where the win comes from XLA fusing the whole step into one kernel chain.

Weights live in the scan carry, so a rollout returns a *changed hen*. Plasticity is
switched by a static config field, which means the non-plastic control conditions
compile to a loop with no learning code in it at all rather than one that multiplies
by zero -- the control is genuinely a different program, not a disabled branch.

Long runs are chunked so that a 30-day rearing does not try to materialise a trace
with 260 million entries.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from coop import sensing, spec, world
from coop.spec import CoopConfig
from hen import brain, neurons, plasticity
from hen.plasticity import PlasticConfig

# No learning and no exploration: a fixed, deterministic hen. This is what assays
# run under -- they measure the policy, and noise would measure the policy plus the
# noise. It is also the control condition for every contrast.
NO_PLASTICITY = PlasticConfig(enabled=False, explore_sigma=0.0)


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
    reward: jax.Array       # (C,) mean neuromodulator input
    synapses: jax.Array     # (C,) mean live synapses per hen
    reflex_drive: jax.Array    # (C,) mean |innate arc| at the motor output
    cortical_drive: jax.Array  # (C,) mean |pallial readout| at the motor output
    w_out_norm: jax.Array      # (C,) mean |W_out|


def _one_step(carry, _, cfg: CoopConfig, pc: PlasticConfig):
    w, x, p, ps, key = carry
    key, k_world, k_grow, k_explore = jax.random.split(key, 4)

    # Exploration decays with age on the same schedule as the critical period: a
    # chick is behaviourally variable, an adult is not. Age comes from the world
    # clock rather than the plastic state so it is defined in every condition.
    age_s = w.t.astype(jnp.float32) * cfg.dt
    sigma = pc.explore_sigma / (1.0 + age_s / pc.explore_tau_s)

    obs = sensing.observe(w, cfg)
    x, motor, drives = brain.step(x, obs, p, cfg.dt, k_explore, sigma)
    w_next = world.step(w, motor, k_world, cfg)

    # Pathway magnitudes, carried out of the loop so a run can report whether the
    # cortical pathway ever gained influence over behaviour.
    mags = jnp.stack([jnp.mean(jnp.abs(drives.reflex)),
                      jnp.mean(jnp.abs(drives.cortical))])

    if not pc.enabled:
        return (w_next, x, p, ps, key), (motor, obs, jnp.zeros(()), mags)

    r = neurons.rate(x)
    reward = plasticity.reward(w, w_next, cfg, pc)
    ps = plasticity.update_traces(ps, r, motor, reward, cfg, pc)

    # Reward prediction error: what just happened, minus what she had come to expect.
    m = reward - ps.baseline

    p = jax.lax.cond(
        w_next.t % pc.interval == 0,
        lambda: plasticity.consolidate(p, ps, m, pc),
        lambda: p)

    if pc.growth_enabled:
        p = jax.lax.cond(
            w_next.t % pc.growth_interval == 0,
            lambda: plasticity.restructure(p, ps, k_grow, pc),
            lambda: p)

    return (w_next, x, p, ps, key), (motor, obs, jnp.mean(reward), mags)


@partial(jax.jit, static_argnames=("cfg", "pc", "n_steps"))
def rollout(w, x, p, key, cfg: CoopConfig, n_steps: int,
            ps=None, pc: PlasticConfig = NO_PLASTICITY):
    """Short run with a full per-step trace. Use for probes, not for lifetimes."""
    if ps is None:
        ps = plasticity.initial_state(p, w.pos.shape[0], pc)
    (w, x, p, ps, key), (motor, obs, _, _) = jax.lax.scan(
        partial(_one_step, cfg=cfg, pc=pc), (w, x, p, ps, key),
        None, length=n_steps)
    return w, x, p, ps, key, Trace(motor=motor, obs=obs)


@partial(jax.jit, static_argnames=("cfg", "pc", "n_steps"))
def rollout_quiet(w, x, p, key, cfg: CoopConfig, n_steps: int,
                  ps=None, pc: PlasticConfig = NO_PLASTICITY):
    """Same dynamics, no trace retained. This is what the benchmark times."""
    if ps is None:
        ps = plasticity.initial_state(p, w.pos.shape[0], pc)

    def body(carry, _):
        carry, _out = _one_step(carry, None, cfg, pc)
        return carry, None

    (w, x, p, ps, key), _ = jax.lax.scan(
        body, (w, x, p, ps, key), None, length=n_steps)
    return w, x, p, ps, key


@partial(jax.jit, static_argnames=("cfg", "pc", "n_chunks", "chunk_steps"))
def _chunked(w, x, p, ps, key, cfg: CoopConfig, pc: PlasticConfig,
             n_chunks: int, chunk_steps: int):
    def chunk(carry, _):
        carry, (motor, _obs, reward, mags) = jax.lax.scan(
            partial(_one_step, cfg=cfg, pc=pc), carry, None, length=chunk_steps)
        w, _x, p, _ps, _k = carry
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
            reward=jnp.mean(reward),
            synapses=jnp.mean(jnp.sum(p.W != 0.0, axis=(1, 2))),
            reflex_drive=jnp.mean(mags[:, 0]),
            cortical_drive=jnp.mean(mags[:, 1]),
            w_out_norm=jnp.mean(jnp.abs(p.W_out)),
        )
        return carry, s
    return jax.lax.scan(chunk, (w, x, p, ps, key), None, length=n_chunks)


def simulate(w, x, p, key, cfg: CoopConfig, seconds: float, chunk_s: float = 60.0,
             pc: PlasticConfig = NO_PLASTICITY, ps=None):
    """Run for `seconds` of biological time, summarising every `chunk_s`."""
    if ps is None:
        ps = plasticity.initial_state(p, w.pos.shape[0], pc)
    chunk_steps = max(1, int(round(chunk_s / cfg.dt)))
    n_chunks = max(1, int(round(seconds / chunk_s)))
    (w, x, p, ps, key), summary = _chunked(
        w, x, p, ps, key, cfg, pc, n_chunks, chunk_steps)
    return w, x, p, ps, key, summary
