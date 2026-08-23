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
    calls: jax.Array        # (C, N_CALLS) emitted
    audio: jax.Array        # (C, N_CALLS) received -- differs whenever the channel is
                            # manipulated, and is the only honest manipulation check
    hunger: jax.Array       # (C,)
    thirst: jax.Array
    cold: jax.Array
    head_down: jax.Array    # (C,) fraction of time with the beak down
    struck: jax.Array       # (C,) cumulative predator contacts, flock total
    exposed: jax.Array      # (C,) cumulative steps in strike range, hiding or not
    at_risk: jax.Array      # (C,) (hen, dive) pairs beginning inside the radius
    caught: jax.Array       # (C,) of those, ones ending in a strike
    fed: jax.Array
    reward: jax.Array       # (C,) mean neuromodulator input
    synapses: jax.Array     # (C,) mean live synapses per hen
    reflex_drive: jax.Array    # (C,) mean |innate arc| at the motor output
    cortical_drive: jax.Array  # (C,) mean |pallial readout| at the motor output
    w_out_norm: jax.Array      # (C,) mean |W_out| -- the READOUT, see w_norm below
    w_norm: jax.Array          # (C,) mean |W| over live synapses -- the RECURRENT matrix
    #
    # Two norms, because under `hebbian_readout` they answer different questions and
    # only one of them answers "is the reward signal reaching the weights?".
    # `hebbian_readout` replaces the neuromodulator with a constant for `W_out` only,
    # so `W_out` drifts at the same rate whether a reward arrived or not -- which is
    # why |W_out| returned a falsely reassuring identical value across E065, E066 and
    # E068 while `sickness_penalty` was in fact supplying ~0.007% of reinforcement
    # (E068). `W` stays reward-gated, so |W| is the diagnostic those experiments
    # needed. Masked, because ~86% of the matrix is structurally zero and averaging
    # over it dilutes any real movement by an order of magnitude.


def _one_step(carry, _, cfg: CoopConfig, pc: PlasticConfig):
    w, x, p, ps, key = carry
    key, k_world, k_grow, k_explore = jax.random.split(key, 4)

    # Exploration decays with age on the same schedule as the critical period: a
    # chick is behaviourally variable, an adult is not. Age comes from the world
    # clock rather than the plastic state so it is defined in every condition.
    age_s = w.t.astype(jnp.float32) * cfg.dt
    sigma = pc.explore_sigma / (1.0 + age_s / pc.explore_tau_s)

    obs = sensing.observe(w, cfg)
    # The prediction pathway's source. Centred under `pred_centred` (E071): `z_lag` is
    # a strictly-positive rate trace whose across-stimulus signal is ~3.7% of its DC
    # baseline, and projecting it raw lets the DC term dominate the prediction.
    pred_from = None
    if pc.pred_enabled:
        pred_from = ps.z_lag - ps.z_lag_bar if pc.pred_centred else ps.z_lag
    x, motor, drives = brain.step(x, obs, p, cfg.dt, k_explore, sigma,
                                  pc.pred_gain, pred_from)
    w_next = world.step(w, motor, k_world, cfg)

    # Pathway magnitudes, carried out of the loop so a run can report whether the
    # cortical pathway ever gained influence over behaviour.
    mags = jnp.stack([jnp.mean(jnp.abs(drives.reflex)),
                      jnp.mean(jnp.abs(drives.cortical))])

    # Traces are STATE, not learning (E098). `z_lag` is what the prediction pathway
    # reads, and it has to keep tracking whenever anything reads it -- including with
    # `enabled=False`, which is how every assay runs. Before E098 this update sat below
    # the early return, so an assay measured a `W_pred` projection against a trace that
    # was either frozen at its reared value or, since `assay()` did not pass `ps` at
    # all, identically zero. A rule trained on a centred lagged trace was tested against
    # instantaneous `rate(x)`. That silently nullified the pathway in E097 and is the
    # fourth instance in this project of reading a quantity in a different regime from
    # the one it was measured in.
    #
    # Only the *weight* updates below stay gated on `enabled`: learning is the weight
    # change, and an assay must not do any. Gated on `pred_enabled` so that nothing
    # without a prediction pathway pays for this or changes behaviour -- see E098's
    # inertness falsifier, which asserts bit-identity for exactly that case.
    if pc.pred_enabled:
        pred_err = sensing.observability(w, cfg) * (obs - drives.predicted)
        ps = plasticity.update_traces(
            ps, neurons.rate(x), motor,
            plasticity.reward(w, w_next, cfg, pc), cfg, pc, pred_err)

    if not pc.enabled:
        return (w_next, x, p, ps, key), (motor, obs, jnp.zeros(()), mags)

    r = neurons.rate(x)
    reward = plasticity.reward(w, w_next, cfg, pc)

    # Prediction error, masked by what she could actually observe. A head-down hen is
    # not seeing an empty sky; she is not looking, and training on that sample would
    # teach her that alarm calls mean no hawk.
    #
    # With `pred_enabled` the traces were already advanced above, so advancing them
    # again here would double-step them. Only the non-pred path needs it now.
    if not pc.pred_enabled:
        ps = plasticity.update_traces(ps, r, motor, reward, cfg, pc, None)

    # Reward prediction error, averaged over the window since the last consolidation
    # (E067), not the instantaneous value at this step. `ps.m_acc` accumulates every
    # step in `update_traces`; a single-step discrete reward event (a strike, a
    # sickness onset) used to be visible to `consolidate()` only if it happened to
    # land exactly on this boundary -- confirmed at a 2% hit rate. Averaging over the
    # whole `interval`-step window means any such event within it is always seen,
    # rather than a lottery over its exact timing.
    at_boundary = w_next.t % pc.interval == 0
    # `legacy_m_sampling` restores the pre-E067 snapshot for bisection (E075).
    m = (reward - ps.baseline) if pc.legacy_m_sampling else (ps.m_acc / pc.interval)

    p = jax.lax.cond(at_boundary, lambda: plasticity.consolidate(p, ps, m, pc),
                     lambda: p)
    ps = ps._replace(m_acc=jnp.where(at_boundary, 0.0, ps.m_acc))

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
            # What flockmates actually *hear*, as distinct from what is emitted. The
            # two come apart the moment the channel is manipulated: a severed channel
            # (E024's C0) has hens calling exactly as much and nobody receiving
            # anything. E024's first manipulation check read `calls` and so reported a
            # severed channel as working.
            audio=jnp.mean(_obs[:, :, spec.AUDIO_LO:spec.AUDIO_HI], axis=(0, 1)),
            hunger=jnp.mean(w.hunger),
            thirst=jnp.mean(w.thirst),
            cold=jnp.mean(w.cold),
            head_down=jnp.mean(jnp.max(
                motor[:, :, list(spec.HEAD_DOWN_ACTIONS)], axis=-1)),
            struck=jnp.sum(w.n_struck),
            exposed=jnp.sum(w.n_exposed),
            at_risk=jnp.sum(w.n_at_risk),
            caught=jnp.sum(w.n_caught),
            fed=jnp.sum(w.n_fed),
            reward=jnp.mean(reward),
            synapses=jnp.mean(jnp.sum(p.W != 0.0, axis=(1, 2))),
            reflex_drive=jnp.mean(mags[:, 0]),
            cortical_drive=jnp.mean(mags[:, 1]),
            w_out_norm=jnp.mean(jnp.abs(p.W_out)),
            w_norm=(jnp.sum(jnp.abs(p.W) * p.mask[None, :, :])
                    / (p.W.shape[0] * jnp.sum(p.mask) + 1e-9)),
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
