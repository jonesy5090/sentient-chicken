"""The coop: state, reset, and one step of world dynamics.

The environment is deliberately impoverished. That is the point -- it is what licenses
replacing the optic tectum and cerebellum with a 59-dim vector and 11 motor channels.
But it is not *arbitrarily* impoverished: it contains exactly the pressures the later
phases need, in particular a foraging/vigilance conflict (see `sensing.py`) and
predators that arrive without warning.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from coop import spec
from coop.spec import CoopConfig


class World(NamedTuple):
    # Hen kinematics and state
    pos: jax.Array          # (H, 2)
    heading: jax.Array      # (H,)
    hunger: jax.Array       # (H,) in [0, 1]
    thirst: jax.Array       # (H,)
    cold: jax.Array         # (H,)
    vigour: jax.Array       # (H,) in [0, 1]; vocal energy, 1 = rested
    head_down: jax.Array    # (H,) in [0, 1], from the previous motor vector
    speed: jax.Array        # (H,) metres/sec last step
    calls: jax.Array        # (H, N_CALLS) amplitudes emitted last step

    # Resources
    food_pos: jax.Array     # (F, 2)
    food_amount: jax.Array  # (F,) in [0, 1]
    water_pos: jax.Array    # (W, 2)

    # Predators
    hawk_pos: jax.Array     # (2,)
    hawk_on: jax.Array      # scalar 0.0/1.0
    hawk_t: jax.Array       # seconds remaining in the current dive
    fox_pos: jax.Array      # (2,)
    fox_on: jax.Array
    fox_t: jax.Array

    # Bookkeeping (not sensed; used by probes and assays)
    t: jax.Array            # step counter
    n_struck: jax.Array     # (H,) cumulative predator contacts
    n_fed: jax.Array        # (H,) cumulative successful pecks
    n_drunk: jax.Array      # (H,)


def reset(key: jax.Array, cfg: CoopConfig = spec.DEFAULT_COOP) -> World:
    """A fresh coop with the flock scattered near the centre, as after brooding."""
    k_pos, k_head, k_food, k_water = jax.random.split(key, 4)
    h, s = cfg.n_hens, cfg.size

    pos = jax.random.uniform(k_pos, (h, 2), minval=0.35 * s, maxval=0.65 * s)
    heading = jax.random.uniform(k_head, (h,), minval=-jnp.pi, maxval=jnp.pi)

    return World(
        pos=pos,
        heading=heading,
        hunger=jnp.full((h,), 0.3),
        thirst=jnp.full((h,), 0.2),
        cold=jnp.full((h,), 0.2),
        vigour=jnp.ones((h,)),
        head_down=jnp.zeros((h,)),
        speed=jnp.zeros((h,)),
        calls=jnp.zeros((h, spec.N_CALLS)),
        food_pos=jax.random.uniform(k_food, (cfg.n_food, 2),
                                    minval=0.1 * s, maxval=0.9 * s),
        food_amount=jnp.ones((cfg.n_food,)),
        water_pos=jax.random.uniform(k_water, (cfg.n_water, 2),
                                     minval=0.1 * s, maxval=0.9 * s),
        hawk_pos=jnp.array([0.5 * s, 0.5 * s]),
        hawk_on=jnp.array(0.0),
        hawk_t=jnp.array(0.0),
        fox_pos=jnp.array([0.0, 0.0]),
        fox_on=jnp.array(0.0),
        fox_t=jnp.array(0.0),
        t=jnp.array(0, dtype=jnp.int32),
        n_struck=jnp.zeros((h,)),
        n_fed=jnp.zeros((h,)),
        n_drunk=jnp.zeros((h,)),
    )


def _wrap_angle(a):
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


def _step_predators(w: World, key: jax.Array, cfg: CoopConfig):
    """Hawks arrive overhead; foxes prowl the fence line.

    Both are Poisson arrivals, so they are genuinely unpredictable -- a hen cannot
    learn a schedule, only learn to watch, or to listen to a flockmate who was.
    """
    k_hawk, k_fox, k_hpos, k_fpos = jax.random.split(key, 4)
    s = cfg.size

    # --- Hawk: appears over a random point and dives for hawk_dive_s ---
    hawk_arrives = jax.random.uniform(k_hawk) < (cfg.dt / cfg.hawk_period_s)
    start_hawk = jnp.logical_and(hawk_arrives, w.hawk_on < 0.5)
    new_hawk_pos = jnp.where(
        start_hawk,
        jax.random.uniform(k_hpos, (2,), minval=0.2 * s, maxval=0.8 * s),
        w.hawk_pos,
    )
    hawk_t = jnp.where(start_hawk, cfg.hawk_dive_s, jnp.maximum(w.hawk_t - cfg.dt, 0.0))
    hawk_on = jnp.where(hawk_t > 0.0, 1.0, 0.0)

    # --- Fox: enters at a wall and walks toward the nearest hen ---
    fox_arrives = jax.random.uniform(k_fox) < (cfg.dt / cfg.ground_pred_period_s)
    start_fox = jnp.logical_and(fox_arrives, w.fox_on < 0.5)
    edge_pos = jax.random.uniform(k_fpos, (2,), minval=0.0, maxval=s)
    edge_pos = edge_pos.at[0].set(jnp.where(edge_pos[1] > 0.5 * s, 0.0, s))
    fox_pos = jnp.where(start_fox, edge_pos, w.fox_pos)
    fox_t = jnp.where(start_fox, cfg.ground_pred_dwell_s,
                      jnp.maximum(w.fox_t - cfg.dt, 0.0))
    fox_on = jnp.where(fox_t > 0.0, 1.0, 0.0)

    to_hens = w.pos - fox_pos[None, :]
    d = jnp.linalg.norm(to_hens, axis=-1) + 1e-6
    target = jnp.argmin(d)
    fox_pos = fox_pos + fox_on * cfg.ground_pred_speed * cfg.dt * (
        to_hens[target] / d[target]
    )

    return new_hawk_pos, hawk_on, hawk_t, fox_pos, fox_on, fox_t


def _emit(motor: jax.Array, cfg: CoopConfig) -> jax.Array:
    """Motor call channels -> emitted call amplitude, (H, N_CALLS).

    Subtracting the resting sigmoid floor is what makes silence silent. See
    `spec.CALL_FLOOR`; `cfg.legacy_audio` restores the pre-E019 behaviour for E021.
    """
    raw = motor[:, list(spec.CALL_MOTOR_IDX)]
    if cfg.legacy_audio:
        return raw
    return jax.nn.relu(raw - spec.CALL_FLOOR) / (1.0 - spec.CALL_FLOOR)


def step(w: World, motor: jax.Array, key: jax.Array,
         cfg: CoopConfig = spec.DEFAULT_COOP) -> World:
    """Advance the coop by one dt given each hen's motor vector (H, MOTOR_DIM)."""
    from coop import actuation

    kin = actuation.apply_motor(w, motor, cfg)
    hawk_pos, hawk_on, hawk_t, fox_pos, fox_on, fox_t = _step_predators(w, key, cfg)

    # --- Feeding ---
    d_food = jnp.linalg.norm(w.pos[:, None, :] - w.food_pos[None, :, :], axis=-1)
    at_food = (d_food < cfg.peck_radius) & (w.food_amount[None, :] > 0.01)
    pecking = motor[:, spec.M_PECK] > 0.5
    fed = jnp.any(at_food, axis=-1) & pecking
    hunger = jnp.clip(
        w.hunger + cfg.dt * (1.0 / cfg.hunger_fill_s
                             - fed * cfg.peck_food_rate * w.hunger),
        0.0, 1.0)

    # --- Vocal effort, on its own budget. Calling spends vigour; silence restores it.
    call_effort = jnp.sum(motor[:, list(spec.CALL_MOTOR_IDX)], axis=-1)
    vigour = jnp.clip(
        w.vigour + cfg.dt * (1.0 / cfg.vigour_recovery_s
                             - call_effort * cfg.call_vigour_drain),
        0.0, 1.0)

    # --- Drinking ---
    d_water = jnp.linalg.norm(w.pos[:, None, :] - w.water_pos[None, :, :], axis=-1)
    drinking = jnp.any(d_water < cfg.drink_radius, axis=-1) & pecking
    thirst = jnp.clip(
        w.thirst + cfg.dt * (1.0 / cfg.thirst_fill_s
                             - drinking * cfg.drink_rate * w.thirst),
        0.0, 1.0)

    # --- Thermoregulation: huddling with flockmates is the only heat source ---
    d_hens = jnp.linalg.norm(kin.pos[:, None, :] - kin.pos[None, :, :], axis=-1)
    d_hens = d_hens + jnp.eye(cfg.n_hens) * 1e6
    n_huddled = jnp.minimum(jnp.sum(d_hens < cfg.huddle_radius, axis=-1),
                            cfg.huddle_max)
    cold = jnp.clip(
        w.cold + cfg.dt * (1.0 / cfg.cold_fill_s
                           - n_huddled * cfg.huddle_warm_rate),
        0.0, 1.0)

    # --- Predation. Crouching hides you from a hawk; fleeing outruns a fox. ---
    d_hawk = jnp.linalg.norm(kin.pos - hawk_pos[None, :], axis=-1)
    crouched = motor[:, spec.M_CROUCH] > 0.5
    hawk_hit = (d_hawk < cfg.hawk_strike_radius) & (hawk_on > 0.5) & ~crouched

    d_fox = jnp.linalg.norm(kin.pos - fox_pos[None, :], axis=-1)
    fleeing = motor[:, spec.M_FLEE] > 0.5
    fox_hit = (d_fox < cfg.hawk_strike_radius) & (fox_on > 0.5) & ~fleeing

    struck = (hawk_hit | fox_hit).astype(jnp.float32)

    return w._replace(
        pos=kin.pos,
        heading=kin.heading,
        speed=kin.speed,
        head_down=kin.head_down,
        # What flockmates actually hear. Two factors: a spent bird cannot call loudly,
        # which makes vocal effort self-limiting without an arbitrary cap; and the
        # resting sigmoid floor is subtracted off, so a hen who is not calling emits
        # nothing at all rather than 0.076 of every call in the repertoire (E019).
        calls=_emit(motor, cfg) * vigour[:, None],
        hunger=hunger,
        thirst=thirst,
        cold=cold,
        vigour=vigour,
        food_amount=w.food_amount,   # patches are not depleted in phase 0
        hawk_pos=hawk_pos, hawk_on=hawk_on, hawk_t=hawk_t,
        fox_pos=fox_pos, fox_on=fox_on, fox_t=fox_t,
        t=w.t + 1,
        n_struck=w.n_struck + struck,
        n_fed=w.n_fed + fed.astype(jnp.float32),
        n_drunk=w.n_drunk + drinking.astype(jnp.float32),
    )
