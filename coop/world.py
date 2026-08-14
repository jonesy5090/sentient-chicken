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
    # Ring buffer of recent emissions, for the yoked control (E026). A receiver in
    # that condition hears the flock's REAL calls from `t - lag_i`, so the lag has to
    # reach back further than one step and the history has to be carried. Only one
    # slot is written and one gathered per step, so the cost is a few hundred bytes of
    # traffic against W's megabytes -- it does not threaten the bandwidth budget.
    call_log: jax.Array     # (cfg.call_log_steps, H, N_CALLS); 1 unless yoked

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
    # (H,) cumulative steps spent inside a live predator's strike radius, whether or
    # not she was hiding. This is the *opportunity* to be struck, and without it
    # `n_struck` is uninterpretable: it counts contacts, which depend overwhelmingly on
    # whether the flock happened to be standing where the hawk came down. E024's smoke
    # test measured a 17x spread between conditions that was positional luck, not
    # behaviour. The quantity that tests the hypothesis is n_struck / n_exposed -- of
    # the moments when a hawk was actually on her, how often did she fail to hide?
    n_exposed: jax.Array    # (H,)
    # Event-anchored risk (E026). `n_exposed` turned out to be moved by the treatment
    # itself: crouching zeroes locomotion, so a hen at crouch 0.269 -- below the hiding
    # threshold but above zero -- lingers inside the strike radius instead of walking
    # out. Exposure varied 15x across conditions in E026's ablation, and `struck /
    # exposed` and raw `struck` disagreed about the *sign* of the effect.
    #
    # These anchor to the predator event instead. At the moment a dive begins, whoever
    # is inside the strike radius is at risk -- a denominator fixed before any response
    # can alter it. The numerator is whether such a hen was caught during that dive.
    # P(caught | at risk at dive onset) is the quantity the hypothesis is about.
    at_risk: jax.Array      # (H,) in the radius when the current dive began
    # ...and could not see it when it began. This is the subset where a call carries
    # information she does not already have, and it is the only subset where the
    # hypothesis can possibly be true. Measured free-running rather than staged: six
    # attempts to stage a blind-but-endangered hen all failed, because posture here is
    # coupled to position and will not hold still. The flock supplies the condition on
    # its own about 47% of the time (E026).
    blind_at_onset: jax.Array   # (H,)
    hit_this_dive: jax.Array   # (H,) struck at least once during the current dive
    n_at_risk: jax.Array    # (H,) dives she began inside the radius
    n_caught: jax.Array     # (H,) of those, dives she was struck in
    n_blind_risk: jax.Array # (H,) dives she began inside the radius AND blind
    n_blind_caught: jax.Array  # (H,) of those, dives she was struck in
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
        call_log=jnp.zeros((cfg.call_log_steps, h, spec.N_CALLS)),
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
        n_exposed=jnp.zeros((h,)),
        at_risk=jnp.zeros((h,)),
        blind_at_onset=jnp.zeros((h,)),
        hit_this_dive=jnp.zeros((h,)),
        n_at_risk=jnp.zeros((h,)),
        n_caught=jnp.zeros((h,)),
        n_blind_risk=jnp.zeros((h,)),
        n_blind_caught=jnp.zeros((h,)),
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
    s = cfg.size

    # --- Hawk: picks a hen and dives near her, for hawk_dive_s ---
    #
    # It used to appear at a uniformly random point in the arena. A hawk is a hunter,
    # not weather: it goes where the chickens are. The uniform version was also
    # quietly fatal to any experiment about alarm calls -- the flock clumps, the strike
    # radius is 1.5 m, and a uniform draw over a 20x20 m run therefore landed near the
    # birds about 5% of the time. E024's smoke test recorded zero strikes across every
    # condition, which is not a safe flock, it is a dead metric.
    #
    # The offset keeps the encounter uncertain rather than scripted: the targeted hen
    # is not guaranteed to be hit, and whether *anyone* is depends on where the flock
    # happens to be standing and who crouches -- which is the thing under test.
    k_hawk, k_fox, k_hpos, k_fpos, k_target = jax.random.split(key, 5)
    hawk_arrives = jax.random.uniform(k_hawk) < (cfg.dt / cfg.hawk_period_s)
    start_hawk = jnp.logical_and(hawk_arrives, w.hawk_on < 0.5)
    target = jax.random.randint(k_target, (), 0, w.pos.shape[0])
    offset = jax.random.uniform(k_hpos, (2,), minval=-cfg.hawk_aim_spread,
                                maxval=cfg.hawk_aim_spread)
    new_hawk_pos = jnp.where(
        start_hawk,
        jnp.clip(w.pos[target] + offset, 0.0, s),
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

    # Patches deplete under pressure and recover when abandoned. This is the only
    # force in the coop that pushes hens *apart*; without it a patch is infinite, the
    # flock piles onto one, and every bird shares every hawk (E024, E025).
    feeders = jnp.sum((at_food & pecking[:, None]).astype(jnp.float32), axis=0)
    food_amount = jnp.clip(
        w.food_amount + cfg.dt * ((1.0 - w.food_amount) / cfg.food_regrow_s
                                  - feeders * cfg.food_deplete_rate),
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

    emitted = _emit(motor, cfg) * vigour[:, None]

    # --- Predation. Crouching hides you from a hawk; fleeing outruns a fox. ---
    d_hawk = jnp.linalg.norm(kin.pos - hawk_pos[None, :], axis=-1)
    crouched = motor[:, spec.M_CROUCH] > 0.5
    # The hawk is overhead and visible for `hawk_approach_s` before it can strike.
    # `hawk_t` counts down from `hawk_dive_s`, so the approach is the first slice of
    # it. This is the interval an alarm call exists to fill; without it a warning can
    # never arrive in time and every condition sits at the same ceiling (E026).
    committed = hawk_t < (cfg.hawk_dive_s - cfg.hawk_approach_s)
    hawk_hit = ((d_hawk < cfg.hawk_strike_radius) & (hawk_on > 0.5)
                & committed & ~crouched)

    d_fox = jnp.linalg.norm(kin.pos - fox_pos[None, :], axis=-1)
    fleeing = motor[:, spec.M_FLEE] > 0.5
    fox_hit = (d_fox < cfg.hawk_strike_radius) & (fox_on > 0.5) & ~fleeing

    struck = (hawk_hit | fox_hit).astype(jnp.float32)
    # In range of a live predator, hiding or not: the denominator that makes `struck`
    # mean something.
    exposed = (((d_hawk < cfg.hawk_strike_radius) & (hawk_on > 0.5) & committed)
               | ((d_fox < cfg.hawk_strike_radius) & (fox_on > 0.5))).astype(jnp.float32)

    # --- Event-anchored risk. See the World fields for why exposure-time is unusable.
    onset = (hawk_on > 0.5) & (w.hawk_on < 0.5)
    ended = (hawk_on < 0.5) & (w.hawk_on > 0.5)
    in_radius = (d_hawk < cfg.hawk_strike_radius).astype(jnp.float32)
    # Who is in the radius the instant the hawk commits, before anyone can react.
    at_risk = jnp.where(onset, in_radius, w.at_risk)
    # Could she see it as it committed? aerial = proximity * (1 - head_down), and the
    # innate crouch needs 8*aerial > 2.5 to cross threshold.
    aerial_now = jnp.clip(1.0 - d_hawk / cfg.vision_range, 0.0, 1.0) * (1.0 - w.head_down)
    blind_at_onset = jnp.where(onset, (aerial_now < 0.3125).astype(jnp.float32),
                               w.blind_at_onset)
    hit_this_dive = jnp.where(onset, 0.0,
                              jnp.maximum(w.hit_this_dive, hawk_hit.astype(jnp.float32)))
    n_at_risk = w.n_at_risk + jnp.where(ended, at_risk, 0.0)
    n_caught = w.n_caught + jnp.where(ended, at_risk * hit_this_dive, 0.0)
    blind_risk = at_risk * blind_at_onset
    n_blind_risk = w.n_blind_risk + jnp.where(ended, blind_risk, 0.0)
    n_blind_caught = w.n_blind_caught + jnp.where(ended, blind_risk * hit_this_dive, 0.0)

    return w._replace(
        pos=kin.pos,
        heading=kin.heading,
        speed=kin.speed,
        head_down=kin.head_down,
        # What flockmates actually hear. Two factors: a spent bird cannot call loudly,
        # which makes vocal effort self-limiting without an arbitrary cap; and the
        # resting sigmoid floor is subtracted off, so a hen who is not calling emits
        # nothing at all rather than 0.076 of every call in the repertoire (E019).
        calls=emitted,
        call_log=w.call_log.at[w.t % cfg.call_log_steps].set(emitted),
        hunger=hunger,
        thirst=thirst,
        cold=cold,
        vigour=vigour,
        food_amount=food_amount,
        hawk_pos=hawk_pos, hawk_on=hawk_on, hawk_t=hawk_t,
        fox_pos=fox_pos, fox_on=fox_on, fox_t=fox_t,
        t=w.t + 1,
        n_struck=w.n_struck + struck,
        n_exposed=w.n_exposed + exposed,
        at_risk=at_risk,
        blind_at_onset=blind_at_onset,
        hit_this_dive=hit_this_dive,
        n_at_risk=n_at_risk,
        n_caught=n_caught,
        n_blind_risk=n_blind_risk,
        n_blind_caught=n_blind_caught,
        n_fed=w.n_fed + fed.astype(jnp.float32),
        n_drunk=w.n_drunk + drinking.astype(jnp.float32),
    )
