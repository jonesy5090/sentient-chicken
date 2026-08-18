"""E062: T2 Stage 1c -- calibrate contamination_period_s. Sweeps the period across
{100, 200, 300, 450, 600}s, 16-hen free-running flock, no learning. Three checks:
audience size (does a longer period buy more reachable audience, or does it saturate
given hear_range already covers the arena?), overlap (does a rotation fire while a
hen from the previous one is still visibly sick?), and rotations-per-run (the
statistical-power tradeoff).
"""
from functools import partial

import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, regions

HENS = 16
DURATION_MIN = 40
STEPS = int(DURATION_MIN * 60 / spec.DEFAULT_COOP.dt)   # 240,000
PERIODS = (100.0, 200.0, 300.0, 450.0, 600.0)
SEEDS = 3


@partial(jax.jit, static_argnames=("cfg",))
def run(w, x, p, key, cfg):
    def step(carry, _):
        w, x, key = carry
        key, k = jax.random.split(key)
        obs = sensing.observe(w, cfg)
        x, motor, _ = brain.step(x, obs, p, cfg.dt)

        d_hens = jnp.linalg.norm(w.pos[:, None, :] - w.pos[None, :, :], axis=-1)
        d_hens = d_hens + jnp.eye(cfg.n_hens) * 1e6
        # audience: for each sick hen, how many OTHER hens are within vision_range.
        # Summed over sick hens, matching E061 Check 3's per-step masking style --
        # counted once per (sick hen, nearby healthy hen) pair, not deduplicated
        # across multiple sick hens, since simultaneous sickness events are rare
        # at these periods and this keeps the estimator simple and unbiased for the
        # single-sick-hen case that dominates.
        near = d_hens < cfg.vision_range
        audience = jnp.sum(near & w.sick_on[:, None], axis=-1)  # per sick hen
        any_sick_before = w.sick_on.any()

        w_next = world.step(w, motor, k, cfg)
        rotated = w_next.contamination_epoch != w.contamination_epoch
        overlap = rotated & any_sick_before

        return (w_next, x, key), (audience, w.sick_on, rotated, overlap)
    return jax.lax.scan(step, (w, x, key), None, length=STEPS)[1]


print(f"{'period':>8}{'mean audience':>16}{'rotations':>12}{'overlap frac':>15}")
for period in PERIODS:
    cfg = spec.DEFAULT_COOP._replace(n_hens=HENS, food_deplete_rate=0.0,
                                     contamination_period_s=period)
    audiences, rotation_counts, overlap_fracs = [], [], []
    for seed in range(SEEDS):
        key = jax.random.key(seed)
        w = world.reset(key, cfg)
        p = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS,
                             n_hens=HENS)
        x = brain.initial_state(p, HENS)
        audience, sick_on, rotated, overlap = run(w, x, p, jax.random.key(99), cfg)

        aud, sick = np.asarray(audience), np.asarray(sick_on)   # (T,H) each
        sick_steps = sick.sum()
        audiences.append(aud.sum() / sick_steps if sick_steps else float("nan"))

        rot = np.asarray(rotated)
        n_rot = int(rot.sum())
        rotation_counts.append(n_rot)
        ov = np.asarray(overlap)
        overlap_fracs.append(ov.sum() / n_rot if n_rot else float("nan"))

    print(f"{period:>8.0f}{np.nanmean(audiences):>16.3f}"
          f"{np.mean(rotation_counts):>12.1f}{np.nanmean(overlap_fracs):>15.3f}")

print(f"\n({DURATION_MIN} min runs, {SEEDS} seeds/period, vision_range="
      f"{spec.DEFAULT_COOP.vision_range}, sickness_duration_s="
      f"{spec.DEFAULT_COOP.sickness_duration_s})")
