"""Diagnose why test_being_caught_does_not_dominate_the_reward_where_hawks_are_common
went from >=1 strike to 0 strikes after the wall-escape reflex was added.

Reproduces the failing test's exact setup (seed 0, 16 hens, hawk_period_s=20, 10000
steps) and logs hawk dives, closest approach per dive, and time spent near a wall.
"""
import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, regions

CFG = spec.DEFAULT_COOP._replace(hawk_period_s=20.0)
w = world.reset(jax.random.key(0), CFG)
p = connectome.build(jax.random.key(1), regions.DEFAULT_REGIONS, n_hens=CFG.n_hens)
x = brain.initial_state(p, CFG.n_hens)

n_dives = 0
was_on = False
min_d_this_dive = np.inf
near_wall_steps = 0
events = 0.0
for t in range(10_000):
    obs = sensing.observe(w, CFG)
    x, motor, _ = brain.step(x, obs, p, CFG.dt)
    wn = world.step(w, motor, jax.random.fold_in(jax.random.key(4), t), CFG)

    on = bool(wn.hawk_on > 0.5)
    if on and not was_on:
        n_dives += 1
        min_d_this_dive = np.inf
    if on:
        d = float(jnp.min(jnp.linalg.norm(wn.pos - wn.hawk_pos[None, :], axis=-1)))
        min_d_this_dive = min(min_d_this_dive, d)
    if not on and was_on:
        print(f"dive {n_dives} ended, closest approach {min_d_this_dive:.3f} m "
              f"(strike radius {CFG.hawk_strike_radius})")
    was_on = on

    wall_prox = np.asarray(obs[:, spec.IDX_WALL])
    near_wall_steps += int((wall_prox > 0.01).sum())

    events += float(jnp.sum(wn.n_strike_events - w.n_strike_events))
    w = wn

print(f"\ntotal dives: {n_dives}, total strike events: {events}")
print(f"hen-steps with wall proximity > 0: {near_wall_steps} / {10_000 * CFG.n_hens} "
      f"({100*near_wall_steps/(10_000*CFG.n_hens):.1f}%)")
print(f"final mean hen position: {np.asarray(w.pos).mean(axis=0)}, "
      f"arena size {CFG.size}")
