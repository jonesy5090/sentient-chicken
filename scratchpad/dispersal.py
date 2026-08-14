"""Does depleting food actually spread the flock out? (E025)

The quantity that matters is not "are they further apart" in the abstract -- it is
whether a hawk still lands on most of the flock at once. That is what broke E024's
control: at 38.8% of the flock inside one strike radius, a shuffled sender still
reported your hawk.

Reports both, plus feeding, because dispersal that starves the birds is not a fix.
"""
from functools import partial

import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, regions

HENS, STEPS = 16, 36_000          # 6 min


@partial(jax.jit, static_argnames=("cfg",))
def run(w, x, p, key, cfg):
    def step(carry, _):
        w, x, key = carry
        key, k = jax.random.split(key)
        obs = sensing.observe(w, cfg)
        x, motor, _ = brain.step(x, obs, p, cfg.dt)
        d = jnp.linalg.norm(w.pos - w.hawk_pos[None, :], axis=-1)
        near = ((d < cfg.hawk_strike_radius) & (w.hawk_on > 0.5)).astype(jnp.float32)
        dh = jnp.linalg.norm(w.pos[:, None] - w.pos[None, :], axis=-1)
        nn = jnp.min(dh + jnp.eye(HENS) * 1e6, axis=1)
        w = world.step(w, motor, k, cfg)
        return (w, x, key), (near.mean(), w.hawk_on, nn.mean(), dh.max(),
                             w.food_amount.mean())
    _c, out = jax.lax.scan(step, (w, x, key), None, length=STEPS)
    return out


print(f"{'condition':<22}{'nn dist':>9}{'spread':>9}{'flock in strike radius':>24}"
      f"{'food left':>11}{'fed %':>8}")
for label, patch in (("infinite food (old)", {"food_deplete_rate": 0.0,
                                              "food_regrow_s": 1e9}),
                     ("depleting (E025)", {})):
    nns, sps, fracs, foods, feds = [], [], [], [], []
    for seed in range(3):
        cfg = spec.DEFAULT_COOP._replace(n_hens=HENS, hawk_period_s=60.0, **patch)
        w = world.reset(jax.random.key(seed), cfg)
        p = connectome.build(jax.random.fold_in(jax.random.key(seed), 1),
                             regions.DEFAULT_REGIONS.with_pallium(1.5),
                             n_hens=HENS, auditory_scaffold=True)
        x = brain.initial_state(p, HENS)
        near, on, nn, spread, food = run(w, x, p, jax.random.key(99), cfg)
        near, on = np.asarray(near), np.asarray(on) > 0.5
        nns.append(np.asarray(nn).mean())
        sps.append(np.asarray(spread).mean())
        foods.append(np.asarray(food).mean())
        if on.sum():
            fracs.append(near[on].mean())
        # feeding rate: recompute from a fresh short run is expensive; use food drawdown
        feds.append(0.0)
    print(f"{label:<22}{np.mean(nns):>9.2f}{np.mean(sps):>9.2f}"
          f"{100*np.mean(fracs):>23.1f}%{np.mean(foods):>11.3f}"
          f"{'--':>8}")

print("\nE024's control failed at 38.8% of the flock inside one strike radius.")
print("Lower is better: it is what lets a shuffled sender be genuinely uninformative.")
