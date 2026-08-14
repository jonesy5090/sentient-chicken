"""Distribution of the primary metric across many seeds, fixed condition."""
import json, time
import jax, jax.numpy as jnp
from coop import spec, world
from hen import brain, connectome, regions
from hen.plasticity import PlasticConfig
from run import simulate

cfg = spec.DEFAULT_COOP
MIN = 10.0
out = []
t0 = time.perf_counter()
for seed in range(36):
    key = jax.random.key(seed)
    w = world.reset(key, cfg)
    p = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS, n_hens=cfg.n_hens)
    x = brain.initial_state(p, cfg.n_hens)
    # geometry at t=0: distance from each hen to nearest food patch
    d0 = jnp.min(jnp.linalg.norm(w.pos[:, None, :] - w.food_pos[None, :, :], axis=-1), axis=-1)
    w_end, _x, _p, _ps, _k, s = simulate.simulate(
        w, x, p, jax.random.fold_in(key, 2), cfg, MIN*60.0, 60.0,
        PlasticConfig(enabled=False, explore_sigma=0.0))
    n = len(s.hunger); third = max(1, n//3)
    dend = jnp.min(jnp.linalg.norm(w_end.pos[:, None, :] - w_end.food_pos[None, :, :], axis=-1), axis=-1)
    out.append(dict(
        seed=seed,
        early=float(jnp.mean(s.hunger[:third])),
        late=float(jnp.mean(s.hunger[-third:])),
        change=float(jnp.mean(s.hunger[-third:]) - jnp.mean(s.hunger[:third])),
        fed=float(jnp.sum(w_end.n_fed)/(cfg.n_hens*MIN*60.0/cfg.dt)),
        struck=float(jnp.sum(w_end.n_struck)),
        d0_min=float(jnp.min(d0)), d0_mean=float(jnp.mean(d0)),
        dend_min=float(jnp.min(dend)), dend_mean=float(jnp.mean(dend)),
        n_at_food_end=float(jnp.sum(dend < 0.3)),
        traj=[float(v) for v in s.hunger],
    ))
    print(seed, f"{out[-1]['change']:+.4f}", f"fed={out[-1]['fed']*100:.2f}%",
          f"dend_min={out[-1]['dend_min']:.2f}", f"n_at_food={out[-1]['n_at_food_end']:.0f}",
          f"[{time.perf_counter()-t0:.0f}s]", flush=True)
json.dump(out, open("/home/user/sentient-chicken/scratchpad/seed_dist.json","w"))
print("done", time.perf_counter()-t0)
