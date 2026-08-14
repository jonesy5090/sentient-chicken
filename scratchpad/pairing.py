"""Does the matched-seed pairing actually reduce variance? Per-seed spread, two blocks."""
import json, time
import jax, jax.numpy as jnp
from coop import spec, world
from hen import brain, connectome, regions
from hen.plasticity import PlasticConfig
from run import simulate

cfg = spec.DEFAULT_COOP; MIN = 10.0
CONDS = {
  "fixed":  PlasticConfig(enabled=False, explore_sigma=0.0),
  "noise":  PlasticConfig(enabled=False, explore_sigma=0.6),
  "learn":  PlasticConfig(enabled=True, growth_enabled=False, explore_sigma=0.6),
}
def run(seed, pc):
    key = jax.random.key(seed)
    w = world.reset(key, cfg)
    p = connectome.build(jax.random.fold_in(key,1), regions.DEFAULT_REGIONS, n_hens=cfg.n_hens)
    x = brain.initial_state(p, cfg.n_hens)
    w_end, _x, _p, _ps, _k, s = simulate.simulate(
        w, x, p, jax.random.fold_in(key,2), cfg, MIN*60.0, 60.0, pc)
    n = len(s.hunger); third = max(1, n//3)
    return (float(jnp.mean(s.hunger[-third:]) - jnp.mean(s.hunger[:third])),
            float(jnp.sum(w_end.n_fed)/(cfg.n_hens*MIN*60.0/cfg.dt)))

res = {k: {} for k in CONDS}
t0 = time.perf_counter()
for seed in range(24):
    for k, pc in CONDS.items():
        res[k][seed] = run(seed, pc)
    print(seed, {k: (round(v[seed][0],4), round(v[seed][1]*100,2)) for k,v in res.items()},
          f"[{time.perf_counter()-t0:.0f}s]", flush=True)
json.dump(res, open("/home/user/sentient-chicken/scratchpad/pairing.json","w"))
