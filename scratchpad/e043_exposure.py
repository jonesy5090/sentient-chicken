"""E043: does raising predator density (exposure) move |W_pred| further than E042's
hawk_period_s=20 did, and does comprehension track it? See docs/experiments/E043.

Two conditions only -- E042 already established the no-association control near zero.
"""
import argparse, json, os, time

import jax, jax.numpy as jnp, numpy as np
from coop import spec, world
from hen import brain, connectome, regions
from hen.plasticity import PlasticConfig
from run import simulate
from run.audience import comprehension
from run.experiment import _t_critical

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--minutes", type=float, default=20.0)
ap.add_argument("--hawk-period", type=float, default=10.0)
ap.add_argument("--cache", default="scratchpad/e043_cache.json")
ap.add_argument("--budget", type=float, default=100000.0)
a = ap.parse_args()

cfg = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=a.hawk_period,
                                 food_deplete_rate=0.0)
ASSOC = PlasticConfig(enabled=True, growth_enabled=False, explore_sigma=0.6,
                      pred_enabled=True)

CONDITIONS = {
    "assoc, default density": None,
    "assoc, full density": 1.0,
}

cache = json.load(open(a.cache)) if os.path.exists(a.cache) else {}
t0 = time.perf_counter()


def run_one(seed, density):
    key = jax.random.key(seed)
    w = world.reset(key, cfg)
    kwargs = {} if density is None else {"sensory_pallium_density": density}
    p = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS,
                         n_hens=16, **kwargs)
    x = brain.initial_state(p, 16)
    comp_before = comprehension(p, cfg, 16, pred_gain=1.0)
    _w, _x, p_end, *_ = simulate.simulate(
        w, x, p, jax.random.fold_in(key, 2), cfg, a.minutes * 60.0, 60.0, ASSOC)
    comp_after = comprehension(p_end, cfg, 16, pred_gain=1.0)
    w_pred_norm = float(jnp.mean(jnp.abs(p_end.W_pred)))
    w_pred_max = float(jnp.max(jnp.abs(p_end.W_pred)))
    return [comp_before, comp_after, w_pred_norm, w_pred_max]


rows = {name: [] for name in CONDITIONS}
for name, density in CONDITIONS.items():
    for s in range(a.seeds):
        ck = f"{name}|{s}|{a.minutes}|{a.hawk_period}"
        if ck not in cache:
            if time.perf_counter() - t0 > a.budget:
                print("budget reached; stopping")
                json.dump(cache, open(a.cache, "w"))
                raise SystemExit(0)
            cache[ck] = run_one(s, density)
            json.dump(cache, open(a.cache, "w"))
        rows[name].append(cache[ck])

print(f"E043 -- exposure escalation, {a.seeds} seeds, {a.minutes:.0f} min rearing, "
      f"hawk every {a.hawk_period:.0f}s (E042 used 20s)\n")
print(f"{'condition':<28}{'comp before':>13}{'comp after':>12}{'mean|W_pred|':>13}{'max|W_pred|':>12}")
for name in CONDITIONS:
    arr = np.array(rows[name])
    print(f"{name:<28}{arr[:,0].mean():>13.4f}{arr[:,1].mean():>12.4f}"
          f"{arr[:,2].mean():>13.5f}{arr[:,3].mean():>12.5f}")

n = a.seeds
crit = _t_critical(n - 1)


def contrast(name_t, name_c, col):
    d = np.array(rows[name_t])[:, col] - np.array(rows[name_c])[:, col]
    m, se = d.mean(), d.std(ddof=1) / n ** 0.5
    t = abs(m) / (se + 1e-12)
    return m, se, t


print("\nPRIMARY -- |W_pred| growth vs E042's hawk_period=20 baseline (reported, not a live contrast):")
for name in CONDITIONS:
    arr = np.array(rows[name])
    e042_baseline = 0.00058 if "default" in name else 0.00047
    print(f"  {name}: mean|W_pred|={arr[:,2].mean():.5f} vs E042's {e042_baseline:.5f}")

print("\nSECONDARY -- comprehension after rearing, full density vs default density:")
m, se, t = contrast("assoc, full density", "assoc, default density", 1)
print(f"  {m:+.4f} +/- {se:.4f}  t={t:.2f}  threshold(df={n-1})={crit:.3f}  -> "
      f"{'SIGNIFICANT' if t > crit else 'not significant'}")

print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
