"""E042: does E041's density fix unblock H2c's comprehension mechanism?

E008/E009 tested exactly this associative mechanism (W_pred, a Pavlovian delta rule --
see hen/plasticity.py:349-359, no reward term) and found it null: the pallium's states
for "heard a call" and "saw a hawk" were too similar to condition on (H2d). E041 found a
connectome fix that dramatically improves that separability (sensory_pallium_density).
Does comprehension -- crouching to a played-back call with no predator present -- rise
above zero when reared with that fix, where it didn't at the default density?
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
ap.add_argument("--hawk-period", type=float, default=20.0)
ap.add_argument("--cache", default="scratchpad/e042_cache.json")
ap.add_argument("--budget", type=float, default=100000.0)
a = ap.parse_args()

cfg = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=a.hawk_period,
                                 food_deplete_rate=0.0)
NO_ASSOC = PlasticConfig(enabled=True, growth_enabled=False, explore_sigma=0.6,
                         pred_enabled=False)
ASSOC = PlasticConfig(enabled=True, growth_enabled=False, explore_sigma=0.6,
                      pred_enabled=True)

CONDITIONS = {
    "no assoc, default density": (NO_ASSOC, None),
    "assoc, default density (E009 replication)": (ASSOC, None),
    "assoc, full density (E041 fix)": (ASSOC, 1.0),
}

cache = json.load(open(a.cache)) if os.path.exists(a.cache) else {}
t0 = time.perf_counter()


def run_one(seed, pc, density):
    key = jax.random.key(seed)
    w = world.reset(key, cfg)
    kwargs = {} if density is None else {"sensory_pallium_density": density}
    p = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS,
                         n_hens=16, **kwargs)
    x = brain.initial_state(p, 16)
    gain = pc.pred_gain if pc.pred_enabled else 0.0
    comp_before = comprehension(p, cfg, 16, pred_gain=gain)
    _w, _x, p_end, *_ = simulate.simulate(
        w, x, p, jax.random.fold_in(key, 2), cfg, a.minutes * 60.0, 60.0, pc)
    comp_after = comprehension(p_end, cfg, 16, pred_gain=gain)
    w_pred_norm = float(jnp.mean(jnp.abs(p_end.W_pred)))
    return [comp_before, comp_after, w_pred_norm]


rows = {name: [] for name in CONDITIONS}
for name, (pc, density) in CONDITIONS.items():
    for s in range(a.seeds):
        ck = f"{name}|{s}|{a.minutes}|{a.hawk_period}"
        if ck not in cache:
            if time.perf_counter() - t0 > a.budget:
                print("budget reached; stopping")
                json.dump(cache, open(a.cache, "w"))
                raise SystemExit(0)
            cache[ck] = run_one(s, pc, density)
            json.dump(cache, open(a.cache, "w"))
        rows[name].append(cache[ck])

print(f"E042 -- comprehension after the E041 density fix, {a.seeds} seeds, "
      f"{a.minutes:.0f} min rearing, hawk every {a.hawk_period:.0f}s\n")
print(f"{'condition':<42}{'comp before':>13}{'comp after':>12}{'|W_pred|':>10}")
for name in CONDITIONS:
    arr = np.array(rows[name])
    print(f"{name:<42}{arr[:,0].mean():>13.4f}{arr[:,1].mean():>12.4f}{arr[:,2].mean():>10.5f}")

n = a.seeds
crit = _t_critical(n - 1)


def contrast(name_t, name_c):
    d = np.array(rows[name_t])[:, 1] - np.array(rows[name_c])[:, 1]
    m, se = d.mean(), d.std(ddof=1) / n ** 0.5
    t = abs(m) / (se + 1e-12)
    return m, se, t


print("\nPRIMARY -- comprehension after rearing, full density vs default density (both assoc):")
m, se, t = contrast("assoc, full density (E041 fix)", "assoc, default density (E009 replication)")
print(f"  {m:+.4f} +/- {se:.4f}  t={t:.2f}  threshold(df={n-1})={crit:.3f}  -> "
      f"{'SIGNIFICANT' if t > crit else 'not significant'}")

print("\nSECONDARY -- does full-density association exceed the no-association control:")
m, se, t = contrast("assoc, full density (E041 fix)", "no assoc, default density")
print(f"  {m:+.4f} +/- {se:.4f}  t={t:.2f}  threshold(df={n-1})={crit:.3f}  -> "
      f"{'SIGNIFICANT' if t > crit else 'not significant'}")

print("\nSECONDARY -- default-density association vs no-association (E009 replication check):")
m, se, t = contrast("assoc, default density (E009 replication)", "no assoc, default density")
print(f"  {m:+.4f} +/- {se:.4f}  t={t:.2f}  threshold(df={n-1})={crit:.3f}  -> "
      f"{'SIGNIFICANT' if t > crit else 'not significant'}")

print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
