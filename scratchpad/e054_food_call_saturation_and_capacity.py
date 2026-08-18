"""E054: does removing food-call saturation (E053) change H2c's comprehension outcome?

E042 found comprehension null even after E041's density fix -- "nowhere near a working
mechanism." The user's live question was whether near-constant food-calling (pre-E053:
42.8% of hen-steps) was crowding out pallium capacity that could otherwise represent the
rarer alarm channel. E053 fixed the saturation; this asks whether that mattered.

Same harness as E042 (`run.audience.comprehension`, `run.simulate.simulate`), same
density fix held constant across both arms (E041, 1.0) so this isolates the food-call
variable alone. The only manipulated variable is `legacy_food_call`
(`hen/connectome.build`): False = E053's discovery pulse (the current default), True =
the pre-E053 continuous-on-sight food call, recreated as an ablation condition.
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
ap.add_argument("--cache", default="scratchpad/e054_cache.json")
ap.add_argument("--budget", type=float, default=100000.0)
a = ap.parse_args()

cfg = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=a.hawk_period,
                                 food_deplete_rate=0.0)
NO_ASSOC = PlasticConfig(enabled=True, growth_enabled=False, explore_sigma=0.6,
                         pred_enabled=False)
ASSOC = PlasticConfig(enabled=True, growth_enabled=False, explore_sigma=0.6,
                      pred_enabled=True)

# density held at E041's fix (1.0) throughout -- legacy_food_call is the only thing
# that varies between the two arms this experiment is actually about.
CONDITIONS = {
    "no assoc, discovery pulse (baseline)": (NO_ASSOC, False),
    "assoc, discovery pulse (E053, current default)": (ASSOC, False),
    "assoc, legacy continuous food call (pre-E053 ablation)": (ASSOC, True),
}

cache = json.load(open(a.cache)) if os.path.exists(a.cache) else {}
t0 = time.perf_counter()


def run_one(seed, pc, legacy_food_call):
    key = jax.random.key(seed)
    w = world.reset(key, cfg)
    p = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS,
                         n_hens=16, sensory_pallium_density=1.0,
                         legacy_food_call=legacy_food_call)
    x = brain.initial_state(p, 16)
    gain = pc.pred_gain if pc.pred_enabled else 0.0
    comp_before = comprehension(p, cfg, 16, pred_gain=gain)
    _w, _x, p_end, *_ = simulate.simulate(
        w, x, p, jax.random.fold_in(key, 2), cfg, a.minutes * 60.0, 60.0, pc)
    comp_after = comprehension(p_end, cfg, 16, pred_gain=gain)
    w_pred_norm = float(jnp.mean(jnp.abs(p_end.W_pred)))
    return [comp_before, comp_after, w_pred_norm]


rows = {name: [] for name in CONDITIONS}
for name, (pc, legacy) in CONDITIONS.items():
    for s in range(a.seeds):
        ck = f"{name}|{s}|{a.minutes}|{a.hawk_period}"
        if ck not in cache:
            if time.perf_counter() - t0 > a.budget:
                print("budget reached; stopping")
                json.dump(cache, open(a.cache, "w"))
                raise SystemExit(0)
            cache[ck] = run_one(s, pc, legacy)
            json.dump(cache, open(a.cache, "w"))
        rows[name].append(cache[ck])

print(f"E054 -- does removing food-call saturation change comprehension, {a.seeds} seeds, "
      f"{a.minutes:.0f} min rearing, hawk every {a.hawk_period:.0f}s, density=1.0 throughout\n")
print(f"{'condition':<52}{'comp before':>13}{'comp after':>12}{'|W_pred|':>10}")
for name in CONDITIONS:
    arr = np.array(rows[name])
    print(f"{name:<52}{arr[:,0].mean():>13.4f}{arr[:,1].mean():>12.4f}{arr[:,2].mean():>10.5f}")

n = a.seeds
crit = _t_critical(n - 1)


def contrast(name_t, name_c):
    d = np.array(rows[name_t])[:, 1] - np.array(rows[name_c])[:, 1]
    m, se = d.mean(), d.std(ddof=1) / n ** 0.5
    t = abs(m) / (se + 1e-12)
    return m, se, t


print("\nPRIMARY -- comprehension after rearing, discovery pulse vs legacy continuous food call:")
m, se, t = contrast("assoc, discovery pulse (E053, current default)",
                    "assoc, legacy continuous food call (pre-E053 ablation)")
print(f"  {m:+.4f} +/- {se:.4f}  t={t:.2f}  threshold(df={n-1})={crit:.3f}  -> "
      f"{'SIGNIFICANT' if t > crit else 'not significant'}")

print("\nSECONDARY -- discovery-pulse association exceeds the no-association control:")
m, se, t = contrast("assoc, discovery pulse (E053, current default)",
                    "no assoc, discovery pulse (baseline)")
print(f"  {m:+.4f} +/- {se:.4f}  t={t:.2f}  threshold(df={n-1})={crit:.3f}  -> "
      f"{'SIGNIFICANT' if t > crit else 'not significant'}")

print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
