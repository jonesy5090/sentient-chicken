"""E039: does H4's registered L vs C? contrast (caught/dive) survive without food
depletion? See docs/experiments/E039.

Mirrors scratchpad/e030.py's own pattern (three conditions, run_condition called
directly), with food_deplete_rate parametrised so a clean run is one flag away.
"""
import argparse, time

import jax, jax.numpy as jnp, numpy as np
from coop import spec
from run.experiment import Condition, _t_critical
from run.h4 import EXPANDED, INNATE, _cache_load, _cache_save, run_condition

CONDS = (
    Condition("C? yoked", INNATE,
              cfg_patch=(("channel_mode", "yoked"),
                         ("call_log_steps", spec.YOKE_LOG_STEPS)),
              pallium_scale=EXPANDED, scaffold=True),
    Condition("L  language", INNATE, cfg_patch=(("channel_mode", "intact"),),
              pallium_scale=EXPANDED, scaffold=True),
    Condition("Lx lesioned", INNATE, cfg_patch=(("channel_mode", "intact"),),
              pallium_scale=EXPANDED, scaffold=True, lesion_readout=True),
)

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--seed-offset", type=int, default=60)
ap.add_argument("--minutes", type=float, default=10.0)
ap.add_argument("--hawk-period", type=float, default=20.0)
ap.add_argument("--food-deplete-rate", type=float, default=spec.DEFAULT_COOP.food_deplete_rate)
ap.add_argument("--budget", type=float, default=100000.0)
ap.add_argument("--cache", default="scratchpad/e039_cache.json")
a = ap.parse_args()

cfg = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=a.hawk_period,
                                 food_deplete_rate=a.food_deplete_rate)
seeds = list(range(a.seed_offset, a.seed_offset + a.seeds))
cache = _cache_load(a.cache)
t0 = time.perf_counter()

for c in CONDS:
    for sd in seeds:
        k = f"{c.name}|{sd}|{a.minutes}|deplete={a.food_deplete_rate}"
        if k in cache:
            continue
        if time.perf_counter() - t0 > a.budget:
            print(f"budget reached; stopping")
            raise SystemExit(0)
        cache[k] = list(run_condition(c, sd, cfg, a.minutes * 60.0))
        _cache_save(a.cache, cache)

print(f"E039 -- H4 depletion audit, {len(seeds)} seeds ({seeds[0]}-{seeds[-1]}), "
      f"{a.minutes:.0f} min, hawk every {a.hawk_period:.0f}s, "
      f"food_deplete_rate={a.food_deplete_rate}\n")

# H4Result field order: fed_rate, struck, exposed, caught_rate, at_risk, caught,
# caught_per_event, head_down, hunger, heard, dives, blind_risk, blind_caught,
# caught_any, caught_itt
IDX_CAUGHT_ITT = 14
IDX_DIVES = 10
IDX_FED = 0

by_cond = {}
for c in CONDS:
    vals = [cache[f"{c.name}|{sd}|{a.minutes}|deplete={a.food_deplete_rate}"] for sd in seeds]
    by_cond[c.name] = np.array(vals)
    itt = by_cond[c.name][:, IDX_CAUGHT_ITT]
    dives = by_cond[c.name][:, IDX_DIVES]
    fed = by_cond[c.name][:, IDX_FED]
    print(f"{c.name:<14} caught/dive={itt.mean():.4f}  dives={dives.mean():.1f}  "
          f"fed%={fed.mean():.2f}")

n = len(seeds)


def contrast(name_t, name_c):
    d = by_cond[name_t][:, IDX_CAUGHT_ITT] - by_cond[name_c][:, IDX_CAUGHT_ITT]
    m, se = d.mean(), d.std(ddof=1) / n ** 0.5
    t = abs(m) / (se + 1e-12)
    crit = _t_critical(n - 1)
    return m, se, t, crit


print("\nPRIMARY -- L vs C?, caught/dive:")
m, se, t, crit = contrast("L  language", "C? yoked")
print(f"  {m:+.4f} +/- {se:.4f}  t={t:.2f}  threshold(df={n-1})={crit:.3f}  -> "
      f"{'SIGNIFICANT' if t > crit else 'not significant'}")

print("\nSECONDARY -- L vs Lx, caught/dive (the standing brain check):")
m, se, t, crit = contrast("L  language", "Lx lesioned")
print(f"  {m:+.4f} +/- {se:.4f}  t={t:.2f}  threshold(df={n-1})={crit:.3f}  -> "
      f"{'SIGNIFICANT' if t > crit else 'not significant'}")

print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
