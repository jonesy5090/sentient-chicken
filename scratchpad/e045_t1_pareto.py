"""E045: T1's intake/risk trade-off traced across a pallium-capacity sweep.
See docs/experiments/E045. Reuses run.h4's machinery directly, matching
scratchpad/e030.py / e039_h4_depletion.py's pattern.
"""
import argparse, time

import numpy as np
from coop import spec
from run.experiment import Condition, _t_critical
from run.h4 import INNATE, _cache_load, _cache_save, run_condition

CAPACITIES = [0.5, 1.0, 1.5, 2.0, 4.0]


def conds_at(scale):
    return {
        "C?": Condition("C? yoked", INNATE,
                        cfg_patch=(("channel_mode", "yoked"),
                                   ("call_log_steps", spec.YOKE_LOG_STEPS)),
                        pallium_scale=scale, scaffold=True),
        "L": Condition("L  language", INNATE, cfg_patch=(("channel_mode", "intact"),),
                       pallium_scale=scale, scaffold=True),
    }


ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--minutes", type=float, default=10.0)
ap.add_argument("--hawk-period", type=float, default=20.0)
ap.add_argument("--cache", default="scratchpad/e045_cache.json")
ap.add_argument("--budget", type=float, default=100000.0)
a = ap.parse_args()

cfg = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=a.hawk_period,
                                 food_deplete_rate=0.0)
seeds = list(range(a.seeds))
cache = _cache_load(a.cache)
t0 = time.perf_counter()

# H4Result field order: fed_rate, struck, exposed, caught_rate, at_risk, caught,
# caught_per_event, head_down, hunger, heard, dives, blind_risk, blind_caught,
# caught_any, caught_itt
FED, CAUGHT_ITT = 0, 14

for scale in CAPACITIES:
    for label, c in conds_at(scale).items():
        for sd in seeds:
            k = f"{label}|{scale}|{sd}|{a.minutes}|{a.hawk_period}"
            if k in cache:
                continue
            if time.perf_counter() - t0 > a.budget:
                print(f"budget reached; stopping")
                raise SystemExit(0)
            cache[k] = list(run_condition(c, sd, cfg, a.minutes * 60.0))
            _cache_save(a.cache, cache)
    print(f"capacity {scale}x done ({time.perf_counter()-t0:.0f}s)")

print(f"\nE045 -- T1 Pareto frontier, {a.seeds} seeds, {a.minutes:.0f} min, "
      f"hawk every {a.hawk_period:.0f}s, food_deplete_rate=0\n")

n = a.seeds
crit = _t_critical(n - 1)

print(f"{'capacity':<10}{'cond':<6}{'fed %':>8}{'caught/dive':>13}")
rows = {}
for scale in CAPACITIES:
    for label in ("L", "C?"):
        vals = [cache[f"{label}|{scale}|{sd}|{a.minutes}|{a.hawk_period}"] for sd in seeds]
        arr = np.array(vals)
        rows[(scale, label)] = arr
        print(f"{scale:<10}{label:<6}{arr[:,FED].mean():>8.3f}{arr[:,CAUGHT_ITT].mean():>13.4f}")

print(f"\n{'capacity':<10}{'fed% L-C?':>16}{'t':>7}{'caught/dive L-C?':>19}{'t':>7}")
for scale in CAPACITIES:
    L, C = rows[(scale, "L")], rows[(scale, "C?")]
    d_fed = L[:, FED] - C[:, FED]
    d_risk = L[:, CAUGHT_ITT] - C[:, CAUGHT_ITT]
    m1, se1 = d_fed.mean(), d_fed.std(ddof=1) / np.sqrt(n)
    m2, se2 = d_risk.mean(), d_risk.std(ddof=1) / np.sqrt(n)
    t1, t2 = abs(m1) / (se1 + 1e-12), abs(m2) / (se2 + 1e-12)
    print(f"{scale:<10}{m1:>+9.3f}{'*' if t1>crit else ' ':<7}{t1:>7.2f}"
          f"{m2:>+12.4f}{'*' if t2>crit else ' ':<7}{t2:>7.2f}")
print(f"\n* = clears threshold t={crit:.3f} (df={n-1})")
print(f"wall clock: {time.perf_counter() - t0:.0f} s")
