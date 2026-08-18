"""E046: T1's other half -- does per-hen vigilance (head_down) rise with flock size,
specifically for the informative channel? See docs/experiments/E046.
"""
import argparse, time

import numpy as np
from coop import spec
from run.experiment import Condition, _t_critical
from run.h4 import EXPANDED, INNATE, _cache_load, _cache_save, run_condition

FLOCK_SIZES = [4, 8, 16, 32]


def conds():
    return {
        "C?": Condition("C? yoked", INNATE,
                        cfg_patch=(("channel_mode", "yoked"),
                                   ("call_log_steps", spec.YOKE_LOG_STEPS)),
                        pallium_scale=EXPANDED, scaffold=True),
        "L": Condition("L  language", INNATE, cfg_patch=(("channel_mode", "intact"),),
                       pallium_scale=EXPANDED, scaffold=True),
    }


ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--minutes", type=float, default=10.0)
ap.add_argument("--hawk-period", type=float, default=20.0)
ap.add_argument("--cache", default="scratchpad/e046_cache.json")
ap.add_argument("--budget", type=float, default=100000.0)
a = ap.parse_args()

seeds = list(range(a.seeds))
cache = _cache_load(a.cache)
t0 = time.perf_counter()

# H4Result field order: fed_rate, struck, exposed, caught_rate, at_risk, caught,
# caught_per_event, head_down, hunger, heard, dives, blind_risk, blind_caught,
# caught_any, caught_itt
HEAD_DOWN, CAUGHT_ITT = 7, 14

for hens in FLOCK_SIZES:
    cfg = spec.DEFAULT_COOP._replace(n_hens=hens, hawk_period_s=a.hawk_period,
                                     food_deplete_rate=0.0)
    for label, c in conds().items():
        for sd in seeds:
            k = f"{label}|{hens}|{sd}|{a.minutes}|{a.hawk_period}"
            if k in cache:
                continue
            if time.perf_counter() - t0 > a.budget:
                print("budget reached; stopping")
                raise SystemExit(0)
            cache[k] = list(run_condition(c, sd, cfg, a.minutes * 60.0))
            _cache_save(a.cache, cache)
    print(f"n_hens={hens} done ({time.perf_counter()-t0:.0f}s)")

print(f"\nE046 -- T1 flock-size sweep, {a.seeds} seeds, {a.minutes:.0f} min, "
      f"hawk every {a.hawk_period:.0f}s, food_deplete_rate=0\n")

n = a.seeds
crit = _t_critical(n - 1)

print(f"{'n_hens':<8}{'cond':<6}{'head_down':>11}{'caught/dive':>13}")
rows = {}
for hens in FLOCK_SIZES:
    for label in ("L", "C?"):
        vals = [cache[f"{label}|{hens}|{sd}|{a.minutes}|{a.hawk_period}"] for sd in seeds]
        arr = np.array(vals)
        rows[(hens, label)] = arr
        print(f"{hens:<8}{label:<6}{arr[:,HEAD_DOWN].mean():>11.4f}{arr[:,CAUGHT_ITT].mean():>13.4f}")

print(f"\n{'n_hens':<8}{'head_down L':>13}{'head_down C?':>14}{'L-C? t':>9}"
      f"{'caught/dive L-C?':>19}{'t':>7}")
for hens in FLOCK_SIZES:
    L, C = rows[(hens, "L")], rows[(hens, "C?")]
    d_hd = L[:, HEAD_DOWN] - C[:, HEAD_DOWN]
    d_risk = L[:, CAUGHT_ITT] - C[:, CAUGHT_ITT]
    t_hd = abs(d_hd.mean()) / (d_hd.std(ddof=1) / np.sqrt(n) + 1e-12)
    m2, se2 = d_risk.mean(), d_risk.std(ddof=1) / np.sqrt(n)
    t2 = abs(m2) / (se2 + 1e-12)
    print(f"{hens:<8}{L[:,HEAD_DOWN].mean():>13.4f}{C[:,HEAD_DOWN].mean():>14.4f}"
          f"{t_hd:>9.2f}{m2:>+15.4f}{'*' if t2>crit else ' ':<4}{t2:>7.2f}")

print(f"\n* = caught/dive contrast clears threshold t={crit:.3f} (df={n-1})")

# Does head_down for L trend up with n_hens? Simple linear check across the 4 points.
hd_L = np.array([rows[(h, "L")][:, HEAD_DOWN].mean() for h in FLOCK_SIZES])
hd_C = np.array([rows[(h, "C?")][:, HEAD_DOWN].mean() for h in FLOCK_SIZES])
slope_L = np.polyfit(FLOCK_SIZES, hd_L, 1)[0]
slope_C = np.polyfit(FLOCK_SIZES, hd_C, 1)[0]
print(f"\nhead_down vs n_hens, linear slope: L={slope_L:+.6f}/hen  C?={slope_C:+.6f}/hen")
print(f"wall clock: {time.perf_counter() - t0:.0f} s")
