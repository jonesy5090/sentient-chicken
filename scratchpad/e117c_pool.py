"""E117 -- pool the two seed blocks.

Block 1 (genome seeds 1000-1007) and block 2 (1008-1015) disagree on two arms, which is
the whole reason this project requires a second block. Pooled over 16 seeds via Fisher z.

Run:  PYTHONPATH=. python scratchpad/e117c_pool.py
"""

import json

import numpy as np

from scratchpad.e117_standing_variation import ARMS, fisher_mean

b1 = json.load(open("scratchpad/e117_cache.json"))
b2 = json.load(open("scratchpad/e117_cache_b2.json"))

print(f"{'arm':>14} | {'block 1':>16} {'block 2':>16} | {'POOLED (16 seeds)':>24}")
print("-" * 82)
print("r(-drives): the homeostatic half of fitness")
print("-" * 82)
for arm in ARMS:
    if arm not in b1:
        continue
    r1, s1, t1 = fisher_mean(b1[arm]["per_seed_drives"])
    line = f"{arm:>14} | {r1:+.3f}+-{s1:.3f} t={t1:+5.2f} "
    if arm in b2:
        r2, s2, t2 = fisher_mean(b2[arm]["per_seed_drives"])
        pool = b1[arm]["per_seed_drives"] + b2[arm]["per_seed_drives"]
        rp, sp, tp = fisher_mean(pool)
        line += (f"{r2:+.3f}+-{s2:.3f} t={t2:+5.2f} | "
                 f"{rp:+.3f}+-{sp:.3f} t={tp:+6.2f}  (df=15, bar 2.131)")
    else:
        line += f"{'--':>16} {'--':>16} | block 1 only"
    print(line)

print("-" * 82)
print("r(fitness): drives plus the predation term")
print("-" * 82)
for arm in ARMS:
    if arm not in b1:
        continue
    r1, s1, t1 = fisher_mean(b1[arm]["per_seed_fit"])
    line = f"{arm:>14} | {r1:+.3f}+-{s1:.3f} t={t1:+5.2f} "
    if arm in b2:
        r2, s2, t2 = fisher_mean(b2[arm]["per_seed_fit"])
        pool = b1[arm]["per_seed_fit"] + b2[arm]["per_seed_fit"]
        rp, sp, tp = fisher_mean(pool)
        line += (f"{r2:+.3f}+-{s2:.3f} t={t2:+5.2f} | "
                 f"{rp:+.3f}+-{sp:.3f} t={tp:+6.2f}")
    else:
        line += f"{'--':>16} {'--':>16} | block 1 only"
    print(line)

print("-" * 82)
print("r(predation): is being caught ever a property of the hen?")
print("-" * 82)
for arm in ARMS:
    # `per_seed_caught` was added to the cache after the first arms had already run, so
    # only the later ones carry it. The per-block aggregates are in the run logs either
    # way, and none of them reach significance in either block.
    if arm not in b1 or "per_seed_caught" not in b1.get(arm, {}):
        continue
    r1, s1, t1 = fisher_mean(b1[arm]["per_seed_caught"])
    line = f"{arm:>14} | {r1:+.3f}+-{s1:.3f} t={t1:+5.2f} "
    if arm in b2 and "per_seed_caught" in b2[arm]:
        r2, s2, t2 = fisher_mean(b2[arm]["per_seed_caught"])
        pool = b1[arm]["per_seed_caught"] + b2[arm]["per_seed_caught"]
        rp, sp, tp = fisher_mean(pool)
        line += (f"{r2:+.3f}+-{s2:.3f} t={t2:+5.2f} | "
                 f"{rp:+.3f}+-{sp:.3f} t={tp:+6.2f}")
    else:
        line += f"{'--':>16} {'--':>16} | block 1 only"
    print(line)

print("\nContrast against `default`, pooled, on r(-drives) -- unpaired two-sample t on z")
print("-" * 82)
zd = np.arctanh(np.clip(b1["default"]["per_seed_drives"]
                        + b2["default"]["per_seed_drives"], -0.999, 0.999))
for arm in ARMS:
    if arm == "default" or arm not in b1 or arm not in b2:
        continue
    za = np.arctanh(np.clip(b1[arm]["per_seed_drives"]
                            + b2[arm]["per_seed_drives"], -0.999, 0.999))
    se = np.sqrt(za.var(ddof=1) / len(za) + zd.var(ddof=1) / len(zd))
    t = (za.mean() - zd.mean()) / se
    print(f"{arm:>14} vs default: dz={za.mean()-zd.mean():+.3f}+-{se:.3f}  t={t:+6.2f}"
          f"   (df~30, bar 2.04)")
