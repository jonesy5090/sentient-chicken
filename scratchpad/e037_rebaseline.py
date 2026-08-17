"""E037: does H2's clean null (E020/E021, +0.011 +/- 0.012, t=0.95) survive on the
corrected (E023) connectome? See docs/experiments/E037.

E020/E021 ran at gain=0.70, before E023 fixed the E/I bug (pallium had zero inhibitory
neurons) and re-baselined gain to 0.95. `run/experiment.py`'s PHASE1 conditions call
`connectome.build` with no gain override, so simply re-running today uses the corrected
connectome automatically -- no code change needed, just fresh data.

Caches per-seed, per-condition hunger_early/late so a long run survives interruption,
matching scratchpad/e032.py's pattern. Two blocks (0-11, 12-23), pooled per the
E029/E030 template, declared in E037's pre-registration before this ran.
"""
import argparse, json, os, time

import jax, jax.numpy as jnp, numpy as np
from run.experiment import PHASE1, run_condition, _t_critical
from coop import spec

ap = argparse.ArgumentParser()
ap.add_argument("--minutes", type=float, default=20.0)
ap.add_argument("--seeds", type=int, default=24)
ap.add_argument("--hens", type=int, default=spec.DEFAULT_COOP.n_hens)
ap.add_argument("--chunk", type=float, default=60.0)
ap.add_argument("--cache", default="scratchpad/e037_cache.json")
ap.add_argument("--budget", type=float, default=100000.0)
a = ap.parse_args()

cfg = spec.DEFAULT_COOP._replace(n_hens=a.hens)
seconds = a.minutes * 60.0
cache = json.load(open(a.cache)) if os.path.exists(a.cache) else {}
t0 = time.perf_counter()

rows = {c.name: {} for c in PHASE1}
for cond in PHASE1:
    for s in range(a.seeds):
        ck = f"{cond.name}|{s}|{a.minutes}"
        if ck not in cache:
            if time.perf_counter() - t0 > a.budget:
                print(f"budget reached; stopping early")
                json.dump(cache, open(a.cache, "w"))
                raise SystemExit(0)
            r = run_condition(cond, s, cfg, seconds, a.chunk)
            cache[ck] = [r.hunger_early, r.hunger_late, r.fed_rate, r.struck, r.synapses]
            json.dump(cache, open(a.cache, "w"))
        rows[cond.name][s] = cache[ck]

print(f"E037 -- H2 rebaseline on the corrected connectome, {a.seeds} seeds, "
      f"{a.minutes:.0f} min, {cfg.n_hens} hens\n")


def change(name, s):
    early, late = rows[name][s][0], rows[name][s][1]
    return late - early


def block_stats(name_t, name_c, seeds):
    d = np.array([change(name_t, s) - change(name_c, s) for s in seeds])
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / n ** 0.5
    t = abs(m) / (se + 1e-12)
    return m, se, t, n, d


FIXED = "fixed (innate only)"
NOISE = "noise only (no learning)"
LEARN = "learning, no growth"
GROWTH = "learning + growth"

block_a = list(range(0, min(12, a.seeds)))
block_b = list(range(12, a.seeds))

print(f"{'condition':<26}{'hunger change (mean, all seeds run so far)':>44}")
for name in rows:
    seeds_done = sorted(rows[name].keys())
    vals = [change(name, s) for s in seeds_done]
    print(f"{name:<26}{np.mean(vals):>+44.4f}")

print("\nPRIMARY -- learning, no growth vs fixed, hunger change (E020's contrast):")
for label, seeds in (("block A (0-11)", block_a), ("block B (12-23)", block_b)):
    if not seeds or not all(s in rows[LEARN] for s in seeds):
        continue
    m, se, t, n, _ = block_stats(LEARN, FIXED, seeds)
    print(f"  {label:<18} n={n:<3} {m:+.4f} +/- {se:.4f}  t={t:.2f}")

if all(s in rows[LEARN] for s in block_a + block_b):
    m, se, t, n, d = block_stats(LEARN, FIXED, block_a + block_b)
    crit = _t_critical(n - 1)
    print(f"  {f'POOLED ({n} seeds)':<18} n={n:<3} {m:+.4f} +/- {se:.4f}  t={t:.2f}"
          f"  threshold(df={n-1})={crit:.3f}  -> "
          f"{'SIGNIFICANT' if t > crit else 'not significant'}")

    print("\nSECONDARY (reported for comparability with E020/E021, not the primary test):")
    for tname, ctrl in ((NOISE, FIXED), (GROWTH, FIXED)):
        m2, se2, t2, n2, _ = block_stats(tname, ctrl, block_a + block_b)
        print(f"  {tname} vs {ctrl}: {m2:+.4f} +/- {se2:.4f}  t={t2:.2f}")

print(f"\nwall clock so far: {time.perf_counter() - t0:.0f} s")
