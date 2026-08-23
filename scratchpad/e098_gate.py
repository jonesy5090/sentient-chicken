"""E098 hard gate -- the inertness falsifier (docs/experiments/E098 section 4).

Arm S (plasticity off, `pred_enabled` False) reared and assayed exactly as E097 did.
Run once BEFORE repair 1 and once AFTER; the four assay numbers must be bit-identical,
and must reproduce E097's published S row.

    PYTHONPATH=. .venv/bin/python scratchpad/e098_gate.py --out scratchpad/e098_gate_pre.json
"""
import argparse
import json
import time

import jax
import numpy as np

from coop import spec, world
from hen import brain, connectome, plasticity, regions
from hen.plasticity import PlasticConfig
from run import audience, simulate

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--minutes", type=float, default=30.0)
ap.add_argument("--with-ps", action="store_true",
                help="also assay carrying the reared ps (post-repair only)")
a = ap.parse_args()

CFG = spec.DEFAULT_COOP._replace(n_hens=16, food_deplete_rate=0.0)
STEPS = int(a.minutes * 60.0 / CFG.dt)
ARM_S = PlasticConfig(enabled=False, explore_sigma=0.0)

E097_S_INTACT = [0.3196, 0.3853, 0.5005, 0.5013]   # alarm_alone, alarm_aud, food_alone, food_aud

t0 = time.perf_counter()
rows = []
for seed in range(a.seeds):
    k = jax.random.key(seed)
    p = connectome.build(jax.random.fold_in(k, 1), regions.DEFAULT_REGIONS,
                         n_hens=CFG.n_hens, auditory_scaffold=True)
    w = world.reset(k, CFG)
    x = brain.initial_state(p, CFG.n_hens)
    ps = plasticity.initial_state(p, CFG.n_hens, ARM_S)
    _w, _x, p_end, ps_end, *_ = simulate.rollout_quiet(
        w, x, p, jax.random.fold_in(k, 2), CFG, STEPS, pc=ARM_S, ps=ps)
    r = audience.assay(p_end, CFG, CFG.n_hens)
    row = {"seed": seed, "intact4": [float(r[i]) for i in range(4)]}
    if a.with_ps:
        r2 = audience.assay(p_end, CFG, CFG.n_hens, ps=ps_end)
        row["intact4_with_ps"] = [float(r2[i]) for i in range(4)]
    rows.append(row)
    print(f"  seed {seed}: {row['intact4']}  ({time.perf_counter() - t0:.0f} s)",
          flush=True)

json.dump(rows, open(a.out, "w"))

arr = np.array([r["intact4"] for r in rows])
mean = arr.mean(axis=0)
names = ["alarm_alone", "alarm_aud", "food_alone", "food_aud"]
print("\narm S, intact assay, mean over "
      f"{a.seeds} seeds ({a.minutes:.0f} min rearing):")
for n, m, e in zip(names, mean, E097_S_INTACT):
    print(f"  {n:<12}{m:>10.6f}   E097 {e:.4f}   delta {m - e:+.6f}")

# Per-seed bit-identity against E097's own cache, which is the strongest available
# form of the check: the published table is rounded to four places.
try:
    cache = json.load(open("scratchpad/e097_cache.json"))
except FileNotFoundError:
    cache = {}
n_exact, n_cmp = 0, 0
for r in rows:
    ck = f"S|{r['seed']}|{a.minutes}"
    if ck in cache:
        n_cmp += 1
        n_exact += (cache[ck]["intact"] == r["intact4"])
if n_cmp:
    print(f"\nper-seed vs scratchpad/e097_cache.json: {n_exact}/{n_cmp} bit-identical")

if a.with_ps:
    same = sum(r["intact4"] == r["intact4_with_ps"] for r in rows)
    print(f"carrying the reared ps changes arm S: {a.seeds - same}/{a.seeds} seeds "
          f"(expected 0 -- arm S never updates a trace)")

print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
