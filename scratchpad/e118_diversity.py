"""E118 -- can the generational search be made to keep improving past generation 3?

E116's loop stalls: all improvement by generation 3, diversity down to 12-13% of its
founding value by generation 15. Six arms, each changing one search parameter, each with
a matched unselected control.

Run:  PYTHONPATH=. python scratchpad/e118_diversity.py [arm ...]
"""

import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from coop import spec, world
from hen import brain, plasticity, regions
from run import evolve, simulate

CFG = spec.DEFAULT_COOP._replace(hawk_period_s=50.0)
N_LINEAGES = 4
GENERATIONS = 16
BLOCK = int(os.environ.get("E118_BLOCK", "1"))
LIN_BASE = 0 + (BLOCK - 1) * N_LINEAGES
CACHE = f"scratchpad/e118_cache{'' if BLOCK == 1 else f'_b{BLOCK}'}.json"

BASE = evolve.EvoConfig(generations=GENERATIONS, lifetime_s=600.0,
                        founder_sigma=2.0)

ARMS = {
    "baseline":    BASE,
    "mut_0.15":    BASE._replace(mutation=0.15),
    "mut_0.30":    BASE._replace(mutation=0.30),
    "parents_8":   BASE._replace(n_parents=8),
    "recomb":      BASE._replace(recombine=True),
    "drives_only": BASE._replace(caught_weight=0.0),
}


def motor_profile(p, key):
    """Mean drive per motor channel over 60 s -- the saturation check from section 4."""
    pc = evolve.NO_LEARNING
    w0 = world.reset(key, CFG)
    _, _, _, _, _, tr = simulate.rollout(
        w0, brain.initial_state(p, CFG.n_hens), p, jax.random.fold_in(key, 5),
        CFG, int(60.0 / CFG.dt), plasticity.initial_state(p, CFG.n_hens, pc), pc)
    return np.asarray(jnp.mean(tr.motor, axis=(0, 1)))


def run_arm(arm):
    evo = ARMS[arm]
    out = {"selected": [], "control": []}
    for lin in range(N_LINEAGES):
        key = jax.random.key(7000 + LIN_BASE + lin)
        for cond, select in (("selected", True), ("control", False)):
            t0 = time.time()
            hist, p_final = evolve.run_lineage(key, CFG, evo._replace(select=select))
            hung = [float(h["hunger"]) for h in hist]
            div = [float(h["diversity"]) for h in hist]
            div_w = [float(h["diversity_w"]) for h in hist]
            m = motor_profile(p_final, jax.random.fold_in(key, 99))
            out[cond].append(dict(
                lineage=LIN_BASE + lin, hunger=hung, diversity=div, diversity_w=div_w,
                improve=hung[0] - hung[-1],          # positive = got better
                div_frac=div[-1] / (div[0] + 1e-12),
                motor=m.tolist(), secs=time.time() - t0))
            print(f"    lin {LIN_BASE+lin} {cond:>8}: hunger {hung[0]:.4f} -> "
                  f"{hung[-1]:.4f} (improve {hung[0]-hung[-1]:+.4f})  "
                  f"div {div[-1]/(div[0]+1e-12)*100:5.1f}%  "
                  f"crouch={m[spec.M_CROUCH]:.2f} flee={m[spec.M_FLEE]:.2f} "
                  f"fwd={m[spec.M_FORWARD]:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    return out


def summarise(cache):
    print("\n" + "=" * 100)
    print(f"{'arm':>12} {'selected':>10} {'control':>10} {'sel-ctl':>18} "
          f"{'div%':>7} {'gen@90%':>8} {'crouch':>7} {'flee':>6}")
    print("=" * 100)
    for arm in ARMS:
        if arm not in cache:
            continue
        c = cache[arm]
        s = np.array([r["improve"] for r in c["selected"]])
        k = np.array([r["improve"] for r in c["control"]])
        d = s - k                      # paired by lineage: same world key, same founders
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        divf = np.mean([r["div_frac"] for r in c["selected"]]) * 100
        # generation by which 90% of the total improvement has happened
        g90 = []
        for r in c["selected"]:
            h = np.array(r["hunger"])
            tot = h[0] - h[-1]
            if abs(tot) < 1e-9:
                continue
            reached = np.where((h[0] - h) >= 0.9 * tot)[0]
            g90.append(int(reached[0]) if len(reached) else len(h))
        cr = np.mean([r["motor"][spec.M_CROUCH] for r in c["selected"]])
        fl = np.mean([r["motor"][spec.M_FLEE] for r in c["selected"]])
        print(f"{arm:>12} {s.mean():+10.4f} {k.mean():+10.4f} "
              f"{d.mean():+.4f}+-{se:.4f} t={d.mean()/se if se else 0:+5.2f} "
              f"{divf:6.1f}% {np.mean(g90) if g90 else float('nan'):8.1f} "
              f"{cr:7.2f} {fl:6.2f}")
    print("=" * 100)
    print("improve = hunger(gen 0) - hunger(gen 15); positive is better.")
    print("sel-ctl paired per lineage; threshold |t| = 3.182 (df=3).")
    print("E116's baseline for comparison: sel-ctl -0.0359 and -0.0315 on its own sign")
    print("convention (it reported the difference in *change*, so its sign is flipped")
    print("relative to `improve` here); div collapsed to 12-13%.")


def main():
    only = sys.argv[1:] or list(ARMS)
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    print(f"E118 block {BLOCK}: {GENERATIONS} generations, {N_LINEAGES} lineages, "
          f"lifetime {BASE.lifetime_s:.0f}s, founder_sigma={BASE.founder_sigma}\n")
    for arm in only:
        if arm in cache:
            print(f"[{arm}] cached")
            continue
        print(f"[{arm}] {ARMS[arm]}", flush=True)
        cache[arm] = run_arm(arm)
        json.dump(cache, open(CACHE, "w"), indent=1)
        print(flush=True)
    summarise(cache)


if __name__ == "__main__":
    main()
