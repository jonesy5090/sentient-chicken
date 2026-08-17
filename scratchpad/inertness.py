"""Can the cortical pathway move H2's metric at all? (E031, second diagnostic)

The credit-window story does not survive measurement (see `creditgap.py`): reward
arrives every 0.3 s, and two thirds of the peak peck-reward correlation is already
available at lag 0, inside the rule's 0.2 s window.

A better candidate came out of H4. E027 and E030 both found that zeroing `W_out` --
severing every route from the pallium to a muscle -- changes predation outcomes by
+0.010 (t=0.46), i.e. nothing. If the cortical pathway is behaviourally inert, then **no
learning rule operating on it can produce a measurable improvement**, and H2 is
unfalsifiable as constructed. That would explain every null from E001 onward without
appealing to the rule at all.

This measures it on H2's own metric rather than H4's. Three conditions, matched seeds:

    fixed        the normal hen, no plasticity
    lesioned     W_out = 0, no plasticity
    amplified    W_out x 10, no plasticity

If `fixed` and `lesioned` feed identically, the pathway contributes nothing and H2
cannot detect learning through it. `amplified` is the positive control for the
measurement itself: if a 10x cortical drive ALSO changes nothing, the metric is blind
rather than the pathway being inert, and that is a different problem.
"""
import argparse

import json, os, time

import jax, jax.numpy as jnp, numpy as np
from coop import spec, world
from hen import brain, connectome, regions
from run import simulate
from run.experiment import _t_critical

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--minutes", type=float, default=5.0)
ap.add_argument("--cache", default="scratchpad/e031_cache.json")
ap.add_argument("--budget", type=float, default=500.0)
args = ap.parse_args()

cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}
t0 = time.perf_counter()

# The last one is a positive control on the METRIC, not on the pathway: halving the
# innate peck reflex must change how often a hen eats. If fed % cannot see that either,
# the metric is blind and nothing else here means anything (E029's lesson, reapplied).
CONDS = (("fixed", 1.0), ("lesioned (W_out=0)", 0.0), ("amplified (W_out x10)", 10.0),
         ("peck reflex x0.5", "peck"))
cfg = spec.DEFAULT_COOP._replace(n_hens=16)

out = {}
for name, scale in CONDS:
    rows = []
    for s in range(args.seeds):
        ck = f"{name}|{s}|{args.minutes}"
        if ck in cache:
            rows.append(cache[ck]); continue
        if time.perf_counter() - t0 > args.budget:
            print(f"budget reached; {len(cache)} cells cached"); raise SystemExit(0)
        w = world.reset(jax.random.key(s), cfg)
        p = connectome.build(jax.random.fold_in(jax.random.key(s), 1),
                             regions.DEFAULT_REGIONS, n_hens=16)
        if scale == "peck":
            r = np.asarray(p.reflex).copy()
            r[spec.M_PECK, :] *= 0.5
            p = p._replace(reflex=jnp.asarray(r))
        else:
            p = p._replace(W_out=p.W_out * scale)
        x = brain.initial_state(p, 16)
        w_end, _x, _p, _ps, _k, summ = simulate.simulate(
            w, x, p, jax.random.fold_in(jax.random.key(s), 2), cfg,
            args.minutes * 60.0, 60.0, simulate.NO_PLASTICITY)
        steps = 16 * args.minutes * 60.0 / cfg.dt
        cache[ck] = [float(jnp.sum(w_end.n_fed)) / steps * 100,
                     float(jnp.mean(summ.hunger))]
        json.dump(cache, open(args.cache, "w"))
        rows.append(cache[ck])
    out[name] = np.array(rows)

print(f"H2's metric under cortical manipulation -- {args.seeds} matched seeds x "
      f"{args.minutes:.0f} min, 16 hens, no plasticity\n")
print(f"{'condition':<24}{'fed %':>9}{'hunger':>9}")
print("-" * 42)
for name, _ in CONDS:
    a = out[name]
    print(f"{name:<24}{a[:,0].mean():>9.3f}{a[:,1].mean():>9.3f}")

print("\npaired against `fixed`:")
base = out["fixed"]
for name, _ in CONDS[1:]:
    d = out[name][:, 0] - base[:, 0]
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / n ** 0.5
    t = abs(m) / (se + 1e-12)
    crit = _t_critical(n - 1)
    print(f"  fed %  {name:<24}{m:+.4f} +/- {se:.4f}  t={t:.2f}  "
          f"{'DIFFERENT' if t > crit else 'indistinguishable'}")

print("\nif lesioning changes nothing, the cortical pathway is behaviourally inert and")
print("H2 cannot detect learning through it, whatever the rule does. if amplifying")
print("ALSO changes nothing, the metric is blind and that is a different problem.")
