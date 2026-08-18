"""E055: does a non-reward-gated (Hebbian) readout rule produce the audience-conditional
alarm-calling effect the reward-modulated rule failed to produce (E036)?

Reuses run/audience.py's exact instrument (_run_cell, the S/S+L scaffold_2x2 cells)
unmodified -- only the S+L cell's PlasticConfig gets hebbian_readout=True in place of
the default reward-modulated readout. Same task, same innate comprehension scaffold,
same everything else E036/E040 used.
"""
import argparse, json, os, time

import jax, jax.numpy as jnp
from coop import spec
from hen.plasticity import PlasticConfig
from run.audience import _run_cell
from run.experiment import _t_critical

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--minutes", type=float, default=30.0)
ap.add_argument("--cache", default="scratchpad/e055_cache.json")
ap.add_argument("--budget", type=float, default=100000.0)
a = ap.parse_args()

cfg = spec.DEFAULT_COOP._replace(n_hens=16, food_deplete_rate=0.0)
seconds = a.minutes * 60.0

FIXED = PlasticConfig(enabled=False, explore_sigma=0.0)
LEARN_HEBBIAN = PlasticConfig(enabled=True, growth_enabled=False, kin_audible=True,
                              explore_sigma=0.6, hebbian_readout=True)

CONDITIONS = {
    "S   (scaffold, fixed)": (FIXED, True),
    "S+L-hebbian (scaffold, hebbian readout)": (LEARN_HEBBIAN, True),
}

cache = json.load(open(a.cache)) if os.path.exists(a.cache) else {}
t0 = time.perf_counter()

rows = {name: [] for name in CONDITIONS}
for name, (pc, scaffold) in CONDITIONS.items():
    for s in range(a.seeds):
        ck = f"{name}|{s}|{a.minutes}"
        if ck not in cache:
            if time.perf_counter() - t0 > a.budget:
                print("budget reached; stopping")
                json.dump(cache, open(a.cache, "w"))
                raise SystemExit(0)
            c = _run_cell(s, cfg, seconds, pc, scaffold)
            cache[ck] = [c.audience, c.comprehension, c.strikes, c.hunger, c.synapses]
            json.dump(cache, open(a.cache, "w"))
        rows[name].append(cache[ck])

print(f"E055 -- Hebbian (non-reward-gated) readout vs the audience-effect null, "
      f"{a.seeds} seeds, {a.minutes:.0f} min rearing, 16 hens, food_deplete_rate=0\n")
hdr = f"{'condition':<42}{'audience':>10}{'compreh.':>10}{'strikes/hen':>13}{'hunger':>9}{'synapses':>10}"
print(hdr)
print("-" * len(hdr))
for name in CONDITIONS:
    arr = jnp.array(rows[name])
    print(f"{name:<42}{arr[:,0].mean():>+10.3f}{arr[:,1].mean():>10.4f}"
          f"{arr[:,2].mean():>13.2f}{arr[:,3].mean():>9.3f}{arr[:,4].mean():>10.0f}")

n = a.seeds
crit = _t_critical(n - 1)
S = "S   (scaffold, fixed)"
SLh = "S+L-hebbian (scaffold, hebbian readout)"

d = jnp.array(rows[SLh])[:, 0] - jnp.array(rows[S])[:, 0]
mean, se = float(d.mean()), float(d.std(ddof=1) / n ** 0.5)
t = abs(mean) / (se + 1e-12)
print(f"\nPRIMARY -- audience effect, S+L-hebbian - S:")
print(f"  {mean:+.4f} +/- {se:.4f}  t={t:.2f}  threshold(df={n-1})={crit:.3f}  -> "
      f"{'SIGNIFICANT' if t > crit else 'not significant'}")

print(f"\nMANIPULATION CHECK -- comprehension, scaffold+hebbian (expect ~0.19-0.25, "
      f"unaffected by the readout-rule change):")
print(f"  {jnp.array(rows[SLh])[:,1].mean():.4f}")

print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
