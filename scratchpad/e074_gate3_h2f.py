"""E074 gate 3: does H2f's audience effect -- this project's only genuine, replicated
positive learning result (E057) -- survive balanced E/I?

Run internally rather than against E057's recorded numbers: the codebase has moved a
long way since (OBS_DIM 74 -> 138, three new sensory blocks, and E067's m_acc fix
changed the learning rule itself), so historical figures are not a valid comparison.
Both arms run here, same code, same seeds, paired.
"""
import argparse, json, os, time
import jax, jax.numpy as jnp
from coop import spec, world
from hen import brain, connectome, regions
from hen.plasticity import PlasticConfig
from run import simulate
from run.audience import assay
from run.experiment import _t_critical

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--minutes", type=float, default=30.0)
ap.add_argument("--cache", default="scratchpad/e074_cache.json")
a = ap.parse_args()

cfg = spec.DEFAULT_COOP._replace(n_hens=16, food_deplete_rate=0.0)
FIXED = PlasticConfig(enabled=False, explore_sigma=0.0)
LEARN = PlasticConfig(enabled=True, growth_enabled=False, kin_audible=True,
                      explore_sigma=0.6, hebbian_readout=True,
                      readout_scaling_strength=0.3)


def run_one(seed, pc, balanced):
    key = jax.random.key(seed)
    w = world.reset(key, cfg)
    p = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS,
                         n_hens=cfg.n_hens, auditory_scaffold=True,
                         balanced_ei=balanced)
    x = brain.initial_state(p, cfg.n_hens)
    _w, _x, p_end, *_ = simulate.simulate(
        w, x, p, jax.random.fold_in(key, 2), cfg, a.minutes * 60.0, 60.0, pc)
    r = assay(p_end, cfg, cfg.n_hens)
    return [r.alarm_alone, r.alarm_audience, r.food_alone, r.food_audience]


cache = json.load(open(a.cache)) if os.path.exists(a.cache) else {}
t0 = time.perf_counter()
n = a.seeds
crit = _t_critical(n - 1)
print(f"E074 gate 3 -- H2f's audience effect under balanced E/I, {n} seeds, "
      f"{a.minutes:.0f} min\nthreshold t({n-1})={crit:.3f}\n")

for balanced in (False, True):
    rows = {}
    for name, pc in (("S", FIXED), ("L", LEARN)):
        vals = []
        for s in range(n):
            ck = f"{name}|{s}|{balanced}|{a.minutes}"
            if ck not in cache:
                cache[ck] = run_one(s, pc, balanced)
                json.dump(cache, open(a.cache, "w"))
            vals.append(cache[ck])
        rows[name] = jnp.array(vals)
    S, L = rows["S"], rows["L"]

    def one(d, label):
        m, se = float(d.mean()), float(d.std(ddof=1) / n ** 0.5)
        t = abs(m) / (se + 1e-12)
        print(f"  {label:<44}{m:+.4f} +/- {se:.4f}  t={t:.2f}  "
              f"{'SIGNIFICANT' if t > crit else 'not significant'}")

    print(f"balanced_ei={balanced}")
    one(L[:, 0] - S[:, 0], "general elevation (alarm_alone L-S)")
    one((L[:, 1] - L[:, 0]) - (S[:, 1] - S[:, 0]), "AUDIENCE-SPECIFIC (diff-in-diff)")
    one((L[:, 3] - L[:, 2]) - (S[:, 3] - S[:, 2]), "food control (should stay null)")
    print()

print(f"wall clock: {time.perf_counter() - t0:.0f} s")
