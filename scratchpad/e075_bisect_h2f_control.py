"""E075: what broke H2f's food-channel control?

E074 found it firing at +0.1054, t=10.04 on current code, where E057 reported it null.
Three candidates, all recent. This bisects the one that is cheapest to revert and most
likely on mechanism -- E067's m_acc fix, which changed what `W` learns from, and `W`
feeds the pallial states `W_out` reads.

Arms, all otherwise identical to E057's contrast (8 seeds, 30 min, auditory scaffold):
  current   -- as shipped
  legacy_m  -- E067's fix reverted via legacy_m_sampling
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
ap.add_argument("--cache", default="scratchpad/e075_cache.json")
a = ap.parse_args()

cfg = spec.DEFAULT_COOP._replace(n_hens=16, food_deplete_rate=0.0)
BASE = dict(growth_enabled=False, kin_audible=True, explore_sigma=0.6,
            hebbian_readout=True, readout_scaling_strength=0.3)

ARMS = {
    "current":  (PlasticConfig(enabled=False, explore_sigma=0.0),
                 PlasticConfig(enabled=True, **BASE)),
    "legacy_m": (PlasticConfig(enabled=False, explore_sigma=0.0, legacy_m_sampling=True),
                 PlasticConfig(enabled=True, legacy_m_sampling=True, **BASE)),
}


def run_one(seed, pc, tag):
    key = jax.random.key(seed)
    w = world.reset(key, cfg)
    p = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS,
                         n_hens=cfg.n_hens, auditory_scaffold=True)
    x = brain.initial_state(p, cfg.n_hens)
    _w, _x, p_end, *_ = simulate.simulate(
        w, x, p, jax.random.fold_in(key, 2), cfg, a.minutes * 60.0, 60.0, pc)
    r = assay(p_end, cfg, cfg.n_hens)
    return [r.alarm_alone, r.alarm_audience, r.food_alone, r.food_audience]


cache = json.load(open(a.cache)) if os.path.exists(a.cache) else {}
t0, n = time.perf_counter(), a.seeds
crit = _t_critical(n - 1)
print(f"E075 -- bisecting H2f's food control, {n} seeds, {a.minutes:.0f} min")
print(f"threshold t({n-1})={crit:.3f}   (E057 reported the food control NULL)\n")

for arm, (pc_s, pc_l) in ARMS.items():
    rows = {}
    for name, pc in (("S", pc_s), ("L", pc_l)):
        vals = []
        for s in range(n):
            ck = f"{arm}|{name}|{s}|{a.minutes}"
            if ck not in cache:
                cache[ck] = run_one(s, pc, arm)
                json.dump(cache, open(a.cache, "w"))
            vals.append(cache[ck])
        rows[name] = jnp.array(vals)
    S, L = rows["S"], rows["L"]

    def one(d, label):
        m, se = float(d.mean()), float(d.std(ddof=1) / n ** 0.5)
        t = abs(m) / (se + 1e-12)
        print(f"  {label:<40}{m:+.4f} +/- {se:.4f}  t={t:.2f}  "
              f"{'SIGNIFICANT' if t > crit else 'null'}")

    print(f"{arm}")
    one(L[:, 0] - S[:, 0], "general elevation")
    one((L[:, 1] - L[:, 0]) - (S[:, 1] - S[:, 0]), "audience-specific")
    one((L[:, 3] - L[:, 2]) - (S[:, 3] - S[:, 2]), "FOOD CONTROL (should be null)")
    print()

print(f"wall clock: {time.perf_counter() - t0:.0f} s")
