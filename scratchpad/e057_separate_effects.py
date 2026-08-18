"""E057: separate the audience-specific component of E056's effect from the general
elevation component, with a matched (difference-in-differences) design across the full
8-seed sample, rather than a 3-seed eyeballed diagnostic.

Reruns S and S+L-hebbian-scaled at the exact seeds/config E056 used, but calls
run.audience.assay() directly to capture the full AudienceResult per seed instead of
just the scalar .alarm_effect _run_cell returns.
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
ap.add_argument("--seed-offset", type=int, default=0)
ap.add_argument("--minutes", type=float, default=30.0)
ap.add_argument("--cache", default="scratchpad/e057_cache.json")
ap.add_argument("--budget", type=float, default=100000.0)
a = ap.parse_args()
SEED_OFFSET = a.seed_offset

cfg = spec.DEFAULT_COOP._replace(n_hens=16, food_deplete_rate=0.0)
seconds = a.minutes * 60.0

FIXED = PlasticConfig(enabled=False, explore_sigma=0.0)
LEARN_HEBBIAN_SCALED = PlasticConfig(enabled=True, growth_enabled=False, kin_audible=True,
                                     explore_sigma=0.6, hebbian_readout=True,
                                     readout_scaling_strength=0.3)

CONDITIONS = {"S": (FIXED, True), "L": (LEARN_HEBBIAN_SCALED, True)}


def run_one(seed, pc, scaffold):
    key = jax.random.key(seed)
    w = world.reset(key, cfg)
    p = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS,
                         n_hens=cfg.n_hens, auditory_scaffold=scaffold)
    x = brain.initial_state(p, cfg.n_hens)
    _w, _x, p_end, *_ = simulate.simulate(
        w, x, p, jax.random.fold_in(key, 2), cfg, seconds, 60.0, pc)
    r = assay(p_end, cfg, cfg.n_hens)
    return [r.alarm_alone, r.alarm_audience, r.food_alone, r.food_audience]


cache = json.load(open(a.cache)) if os.path.exists(a.cache) else {}
t0 = time.perf_counter()

rows = {name: [] for name in CONDITIONS}
for name, (pc, scaffold) in CONDITIONS.items():
    for s in range(SEED_OFFSET, SEED_OFFSET + a.seeds):
        ck = f"{name}|{s}|{a.minutes}"
        if ck not in cache:
            if time.perf_counter() - t0 > a.budget:
                print("budget reached; stopping")
                json.dump(cache, open(a.cache, "w"))
                raise SystemExit(0)
            cache[ck] = run_one(s, pc, scaffold)
            json.dump(cache, open(a.cache, "w"))
        rows[name].append(cache[ck])

S = jnp.array(rows["S"])    # (seeds, 4): alarm_alone, alarm_audience, food_alone, food_audience
L = jnp.array(rows["L"])

print(f"E057 -- separating audience-specific from general-elevation, {a.seeds} seeds, "
      f"{a.minutes:.0f} min rearing\n")
print(f"{'':>4}{'alarm_alone':>12}{'alarm_aud':>10}{'food_alone':>11}{'food_aud':>9}")
for name, arr in (("S", S), ("L", L)):
    print(f"{name:>4}{arr[:,0].mean():>12.4f}{arr[:,1].mean():>10.4f}"
          f"{arr[:,2].mean():>11.4f}{arr[:,3].mean():>9.4f}")

n = a.seeds
crit = _t_critical(n - 1)


def one_sample(d, label):
    mean, se = float(d.mean()), float(d.std(ddof=1) / n ** 0.5)
    t = abs(mean) / (se + 1e-12)
    print(f"{label:<55}{mean:+.4f} +/- {se:.4f}  t={t:.2f}  "
          f"{'SIGNIFICANT' if t > crit else 'not significant'}")
    return mean, se, t


print(f"\nthreshold(df={n-1})={crit:.3f}\n")

general_elevation = L[:, 0] - S[:, 0]
m_gen, se_gen, t_gen = one_sample(general_elevation, "General elevation: alarm_alone(L) - alarm_alone(S)")

targeted = (L[:, 1] - L[:, 0]) - (S[:, 1] - S[:, 0])
m_tgt, se_tgt, t_tgt = one_sample(targeted, "Audience-specific (diff-in-diff)")

print(f"\n|audience-specific| {'>' if abs(m_tgt) > abs(m_gen) else '<='} |general elevation|: "
      f"{abs(m_tgt):.4f} vs {abs(m_gen):.4f}")

print("\n--- SECONDARY: food channel (task gives it no mechanistic route to an")
print("    audience effect -- if it shows one too, that's evidence for elevation) ---")
food_general = L[:, 2] - S[:, 2]
one_sample(food_general, "food_alone(L) - food_alone(S)")
food_targeted = (L[:, 3] - L[:, 2]) - (S[:, 3] - S[:, 2])
one_sample(food_targeted, "food audience-specific (diff-in-diff)")

print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
