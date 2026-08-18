"""E058: does H2f's validated non-reward-gated readout rule build comprehension
(crouch on hearing an alarm call) from nothing -- no scaffold, pred_gain=0.0, so only
the readout pathway (W_out) can contribute?

Reimplements run.audience.comprehension() locally to return the full motor breakdown
(crouch, peck, scratch, flee), not just the crouch scalar -- the mandatory diagnostic
E055 established: a positive crouch result only counts as targeted if the other,
unrelated channels do not show a comparable rise.
"""
import argparse, json, os, time

import jax, jax.numpy as jnp
from coop import spec, world
from hen import brain, connectome, regions
from hen.plasticity import PlasticConfig
from run import simulate
from run.experiment import _t_critical

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--seed-offset", type=int, default=0)
ap.add_argument("--minutes", type=float, default=20.0)
ap.add_argument("--hawk-period", type=float, default=20.0)
ap.add_argument("--cache", default="scratchpad/e058_cache.json")
ap.add_argument("--budget", type=float, default=100000.0)
a = ap.parse_args()

cfg = spec.DEFAULT_COOP._replace(n_hens=16, food_deplete_rate=0.0,
                                 hawk_period_s=a.hawk_period)
seconds = a.minutes * 60.0
CHANNELS = {"crouch": spec.M_CROUCH, "peck": spec.M_PECK,
           "scratch": spec.M_SCRATCH, "flee": spec.M_FLEE}

FIXED = PlasticConfig(enabled=False, explore_sigma=0.0)
LEARN_HEBBIAN = PlasticConfig(enabled=True, growth_enabled=False, kin_audible=True,
                              explore_sigma=0.6, hebbian_readout=True,
                              readout_scaling_strength=0.3)


def full_comprehension(p, n_hens, steps=200):
    """Same design as run.audience.comprehension: hold an observation with/without an
    aerial-alarm call, pred_gain=0.0 so only the readout can respond. Returns a dict
    of channel -> (rate under call - rate under silence).
    """
    def rates_under(call_level):
        obs = jnp.zeros((n_hens, spec.OBS_DIM))
        obs = obs.at[:, spec.AUDIO_LO + 2].set(call_level)   # aerial alarm channel
        x = brain.initial_state(p, n_hens)
        totals = {name: 0.0 for name in CHANNELS}
        for _ in range(steps):
            x, motor, _ = brain.step(x, obs, p, cfg.dt, pred_gain=0.0)
            for name, ch in CHANNELS.items():
                totals[name] += float(jnp.mean(motor[:, ch]))
        return {name: v / steps for name, v in totals.items()}

    with_call = rates_under(1.0)
    without = rates_under(0.0)
    return {name: with_call[name] - without[name] for name in CHANNELS}


def run_one(seed, pc):
    key = jax.random.key(seed)
    w = world.reset(key, cfg)
    p = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS,
                         n_hens=cfg.n_hens, auditory_scaffold=False)
    x = brain.initial_state(p, cfg.n_hens)
    _w, _x, p_end, *_ = simulate.simulate(
        w, x, p, jax.random.fold_in(key, 2), cfg, seconds, 60.0, pc)
    d = full_comprehension(p_end, cfg.n_hens)
    return [d["crouch"], d["peck"], d["scratch"], d["flee"]]


cache = json.load(open(a.cache)) if os.path.exists(a.cache) else {}
t0 = time.perf_counter()

rows = {"FIXED": [], "LEARN_HEBBIAN": []}
for name, pc in (("FIXED", FIXED), ("LEARN_HEBBIAN", LEARN_HEBBIAN)):
    for s in range(a.seed_offset, a.seed_offset + a.seeds):
        ck = f"{name}|{s}|{a.minutes}|{a.hawk_period}"
        if ck not in cache:
            if time.perf_counter() - t0 > a.budget:
                print("budget reached; stopping")
                json.dump(cache, open(a.cache, "w"))
                raise SystemExit(0)
            cache[ck] = run_one(s, pc)
            json.dump(cache, open(a.cache, "w"))
        rows[name].append(cache[ck])

F = jnp.array(rows["FIXED"])
L = jnp.array(rows["LEARN_HEBBIAN"])
n = a.seeds
crit = _t_critical(n - 1)

print(f"E058 -- H2c comprehension under H2f's validated readout rule, {a.seeds} seeds "
      f"(offset {a.seed_offset}), {a.minutes:.0f} min rearing, hawk every "
      f"{a.hawk_period:.0f}s, no scaffold, pred_gain=0.0\n")
print(f"{'':>4}{'crouch':>10}{'peck':>10}{'scratch':>10}{'flee':>10}")
for name, arr in (("FIXED", F), ("LEARN", L)):
    print(f"{name:>4}{arr[:,0].mean():>+10.4f}{arr[:,1].mean():>+10.4f}"
          f"{arr[:,2].mean():>+10.4f}{arr[:,3].mean():>+10.4f}")

print(f"\nthreshold(df={n-1})={crit:.3f}\n")
for i, name in enumerate(("crouch", "peck", "scratch", "flee")):
    d = L[:, i] - F[:, i]
    mean, se = float(d.mean()), float(d.std(ddof=1) / n ** 0.5)
    t = abs(mean) / (se + 1e-12)
    tag = "SIGNIFICANT" if t > crit else "not significant"
    star = "  <-- PRIMARY" if name == "crouch" else ""
    print(f"{name:<10} LEARN-FIXED: {mean:+.4f} +/- {se:.4f}  t={t:.2f}  {tag}{star}")

print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
