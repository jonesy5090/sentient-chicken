"""E032 causal efficacy: does a TRAINED readout do anything? See docs/experiments/E032.

Rear, then fork. Both test branches continue from the identical end-of-rearing world and
brain state, so lesioning `W_out` is the only difference between them -- a within-subject
manipulation, not two separate runs that happen to share a seed.

The interaction is the quantity:

    (trained: intact - lesioned)  -  (fixed: intact - lesioned)

The fixed pair prices the lesion itself. Anything above that is what learning bought.
"""
import argparse, json, os, time

import jax, jax.numpy as jnp, numpy as np
from coop import spec, world
from hen import brain, connectome, plasticity, regions
from hen.plasticity import PlasticConfig
from run import simulate
from run.experiment import _t_critical

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=12)
ap.add_argument("--seed-offset", type=int, default=0, help="first seed index (for a second block)")
ap.add_argument("--rear", type=float, default=20.0, help="rearing minutes")
ap.add_argument("--test", type=float, default=5.0, help="test minutes")
ap.add_argument("--cache", default="scratchpad/e032_cache.json")
ap.add_argument("--budget", type=float, default=500.0)
ap.add_argument("--food-deplete-rate", type=float, default=spec.DEFAULT_COOP.food_deplete_rate,
                help="E037 found this silently confounds 20-min/16-hen runs; pass 0.0 "
                     "to audit whether E032/E033's interaction survives without it")
a = ap.parse_args()

cfg = spec.DEFAULT_COOP._replace(n_hens=16, food_deplete_rate=a.food_deplete_rate)
LEARN = PlasticConfig(enabled=True, explore_sigma=0.6)
FIXED = PlasticConfig(enabled=False, explore_sigma=0.0)
cache = json.load(open(a.cache)) if os.path.exists(a.cache) else {}
t0 = time.perf_counter()


def one(seed: int, plastic: bool):
    """Rear one flock, then measure fed % with the readout intact and lesioned."""
    key = jax.random.key(seed)
    w = world.reset(key, cfg)
    p0 = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS, n_hens=16)
    x = brain.initial_state(p0, 16)
    pc = LEARN if plastic else FIXED

    w1, x1, p1, _ps, _k, _s = simulate.simulate(
        w, x, p0, jax.random.fold_in(key, 2), cfg, a.rear * 60.0, 60.0, pc)

    # Manipulation check: did rearing actually move the readout?
    drift = float(jnp.mean(jnp.abs(p1.W_out - p0.W_out))
                  / (jnp.mean(jnp.abs(p0.W_out)) + 1e-12))

    steps = 16 * a.test * 60.0 / cfg.dt
    out = {}
    for label, pt in (("intact", p1),
                      ("lesioned", p1._replace(W_out=jnp.zeros_like(p1.W_out)))):
        # Same key for both branches: the lesion is the only difference.
        w2, *_ = simulate.simulate(w1, x1, pt, jax.random.fold_in(key, 3), cfg,
                                   a.test * 60.0, 60.0, FIXED)
        out[label] = float(jnp.sum(w2.n_fed)) / steps * 100
    return [out["intact"], out["lesioned"], drift]


rows = {}
for plastic in (True, False):
    name = "trained" if plastic else "fixed"
    acc = []
    for s in range(a.seed_offset, a.seed_offset + a.seeds):
        ck = f"{name}|{s}|{a.rear}|{a.test}|deplete={a.food_deplete_rate}"
        if ck not in cache:
            if time.perf_counter() - t0 > a.budget:
                print(f"budget reached; {len(cache)}/{2*a.seeds} cells cached")
                raise SystemExit(0)
            cache[ck] = one(s, plastic)
            json.dump(cache, open(a.cache, "w"))
        acc.append(cache[ck])
    rows[name] = np.array(acc)

print(f"E032 causal efficacy -- {a.seeds} matched seeds, {a.rear:.0f} min rearing "
      f"+ {a.test:.0f} min test, 16 hens\n")

drift = rows["trained"][:, 2].mean()
print(f"MANIPULATION CHECK: mean |dW_out| / |W_out| during rearing = {drift:.4f}")
print(f"  gate is > 0.05 -> {'PASSES' if drift > 0.05 else 'FAILS -- result is void'}")
print(f"  (fixed flocks drift {rows['fixed'][:, 2].mean():.6f}, must be 0)\n")

print(f"{'rearing':<12}{'intact':>10}{'lesioned':>10}{'drop':>10}")
print("-" * 42)
drops = {}
for name in ("trained", "fixed"):
    r = rows[name]
    d = r[:, 0] - r[:, 1]
    drops[name] = d
    print(f"{name:<12}{r[:,0].mean():>10.3f}{r[:,1].mean():>10.3f}{d.mean():>10.3f}")

inter = drops["trained"] - drops["fixed"]
n = len(inter)
m, se = inter.mean(), inter.std(ddof=1) / n ** 0.5
t = abs(m) / (se + 1e-12)
crit = _t_critical(n - 1)
print(f"\nPRIMARY -- interaction (trained drop) - (fixed drop):")
print(f"  {m:+.4f} +/- {se:.4f}  t={t:.2f}  threshold {crit:.3f}  -> "
      f"{'SIGNIFICANT' if t > crit else 'not significant'}")
print("  H2e predicts ~0: lesioning a trained readout costs no more than a random one.")
print("  a significant NEGATIVE value falsifies H2e and hands H2 back to the rule.")

sec = rows["trained"][:, 0] - rows["fixed"][:, 0]
m2, se2 = sec.mean(), sec.std(ddof=1) / n ** 0.5
print(f"\nsecondary -- trained vs fixed, both intact (H2's own question):")
print(f"  fed % {m2:+.4f} +/- {se2:.4f}  t={abs(m2)/(se2+1e-12):.2f}")
