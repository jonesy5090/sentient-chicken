"""Of the moments she could NOT see the hawk, how often was she caught? (E026)

This replaces the staged assay, after six attempts to stage a blind-but-endangered
hen all failed. Posture here is coupled to position and will not hold still: she walks
off any patch within a second, head_down falls to ~0.5, and half a gate is wide open at
half a metre. Rather than manufacture the condition, let the flock supply it -- at dive
onset it does, about 47% of the time -- and condition on it.

The denominator is fixed at the instant the hawk commits, before any response, and is
further restricted to hens who were blind at that instant. That is the only subset in
which an alarm call can carry information the receiver does not already have, so it is
the only subset in which H4 can possibly be true.
"""
import argparse

import jax, jax.numpy as jnp, numpy as np
from coop import spec, world
from hen import brain, connectome, regions
from run import simulate
from run.experiment import _t_critical

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=12)
ap.add_argument("--minutes", type=float, default=10.0)
args = ap.parse_args()

CONDITIONS = (
    ("deaf (no channel)",       dict(channel="none",   scaffold=True)),
    ("intact channel",          dict(channel="intact", scaffold=True)),
    ("yoked control",           dict(channel="yoked",  scaffold=True)),
    ("intact, no scaffold",     dict(channel="intact", scaffold=False)),
)

print(f"blind-at-onset risk, {args.seeds} matched seeds x {args.minutes:.0f} min, "
      f"hawk every 20 s\n")
hdr = (f"{'condition':<24}{'blind risks':>13}{'caught|blind':>14}"
       f"{'all risks':>11}{'caught|risk':>13}{'fed %':>8}")
print(hdr); print("-" * len(hdr))

out = {}
for name, kw in CONDITIONS:
    per_seed = []
    for s in range(args.seeds):
        cfg = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=20.0,
                                         channel_mode=kw["channel"])
        w = world.reset(jax.random.key(s), cfg)
        p = connectome.build(jax.random.fold_in(jax.random.key(s), 1),
                             regions.DEFAULT_REGIONS.with_pallium(1.5),
                             n_hens=16, auditory_scaffold=kw["scaffold"])
        x = brain.initial_state(p, 16)
        w_end, *_ = simulate.simulate(w, x, p, jax.random.fold_in(jax.random.key(s), 2),
                                      cfg, args.minutes * 60.0, 60.0,
                                      simulate.NO_PLASTICITY)
        br = float(jnp.sum(w_end.n_blind_risk)); bc = float(jnp.sum(w_end.n_blind_caught))
        ar = float(jnp.sum(w_end.n_at_risk));    ac = float(jnp.sum(w_end.n_caught))
        fed = float(jnp.sum(w_end.n_fed) / (16 * args.minutes * 60.0 / cfg.dt)) * 100
        per_seed.append((br, bc, ar, ac, fed))
    out[name] = per_seed
    a = np.array(per_seed)
    print(f"{name:<24}{a[:,0].sum():>13.0f}"
          f"{a[:,1].sum()/max(a[:,0].sum(),1):>14.3f}"
          f"{a[:,2].sum():>11.0f}{a[:,3].sum()/max(a[:,2].sum(),1):>13.3f}"
          f"{a[:,4].mean():>8.2f}")

print("\n--- vs the deaf hen, paired across seeds, on blind risks only ---")
base = np.array([(bc / br if br else np.nan) for br, bc, *_ in out["deaf (no channel)"]])
for name, _ in CONDITIONS[1:]:
    other = np.array([(bc / br if br else np.nan) for br, bc, *_ in out[name]])
    ok = ~(np.isnan(base) | np.isnan(other))
    d = other[ok] - base[ok]
    n = len(d)
    if n < 2:
        print(f"  {name:<24} too few usable seeds ({n})")
        continue
    mean, se = float(d.mean()), float(d.std(ddof=1)) / (n ** 0.5)
    t = abs(mean) / (se + 1e-12)
    crit = _t_critical(n - 1)
    flag = ("SIGNIFICANT" if t > crit
            else f"suggestive (t={t:.2f})" if t > 1.0 else f"noise (t={t:.2f})")
    print(f"  {name:<24} {mean:+.3f} +/- {se:.3f}  (n={n})  {flag}")

print("\nnegative = fewer catches than the deaf hen among the moments she was blind.")
print("intact should beat deaf; if yoked matches intact, the content is doing no work.")
