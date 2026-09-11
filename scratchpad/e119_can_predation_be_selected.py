"""E119 instrument checks -- can an alarm channel ever be selected for in this coop?

The H0 ladder (evolve an intact channel against a yoked one) needs a fitness criterion on
which *hearing about a hawk* pays. E117 found predation repeatability at zero, and
`coop/world.py` turns out to attach no physiological cost to being caught at all -- hunger
moves only with `at_food_any & pecking`. Three things have to be true before the ladder is
worth running, and none has been measured:

A. **Does being caught cost a hen anything she is scored on?**  Within-flock correlation
   between `n_caught_any` and final drives. If it is zero or negative, predation is free,
   and a drives-only criterion is indifferent to hawks -- or worse, rewards ignoring them.

B. **Does anti-predator behaviour cost drives?**  Correlation between a hen's crouch/flee
   rate and her hunger. If crouching costs foraging time, a drives-only criterion selects
   *against* the very behaviour an alarm call is for.

C. **Does predation become a property of the hen at higher predator density?**  r(caught)
   across replicate lifetimes at `hawk_period_s` 50, 20, 10. Catches are rare events; if
   they are Poisson noise around a hen-specific rate, more dives should make the rate
   visible. If it never becomes repeatable, no criterion containing predation can be
   selected on, at any weight.

Founder variation is supplied throughout (`founder_sigma=2.0`), because E117 measured that
without it there are no individual differences for any of these correlations to detect --
a null here would otherwise be a null about the flock, not about predation.

Run:  PYTHONPATH=. python scratchpad/e119_can_predation_be_selected.py
"""

import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from coop import spec, world
from hen import brain, connectome, plasticity, regions
from run import evolve, simulate

N_SEEDS = 8
LIFETIME_S = 600.0
PERIODS = (50.0, 20.0, 10.0)
CACHE = "scratchpad/e119_cache.json"


def pearson(a, b):
    a = np.asarray(a, dtype=float) - np.mean(a)
    b = np.asarray(b, dtype=float) - np.mean(b)
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else np.nan


def fisher(rs):
    rs = [r for r in rs if np.isfinite(r)]
    if len(rs) < 2:
        return np.nan, np.nan, np.nan
    z = np.arctanh(np.clip(rs, -0.999, 0.999))
    m, se = z.mean(), z.std(ddof=1) / np.sqrt(len(z))
    return float(np.tanh(m)), float(se), float(m / se if se > 1e-12 else np.nan)


def run_once(cfg, p, pc, k_world, k_run, steps, trace_s=60.0):
    w0 = world.reset(k_world, cfg)
    x0 = brain.initial_state(p, cfg.n_hens)
    ps = plasticity.initial_state(p, cfg.n_hens, pc)
    w_end, *_ = simulate.rollout_quiet(w0, x0, p, k_run, cfg, steps, ps, pc)
    # Per-hen motor profile over the first `trace_s`, for check B.
    _, _, _, _, _, tr = simulate.rollout(
        world.reset(k_world, cfg), brain.initial_state(p, cfg.n_hens), p, k_run,
        cfg, int(trace_s / cfg.dt),
        plasticity.initial_state(p, cfg.n_hens, pc), pc)
    motor = np.asarray(jnp.mean(tr.motor, axis=0))          # (H, MOTOR_DIM)
    return dict(
        drives=np.asarray(w_end.hunger + w_end.cold + w_end.thirst),
        hunger=np.asarray(w_end.hunger),
        caught=np.asarray(w_end.n_caught_any).astype(float),
        exposed=np.asarray(w_end.n_exposed).astype(float)
        if hasattr(w_end, "n_exposed") else np.zeros(cfg.n_hens),
        crouch=motor[:, spec.M_CROUCH], flee=motor[:, spec.M_FLEE],
        peck=motor[:, spec.M_PECK])


def main():
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    pc = evolve.NO_LEARNING

    for period in PERIODS:
        key_p = f"hawk_{period:.0f}"
        if key_p in cache:
            print(f"[{key_p}] cached")
            continue
        cfg = spec.DEFAULT_COOP._replace(hawk_period_s=period)
        steps = int(LIFETIME_S / cfg.dt)
        r_caught, r_drives, r_cd, r_crouch_h, r_flee_h, r_peck_h = [], [], [], [], [], []
        caught_mean, caught_sd = [], []
        print(f"[{key_p}]", flush=True)
        for g in range(N_SEEDS):
            k = jax.random.key(1000 + g)
            k_gen, k_world, k_run = jax.random.split(k, 3)
            p = connectome.build(k_gen, regions.DEFAULT_REGIONS, n_hens=cfg.n_hens)
            p = evolve._mutate(p, jax.random.fold_in(k_gen, 77), 2.0)
            A = run_once(cfg, p, pc, k_world, jax.random.fold_in(k_run, 0), steps)
            B = run_once(cfg, p, pc, k_world, jax.random.fold_in(k_run, 1), steps)
            # C: is being caught repeatable across replicate lifetimes?
            r_caught.append(pearson(A["caught"], B["caught"]))
            r_drives.append(pearson(-A["drives"], -B["drives"]))
            # A: does being caught cost her anything she is scored on?
            r_cd.append(pearson(A["caught"], A["drives"]))
            # B: does anti-predator behaviour cost drives?
            r_crouch_h.append(pearson(A["crouch"], A["hunger"]))
            r_flee_h.append(pearson(A["flee"], A["hunger"]))
            r_peck_h.append(pearson(A["peck"], A["hunger"]))
            caught_mean.append(float(A["caught"].mean()))
            caught_sd.append(float(A["caught"].std()))
            print(f"    seed {g}: r_caught={r_caught[-1]:+.3f} "
                  f"caught/hen={caught_mean[-1]:.2f}", flush=True)
        cache[key_p] = dict(
            period=period,
            r_caught=r_caught, r_drives=r_drives, r_caught_drives=r_cd,
            r_crouch_hunger=r_crouch_h, r_flee_hunger=r_flee_h,
            r_peck_hunger=r_peck_h,
            caught_mean=float(np.mean(caught_mean)),
            caught_sd=float(np.mean(caught_sd)))
        json.dump(cache, open(CACHE, "w"), indent=1)

    print("\n" + "=" * 96)
    print("C. Is being caught a property of the hen? (repeatability across lifetimes)")
    print("=" * 96)
    print(f"{'hawk_period':>12} {'caught/hen':>11} {'sd':>6} "
          f"{'r(caught)':>22} {'r(-drives)':>22}")
    for period in PERIODS:
        c = cache.get(f"hawk_{period:.0f}")
        if not c:
            continue
        rc, sc, tc = fisher(c["r_caught"])
        rd, sd_, td = fisher(c["r_drives"])
        print(f"{period:>12.0f} {c['caught_mean']:>11.2f} {c['caught_sd']:>6.2f} "
              f"{rc:+.3f}+-{sc:.3f} t={tc:+5.2f}  {rd:+.3f}+-{sd_:.3f} t={td:+5.2f}")

    print("\n" + "=" * 96)
    print("A/B. Does predation, or avoiding it, cost a hen anything she is scored on?")
    print("=" * 96)
    print(f"{'hawk_period':>12} {'r(caught,drives)':>24} {'r(crouch,hunger)':>24} "
          f"{'r(flee,hunger)':>22}")
    for period in PERIODS:
        c = cache.get(f"hawk_{period:.0f}")
        if not c:
            continue
        a = fisher(c["r_caught_drives"])
        b = fisher(c["r_crouch_hunger"])
        d = fisher(c["r_flee_hunger"])
        print(f"{period:>12.0f} {a[0]:+.3f}+-{a[1]:.3f} t={a[2]:+5.2f}  "
              f"{b[0]:+.3f}+-{b[1]:.3f} t={b[2]:+5.2f}  "
              f"{d[0]:+.3f}+-{d[1]:.3f} t={d[2]:+5.2f}")
    print("\nthreshold |t| = 2.365 (df=7). Positive r(caught,drives) = being caught goes")
    print("with being hungrier, i.e. predation costs something. Positive")
    print("r(crouch,hunger) = crouching goes with being hungrier, i.e. anti-predator")
    print("behaviour costs foraging -- which a drives-only criterion would select against.")


if __name__ == "__main__":
    main()
