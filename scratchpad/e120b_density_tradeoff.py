"""E120b -- the density trade-off the ladder has to sit inside.

E119 found predation only becomes a property of the hen at high predator density
(r(caught) +0.049 at hawk_period_s=50, +0.478 at 10). But `hawk_dive_s` is 12 s, so at a
mean onset interval of 10 s dives *overlap*: a hawk is essentially always in the sky, alarm
calling is close to continuous, and hearing an alarm stops meaning "a hawk is coming" --
it means "it is Tuesday". The first pre-check measured intact r(heard, hawk-on-me) at only
**+0.173** against E026's +0.56 at its own configuration.

So the two requirements pull in opposite directions:

  * **predation must be heritable**  -> more hawks
  * **the alarm channel must be informative** -> fewer hawks

This measures both across densities and asks whether any density satisfies both. If none
does, the H0 ladder is blocked by the *world's* parameters rather than by the channel, and
that is a finding about `coop/spec.py`, not about communication.

Run:  PYTHONPATH=. python scratchpad/e120b_density_tradeoff.py
"""

import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from coop import spec, world
from hen import brain, plasticity
from run import evolve, simulate
from scratchpad.e120_ladder_checks import AERIAL_CH, founders, hawk_on_trace

N_SEEDS = 8
TRACE_S = 180.0
PERIODS = (10.0, 20.0, 50.0)
CACHE = "scratchpad/e120b_cache.json"

# E119's measured predation repeatability, for the trade-off table.
R_CAUGHT = {10.0: 0.478, 20.0: 0.179, 50.0: 0.049}


def scaffold_effect(period, n_seeds=N_SEEDS, lifetime_s=600.0):
    """The mandatory positive control, at one predator density.

    Turns on `auditory_scaffold`, which hand-wires hearing-an-alarm to crouching, and asks
    whether it reduces catches. This is the check that failed at 10 s (+0.227 +- 0.503,
    t=0.45) and sent the ladder back here: if a *planted* comprehension cannot save a hen,
    a selected one cannot either, and a null ladder would be about the coop.
    """
    pc = evolve.NO_LEARNING
    out = {}
    for name, kw in (("off", {}), ("on", dict(auditory_scaffold=True))):
        caught, hunger, crouch = [], [], []
        cfg = spec.DEFAULT_COOP._replace(hawk_period_s=period, channel_mode="intact")
        for g in range(n_seeds):
            key = jax.random.key(3000 + g)
            p = founders(jax.random.fold_in(key, 0), cfg, **kw)
            w0 = world.reset(jax.random.fold_in(key, 1), cfg)
            x0 = brain.initial_state(p, cfg.n_hens)
            ps = plasticity.initial_state(p, cfg.n_hens, pc)
            w_end, *_ = simulate.rollout_quiet(
                w0, x0, p, jax.random.fold_in(key, 2), cfg,
                int(lifetime_s / cfg.dt), ps, pc)
            caught.append(float(jnp.mean(w_end.n_caught_any)))
            hunger.append(float(jnp.mean(w_end.hunger)))
        out[name] = dict(caught=caught, hunger=hunger)
    return out


def main():
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    for period in PERIODS:
        k = f"{period:.0f}"
        if k in cache:
            print(f"[{k}s] cached")
            continue
        cfg = spec.DEFAULT_COOP._replace(hawk_period_s=period, channel_mode="intact")
        duty, corrs, rest, onme = [], [], [], []
        print(f"[{k}s]", flush=True)
        for g in range(N_SEEDS):
            key = jax.random.key(2000 + g)
            p = founders(jax.random.fold_in(key, 0), cfg)
            on, dist, obs = hawk_on_trace(cfg, p, key, TRACE_S)
            aer = obs[:, :, AERIAL_CH]
            on_me = (on[:, None] > 0.5) & (dist < 4.0)
            duty.append(float((on > 0.5).mean()))
            if on_me.sum() < 5 or (~on_me).sum() < 5:
                continue
            rest.append(float(aer[~on_me].mean()))
            onme.append(float(aer[on_me].mean()))
            a = aer.reshape(-1) - aer.mean()
            b = on_me.reshape(-1).astype(float)
            b = b - b.mean()
            den = np.sqrt((a * a).sum() * (b * b).sum())
            corrs.append(float((a * b).sum() / den) if den > 1e-12 else np.nan)
        cache[k] = dict(period=period, duty=float(np.mean(duty)),
                        corr=float(np.nanmean(corrs)),
                        corr_se=float(np.nanstd(corrs) / np.sqrt(max(len(corrs), 1))),
                        rest=float(np.mean(rest)), onme=float(np.mean(onme)),
                        per_seed=corrs)
        print(f"  duty={cache[k]['duty']:.3f} r={cache[k]['corr']:+.3f} "
              f"rest={cache[k]['rest']:.3f} on-me={cache[k]['onme']:.3f}", flush=True)
        cache[k]["scaffold"] = scaffold_effect(period)
        off = np.array(cache[k]["scaffold"]["off"]["caught"])
        on = np.array(cache[k]["scaffold"]["on"]["caught"])
        print(f"  scaffold: off={off.mean():.3f} on={on.mean():.3f} "
              f"delta={(on-off).mean():+.3f}", flush=True)
        json.dump(cache, open(CACHE, "w"), indent=1)

    print("\n" + "=" * 90)
    print("The trade-off: does any predator density give both a heritable predation")
    print("rate and an informative alarm channel?")
    print("=" * 90)
    print(f"{'hawk_period':>12} {'sky busy':>9} {'r(channel)':>18} {'r(caught)':>10} "
          f"{'scaffold helps?':>26}")
    for period in PERIODS:
        c = cache.get(f"{period:.0f}")
        if not c:
            continue
        rc = R_CAUGHT.get(period)
        rc_s = f"{rc:+.3f}" if rc is not None else "   --"
        sc = "--"
        if "scaffold" in c:
            off = np.array(c["scaffold"]["off"]["caught"])
            on = np.array(c["scaffold"]["on"]["caught"])
            d = on - off
            se = d.std(ddof=1) / np.sqrt(len(d))
            sc = f"{d.mean():+.3f}+-{se:.3f} t={d.mean()/se:+5.2f}"
        print(f"{period:>12.0f} {c['duty']:>8.1%} {c['corr']:+.3f} +- {c['corr_se']:.3f} "
              f"{rc_s:>10} {sc:>26}")
    print("\nThe ladder needs all three at once: r(caught) well above 0 (selection can")
    print("see predation), r(channel) well above 0 (the signal carries news), and a")
    print("NEGATIVE scaffold delta (a planted comprehension actually saves hens).")
    print("\n`hawk in sky` is the fraction of time a dive is active. hawk_dive_s is 12 s,")
    print("so any onset interval below ~12 s means dives overlap and the sky is never")
    print("empty -- at which point an alarm call carries no news.")


if __name__ == "__main__":
    main()
