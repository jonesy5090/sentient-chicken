"""Measure this machine, then print the brain sizes it can actually afford.

Every neuron count in the project plan was an estimate. This script replaces them
with measurements. Run it before building anything on top of a assumed size.

The quantity that matters is the *real-time factor*: how many seconds of chicken life
pass per second of wall clock. Developmental wall-clock time, not memory, is the
binding constraint on this project -- a hen learns her surroundings over days and her
rank over weeks, and a run that is not many times faster than reality is useless.

Note what dominates. With per-hen weight matrices the recurrent update is a batched
matrix-vector product: it reads H x N x N weights every step and does only one
multiply-add per weight read. That is memory-bandwidth bound, not FLOP bound, which
is why the affordable neuron count lands in the hundreds rather than the thousands.

    usage:  python -m bench.envelope [--sweep] [--target 50]
"""

import argparse
import time

import jax
import jax.numpy as jnp

from coop import spec, world
from coop.spec import CoopConfig
from hen import brain, connectome, neurons, regions
from run import simulate


def measure(reg: regions.Regions, cfg: CoopConfig, n_steps: int = 400,
            repeats: int = 3) -> dict:
    """Wall-clock cost of the full closed loop at one brain size."""
    key = jax.random.key(0)
    w = world.reset(key, cfg)
    p = connectome.build(jax.random.fold_in(key, 1), reg, n_hens=cfg.n_hens)
    x = brain.initial_state(p, cfg.n_hens)

    # Compile and warm up outside the timed region.
    w2, x2, k2 = simulate.rollout_quiet(w, x, p, key, cfg, n_steps)
    jax.block_until_ready((w2.pos, x2))

    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = simulate.rollout_quiet(w, x, p, key, cfg, n_steps)
        jax.block_until_ready((out[0].pos, out[1]))
        best = min(best, time.perf_counter() - t0)

    steps_per_s = n_steps / best
    weight_bytes = cfg.n_hens * reg.total * reg.total * 4
    return {
        "neurons": reg.total,
        "hens": cfg.n_hens,
        "steps_per_s": steps_per_s,
        "realtime_factor": steps_per_s * cfg.dt,
        "weight_mb": weight_bytes / 1e6,
        "read_gb_s": steps_per_s * weight_bytes / 1e9,
        "chicken_day_hours": (86400.0 / (steps_per_s * cfg.dt)) / 3600.0,
    }


def _row(m: dict) -> str:
    return (f"{m['neurons']:>8} {m['hens']:>6} {m['steps_per_s']:>12,.0f} "
            f"{m['realtime_factor']:>10.1f}x {m['weight_mb']:>10.1f} "
            f"{m['read_gb_s']:>11.1f} {m['chicken_day_hours']:>10.2f}")


HEADER = (f"{'neurons':>8} {'hens':>6} {'steps/s':>12} {'real-time':>11} "
          f"{'W (MB)':>10} {'GB/s read':>11} {'hrs/day':>10}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true",
                    help="sweep brain size to find the envelope")
    ap.add_argument("--target", type=float, default=50.0,
                    help="desired real-time factor")
    ap.add_argument("--hens", type=int, default=spec.DEFAULT_COOP.n_hens)
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()

    cfg = spec.DEFAULT_COOP._replace(n_hens=args.hens)
    reg = regions.DEFAULT_REGIONS

    print(f"device        : {jax.devices()[0]}")
    print(f"dt            : {cfg.dt * 1000:.0f} ms")
    print(f"stability     : max(dt/tau) = "
          f"{neurons.stability_ratio(jnp.asarray(regions.REGION_TAU), cfg.dt):.2f} "
          f"(forward Euler wants << 1)")

    p = connectome.build(jax.random.key(1), reg, n_hens=cfg.n_hens)
    st = connectome.stats(p, reg)
    print(f"connectome    : {st['neurons']} neurons, {st['synapses']:,} synapses, "
          f"fan-out {st['mean_fan_out']:.0f}, density {st['density']:.1%}")
    print(f"regions       : {st['regions']}")
    print()
    print(HEADER)
    print("-" * len(HEADER))

    if not args.sweep:
        print(_row(measure(reg, cfg, args.steps)))
        return

    results = []
    for factor in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        scaled = reg.scaled(factor)
        m = measure(scaled, cfg, args.steps)
        results.append(m)
        print(_row(m))

    print()
    ok = [m for m in results if m["realtime_factor"] >= args.target]
    if ok:
        best = max(ok, key=lambda m: m["neurons"])
        print(f"Largest brain sustaining {args.target:.0f}x real time with "
              f"{cfg.n_hens} hens: {best['neurons']} neurons per hen.")
        print(f"  One chicken-day  : {best['chicken_day_hours']:.2f} hours wall clock")
        print(f"  30-day rearing   : {best['chicken_day_hours'] * 30 / 24:.1f} days")
    else:
        print(f"No tested size reaches {args.target:.0f}x real time. "
              f"Reduce the flock, the brain, or the target.")


if __name__ == "__main__":
    main()
