"""Rear a flock for a stretch of biological time and report what they did.

In phase 0 nothing is learned, so a "lifetime" here is a stationarity check rather
than a development: drives should cycle, hens should feed and huddle and take losses
to predators, and nothing should drift, saturate or blow up over millions of steps.
When phase 1 adds plasticity, the same harness becomes the rearing run and these
columns become the developmental curves.

    usage:  python -m run.lifetime --minutes 30
            python -m run.lifetime --days 1 --chunk 600
"""

import argparse
import time

import jax
import jax.numpy as jnp

from coop import spec, world
from hen import brain, connectome, plasticity, regions
from run import simulate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--days", type=float, default=None,
                    help="overrides --minutes")
    ap.add_argument("--chunk", type=float, default=60.0,
                    help="seconds of biological time per reported row")
    ap.add_argument("--hens", type=int, default=spec.DEFAULT_COOP.n_hens)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plastic", action="store_true",
                    help="enable phase 1 learning (default: a fixed, innate hen)")
    ap.add_argument("--no-growth", action="store_true",
                    help="learning without structural growth")
    ap.add_argument("--record", action="store_true",
                    help="also write a viz/ trajectory of this run to runs/ "
                         "(same seed, same config -- off by default: it's a second "
                         "rollout, so only pay for it when you want to watch)")
    ap.add_argument("--record-fps", type=float, default=15.0)
    args = ap.parse_args()

    seconds = args.days * 86400.0 if args.days else args.minutes * 60.0
    cfg = spec.DEFAULT_COOP._replace(n_hens=args.hens)
    pc = plasticity.PlasticConfig(enabled=args.plastic,
                                  growth_enabled=not args.no_growth)

    key = jax.random.key(args.seed)
    w = world.reset(key, cfg)
    p = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS,
                         n_hens=cfg.n_hens)
    x = brain.initial_state(p, cfg.n_hens)

    st = connectome.stats(p)
    print(f"flock of {cfg.n_hens}, {st['neurons']} neurons each "
          f"({st['synapses']:,} synapses, fan-out {st['mean_fan_out']:.0f})")
    print(f"rearing for {seconds / 60:.0f} min of chicken time "
          f"({int(seconds / cfg.dt):,} steps)\n")

    t0 = time.perf_counter()
    w_end, x_end, p_end, _ps, _k, s = simulate.simulate(
        w, x, p, jax.random.fold_in(key, 2), cfg, seconds, args.chunk, pc)
    jax.block_until_ready(w_end.pos)
    wall = time.perf_counter() - t0

    hdr = (f"{'t (min)':>8} {'hunger':>7} {'cold':>6} {'head-down':>10} "
           f"{'contact':>8} {'food':>6} {'aerial':>7} {'ground':>7} "
           f"{'fed':>8} {'struck':>7} {'reward':>8} {'synapses':>9}")
    print(hdr)
    print("-" * len(hdr))
    n = len(s.t_s)
    for i in range(0, n, max(1, n // 12)):
        print(f"{float(s.t_s[i]) / 60:>8.1f} {float(s.hunger[i]):>7.2f} "
              f"{float(s.cold[i]):>6.2f} {float(s.head_down[i]):>10.2f} "
              f"{float(s.calls[i, 0]):>8.2f} {float(s.calls[i, 1]):>6.2f} "
              f"{float(s.calls[i, 2]):>7.2f} {float(s.calls[i, 3]):>7.2f} "
              f"{float(s.fed[i]):>8.0f} {float(s.struck[i]):>7.0f} "
              f"{float(s.reward[i]):>8.3f} {float(s.synapses[i]):>9.0f}")

    if args.plastic:
        d = plasticity.divergence_from_genome(p_end)
        print(f"\nconnectome drift: {float(d['grown'].mean()):.0f} synapses grown, "
              f"{float(d['pruned'].mean()):.0f} pruned "
              f"(from {int(jnp.sum(p.mask)):,} innate)")

    print(f"\nwall clock      : {wall:.1f} s")
    print(f"real-time factor: {seconds / wall:.0f}x")
    print(f"one chicken-day : {86400.0 / (seconds / wall) / 3600.0:.2f} h")
    print(f"finite state    : {bool(jnp.all(jnp.isfinite(x_end)))}, "
          f"|x| max {float(jnp.max(jnp.abs(x_end))):.1f}")

    if args.record:
        from run import record
        record.record_and_save(key, cfg, pc, seconds / 60.0, args.record_fps,
                               args.seed, desc=f"run.lifetime --minutes {seconds/60:.0f}")


if __name__ == "__main__":
    main()
