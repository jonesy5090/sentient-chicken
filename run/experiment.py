"""Matched-seed comparison between conditions.

Every experiment in this project is a contrast between flocks that differ in exactly
one thing. This module is the harness for that, and `docs/backlog.md` specifies the
six-way condition ladder it will eventually need to support.

Two disciplines are enforced here rather than left to the caller:

**Seeds are matched across conditions.** Same world layout, same genome, same predator
arrival times. Without that, a difference between conditions is indistinguishable
from a difference between coops.

**The primary readout is within-run change, not absolute level.** A flock that starts
lucky looks good all run. What learning predicts is *improvement*, so we compare the
last third of a run against the first third, and then compare that change against the
control's change over the same period.

    usage:  python -m run.experiment --minutes 30 --seeds 4
"""

import argparse
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp

from coop import spec, world
from coop.spec import CoopConfig
from hen import brain, connectome, plasticity, regions
from hen.plasticity import PlasticConfig
from run import simulate


class Condition(NamedTuple):
    name: str
    pc: PlasticConfig


# The phase 1 contrast: does a hen that learns regulate herself better than one that
# cannot? Everything else -- genome, coop, predators -- is held identical.
PHASE1 = (
    Condition("fixed (innate only)", PlasticConfig(enabled=False)),
    Condition("learning, no growth",
              PlasticConfig(enabled=True, growth_enabled=False)),
    Condition("learning + growth", PlasticConfig(enabled=True)),
)


class Result(NamedTuple):
    hunger_early: float
    hunger_late: float
    fed_rate: float
    struck: float
    synapses: float


def run_condition(cond: Condition, seed: int, cfg: CoopConfig,
                  seconds: float, chunk_s: float) -> Result:
    key = jax.random.key(seed)
    w = world.reset(key, cfg)
    p = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS,
                         n_hens=cfg.n_hens)
    x = brain.initial_state(p, cfg.n_hens)

    w_end, _x, p_end, _ps, _k, s = simulate.simulate(
        w, x, p, jax.random.fold_in(key, 2), cfg, seconds, chunk_s, cond.pc)

    n = len(s.hunger)
    third = max(1, n // 3)
    return Result(
        hunger_early=float(jnp.mean(s.hunger[:third])),
        hunger_late=float(jnp.mean(s.hunger[-third:])),
        fed_rate=float(jnp.sum(w_end.n_fed) / (cfg.n_hens * seconds / cfg.dt)),
        struck=float(jnp.sum(w_end.n_struck)),
        synapses=float(jnp.mean(jnp.sum(p_end.W != 0.0, axis=(1, 2)))),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--chunk", type=float, default=60.0)
    ap.add_argument("--hens", type=int, default=spec.DEFAULT_COOP.n_hens)
    args = ap.parse_args()

    cfg = spec.DEFAULT_COOP._replace(n_hens=args.hens)
    seconds = args.minutes * 60.0
    seeds = list(range(args.seeds))

    print(f"phase 1 contrast: {args.minutes:.0f} min of chicken time, "
          f"{len(seeds)} matched seeds, {cfg.n_hens} hens\n")
    hdr = (f"{'condition':<22} {'hunger early':>13} {'hunger late':>12} "
           f"{'change':>9} {'fed %':>7} {'struck':>7} {'synapses':>9}")
    print(hdr)
    print("-" * len(hdr))

    t0 = time.perf_counter()
    table = {}
    for cond in PHASE1:
        rs = [run_condition(cond, s, cfg, seconds, args.chunk) for s in seeds]
        early = jnp.array([r.hunger_early for r in rs])
        late = jnp.array([r.hunger_late for r in rs])
        struck = jnp.array([r.struck for r in rs])
        table[cond.name] = {"hunger_change": late - early, "struck": struck}
        print(f"{cond.name:<22} {float(early.mean()):>13.3f} "
              f"{float(late.mean()):>12.3f} "
              f"{float(jnp.mean(late - early)):>+9.3f} "
              f"{float(jnp.mean(jnp.array([r.fed_rate for r in rs]))) * 100:>7.1f} "
              f"{float(struck.mean()):>7.0f} "
              f"{float(jnp.mean(jnp.array([r.synapses for r in rs]))):>9.0f}")

    print()
    _report(table, "hunger change", lambda v: v["hunger_change"], seeds)
    _report(table, "predator exposure", lambda v: v["struck"], seeds)

    print("\nnegative = better (drives regulated, or less time exposed to predators)")
    print("effects within 2 SE are noise -- say so rather than reading the sign")
    print(f"wall clock: {time.perf_counter() - t0:.0f} s")


def _report(table: dict, label: str, pick, seeds) -> None:
    """Paired contrast against the fixed control.

    Paired because the seeds are matched: same coop, same genome, same predator
    arrivals, so the per-seed difference cancels most of the between-coop variance.
    """
    control = pick(table[PHASE1[0].name])
    for cond in PHASE1[1:]:
        delta = pick(table[cond.name]) - control
        mean = float(jnp.mean(delta))
        if len(seeds) > 1:
            se = float(jnp.std(delta, ddof=1)) / (len(seeds) ** 0.5)
            verdict = "better" if mean < 0 else "worse"
            sig = "  <-- exceeds 2 SE" if abs(mean) > 2 * se else "  (within noise)"
            print(f"{label:<18} {cond.name:<22} {mean:+.3f} +/- {se:.3f} SE "
                  f"{verdict}{sig}")
        else:
            print(f"{label:<18} {cond.name:<22} {mean:+.3f} (single seed)")


if __name__ == "__main__":
    main()
