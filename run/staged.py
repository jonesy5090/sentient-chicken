"""E016: does teaching the two learned pathways one at a time avoid their interference?

Two things learn in this model -- the wiring inside the pallium (`W`) and the
connection out to the motor stub (`W_out`) -- and E015 found that doing both at once
costs far more than doing either alone. The suspected reason is a moving target: the
outgoing connection tries to learn what pallial states mean while those states are
being rewritten underneath it.

Staging is the obvious test. Let the pallium settle first with the readout held still,
then train the readout against a pallium that is no longer changing.

    usage:  python -m run.staged --minutes 20 --seeds 6
"""

import argparse
import time

import jax
import jax.numpy as jnp

from coop import spec, world
from coop.spec import CoopConfig
from hen import brain, connectome, plasticity, regions
from hen.plasticity import PlasticConfig
from run import simulate
from run.experiment import _t_critical

BASE = PlasticConfig(enabled=True, growth_enabled=False, explore_sigma=0.6)


def _measure(sm_early, sm_late, w_end, cfg, seconds):
    """Within-run hunger change, first third of stage one vs last third of stage two."""
    e = jnp.concatenate([sm_early.hunger, sm_late.hunger])
    third = max(1, len(e) // 3)
    return (float(jnp.mean(e[-third:]) - jnp.mean(e[:third])),
            float(jnp.sum(w_end.n_fed) / (cfg.n_hens * seconds / cfg.dt)))


def run_condition(name, seed, cfg: CoopConfig, seconds: float):
    key = jax.random.key(seed)
    w = world.reset(key, cfg)
    p = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS,
                         n_hens=cfg.n_hens)
    x = brain.initial_state(p, cfg.n_hens)
    k = jax.random.fold_in(key, 2)

    if name == "fixed":
        stages = [(PlasticConfig(enabled=False, explore_sigma=0.0), seconds)]
    elif name == "simultaneous":
        stages = [(BASE, seconds)]
    elif name == "staged (pallium first)":
        stages = [(BASE._replace(eta_out=0.0), seconds / 2),
                  (BASE._replace(eta=0.0), seconds / 2)]
    elif name == "staged (muscles first)":
        stages = [(BASE._replace(eta=0.0), seconds / 2),
                  (BASE._replace(eta_out=0.0), seconds / 2)]
    else:
        raise ValueError(name)

    ps, summaries = None, []
    for pc, dur in stages:
        # Plastic state carries across the boundary: the traces and the age clock are
        # continuous, only which weights are writable changes.
        if ps is None:
            ps = plasticity.initial_state(p, cfg.n_hens, pc)
        w, x, p, ps, k, sm = simulate.simulate(w, x, p, k, cfg, dur, 60.0, pc, ps)
        summaries.append(sm)

    if len(summaries) == 1:
        summaries.append(summaries[0]._replace(
            hunger=jnp.zeros((0,), dtype=summaries[0].hunger.dtype)))
    change, fed = _measure(summaries[0], summaries[1], w, cfg, seconds)
    return change, fed, float(jnp.mean(jnp.sum(p.W != 0, axis=(1, 2))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--seeds", type=int, default=6)
    args = ap.parse_args()

    cfg = spec.DEFAULT_COOP
    seconds = args.minutes * 60.0
    seeds = list(range(args.seeds))
    names = ["fixed", "simultaneous",
             "staged (pallium first)", "staged (muscles first)"]

    print(f"E016 -- staged vs simultaneous learning, {args.minutes:.0f} min, "
          f"{len(seeds)} matched seeds\n")
    hdr = f"{'condition':<26} {'hunger change':>14} {'harm':>8} {'fed %':>7} {'synapses':>9}"
    print(hdr); print("-" * len(hdr))

    t0 = time.perf_counter()
    table = {}
    for name in names:
        rs = [run_condition(name, s, cfg, seconds) for s in seeds]
        ch = jnp.array([r[0] for r in rs])
        table[name] = ch
        harm = float(jnp.mean(ch - table["fixed"])) if name != "fixed" else 0.0
        print(f"{name:<26} {float(jnp.mean(ch)):>+14.3f} {harm:>+8.3f} "
              f"{100 * sum(r[1] for r in rs) / len(rs):>7.1f} "
              f"{sum(r[2] for r in rs) / len(rs):>9,.0f}", flush=True)

    print()
    n = len(seeds)
    for name in names[1:]:
        d = table[name] - table["simultaneous"]
        mean = float(jnp.mean(d))
        if n > 1:
            se = float(jnp.std(d, ddof=1)) / (n ** 0.5)
            t = abs(mean) / (se + 1e-12); crit = _t_critical(n - 1)
            flag = ("SIGNIFICANT" if t > crit
                    else f"suggestive (t={t:.2f}, need {crit:.2f})" if t > 1.0
                    else f"noise (t={t:.2f})")
            print(f"vs simultaneous: {name:<26} {mean:+.3f} +/- {se:.3f} SE  {flag}")
    print("\nnegative vs simultaneous = staging helped")
    print(f"wall clock: {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
