"""E122b -- the asymmetry is in the scoreboard, not the coop.

Part A swept patch size and intake rate and could not make unconditional vigilance stop
paying: the drives cost of crouching stayed at ~0 in every cell, and hunger barely moved.
The reason is structural and has nothing to do with patch size.

`evolve.fitness` scores **terminal drives** against a **cumulative catch count**. Hunger is
a regulated homeostat -- intake is `peck_food_rate * hunger`, so a hen who loses feeding
time gets hungrier and then eats faster per second, and the equilibrium barely moves. Being
caught has no such regulator; catches simply accumulate. **So the foraging side of fitness
is compressed by self-correction and the predation side is not**, and any trade of foraging
for safety looks good regardless of the coop's parameters.

This is E019's oldest lesson in a new place -- "hens start at hunger 0.30, which IS the
equilibrium; the metric was a coin flip" -- and it is why E111 found only 0.21 hunger units
between a reflex hen and a camped oracle.

`coop/world.py` already tracks `n_fed`, a cumulative successful-peck counter, and
`evolve.fitness` does not use it. This measures whether the cumulative quantity shows the
cost the equilibrium hides.

Run:  PYTHONPATH=. python scratchpad/e122b_cumulative_intake.py
"""

import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from coop import spec, world
from hen import brain, plasticity
from run import evolve, simulate
from scratchpad.e120_ladder_checks import founders

N_SEEDS = 6
LIFETIME_S = 600.0
BIASES = (0.0, 0.5, 1.0, 1.5)
CAUGHT_W = 0.1049
CACHE = "scratchpad/e122b_cache.json"


def run_one(bias, seed):
    c = spec.DEFAULT_COOP._replace(hawk_period_s=20.0, channel_mode="intact",
                                   call_log_steps=spec.YOKE_LOG_STEPS)
    pc = evolve.NO_LEARNING
    key = jax.random.key(8800 + seed)
    p = founders(jax.random.fold_in(key, 0), c)
    if bias:
        bm = np.asarray(p.b_motor).copy()
        bm[spec.M_CROUCH] += bias
        p = p._replace(b_motor=jnp.asarray(bm))
    w0 = world.reset(jax.random.fold_in(key, 1), c)
    x0 = brain.initial_state(p, c.n_hens)
    ps = plasticity.initial_state(p, c.n_hens, pc)
    w_end, *_ = simulate.rollout_quiet(
        w0, x0, p, jax.random.fold_in(key, 2), c, int(LIFETIME_S / c.dt), ps, pc)
    _, _, _, _, _, tr = simulate.rollout(
        world.reset(jax.random.fold_in(key, 1), c),
        brain.initial_state(p, c.n_hens), p, jax.random.fold_in(key, 2), c,
        int(60.0 / c.dt), plasticity.initial_state(p, c.n_hens, pc), pc)
    return dict(crouch=float(jnp.mean(tr.motor[:, :, spec.M_CROUCH])),
                n_fed=float(jnp.mean(w_end.n_fed)),
                caught=float(jnp.mean(w_end.n_caught_any)),
                hunger=float(jnp.mean(w_end.hunger)),
                drives=float(jnp.mean(w_end.hunger + w_end.cold + w_end.thirst)))


def main():
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    for bias in BIASES:
        k = str(bias)
        if k in cache:
            continue
        cache[k] = [run_one(bias, s) for s in range(N_SEEDS)]
        json.dump(cache, open(CACHE, "w"), indent=1)
        print(f"  bias {bias}: crouch {np.mean([r['crouch'] for r in cache[k]]):.3f} "
              f"n_fed {np.mean([r['n_fed'] for r in cache[k]]):.0f} "
              f"caught {np.mean([r['caught'] for r in cache[k]]):.2f}", flush=True)

    b0 = cache["0.0"]
    f0 = np.mean([r["n_fed"] for r in b0])
    c0 = np.mean([r["caught"] for r in b0])
    d0 = np.mean([r["drives"] for r in b0])
    cr0 = np.mean([r["crouch"] for r in b0])

    print("\n" + "=" * 94)
    print("Does the CUMULATIVE intake counter show the cost terminal hunger hides?")
    print("=" * 94)
    print(f"{'bias':>5} {'crouch':>7} {'n_fed':>9} {'vs base':>9} {'caught':>7} "
          f"{'hunger':>7} {'drives cost':>12} {'intake cost':>12}")
    for bias in BIASES:
        v = cache[str(bias)]
        cr = np.mean([r["crouch"] for r in v])
        fe = np.mean([r["n_fed"] for r in v])
        cg = np.mean([r["caught"] for r in v])
        hu = np.mean([r["hunger"] for r in v])
        dr = np.mean([r["drives"] for r in v])
        print(f"{bias:>5.1f} {cr:>7.3f} {fe:>9.0f} {(fe/f0-1)*100:>+8.1f}% {cg:>7.2f} "
              f"{hu:>7.4f} {-(dr-d0):>+12.4f} {(fe/f0-1):>+12.3f}")

    print("\n" + "=" * 94)
    print("Re-scoring the SAME runs two ways. Predation term identical in both;")
    print("only the foraging term changes from a regulated equilibrium to a running total.")
    print("=" * 94)
    print(f"{'bias':>5} {'net (terminal drives)':>24} {'net (cumulative intake)':>26}")
    for bias in BIASES[1:]:
        v = cache[str(bias)]
        fe = np.mean([r["n_fed"] for r in v])
        cg = np.mean([r["caught"] for r in v])
        dr = np.mean([r["drives"] for r in v])
        pred = CAUGHT_W * (c0 - cg)
        net_drives = pred - (dr - d0)
        # Scale the intake term so its baseline spread matches the predation term's, the
        # same commensurability rule evolve.fitness's caught_weight was chosen by.
        sd_fed = np.std([r["n_fed"] for r in b0]) / f0
        sd_caught = np.std([r["caught"] for r in b0])
        w_intake = (CAUGHT_W * sd_caught) / (sd_fed + 1e-9)
        net_intake = pred + w_intake * (fe / f0 - 1.0)
        print(f"{bias:>5.1f} {net_drives:>+24.4f} {net_intake:>+26.4f}")
    print(f"\n  (intake weight {w_intake:.4f}, set so the two terms' baseline spreads "
          f"match)")
    print("  Positive = unconditional vigilance still pays. E121's reference: +0.233.")


if __name__ == "__main__":
    main()
