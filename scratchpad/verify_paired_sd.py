"""The quantity that actually decides seed budgets: paired-difference sd.

Both my E022 check and the review measured the *marginal* between-seed spread. That is
not what a matched-seed contrast is powered by -- pairing cancels whatever the two
conditions share, so what matters is sd of the per-seed difference. A nuisance that
affects both conditions equally is already removed by the pairing and blocking it buys
nothing.

Measures sd_d for noise-vs-fixed, with the food layout varying and pinned.
"""
import jax, jax.numpy as jnp, numpy as np
from coop import spec, world
from hen import brain, connectome, regions
from hen.plasticity import PlasticConfig
from run import simulate

CFG = spec.DEFAULT_COOP
SECONDS, SEEDS = 600.0, list(range(8))
_ref = world.reset(jax.random.key(99), CFG)
FIXED_FOOD, FIXED_WATER = _ref.food_pos, _ref.water_pos

FIXED = PlasticConfig(enabled=False, explore_sigma=0.0)
NOISE = PlasticConfig(enabled=False, explore_sigma=0.6)


def run(seed: int, pc, pin: bool):
    key = jax.random.key(seed)
    w = world.reset(key, CFG)
    if pin:
        w = w._replace(food_pos=FIXED_FOOD, water_pos=FIXED_WATER)
    p = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS,
                         n_hens=CFG.n_hens)
    x = brain.initial_state(p, CFG.n_hens)
    w_end, _x, _p, _ps, _k, s = simulate.simulate(
        w, x, p, jax.random.fold_in(key, 2), CFG, SECONDS, 60.0, pc)
    third = max(1, len(s.hunger) // 3)
    change = float(jnp.mean(s.hunger[-third:]) - jnp.mean(s.hunger[:third]))
    fed = float(jnp.sum(w_end.n_fed) / (CFG.n_hens * SECONDS / CFG.dt)) * 100
    return change, fed


for label, pin in (("layout varies", False), ("layout pinned", True)):
    f = np.array([run(s, FIXED, pin) for s in SEEDS])
    n = np.array([run(s, NOISE, pin) for s in SEEDS])
    for j, metric in ((0, "hunger change"), (1, "fed %")):
        d = n[:, j] - f[:, j]
        marginal = np.std(f[:, j], ddof=1)
        rho = np.corrcoef(f[:, j], n[:, j])[0, 1]
        print(f"{label:<15} {metric:<14} marginal sd {marginal:.4f} | "
              f"paired sd_d {np.std(d, ddof=1):.4f} | pairing rho {rho:+.3f} | "
              f"mean diff {d.mean():+.4f}")
    print()
