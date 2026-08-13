"""Independent check: how much of the primary metric's variance is food layout?

Runs the fixed (innate-only) condition twice over the same 12 seeds:
  A. everything varies with the seed, as run/experiment.py does today
  B. food and water positions pinned to one layout; genome, hen start positions
     and the predator stream still vary

If B's spread collapses, "seed" has been confounding the coop with the bird.
"""
import jax, jax.numpy as jnp, numpy as np
from coop import spec, world
from hen import brain, connectome, plasticity, regions
from hen.plasticity import PlasticConfig
from run import simulate

CFG = spec.DEFAULT_COOP
PC = PlasticConfig(enabled=False, explore_sigma=0.0)
SECONDS, SEEDS = 600.0, list(range(12))

# One fixed layout, drawn from a seed that is not in the test set.
_ref = world.reset(jax.random.key(99), CFG)
FIXED_FOOD, FIXED_WATER = _ref.food_pos, _ref.water_pos


def run(seed: int, pin_layout: bool):
    key = jax.random.key(seed)
    w = world.reset(key, CFG)
    if pin_layout:
        w = w._replace(food_pos=FIXED_FOOD, water_pos=FIXED_WATER)
    p = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS,
                         n_hens=CFG.n_hens)
    x = brain.initial_state(p, CFG.n_hens)
    w_end, _x, _p, _ps, _k, s = simulate.simulate(
        w, x, p, jax.random.fold_in(key, 2), CFG, SECONDS, 60.0, PC)
    third = max(1, len(s.hunger) // 3)
    change = float(jnp.mean(s.hunger[-third:]) - jnp.mean(s.hunger[:third]))
    fed = float(jnp.sum(w_end.n_fed) / (CFG.n_hens * SECONDS / CFG.dt)) * 100
    return change, fed


for label, pin in (("A: layout varies (current)", False),
                   ("B: layout pinned", True)):
    out = [run(s, pin) for s in SEEDS]
    ch = np.array([o[0] for o in out])
    fed = np.array([o[1] for o in out])
    print(f"{label}")
    print(f"   hunger change  mean {ch.mean():+.4f}  sd {ch.std(ddof=1):.4f}  "
          f"range {ch.min():+.4f} .. {ch.max():+.4f}")
    print(f"   fed %          mean {fed.mean():.2f}   sd {fed.std(ddof=1):.2f}   "
          f"range {fed.min():.2f} .. {fed.max():.2f}")
    print(f"   corr(hunger change, fed %) = {np.corrcoef(ch, fed)[0,1]:+.3f}")
    print()

# The equilibrium arithmetic the reviewer derived, checked against the config.
h_star = lambda f: (1.0 / CFG.hunger_fill_s) / (CFG.peck_food_rate * f)
print(f"hunger equilibrium h* = (1/hunger_fill_s)/(peck_food_rate*f)")
print(f"  at f=6.17%: h* = {h_star(0.0617):.3f}   (world.reset starts hunger at 0.30)")
print(f"  so the metric's sign flips around f = "
      f"{100*(1.0/CFG.hunger_fill_s)/(CFG.peck_food_rate*0.30):.2f}% feeding")
