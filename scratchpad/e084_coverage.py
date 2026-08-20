"""Does a free-running flock actually explore the arena?

E084 Part A hit a degenerate split: with one world key, 0 of 19200 samples fell within
3.33 m of the planted feeder over 20 simulated minutes with 16 hens, while another key
gave 0.424. If occupancy of a given cell is set by the world key rather than by anything
the hen does, then every occupancy-based contrast in the T2 arc has been measuring
initial conditions.

Reports per world key: how many of the 25 grid cells the flock ever enters, total area
covered, and occupancy of P and P' specifically.
"""
import sys, time
sys.path.insert(0, 'scratchpad')
import jax, jax.numpy as jnp, numpy as np
from functools import partial
import e083_leaving_anchor as E
from coop import world
from hen import brain, connectome, plasticity, regions
from run import simulate

CEN, P, P2, CFG, HENS, STEPS = E.CEN, E.P, E.P2, E.CFG, E.HENS, E.STEPS
CENJ = jnp.asarray(CEN, dtype=jnp.float32)


@partial(jax.jit, static_argnames=("cfg", "pc", "n"))
def run(w, x, p, ps, key, cfg, pc, n):
    def step(carry, _):
        carry, _o = simulate._one_step(carry, None, cfg=cfg, pc=pc)
        pos = carry[0].pos                                   # (H,2)
        d = jnp.linalg.norm(pos[:, None, :] - CENJ[None, :, :], axis=-1)   # (H,25)
        nearest = jnp.argmin(d, axis=-1)
        vis = jnp.sum(jax.nn.one_hot(nearest, 25), axis=0)   # (25,) counts this step
        return carry, (vis, jnp.mean(pos, 0), jnp.std(pos[:, 0]) + jnp.std(pos[:, 1]))
    return jax.lax.scan(step, (w, x, p, ps, key), None, length=n)[1]


pc = plasticity.PlasticConfig(**E.FROZEN, pred_gain=0.0)
t0 = time.perf_counter()
print(f"free-running flock, {HENS} hens, {E.MINUTES:.0f} min, arena {CFG.size} m, "
      f"25 cells, no plant\n")
print(f"{'seed':>5}{'world key':>11}{'cells visited':>15}{'occ P':>8}{'occ P2':>8}"
      f"{'flock spread':>14}")
for s in range(4):
    k = jax.random.key(s)
    p = connectome.build(jax.random.fold_in(k, 1), regions.DEFAULT_REGIONS, n_hens=HENS,
                         gakel_scaffold=True, shared_place_map=True)
    for tag, wk, rk in (("fold 0", k, jax.random.fold_in(k, 2)),
                        ("fold 6", jax.random.fold_in(k, 6), jax.random.fold_in(k, 7))):
        w = world.reset(wk, CFG)
        w = w._replace(food_pos=jnp.asarray(np.stack([CEN[P], CEN[P2]]), dtype=jnp.float32))
        x = brain.initial_state(p, HENS); ps = plasticity.initial_state(p, HENS, pc)
        vis, ctr, spread = run(w, x, p, ps, rk, CFG, pc, STEPS)
        tot = np.asarray(jnp.sum(vis, 0))
        occ = tot / tot.sum()
        print(f"{s:>5}{tag:>11}{int((tot > 0).sum()):>15}{occ[P]:>8.3f}{occ[P2]:>8.3f}"
              f"{float(jnp.mean(spread)):>14.2f}")
print(f"\nwall clock: {time.perf_counter()-t0:.0f} s")
print("25 cells visited = full coverage. A handful = the flock sits where it started,")
print("and occupancy of any named cell is a property of the world key, not the hen.")
