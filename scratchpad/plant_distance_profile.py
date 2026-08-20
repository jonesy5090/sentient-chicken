"""relu(pred@gakel) as a function of distance from P, live.

The at-P/elsewhere split used radius = one grid spacing (3.33 m) while place_sigma is
2.0, so the disc may simply be wider than the place code. If the plant is selective at
all, pred must be highest in the innermost bin and fall with distance. If it rises with
distance the plant is genuinely inverted in the live regime and both E082 and E083 were
driving the gakel channel hardest where the hen was NOT supposed to be avoiding.
"""
import time, sys
from functools import partial
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, 'scratchpad')
import e083_leaving_anchor as E
from coop import world
from hen import brain, connectome, plasticity, regions
from run import simulate

CEN, P, P2, CFG = E.CEN, E.P, E.P2, E.CFG
EDGES = jnp.asarray([0.0, 1.0, 2.0, 3.33, 5.0, 7.0, 10.0, 99.0])
NB = len(EDGES) - 1


@partial(jax.jit, static_argnames=("cfg", "pc", "n"))
def run(w, x, p, ps, key, cfg, pc, n):
    def step(carry, _):
        carry, (motor, obs, _r, _m) = simulate._one_step(carry, None, cfg=cfg, pc=pc)
        dP = jnp.linalg.norm(carry[0].pos - jnp.asarray(CEN[P], dtype=jnp.float32), axis=-1)
        pred = jnp.einsum("hon,hn->ho", p.W_pred,
                          (carry[3].z_lag - carry[3].z_lag_bar) * p.pred_src[None, :])
        g = jax.nn.relu(pred[:, E.GAKEL_CH])
        b = jnp.clip(jnp.searchsorted(EDGES, dP) - 1, 0, NB - 1)
        oh = jax.nn.one_hot(b, NB)
        return carry, (oh.T @ g, jnp.sum(oh, axis=0))
    return jax.lax.scan(step, (w, x, p, ps, key), None, length=n)[1]


pc = plasticity.PlasticConfig(**E.FROZEN, pred_gain=1.0)
t0 = time.perf_counter()
tot = np.zeros(NB); cnt = np.zeros(NB)
for s in range(E.SEEDS):
    k = jax.random.key(s)
    p0 = connectome.build(jax.random.fold_in(k, 1), regions.DEFAULT_REGIONS, n_hens=E.HENS,
                          gakel_scaffold=True, shared_place_map=True)
    p, _ = E.plant(p0, E._PLANT_CFG)
    w = world.reset(k, CFG)
    w = w._replace(food_pos=jnp.asarray(np.stack([CEN[P], CEN[P2]]), dtype=jnp.float32))
    x = brain.initial_state(p, E.HENS); ps = plasticity.initial_state(p, E.HENS, pc)
    g, c = run(w, x, p, ps, jax.random.fold_in(k, 2), CFG, pc, E.STEPS)
    # normalise per seed: live magnitude varies ~9x across seeds, so pooling raw sums
    # would let one seed set the whole profile
    gs, cs = np.asarray(jnp.sum(g, 0)), np.asarray(jnp.sum(c, 0))
    prof = gs / np.maximum(cs, 1e-9)
    tot += prof / max(prof.mean(), 1e-9); cnt += 1

print(f"pred@gakel vs distance from P (per-seed normalised to its own mean), "
      f"{E.SEEDS} seeds\n")
print(f"{'bin (m)':>14}{'rel. pred':>12}")
for i in range(NB):
    print(f"{f'{EDGES[i]:.1f}-{EDGES[i+1]:.1f}':>14}{tot[i]/cnt[i]:>12.3f}")
print(f"\nwall clock: {time.perf_counter()-t0:.0f} s")
print("selective => innermost bin highest, falling with distance.")
