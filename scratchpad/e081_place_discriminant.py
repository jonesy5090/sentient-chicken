"""E070 planted a MATCHED FILTER (copy P's pallial pattern) and found 3.6% selectivity.
A correlational rule converges to something matched-filter-like; a delta rule converges
toward a DISCRIMINANT. Does a discriminative readout separate places where a matched
filter cannot?
"""
import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, neurons, regions

reg = regions.DEFAULT_REGIONS
p_lo, p_hi = reg.bounds(regions.PALLIUM)
CFG = spec.DEFAULT_COOP._replace(n_hens=2, food_deplete_rate=0.0, place_cells_enabled=True)
E = np.linspace(0.0, CFG.size, spec.PLACE_GRID+2)[1:-1]
C = np.stack(np.meshgrid(E, E, indexing="ij"), -1).reshape(-1, 2)
CELLS = [2, 6, 12, 18, 22]

def settle(p, cell, jitter, rng, steps=200):
    w = world.reset(jax.random.key(0), CFG)
    pos = C[cell] + rng.normal(0, jitter, 2)
    w = w._replace(pos=jnp.broadcast_to(jnp.asarray(pos, dtype=jnp.float32), (2,2)),
                   heading=jnp.zeros((2,)))
    x = brain.initial_state(p, 2); o = sensing.observe(w, CFG)
    for _ in range(steps): x, _m, _d = brain.step(x, o, p, CFG.dt)
    return np.asarray(neurons.rate(x)[0])[p_lo:p_hi]

M, JIT = 24, 0.35
accs_mf, accs_di = [], []
for s in range(4):
    p = connectome.build(jax.random.key(s), reg, n_hens=2, shared_place_map=True)
    rng = np.random.default_rng(s)
    S = {c: np.stack([settle(p, c, JIT, rng) for _ in range(M)]) for c in CELLS}
    tr, te = slice(0, M//2), slice(M//2, M)
    P = CELLS[0]; others = CELLS[1:]
    OT_tr = np.concatenate([S[o][tr] for o in others])
    OT_te = np.concatenate([S[o][te] for o in others])

    def acc(w):
        w = w / (np.linalg.norm(w) + 1e-9)
        thr = 0.5*((S[P][tr]@w).mean() + (OT_tr@w).mean())
        return 0.5*(((S[P][te]@w) > thr).mean() + ((OT_te@w) <= thr).mean())

    accs_mf.append(acc(S[P][tr].mean(0)))                      # matched filter
    accs_di.append(acc(S[P][tr].mean(0) - OT_tr.mean(0)))      # discriminant

print("can a linear readout tell 'at place P' from 'anywhere else'?")
print(f"  {M} samples/place, {JIT} m jitter, {len(CELLS)} places, 4 genomes, held-out\n")
print(f"  matched filter (what E070 planted):       {np.mean(accs_mf):.1%}")
print(f"  discriminant   (what a delta rule finds): {np.mean(accs_di):.1%}")

