"""Where is place information lost -- at the stub, or in the pallium?"""
import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, neurons, regions

CFG = spec.DEFAULT_COOP._replace(n_hens=2, food_deplete_rate=0.0)
EDGES = np.linspace(0.0, CFG.size, spec.PLACE_GRID + 2)[1:-1]
CENTRES = np.stack(np.meshgrid(EDGES, EDGES, indexing="ij"), -1).reshape(-1,2)
reg = regions.DEFAULT_REGIONS
s_lo, s_hi = reg.bounds(regions.SENSORY); p_lo, p_hi = reg.bounds(regions.PALLIUM)
p = connectome.build(jax.random.key(1), reg, n_hens=2, shared_place_map=True)

def state(cell, steps):
    w = world.reset(jax.random.key(0), CFG)
    w = w._replace(pos=jnp.broadcast_to(jnp.asarray(CENTRES[cell],dtype=jnp.float32),(2,2)),
                   heading=jnp.zeros((2,)))
    x = brain.initial_state(p, 2)
    obs = sensing.observe(w, CFG)
    for _ in range(steps):
        x, _m, _d = brain.step(x, obs, p, CFG.dt)
    return np.asarray(neurons.rate(x)[0]), np.asarray(obs[0])

def cos(a,b): return float(a@b/((np.linalg.norm(a)*np.linalg.norm(b))+1e-12))

print(f"{'settle steps':>13}{'obs place block':>18}{'sensory stub':>15}{'pallium':>10}")
for steps in (1, 10, 50, 300):
    rP, oP = state(2, steps); rQ, oQ = state(22, steps)
    pl = slice(spec.PLACE_LO, spec.PLACE_HI)
    print(f"{steps:>13}{cos(oP[pl],oQ[pl]):>18.4f}"
          f"{cos(rP[s_lo:s_hi],rQ[s_lo:s_hi]):>15.4f}{cos(rP[p_lo:p_hi],rQ[p_lo:p_hi]):>10.4f}")
print("\n(cosine similarity between two DIFFERENT places. 1.0 = indistinguishable.)")
