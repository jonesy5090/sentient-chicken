"""Is place information ABSENT from the pallium, or present but riding on a DC offset?"""
import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, neurons, regions

CFG = spec.DEFAULT_COOP._replace(n_hens=2, food_deplete_rate=0.0)
EDGES = np.linspace(0.0, CFG.size, spec.PLACE_GRID+2)[1:-1]
CENTRES = np.stack(np.meshgrid(EDGES,EDGES,indexing="ij"),-1).reshape(-1,2)
reg = regions.DEFAULT_REGIONS
p_lo, p_hi = reg.bounds(regions.PALLIUM)
p = connectome.build(jax.random.key(1), reg, n_hens=2, shared_place_map=True)

def pallial(cell, steps=10):
    w = world.reset(jax.random.key(0), CFG)
    w = w._replace(pos=jnp.broadcast_to(jnp.asarray(CENTRES[cell],dtype=jnp.float32),(2,2)),
                   heading=jnp.zeros((2,)))
    x = brain.initial_state(p,2); obs = sensing.observe(w,CFG)
    for _ in range(steps): x,_m,_d = brain.step(x,obs,p,CFG.dt)
    return np.asarray(neurons.rate(x)[0])[p_lo:p_hi]

cells = [0, 2, 12, 22, 24]
states = {c: pallial(c) for c in cells}
M = np.stack([states[c] for c in cells])
print("pallial state, place cell 2 vs others:")
print(f"{'other cell':>11}{'cosine':>9}{'correlation':>13}{'|diff|/|mean|':>15}")
for c in cells[1:]:
    a, b = states[2], states[c]
    cosv = a@b/(np.linalg.norm(a)*np.linalg.norm(b))
    corr = np.corrcoef(a,b)[0,1]
    rel = np.linalg.norm(a-b)/np.linalg.norm((a+b)/2)
    print(f"{c:>11}{cosv:>9.4f}{corr:>13.4f}{rel:>15.4f}")

print(f"\nDC: mean pallial rate {M.mean():.4f}, "
      f"across-place std of each unit {M.std(axis=0).mean():.6f}")
print(f"signal-to-DC ratio: {M.std(axis=0).mean()/M.mean():.4%}")

# Can a linear readout separate places from the RAW (uncentred) rates W_pred sees?
X = M - M.mean(axis=0, keepdims=True)     # centred across places
u,s,vt = np.linalg.svd(X, full_matrices=False)
print(f"top singular values of the across-place variation: {np.round(s[:4],5)}")
print("If these are nonzero the information IS there -- it is a readout problem,")
print("not an absence problem.")
