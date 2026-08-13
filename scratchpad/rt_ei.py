"""Does the region-segregated E/I assignment cause the knife-edge gain / saturation?

connectome.py assigns dale by flat index over a region-ordered array, so the pallium
ends up 100% excitatory. Compare against dale drawn per-neuron at the same 80% rate.
Metric mirrors E017: settle on 'hawk overhead' vs 'alarm call heard', report mean
pallial rate and relative separability.
"""
import jax, jax.numpy as jnp, numpy as np
from coop import spec
from hen import connectome, regions, neurons

reg = regions.DEFAULT_REGIONS
P_LO, P_HI = reg.bounds(regions.PALLIUM)
AER_CALL = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)

def build(key, gain, mixed):
    p = connectome.build(key, reg, n_hens=1, gain=gain)
    if not mixed:
        return p
    n = p.b.shape[0]
    d = jnp.where(jax.random.uniform(jax.random.fold_in(key, 77), (n,)) < 0.8, 1.0, -1.0)
    W = jnp.abs(p.W) * d[None, None, :]
    return p._replace(W=W, dale=d)

def settle(p, obs, steps=400):
    x = jnp.broadcast_to(p.b[None, :], (1, p.b.shape[0])).copy()
    cur = obs @ p.W_in.T
    for _ in range(steps):
        x, _ = neurons.ctrnn_step(x, p.W, cur, p.b, p.tau, 0.01)
    return neurons.rate(x)[0]

def probe(gain, mixed, seeds=6):
    rates, seps = [], []
    for s in range(seeds):
        p = build(jax.random.key(s), gain, mixed)
        o_h = jnp.zeros((1, spec.OBS_DIM)).at[0, spec.IDX_AERIAL].set(1.0)
        o_c = jnp.zeros((1, spec.OBS_DIM)).at[0, AER_CALL].set(1.0)
        o_0 = jnp.zeros((1, spec.OBS_DIM))
        a, b, z = settle(p, o_h), settle(p, o_c), settle(p, o_0)
        sl = slice(P_LO, P_HI)
        m = float(jnp.mean(z[sl]))
        rates.append(m)
        seps.append(float(jnp.sqrt(jnp.mean((a[sl]-b[sl])**2))) / (m + 1e-9))
    return np.mean(rates), np.std(rates), np.mean(seps), np.std(seps)

print(f"{'gain':>5} {'E/I':>10} {'mean pallial rate':>19} {'relative separability':>23}")
for mixed in (False, True):
    for g in (0.60, 0.70, 0.75, 0.78, 0.90, 1.2, 1.6):
        mr, sr, ms, ss = probe(g, mixed)
        print(f"{g:5.2f} {'mixed 80/20' if mixed else 'segregated':>10} "
              f"{mr:9.3f} +/- {sr:5.3f}      {ms:11.3f} +/- {ss:5.3f}")
