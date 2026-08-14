"""Can the cortical pathway even SEE hunger? H2 asks it to regulate drives.

Hunger enters at the hypothalamus (connectome.py:121), which the flat-index Dale cut
makes 100% inhibitory, and it is 16 units of 512. Measure how much of the pallial and
motor-stub state a change in hunger actually moves, against how much a visual percept
moves it.
"""
import jax, jax.numpy as jnp, numpy as np
from coop import spec
from hen import connectome, regions, neurons
reg = regions.DEFAULT_REGIONS
P = reg.bounds(regions.PALLIUM); M = reg.bounds(regions.MOTOR); H = reg.bounds(regions.HYPOTHALAMUS)
def settle(p, obs, steps=600):
    x = jnp.broadcast_to(p.b[None,:], (1, p.b.shape[0])).copy()
    cur = obs @ p.W_in.T
    for _ in range(steps): x,_ = neurons.ctrnn_step(x, p.W, cur, p.b, p.tau, 0.01)
    return neurons.rate(x)[0]
def rel(a,b,base,sl): 
    s=slice(*sl); return float(jnp.sqrt(jnp.mean((a[s]-b[s])**2))/(jnp.mean(base[s])+1e-9))
res={}
for s in range(6):
    p = connectome.build(jax.random.key(s), reg, n_hens=1)
    z = jnp.zeros((1, spec.OBS_DIM))
    lo = z.at[0, spec.IDX_HUNGER].set(0.1); hi = z.at[0, spec.IDX_HUNGER].set(0.9)
    fo = z.at[0, spec.vis_index(6, spec.CLS_FOOD)].set(1.0)
    a,b,c,base = settle(p,lo), settle(p,hi), settle(p,fo), settle(p,z)
    for name, sl, pair in (("hunger 0.1 vs 0.9", P,(a,b)), ("food seen vs not", P,(c,base)),
                           ("hunger 0.1 vs 0.9", M,(a,b)), ("food seen vs not", M,(c,base)),
                           ("hunger 0.1 vs 0.9", H,(a,b))):
        key=(name, "pallium" if sl==P else "motor stub" if sl==M else "hypothalamus")
        res.setdefault(key,[]).append(rel(pair[0],pair[1],base,sl))
for k,v in res.items():
    print(f"  {k[1]:<13} {k[0]:<20} relative separability {np.mean(v):.4f} +/- {np.std(v):.4f}")
