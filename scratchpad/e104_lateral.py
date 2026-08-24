"""E104: does lateral inhibition at the relay restore situation-dependence?"""
import time
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, neurons, plasticity, regions
from run import simulate

BASE = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=60.0)
REAR, PROBE, SEEDS = int(30*60/BASE.dt), int(2*60/BASE.dt), 4
PC = plasticity.PlasticConfig(enabled=True, hebbian_readout=True,
                              readout_scaling_strength=0.3)
reg = regions.DEFAULT_REGIONS
S_LO, S_HI = reg.bounds(regions.SENSORY)


def stability(a):
    a = np.asarray(a).reshape(-1, a.shape[-1])
    a = a[np.linalg.norm(a, axis=1) > 1e-8]
    if len(a) == 0: return float("nan")
    m = a.mean(0); m /= np.linalg.norm(m) + 1e-12
    return float(((a @ m) / (np.linalg.norm(a, axis=1) + 1e-12)).mean())


def dc_share(a):
    a = np.asarray(a).reshape(-1, a.shape[-1])
    m = a.mean(0)
    return float(np.linalg.norm(m) / max(np.linalg.norm(a, axis=1).mean(), 1e-12))


@partial(jax.jit, static_argnames=("cfg", "pc", "n"))
def probe(w, x, p, ps, key, cfg, pc, n):
    def step(c, _):
        w, x, p, ps, key = c
        key, kw = jax.random.split(key)
        obs = sensing.observe(w, cfg)
        x, motor, d = brain.step(x, obs, p, cfg.dt,
                                 sensory_lateral=cfg.sensory_lateral)
        r = neurons.rate(x)
        return (world.step(w, motor, kw, cfg), x, p, ps, key), (
            obs, r[:, S_LO:S_HI], d.cortical, d.reflex)
    return jax.lax.scan(step, (w, x, p, ps, key), None, length=n)[1]


t0 = time.perf_counter()
print(f"E104 -- lateral inhibition at the relay. {SEEDS} seeds\n")
print(f"{'lateral':>9}{'stub stab':>11}{'stub DC%':>10}{'cortical stab':>15}"
      f"{'obs stab':>10}{'reflex':>9}")
for lat in (0.0, 0.5, 1.0):
    cfg = BASE._replace(sensory_lateral=lat)
    ss, dc, cs, os_, rs = [], [], [], [], []
    for s in range(SEEDS):
        k = jax.random.key(s)
        p0 = connectome.build(jax.random.fold_in(k,1), regions.DEFAULT_REGIONS, n_hens=16)
        w = world.reset(k, cfg); x = brain.initial_state(p0, 16)
        ps = plasticity.initial_state(p0, 16, PC)
        _w,_x,p2,ps2,_k,_t = simulate.rollout(w,x,p0,jax.random.fold_in(k,2),cfg,REAR,pc=PC,ps=ps)
        w3 = world.reset(k, cfg); x3 = brain.initial_state(p2, 16)
        obs, stub, cort, refl = probe(w3, x3, p2, ps2, jax.random.fold_in(k,5), cfg, PC, PROBE)
        ss.append(stability(stub)); dc.append(dc_share(stub))
        cs.append(stability(cort)); os_.append(stability(obs)); rs.append(stability(refl))
    print(f"{lat:>9.1f}{np.mean(ss):>11.4f}{100*np.mean(dc):>9.1f}%{np.mean(cs):>15.4f}"
          f"{np.mean(os_):>10.4f}{np.mean(rs):>9.4f}")
print(f"\nE103 reference: stub 0.9707/97.8% DC, cortical reared 0.9587, obs 0.6573")
print(f"wall clock: {time.perf_counter()-t0:.0f} s")
