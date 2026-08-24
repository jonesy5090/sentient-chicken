"""E103: trace direction stability through every stage of the chain.

1.0 = one fixed pattern whose magnitude alone varies. The reflex arc is the reference:
it is state-dependent by construction, so if IT looks fixed the probe is broken.
"""
import time
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, neurons, plasticity, regions
from run import simulate

CFG = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=60.0)
REAR, PROBE, SEEDS = int(30*60/CFG.dt), int(2*60/CFG.dt), 4
PC = plasticity.PlasticConfig(enabled=True, hebbian_readout=True,
                              readout_scaling_strength=0.3)
reg = regions.DEFAULT_REGIONS
S_LO, S_HI = reg.bounds(regions.SENSORY)
P_LO, P_HI = reg.bounds(regions.PALLIUM)


def stability(a):
    a = np.asarray(a).reshape(-1, a.shape[-1])
    a = a[np.linalg.norm(a, axis=1) > 1e-8]
    if len(a) == 0:
        return float("nan")
    m = a.mean(0); m /= np.linalg.norm(m) + 1e-12
    return float(((a @ m) / (np.linalg.norm(a, axis=1) + 1e-12)).mean())


@partial(jax.jit, static_argnames=("cfg", "pc", "n"))
def probe(w, x, p, ps, key, cfg, pc, n):
    def step(carry, _):
        w, x, p, ps, key = carry
        key, kw = jax.random.split(key)
        obs = sensing.observe(w, cfg)
        pred_from = (ps.z_lag - ps.z_lag_bar) if pc.pred_enabled else None
        x, motor, d = brain.step(x, obs, p, cfg.dt, pred_gain=pc.pred_gain,
                                 pred_from=pred_from)
        r = neurons.rate(x)
        ps2 = plasticity.update_traces(ps, r, motor, jnp.zeros((cfg.n_hens,)), cfg, pc)
        n_m = p.W_out.shape[-1]
        out = (obs, r[:, S_LO:S_HI], r[:, P_LO:P_HI], r[:, -n_m:],
               ps2.z_slow[:, -n_m:] - ps2.z_slow_bar[:, -n_m:],
               ps2.z_motor - ps2.z_motor_bar, d.cortical, d.reflex)
        return (world.step(w, motor, kw, cfg), x, p, ps2, key), out
    return jax.lax.scan(step, (w, x, p, ps, key), None, length=n)[1]


LABELS = ["1 observation", "2 sensory stub", "3 pallium", "4 MOTOR STUB",
          "5 dz_slow (presyn)", "6 dz_motor (postsyn)", "7 cortical out",
          "8 reflex (reference)"]
t0 = time.perf_counter()
res = {"untrained": [], "reared": []}
for s in range(SEEDS):
    k = jax.random.key(s)
    p0 = connectome.build(jax.random.fold_in(k,1), regions.DEFAULT_REGIONS, n_hens=16)
    w = world.reset(k, CFG); x = brain.initial_state(p0, 16)
    ps = plasticity.initial_state(p0, 16, PC)
    out = probe(w, x, p0, ps, jax.random.fold_in(k,5), CFG, PC, PROBE)
    res["untrained"].append([stability(a) for a in out])
    _w,_x,p2,ps2,_k,_t = simulate.rollout(w,x,p0,jax.random.fold_in(k,2),CFG,REAR,pc=PC,ps=ps)
    w3 = world.reset(k, CFG); x3 = brain.initial_state(p2, 16)
    out = probe(w3, x3, p2, ps2, jax.random.fold_in(k,5), CFG, PC, PROBE)
    res["reared"].append([stability(a) for a in out])

U = np.nanmean(res["untrained"], 0); R = np.nanmean(res["reared"], 0)
print("direction stability -- 1.0 = ONE FIXED PATTERN, magnitude only\n")
print(f"{'stage':>22}{'untrained':>12}{'reared':>10}{'change':>10}")
for i, lab in enumerate(LABELS):
    mark = "  <-- collapses" if (R[i] - U[i]) > 0.15 else ""
    print(f"{lab:>22}{U[i]:>12.4f}{R[i]:>10.4f}{R[i]-U[i]:>+10.4f}{mark}")
print(f"\nwall clock: {time.perf_counter()-t0:.0f} s")
print("\n--- pre-registered falsifiers (E103 section 4) ---")
print(f"triviality  observation stability {R[0]:.4f} (fires if >0.9) -> "
      f"{'FIRES' if R[0] > 0.9 else 'clear'}")
print(f"confound    reflex stability {R[7]:.4f} vs untrained {U[7]:.4f} "
      f"(probe broken if fixed AND obs fixed) -> "
      f"{'CHECK' if R[7] > 0.95 and R[0] > 0.9 else 'clear'}")
print(f"my own      motor stub reared {R[3]:.4f} "
      f"(my 'representation' hypothesis is WRONG if <0.85) -> "
      f"{'WRONG -- cause is downstream' if R[3] < 0.85 else 'consistent with (a)'}")
