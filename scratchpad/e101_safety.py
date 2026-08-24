"""E101 safety falsifier: does a hen with a working reflex gate get eaten more?

Pre-registered: predation up >25% vs the fixed control means suppression is working TOO
well, and argues for the basal ganglia's SELECTIVE release over a free gate.
"""
import time
import jax, jax.numpy as jnp, numpy as np
from coop import spec, world
from hen import brain, connectome, plasticity, regions
from run import simulate

CFG = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=60.0)
REAR, TEST, SEEDS = int(30*60/CFG.dt), int(10*60/CFG.dt), 4
BASE = dict(enabled=True, hebbian_readout=True, readout_scaling_strength=0.3)
ARMS = {
    "off": plasticity.PlasticConfig(**BASE),
    "B":   plasticity.PlasticConfig(**BASE, reflex_gate=True),
    "A+B": plasticity.PlasticConfig(**BASE, pred_enabled=True, pred_gain=1.0,
                                    pred_centred=True, pred_bar_freeze_s=60.0,
                                    pred_signed=True, reflex_gate=True),
}

t0 = time.perf_counter()
print(f"{'arm':>6}{'caught/dive':>13}{'dives':>8}{'hunger':>9}{'crouch':>9}{'vs off':>11}")
base = None
for name, pc in ARMS.items():
    cd, hu, cr, dv = [], [], [], []
    for s in range(SEEDS):
        k = jax.random.key(s)
        p0 = connectome.build(jax.random.fold_in(k, 1), regions.DEFAULT_REGIONS, n_hens=16)
        w = world.reset(k, CFG); x = brain.initial_state(p0, 16)
        ps = plasticity.initial_state(p0, 16, pc)
        _w, _x, p2, ps2, _k, _t = simulate.rollout(
            w, x, p0, jax.random.fold_in(k, 2), CFG, REAR, pc=pc, ps=ps)
        pc_off = plasticity.PlasticConfig(
            enabled=False, pred_enabled=pc.pred_enabled, pred_gain=pc.pred_gain,
            pred_centred=pc.pred_centred, pred_signed=pc.pred_signed,
            reflex_gate=pc.reflex_gate)
        w3 = world.reset(k, CFG); x3 = brain.initial_state(p2, 16)
        wf, _x2, _p, _ps, _k2, tr = simulate.rollout(
            w3, x3, p2, jax.random.fold_in(k, 7), CFG, TEST, pc=pc_off, ps=ps2)
        d = float(jnp.sum(wf.n_dives)); c = float(jnp.sum(wf.n_caught_any))
        cd.append(c / max(d, 1)); dv.append(d)
        hu.append(float(jnp.mean(wf.hunger)))
        cr.append(float(jnp.mean(tr.motor[:, :, spec.M_CROUCH])))
    m = float(np.mean(cd))
    if base is None:
        base = m
    delta = "" if name == "off" else f"{100*(m-base)/max(base,1e-9):+.1f}%"
    print(f"{name:>6}{m:>13.4f}{np.mean(dv):>8.0f}{np.mean(hu):>9.3f}"
          f"{np.mean(cr):>9.4f}{delta:>11}")
print(f"\nsafety falsifier: fires if caught/dive rises >25% vs off")
print(f"wall clock: {time.perf_counter()-t0:.0f} s")
