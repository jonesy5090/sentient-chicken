"""E101: can the higher brain overrule a reflex, and does E100's collapse relax?

Four arms: off / A (signed perception) / B (reflex gate) / A+B. hebbian_readout
throughout, since that is the rule producing the largest behavioural change.

The capability gate runs FIRST and aborts: if neither mechanism can oppose the crouch
reflex, nothing behavioural from it is interpretable.
"""
import time
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, plasticity, regions
from run import simulate

CFG = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=60.0)
REAR, PROBE, SEEDS = int(30*60/CFG.dt), int(2*60/CFG.dt), 4
BASE = dict(enabled=True, hebbian_readout=True, readout_scaling_strength=0.3)
ARMS = {
    "off":  plasticity.PlasticConfig(**BASE),
    "A":    plasticity.PlasticConfig(**BASE, pred_enabled=True, pred_gain=1.0,
                                     pred_centred=True, pred_bar_freeze_s=60.0,
                                     pred_signed=True),
    "B":    plasticity.PlasticConfig(**BASE, reflex_gate=True),
    "A+B":  plasticity.PlasticConfig(**BASE, pred_enabled=True, pred_gain=1.0,
                                     pred_centred=True, pred_bar_freeze_s=60.0,
                                     pred_signed=True, reflex_gate=True),
}


@partial(jax.jit, static_argnames=("cfg", "pc", "n"))
def probe(w, x, p, ps, key, cfg, pc, n):
    def step(c, _):
        c, (motor, obs, _r, _m) = simulate._one_step(c, None, cfg=cfg, pc=pc)
        return c, motor
    return jax.lax.scan(step, (w, x, p, ps, key), None, length=n)[1]


def staged_hawk(p, pc):
    """Hawk directly overhead: the strongest reflex in the model. What reaches the muscle?"""
    w = world.reset(jax.random.key(0), CFG)._replace(
        hawk_pos=jnp.array([10., 10.]), hawk_on=jnp.array(1.0), hawk_t=jnp.array(1e4),
        pos=jnp.broadcast_to(jnp.array([10., 10.]), (16, 2)),
        heading=jnp.zeros((16,)), head_down=jnp.zeros((16,)))
    obs = sensing.observe(w, CFG)
    x = brain.initial_state(p, 16)
    ps = plasticity.initial_state(p, 16, pc)
    for _ in range(200):
        x, motor, d = brain.step(x, obs, p, CFG.dt, pred_gain=pc.pred_gain,
                                 pred_from=(ps.z_lag - ps.z_lag_bar) if pc.pred_enabled else None,
                                 pred_signed=pc.pred_signed, reflex_gate=pc.reflex_gate)
    ci = spec.M_CROUCH
    return (float(jnp.mean(d.reflex[:, ci])), float(jnp.mean(d.cortical[:, ci])),
            float(jnp.mean(motor[:, ci])))


def dir_stability(v):
    a = np.asarray(v).reshape(-1, v.shape[-1])
    a = a[np.linalg.norm(a, axis=1) > 1e-8]
    m = a.mean(0); m /= np.linalg.norm(m) + 1e-12
    return float(((a @ m) / (np.linalg.norm(a, axis=1) + 1e-12)).mean())


print(f"E101 -- top-down suppression. {SEEDS} seeds, 30 min rearing\n")
t0 = time.perf_counter()
reared = {}
print(f"{'arm':>6}{'reflex@crouch':>15}{'cortical':>10}{'M_CROUCH out':>14}")
for name, pc in ARMS.items():
    rs, cs, ms, ps_ = [], [], [], []
    for s in range(SEEDS):
        k = jax.random.key(s)
        p0 = connectome.build(jax.random.fold_in(k,1), regions.DEFAULT_REGIONS, n_hens=16)
        w = world.reset(k, CFG); x = brain.initial_state(p0, 16)
        st = plasticity.initial_state(p0, 16, pc)
        _w,_x,p2,st2,_k,_t = simulate.rollout(w,x,p0,jax.random.fold_in(k,2),CFG,REAR,pc=pc,ps=st)
        r,c,m = staged_hawk(p2, pc)
        rs.append(r); cs.append(c); ms.append(m); ps_.append((p2, st2))
    reared[name] = ps_
    print(f"{name:>6}{np.mean(rs):>15.3f}{np.mean(cs):>10.3f}{np.mean(ms):>14.4f}")

print(f"\n--- CAPABILITY GATE (E101 section 4) ---")
print("Can the arc be opposed at all? `off` is the reference.")

print(f"\n{'arm':>6}{'direction stability':>22}{'(E100 reared: 0.9587)':>24}")
for name, pc in ARMS.items():
    ds = []
    for s in range(SEEDS):
        p2, st2 = reared[name][s]
        k = jax.random.key(s)
        w = world.reset(k, CFG); x = brain.initial_state(p2, 16)
        # direction stability of the drive that actually reaches the muscles
        from hen import neurons
        mt = probe(w, x, p2, st2, jax.random.fold_in(k,5), CFG,
                   plasticity.PlasticConfig(enabled=False, pred_enabled=pc.pred_enabled,
                                            pred_gain=pc.pred_gain,
                                            pred_centred=pc.pred_centred,
                                            pred_signed=pc.pred_signed,
                                            reflex_gate=pc.reflex_gate), PROBE)
        ds.append(dir_stability(mt))
    print(f"{name:>6}{np.mean(ds):>22.4f}")
print(f"\nwall clock: {time.perf_counter()-t0:.0f} s")
