"""Is H2d measuring the wrong thing? `pallial_sep` is RMS distance. The question that
matters for any downstream readout (W_pred, W_out) is LINEAR DECODABILITY.
Two states can be highly correlated and still perfectly separable.
"""
import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, neurons, regions

reg = regions.DEFAULT_REGIONS
p_lo, p_hi = reg.bounds(regions.PALLIUM)
CFG = spec.DEFAULT_COOP._replace(n_hens=4, food_deplete_rate=0.0)
AER = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)

def staged(hawk, call):
    w = world.reset(jax.random.key(0), CFG)
    w = w._replace(pos=jnp.array([[10.,10.],[10.,11.],[3.,3.],[17.,17.]]),
                   heading=jnp.zeros((4,)), head_down=jnp.zeros((4,)))
    if hawk: w = w._replace(hawk_pos=jnp.array([10.,10.5]), hawk_on=jnp.array(1.0), hawk_t=jnp.array(1e4))
    if call:
        c = jnp.zeros((4, spec.N_CALLS))
        w = w._replace(calls=c.at[1, spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)].set(1.0))
    return np.asarray(sensing.observe(w, CFG)[0])

O_H, O_C = staged(True, False), staged(False, True)

def settle(p, obs, steps=200):
    x = brain.initial_state(p, 1); o = jnp.asarray(obs)[None, :]
    for _ in range(steps): x, _m, _d = brain.step(x, o, p, CFG.dt)
    return np.asarray(neurons.rate(x))[0][p_lo:p_hi]

NOISE, M = 0.02, 40
accs, seps, corrs = [], [], []
for s in range(6):
    p = connectome.build(jax.random.key(s), reg, n_hens=1)
    rng = np.random.default_rng(s)
    H = np.stack([settle(p, np.clip(O_H + rng.normal(0, NOISE, spec.OBS_DIM), 0, 1)) for _ in range(M)])
    C = np.stack([settle(p, np.clip(O_C + rng.normal(0, NOISE, spec.OBS_DIM), 0, 1)) for _ in range(M)])

    # H2d's own metric, for reference
    z = settle(p, np.zeros(spec.OBS_DIM, np.float32))
    seps.append(np.sqrt(np.mean((H.mean(0)-C.mean(0))**2)) / (np.mean(np.abs(z))+1e-9))
    corrs.append(np.corrcoef(H.mean(0), C.mean(0))[0,1])

    # linear decodability: difference-of-means direction, held-out split
    tr = slice(0, M//2); te = slice(M//2, M)
    wdir = H[tr].mean(0) - C[tr].mean(0)
    thr = 0.5*(H[tr]@wdir).mean() + 0.5*(C[tr]@wdir).mean()
    acc = 0.5*((H[te]@wdir > thr).mean() + (C[te]@wdir <= thr).mean())
    accs.append(acc)

print(f"pallial states, hawk vs alarm-call, {M} noisy samples each, 6 genomes\n")
print(f"  H2d's 'separability'  (RMS dist / rest):   {np.mean(seps):.4f}   <- the ~7-8% figure")
print(f"  correlation between the two mean states:   {np.mean(corrs):.4f}   <- 'indistinguishable'")
print(f"  LINEAR DECODING accuracy, held-out:        {np.mean(accs):.1%}   <- what a readout can do")
