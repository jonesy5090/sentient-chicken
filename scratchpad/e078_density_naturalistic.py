"""E078: does E041's sensory->pallium density result survive a naturalistic probe?

E041's mechanism -- too few pallial units connect to the 1-2 informative channels at
random-sparse density -- is specific to sparse input. Naturalistic observations have 14
nonzero channels, not 1. Both probes, same genomes, paired, on E076's corrected
baseline.
"""
import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, neurons, regions
from run.experiment import _t_critical

reg = regions.DEFAULT_REGIONS
DT, HOLD, N = 0.01, 200, 12
p_lo, p_hi = reg.bounds(regions.PALLIUM)
DENSITIES = [0.15, 0.30, 0.60, 1.00]
DEFAULT_D = 0.30
CFG = spec.DEFAULT_COOP._replace(n_hens=4, food_deplete_rate=0.0)
AERIAL = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)

o_hawk_s = np.zeros(spec.OBS_DIM, np.float32); o_hawk_s[spec.IDX_AERIAL] = 1.0
o_call_s = np.zeros(spec.OBS_DIM, np.float32); o_call_s[AERIAL] = 1.0
o_rest_s = np.zeros(spec.OBS_DIM, np.float32)


def _staged(hawk: bool, call: bool):
    w = world.reset(jax.random.key(0), CFG)
    w = w._replace(pos=jnp.array([[10., 10.], [10., 11.], [3., 3.], [17., 17.]]),
                   heading=jnp.zeros((CFG.n_hens,)), head_down=jnp.zeros((CFG.n_hens,)))
    if hawk:
        w = w._replace(hawk_pos=jnp.array([10., 10.5]), hawk_on=jnp.array(1.0),
                       hawk_t=jnp.array(1e4))
    if call:
        c = jnp.zeros((CFG.n_hens, spec.N_CALLS))
        w = w._replace(calls=c.at[1, spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)].set(1.0))
    return np.asarray(sensing.observe(w, CFG)[0])


PROBES = {
    "sparse (E009 series)": (o_hawk_s, o_call_s, o_rest_s),
    "naturalistic": (_staged(True, False), _staged(False, True), _staged(False, False)),
}
print(f"nonzero channels -- sparse {int((o_hawk_s != 0).sum())}, "
      f"naturalistic {int((PROBES['naturalistic'][0] != 0).sum())}\n")


def settle(p, obs):
    x = brain.initial_state(p, 1)
    o = jnp.asarray(obs)[None, :]
    for _ in range(HOLD):
        x, _m, _d = brain.step(x, o, p, DT)
    return np.asarray(neurons.rate(x))[0]


def sep_rate(p, oh, oc, oz):
    h, c, z = (settle(p, o)[p_lo:p_hi] for o in (oh, oc, oz))
    rest = float(np.mean(np.abs(z)))
    return float(np.sqrt(np.mean((h - c) ** 2)) / (rest + 1e-9)), rest


crit = _t_critical(N - 1)
res = {}
print(f"{'probe':<22}{'density':>9}{'separability':>14}{'mean rate':>12}")
for pname, (oh, oc, oz) in PROBES.items():
    for d in DENSITIES:
        vals = [sep_rate(connectome.build(jax.random.key(s), reg, n_hens=1,
                                          sensory_pallium_density=d), oh, oc, oz)
                for s in range(N)]
        seps = np.array([v[0] for v in vals])
        res[(pname, d)] = seps
        print(f"{pname:<22}{d:>9.2f}{seps.mean():>14.4f}"
              f"{np.mean([v[1] for v in vals]):>12.4f}")

print(f"\npaired vs the {DEFAULT_D:.2f} default, {N} genomes, threshold t={crit:.3f}")
for pname in PROBES:
    base = res[(pname, DEFAULT_D)]
    print(f"  {pname}")
    for d in DENSITIES:
        if d == DEFAULT_D:
            continue
        diff = res[(pname, d)] - base
        se = diff.std(ddof=1) / np.sqrt(N)
        t = abs(diff.mean()) / (se + 1e-12)
        print(f"    density {d:.2f}: {diff.mean():+.4f} +/- {se:.4f}  t={t:5.2f}  "
              f"{'SIGNIFICANT' if t > crit else 'null':<12}"
              f"({res[(pname, d)].mean()/base.mean():.2f}x)")
