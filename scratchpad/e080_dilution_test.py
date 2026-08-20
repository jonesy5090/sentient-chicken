"""E080: is H2d's loss really fan-in dilution?

Progressively zero the channels that are IDENTICAL between the hawk and call
observations -- by construction they carry nothing about the distinction. If dilution is
the mechanism, separability should climb steeply as they go.

Mean pallial rate is reported alongside because removing input lowers drive, and E079
showed lower drive hurts separability below the 0.95 optimum. Any rise here is a rise
against that headwind.
"""
import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, neurons, regions
from run.experiment import _t_critical

reg = regions.DEFAULT_REGIONS
DT, HOLD, N = 0.01, 200, 12
p_lo, p_hi = reg.bounds(regions.PALLIUM)
FRACS = [0.0, 0.25, 0.50, 0.75, 1.00]
CFG = spec.DEFAULT_COOP._replace(n_hens=4, food_deplete_rate=0.0)


def _staged(hawk, call):
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


O_H, O_C, O_Z = _staged(True, False), _staged(False, True), _staged(False, False)

informative = ~np.isclose(O_H, O_C)                     # differs between the two stimuli
active = (O_H != 0) | (O_C != 0) | (O_Z != 0)
irrelevant_active = active & ~informative               # on, but says nothing
IRR = np.flatnonzero(irrelevant_active)
print(f"active channels {int(active.sum())}, informative {int(informative.sum())}, "
      f"irrelevant-but-active {len(IRR)}\n")


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
print(f"{'frac removed':>13}{'kept irrelevant':>17}{'separability':>14}{'settle rate':>13}")
for f in FRACS:
    seps, rates = [], []
    for s in range(N):
        # fixed random order per genome: the fraction varies, not which channels
        order = np.random.default_rng(1000 + s).permutation(IRR)
        drop = order[:int(round(f * len(IRR)))]
        mask = np.ones(spec.OBS_DIM, np.float32); mask[drop] = 0.0
        p = connectome.build(jax.random.key(s), reg, n_hens=1)
        sp, rt = sep_rate(p, O_H * mask, O_C * mask, O_Z * mask)
        seps.append(sp); rates.append(rt)
    res[f] = np.array(seps)
    print(f"{f:>13.2f}{len(IRR) - int(round(f*len(IRR))):>17}"
          f"{np.mean(seps):>14.4f}{np.mean(rates):>13.4f}")

print(f"\npaired vs no removal, {N} genomes, threshold t={crit:.3f}")
base = res[0.0]
for f in FRACS[1:]:
    d = res[f] - base
    se = d.std(ddof=1) / np.sqrt(N); t = abs(d.mean()) / (se + 1e-12)
    print(f"  remove {f:.0%}: {d.mean():+.4f} +/- {se:.4f}  t={t:5.2f}  "
          f"{'SIGNIFICANT' if t > crit else 'null':<12}({res[f].mean()/base.mean():.2f}x)")
print(f"\nH2d's loss is 14.5-17x. Dilution would need a rise of that order to explain it.")
