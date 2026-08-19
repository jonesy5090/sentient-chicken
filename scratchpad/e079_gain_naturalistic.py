"""E079: where does separability peak under naturalistic input as recurrent gain varies?

E023 set gain=0.95 from a sweep on the sparse probe, where the network sits at ~0.27.
Naturalistically it sits at 0.46 and live at 0.69. Both probes, same genomes, paired,
plus a LIVE rollout rate at the interesting gains -- a settle probe cannot see whether
the network is saturated in operation.
"""
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, neurons, plasticity, regions
from run import simulate
from run.simulate import NO_PLASTICITY
from run.experiment import _t_critical

reg = regions.DEFAULT_REGIONS
DT, HOLD, N = 0.01, 200, 12
p_lo, p_hi = reg.bounds(regions.PALLIUM)
GAINS = [0.40, 0.60, 0.80, 0.95, 1.10]
DEFAULT_G = 0.95
CFG = spec.DEFAULT_COOP._replace(n_hens=4, food_deplete_rate=0.0)
AERIAL = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)

o_h = np.zeros(spec.OBS_DIM, np.float32); o_h[spec.IDX_AERIAL] = 1.0
o_c = np.zeros(spec.OBS_DIM, np.float32); o_c[AERIAL] = 1.0
o_z = np.zeros(spec.OBS_DIM, np.float32)


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


PROBES = {"sparse (E009 series)": (o_h, o_c, o_z),
          "naturalistic": (_staged(True, False), _staged(False, True), _staged(False, False))}


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
print(f"{'probe':<22}{'gain':>7}{'separability':>14}{'settle rate':>13}")
for pname, (oh, oc, oz) in PROBES.items():
    for g in GAINS:
        vals = [sep_rate(connectome.build(jax.random.key(s), reg, n_hens=1, gain=g),
                         oh, oc, oz) for s in range(N)]
        seps = np.array([v[0] for v in vals]); res[(pname, g)] = seps
        print(f"{pname:<22}{g:>7.2f}{seps.mean():>14.4f}"
              f"{np.mean([v[1] for v in vals]):>13.4f}")

print(f"\npaired vs the {DEFAULT_G:.2f} default, {N} genomes, threshold t={crit:.3f}")
for pname in PROBES:
    base = res[(pname, DEFAULT_G)]
    print(f"  {pname}")
    for g in GAINS:
        if g == DEFAULT_G:
            continue
        d = res[(pname, g)] - base
        se = d.std(ddof=1) / np.sqrt(N); t = abs(d.mean()) / (se + 1e-12)
        print(f"    gain {g:.2f}: {d.mean():+.4f} +/- {se:.4f}  t={t:5.2f}  "
              f"{'SIGNIFICANT' if t > crit else 'null':<12}({res[(pname,g)].mean()/base.mean():.2f}x)")

# --- live rate: the check a settle probe cannot make ------------------------
@partial(jax.jit, static_argnames=("cfg", "n"))
def roll(w, x, p, ps, k, cfg, n):
    def step(c, _):
        c, _o = simulate._one_step(c, None, cfg=cfg, pc=NO_PLASTICITY)
        return c, jnp.mean(jnp.abs(neurons.rate(c[1])[:, p_lo:p_hi]))
    return jax.lax.scan(step, (w, x, p, ps, k), None, length=n)[1]

nat_best = max(GAINS, key=lambda g: res[("naturalistic", g)].mean())
print(f"\nlive rollout (16 hens, 5 min, 3 seeds) -- default vs naturalistic best")
live_cfg = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=20.0)
for g in sorted({DEFAULT_G, nat_best, 0.40}):
    rates = []
    for s in range(3):
        k = jax.random.key(s); w = world.reset(k, live_cfg)
        p = connectome.build(jax.random.fold_in(k, 1), reg, n_hens=16, gain=g)
        x = brain.initial_state(p, 16); ps = plasticity.initial_state(p, 16, NO_PLASTICITY)
        rates.append(float(jnp.mean(roll(w, x, p, ps, jax.random.fold_in(k, 2), live_cfg, 30000))))
    tag = " <- default" if g == DEFAULT_G else (" <- naturalistic best" if g == nat_best else "")
    print(f"  gain {g:.2f}: live mean pallial rate {np.mean(rates):.4f}{tag}")
