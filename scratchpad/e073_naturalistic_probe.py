"""E073: does H2d's separability metric behave differently under naturalistic input
than under the hand-injected sparse probe used since E009?

Both probes, same genomes, paired -- so the probe-to-probe comparison cannot be
attributed to genome sampling. Same hawk-vs-alarm-call contrast in both; only the
surrounding input statistics differ.
"""
import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, neurons, regions
from run.experiment import _t_critical

reg = regions.DEFAULT_REGIONS
DT, HOLD, N = 0.01, 200, 12
p_lo, p_hi = reg.bounds(regions.PALLIUM)
CFG = spec.DEFAULT_COOP._replace(n_hens=4, food_deplete_rate=0.0)
AERIAL = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)

# ---- sparse probe: E009/E041's exact stimuli ------------------------------
o_hawk_sparse = np.zeros(spec.OBS_DIM, np.float32); o_hawk_sparse[spec.IDX_AERIAL] = 1.0
o_call_sparse = np.zeros(spec.OBS_DIM, np.float32); o_call_sparse[AERIAL] = 1.0
o_rest_sparse = np.zeros(spec.OBS_DIM, np.float32)

# ---- naturalistic probe: the same contrast, staged in a real coop ---------
def _staged(hawk: bool, call: bool):
    w = world.reset(jax.random.key(0), CFG)
    pos = jnp.array([[10.0, 10.0], [10.0, 11.0], [3.0, 3.0], [17.0, 17.0]])
    w = w._replace(pos=pos, heading=jnp.zeros((CFG.n_hens,)),
                   head_down=jnp.zeros((CFG.n_hens,)))   # not pecking: sky visible
    if hawk:
        w = w._replace(hawk_pos=jnp.array([10.0, 10.5]), hawk_on=jnp.array(1.0),
                       hawk_t=jnp.array(1e4))
    if call:
        calls = jnp.zeros((CFG.n_hens, spec.N_CALLS))
        calls = calls.at[1, spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)].set(1.0)
        w = w._replace(calls=calls)
    return np.asarray(sensing.observe(w, CFG)[0])

o_hawk_nat, o_call_nat, o_rest_nat = _staged(True, False), _staged(False, True), _staged(False, False)

print("input statistics (nonzero channels / mean value over the observation):")
for nm, o in (("sparse hawk", o_hawk_sparse), ("sparse call", o_call_sparse),
              ("natural hawk", o_hawk_nat), ("natural call", o_call_nat)):
    print(f"  {nm:<14}{int((o != 0).sum()):>4} nonzero   mean {o.mean():.4f}")
print()


def settle(p, obs):
    x = brain.initial_state(p, 1)
    o = jnp.asarray(obs)[None, :]
    for _ in range(HOLD):
        x, _m, _d = brain.step(x, o, p, DT)
    return np.asarray(neurons.rate(x))[0]


def sep(p, oh, oc, oz):
    h, c, z = (settle(p, o)[p_lo:p_hi] for o in (oh, oc, oz))
    return (float(np.sqrt(np.mean((h - c) ** 2)) / (np.mean(np.abs(z)) + 1e-9)),
            float(np.mean(np.abs(z))))


PROBES = {"sparse (E009 series)": (o_hawk_sparse, o_call_sparse, o_rest_sparse),
          "naturalistic":         (o_hawk_nat, o_call_nat, o_rest_nat)}
crit = _t_critical(N - 1)
print(f"{'probe':<22}{'condition':<14}{'separability':>14}{'mean rate':>12}")
res = {}
for pname, (oh, oc, oz) in PROBES.items():
    for cname, kw in (("baseline", {}), ("balanced_ei", dict(balanced_ei=True))):
        vals = [sep(connectome.build(jax.random.key(s), reg, n_hens=1, **kw), oh, oc, oz)
                for s in range(N)]
        seps = np.array([v[0] for v in vals]); rates = np.array([v[1] for v in vals])
        res[(pname, cname)] = seps
        print(f"{pname:<22}{cname:<14}{seps.mean():>14.4f}{rates.mean():>12.4f}")

print(f"\npaired contrasts within each probe, {N} genomes, threshold t={crit:.3f}")
for pname in PROBES:
    d = res[(pname, "balanced_ei")] - res[(pname, "baseline")]
    se = d.std(ddof=1) / np.sqrt(N); t = abs(d.mean()) / (se + 1e-12)
    ratio = res[(pname, "balanced_ei")].mean() / res[(pname, "baseline")].mean()
    print(f"  {pname:<22} balanced - baseline = {d.mean():+.4f} +/- {se:.4f}  "
          f"t={t:.2f}  {'SIGNIFICANT' if t > crit else 'not significant'}  ({ratio:.2f}x)")
