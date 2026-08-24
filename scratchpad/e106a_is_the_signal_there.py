"""E106a (diagnostic, not pre-registered): is there anything behind the common mode?

E105 measured the motor stub -- the only thing the learned readout reads -- at 99.98%
DC, direction stability 0.9930 at hatch. Before building a sixth mechanism to remove
that common mode, the instrument question this project runs on: IS a positive result
reachable? If the situation-specific signal has genuinely been destroyed, subtracting
the common mode exposes nothing and no amount of inhibition helps.

So subtract it arithmetically and look. For the pallium and the motor stub, take each
step's rate vector and subtract that population's own mean at that step -- exactly what
a pooled inhibitory interneuron does, and the upper bound on what one could achieve.
Then push the mean-subtracted stub through the REARED `W_out` and see what the learned
pathway would have produced.

Three outcomes:
  signal is there   -> stability falls a long way. A pooled interneuron in the pallium
                       and motor stub is worth building, and E106 has a target.
  signal is thin    -> stability falls a little. The mechanism is worth building but
                       will not be sufficient on its own.
  signal is gone    -> stability stays high. Nothing downstream can recover it and the
                       answer is a different rate code, or acceptance in the negative.
"""
import time
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, neurons, plasticity, regions
from run import simulate

BASE = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=60.0)
REAR, PROBE, SEEDS = int(30 * 60 / BASE.dt), int(2 * 60 / BASE.dt), 4
PC = plasticity.PlasticConfig(enabled=True, hebbian_readout=True,
                              readout_scaling_strength=0.3)
reg = regions.DEFAULT_REGIONS
P_LO, P_HI = reg.bounds(regions.PALLIUM)
S_LO, S_HI = reg.bounds(regions.SENSORY)


def stability(a):
    a = np.asarray(a).reshape(-1, a.shape[-1])
    a = a[np.linalg.norm(a, axis=1) > 1e-8]
    m = a.mean(0)
    m /= np.linalg.norm(m) + 1e-12
    return float(((a @ m) / (np.linalg.norm(a, axis=1) + 1e-12)).mean())


def dc_share(a):
    """||mean over time|| / mean ||x||, per hen. E105's normalisation bug was doing
    this across hens and units at once; this is per hen and per unit-vector. The
    `axis=-1` is load-bearing -- numpy's second POSITIONAL argument to `norm` is `ord`,
    not `axis`, and the first version of this script passed -1 as an order, which
    printed a DC share of 0.14% where E105 had correctly measured 99.98%."""
    a = np.asarray(a)
    out = []
    for h in range(a.shape[1]):
        mh = a[:, h].mean(0)
        out.append(np.linalg.norm(mh) / max(np.mean(np.linalg.norm(a[:, h], axis=-1)), 1e-12))
    return float(np.mean(out))


def demean(a):
    """Subtract each population's own mean across units, at each step. What a pooled
    inhibitory interneuron does, and the ceiling on what one could do."""
    return a - a.mean(-1, keepdims=True)


@partial(jax.jit, static_argnames=("cfg", "n"))
def probe(w, x, p, key, cfg, n):
    n_motor = p.W_out.shape[-1]

    def step(c, _):
        w, x, key = c
        key, kw = jax.random.split(key)
        obs = sensing.observe(w, cfg)
        x, motor, d = brain.step(x, obs, p, cfg.dt)
        r = neurons.rate(x)
        return (world.step(w, motor, kw, cfg), x, key), (
            obs, r[:, S_LO:S_HI], r[:, P_LO:P_HI], r[:, -n_motor:], d.cortical)
    return jax.lax.scan(step, (w, x, key), None, length=n)[1]


t0 = time.perf_counter()
print(f"E106a -- is the situation signal still there behind the common mode? "
      f"{SEEDS} seeds, diagnostic\n")

rows = {k: ([], [], [], []) for k in ("sensory", "pallium", "motor stub")}
cort_raw, cort_dm, obs_st = [], [], []
for s in range(SEEDS):
    k = jax.random.key(s)
    p0 = connectome.build(jax.random.fold_in(k, 1), reg, n_hens=16)
    w = world.reset(k, BASE)
    x = brain.initial_state(p0, 16)
    st = plasticity.initial_state(p0, 16, PC)
    _w, _x, p2, _ps, _k, _t = simulate.rollout(
        w, x, p0, jax.random.fold_in(k, 2), BASE, REAR, pc=PC, ps=st)
    w3 = world.reset(k, BASE)
    x3 = brain.initial_state(p2, 16)
    obs, sens, pal, stub, cort = probe(w3, x3, p2, jax.random.fold_in(k, 5), BASE, PROBE)
    obs_st.append(stability(obs))
    for name, a in (("sensory", np.asarray(sens)), ("pallium", np.asarray(pal)),
                    ("motor stub", np.asarray(stub))):
        d = demean(a)
        rows[name][0].append(stability(a))
        rows[name][1].append(stability(d))
        rows[name][2].append(dc_share(a))
        rows[name][3].append(dc_share(d))
    # What the learned pathway would have produced from a de-meaned stub, with the
    # weights it actually grew. No retraining -- this is the readout's own ceiling.
    w_out = np.asarray(p2.W_out)
    cort_raw.append(stability(np.asarray(cort)))
    cort_dm.append(stability(np.einsum("hmk,thk->thm", w_out, demean(np.asarray(stub)))))

print(f"observation direction stability: {np.mean(obs_st):.4f}  (the world does vary)\n")
print(f"{'population':>12}{'stability':>11}{'de-meaned':>11}"
      f"{'DC share':>11}{'de-meaned':>11}")
for name, (a, b, c, d) in rows.items():
    print(f"{name:>12}{np.mean(a):>11.4f}{np.mean(b):>11.4f}"
          f"{100*np.mean(c):>10.2f}%{100*np.mean(d):>10.2f}%")

print(f"\ncortical output, reared W_out:  as built {np.mean(cort_raw):.4f}"
      f"   from a de-meaned stub {np.mean(cort_dm):.4f}")
print("\nE105 reference: motor stub 0.9930 at hatch / 0.9925 reared, DC 99.98%,")
print("cortical 0.9587. A varying input gave 0.4381 at input stability 0.4421.")
print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
