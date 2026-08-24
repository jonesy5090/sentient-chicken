"""E105c: E105b's positive control did not work, and why it failed is the finding.

E105b planted a "varied" input by replacing the motor stub's deviations with random
directions of the SAME energy. The planted input's direction stability came out at
0.9882 against the real 0.9925 -- it barely varied either, so the control could not have
detected a working readout. That is the failure this project's own rule names: a control
must be measured destroying what it claims to destroy, not argued to.

The reason it failed is the measurement. The motor stub's per-step deviation is a tiny
fraction of its mean, so ANY direction assigned to that deviation leaves the vector
pointing where the mean points. `cortical = W_out @ stub` then has a fixed direction for
any `W_out` whatsoever.

So: sweep the deviation gain. Scale the stub's deviations by f while holding the mean,
and plot input direction stability against the cortical stability it produces. This
answers the question E105 could not: is there ANY input this reared readout would give a
state-dependent output for, and how far from the real one does it have to be?
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
GAINS = (1, 2, 5, 10, 30, 100)
reg = regions.DEFAULT_REGIONS


def stability(a):
    a = np.asarray(a).reshape(-1, a.shape[-1])
    a = a[np.linalg.norm(a, axis=1) > 1e-8]
    m = a.mean(0)
    m /= np.linalg.norm(m) + 1e-12
    return float(((a @ m) / (np.linalg.norm(a, axis=1) + 1e-12)).mean())


@partial(jax.jit, static_argnames=("cfg", "n"))
def probe(w, x, p, adapt_bar, key, cfg, n):
    n_motor = p.W_out.shape[-1]

    def step(c, _):
        w, x, adapt_bar, key = c
        key, kw = jax.random.split(key)
        obs = sensing.observe(w, cfg)
        ab = adapt_bar if cfg.sensory_adapt_tau_s is not None else None
        x, motor, d = brain.step(x, obs, p, cfg.dt,
                                 sensory_lateral=cfg.sensory_lateral, adapt_bar=ab)
        if cfg.sensory_adapt_tau_s is not None:
            adapt_bar = adapt_bar + (cfg.dt / cfg.sensory_adapt_tau_s) * (
                d.current - adapt_bar)
        stub = neurons.rate(x)[:, -n_motor:]
        return (world.step(w, motor, kw, cfg), x, adapt_bar, key), (stub, d.cortical)
    return jax.lax.scan(step, (w, x, adapt_bar, key), None, length=n)[1]


t0 = time.perf_counter()
ARMS = (("current", 0.0, None), ("decorr+adapt", 1.0, 2.0))
print(f"E105c -- how far from the real input does the readout need to be? "
      f"{SEEDS} seeds\n")

for label, decorr, adapt in ARMS:
    cfg = BASE._replace(sensory_lateral=(1.0 if adapt is not None else 0.0),
                        sensory_adapt_tau_s=adapt)
    pc = PC._replace(readout_decorrelate=decorr)
    dev_frac, dc, real_cort = [], [], []
    rows = {g: ([], []) for g in GAINS}
    for s in range(SEEDS):
        k = jax.random.key(s)
        p0 = connectome.build(jax.random.fold_in(k, 1), reg, n_hens=16)
        w = world.reset(k, cfg)
        x = brain.initial_state(p0, 16)
        st = plasticity.initial_state(p0, 16, pc)
        _w, _x, p2, ps2, _k, _t = simulate.rollout(
            w, x, p0, jax.random.fold_in(k, 2), cfg, REAR, pc=pc, ps=st)
        w3 = world.reset(k, cfg)
        x3 = brain.initial_state(p2, 16)
        stub, cort = probe(w3, x3, p2, ps2.adapt_bar,
                           jax.random.fold_in(k, 5), cfg, PROBE)
        a = np.asarray(stub)
        mean = a.mean(0, keepdims=True)
        dev = a - mean
        # Per hen. Norming a (1, H, K) mean over H AND K while norming the
        # deviations over K alone inflates the ratio by sqrt(H) -- it is how the first
        # version of this script reported a "DC share" of 400%, which is impossible.
        for h in range(a.shape[1]):
            mh = mean[0, h]
            nm = np.linalg.norm(mh) + 1e-12
            dev_frac.append(float(np.mean(np.linalg.norm(dev[:, h], axis=-1)) / nm))
            dc.append(float(nm / np.mean(np.linalg.norm(a[:, h], axis=-1))))
        real_cort.append(stability(cort))
        w_out = np.asarray(p2.W_out)
        for g in GAINS:
            planted = mean + g * dev
            rows[g][0].append(stability(planted))
            rows[g][1].append(stability(
                np.einsum("hmk,thk->thm", w_out, planted)))
    print(f"[{label}]  motor-stub deviation = {100*np.mean(dev_frac):.2f}% of its own "
          f"mean;  DC share {100*np.mean(dc):.2f}%;  real cortical "
          f"{np.mean(real_cort):.4f}")
    print(f"{'dev gain':>10}{'input stab':>13}{'cortical stab':>16}")
    for g in GAINS:
        print(f"{g:>10}x{np.mean(rows[g][0]):>12.4f}{np.mean(rows[g][1]):>16.4f}")
    print()

print("reading it: if cortical stability tracks input stability down, the readout is a")
print("faithful map and the fixed output is a fixed INPUT -- an architectural property")
print("present at hatch, not something any learning rule can undo. If cortical stays")
print("high while the input varies, the readout itself is the fault and E105 has more")
print("to try.")
print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
