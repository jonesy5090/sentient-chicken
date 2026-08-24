"""E105b: is the readout's INPUT the bottleneck, and can any W_out overcome it?

E105's primary falsifier fired -- no arm moved cortical direction stability below 0.90,
including the arm whose W_out reached effective rank 3.60, near the untrained 3.53. So
rank and direction stability came apart, and the obvious explanation is that `W_out`
reads the MOTOR STUB, whose own direction stability E103 measured at 0.9930 untrained
and 0.9925 reared -- unchanged by learning, and never touched by E104's sensory fix or
by anything in E105.

If that is right, `cortical = W_out @ stub` has a near-fixed direction for ANY W_out,
and five interventions failed because none of them was applied where the problem is.

Two measurements:

  (a) motor-stub direction stability in all four E105 arms, reared. If it sits at ~0.99
      everywhere, nothing tested moved the readout's input.
  (b) a planted-variation positive control: feed each reared W_out an input with the
      same mean and deliberately varied direction. If cortical stability collapses, the
      readout is fine and the input is the bottleneck. If it stays high, the explanation
      is wrong and the readout is at fault after all -- which is the outcome that would
      keep E105's line of work alive.
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


def stability(a):
    a = np.asarray(a).reshape(-1, a.shape[-1])
    a = a[np.linalg.norm(a, axis=1) > 1e-8]
    if len(a) == 0:
        return float("nan")
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
ARMS = (("current", 0.0, None), ("current+adapt", 0.0, 2.0),
        ("decorrelating", 1.0, None), ("decorr+adapt", 1.0, 2.0))
print(f"E105b -- where the bottleneck is. {SEEDS} seeds\n")
print(f"{'arm':>15}{'stub stab':>11}{'cortical':>10}"
      f"{'| planted stub':>16}{'planted cort':>14}{'shuffled cort':>15}")

for label, decorr, adapt in ARMS:
    cfg = BASE._replace(sensory_lateral=(1.0 if adapt is not None else 0.0),
                        sensory_adapt_tau_s=adapt)
    pc = PC._replace(readout_decorrelate=decorr)
    ms, cs, ps_, pc_, sc = [], [], [], [], []
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
        ms.append(stability(stub))
        cs.append(stability(cort))

        # Positive control. Same mean, same per-step magnitude, deliberately varied
        # DIRECTION: rotate each step's stub toward an independent random unit vector.
        # If the readout can express a state-dependent output at all, this must produce
        # one -- and if it does not, the input is not the bottleneck.
        a = np.asarray(stub)                       # (T, H, K)
        mean = a.mean(0, keepdims=True)
        rng = np.random.default_rng(s)
        noise = rng.normal(size=a.shape).astype(np.float32)
        dev = a - mean
        # Match the deviation energy the real trajectory has, so only the DIRECTION
        # differs from the real input, not how far it moves.
        noise *= np.linalg.norm(dev, axis=-1, keepdims=True) / (
            np.linalg.norm(noise, axis=-1, keepdims=True) + 1e-12)
        planted = mean + noise
        w_out = np.asarray(p2.W_out)
        cort_p = np.einsum("hmk,thk->thm", w_out, planted)
        ps_.append(stability(planted))
        pc_.append(stability(cort_p))
        # And a null for the control itself: the real deviations, shuffled in time.
        # Same distribution of inputs, no relationship to the state -- so if the
        # planted number differs from this one, it is the direction variety that did it.
        idx = rng.permutation(a.shape[0])
        cort_s = np.einsum("hmk,thk->thm", w_out, mean + dev[idx])
        sc.append(stability(cort_s))
    print(f"{label:>15}{np.mean(ms):>11.4f}{np.mean(cs):>10.4f}"
          f"{np.mean(ps_):>16.4f}{np.mean(pc_):>14.4f}{np.mean(sc):>15.4f}")

print("\nE103 reference: motor stub 0.9930 untrained / 0.9925 reared; pallium 0.9934")
print("\nreading it:")
print("  stub stab ~0.99 in every arm  -> nothing tested moved the readout's INPUT")
print("  planted cortical << real      -> the readout CAN vary; the input is the")
print("                                   bottleneck, and E105's line is closed")
print("  planted cortical ~ real       -> the readout cannot vary whatever it is fed,")
print("                                   and the fault is in W_out after all")
print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
