"""E107 verification 3: does E106's headline survive the per-hen statistic?

E106 reported pooled cortical stability finally moving, 0.9587 -> 0.8428 -> 0.5735, and
called it the first movement in the project's history. Verification 1 has now shown the
pooled cortical statistic is a between-hen quantity. The review's reading is that
E106's number fell because |cortical| collapsed 99%, leaving the direction to whatever
per-hen residual survives -- one measurement and its cause, read as two findings.

So: every E106 arm, both statistics, plus the between-hen column that diagnoses which
is which. The pallium and motor stub are included because for those stages pooled and
per-hen agreed at baseline, so E106's representation result should stand or fall on its
own rather than with the cortical one.
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


def _stab(a):
    a = np.asarray(a)
    a = a[np.linalg.norm(a, axis=1) > 1e-8]
    if len(a) == 0:
        return float("nan")
    m = a.mean(0)
    m /= np.linalg.norm(m) + 1e-12
    return float(((a @ m) / (np.linalg.norm(a, axis=1) + 1e-12)).mean())


def pooled(a):
    a = np.asarray(a)
    return _stab(a.reshape(-1, a.shape[-1]))


def per_hen(a):
    a = np.asarray(a)
    return float(np.mean([_stab(a[:, h]) for h in range(a.shape[1])]))


def between_hen(a):
    a = np.asarray(a)
    m = np.stack([a[:, h].mean(0) for h in range(a.shape[1])])
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
    g = m.mean(0)
    g /= np.linalg.norm(g) + 1e-12
    return float(np.mean(m @ g))


@partial(jax.jit, static_argnames=("cfg", "n"))
def probe(w, x, p, key, cfg, n):
    n_motor = p.W_out.shape[-1]

    def step(c, _):
        w, x, key = c
        key, kw = jax.random.split(key)
        obs = sensing.observe(w, cfg)
        x, motor, d = brain.step(x, obs, p, cfg.dt,
                                 sensory_lateral=cfg.sensory_lateral,
                                 recurrent_lateral=cfg.recurrent_lateral)
        raw = neurons.rate(x)
        seen = (neurons.pooled(raw, p.region_pools, cfg.recurrent_lateral)
                if cfg.recurrent_lateral else raw)
        return (world.step(w, motor, kw, cfg), x, key), (
            seen[:, P_LO:P_HI], seen[:, -n_motor:], d.cortical, motor)
    return jax.lax.scan(step, (w, x, key), None, length=n)[1]


ARMS = (("A baseline", 0.0, 0.0), ("D interneuron 1.0", 1.0, 0.0),
        ("E + sensory 1.0", 1.0, 1.0))
t0 = time.perf_counter()
print(f"E107 verification 3 -- E106's arms, per hen. {SEEDS} seeds\n")
print(f"{'arm':>19}{'stage':>13}{'POOLED':>9}{'PER-HEN':>10}{'between':>10}{'|cort|':>9}")
for label, rec, sens in ARMS:
    cfg = BASE._replace(recurrent_lateral=rec, sensory_lateral=sens)
    acc = {s: ([], [], []) for s in ("pallium", "motor stub", "CORTICAL", "motor out")}
    mag = []
    for s in range(SEEDS):
        k = jax.random.key(s)
        p0 = connectome.build(jax.random.fold_in(k, 1), reg, n_hens=16)
        w = world.reset(k, cfg)
        x = brain.initial_state(p0, 16)
        st = plasticity.initial_state(p0, 16, PC)
        _w, _x, p2, _ps, _k, _t = simulate.rollout(
            w, x, p0, jax.random.fold_in(k, 2), cfg, REAR, pc=PC, ps=st)
        out = probe(world.reset(k, cfg), brain.initial_state(p2, 16), p2,
                    jax.random.fold_in(k, 5), cfg, PROBE)
        for name, a in zip(("pallium", "motor stub", "CORTICAL", "motor out"), out):
            acc[name][0].append(pooled(a))
            acc[name][1].append(per_hen(a))
            acc[name][2].append(between_hen(a))
        mag.append(float(np.mean(np.abs(np.asarray(out[2])))))
    for i, name in enumerate(("pallium", "motor stub", "CORTICAL", "motor out")):
        p_, h_, b_ = (np.mean(v) for v in acc[name])
        head = label if i == 0 else ""
        tail = f"{np.mean(mag):>9.4f}" if name == "CORTICAL" else ""
        print(f"{head:>19}{name:>13}{p_:>9.4f}{h_:>10.4f}{b_:>10.4f}{tail}")
    print()

print("E106 published (pooled): baseline cortical 0.9587, D 0.8428, E 0.5735;")
print("motor stub 0.9925 -> 0.7400 -> 0.6733; pallium 0.9927 -> 0.7105 -> 0.6797.")
print("\nreading it: if PER-HEN pallium and motor stub still fall, E106's representation")
print("result is real. If PER-HEN cortical is flat near 1.0 in every arm while the")
print("pooled number moves, E106's headline was the magnitude collapse, not a finding.")
print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
