"""E107 verification 1: is E100's direction-stability collapse a pooling artefact?

An outside review claims `stability()` -- used unchanged in E100, E103, E104, E105 and
E106 -- reshapes (T, H, D) to (T*H, D) before taking the mean direction, so it pools
sixteen hens who each have their own `W_out`. If each hen's cortical drive sits near her
own private direction, the pooled statistic measures BETWEEN-HEN spread, not
within-hen state-dependence, and the reported 0.6193 -> 0.9587 "collapse" is an artefact.

Written from scratch rather than by patching the reviewer's script. Measures both
statistics on the same trajectories so they cannot disagree for any other reason.
"""
import time
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, neurons, plasticity, regions
from run import simulate

BASE = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=60.0)
REAR, PROBE, SEEDS = int(30 * 60 / BASE.dt), int(2 * 60 / BASE.dt), 4
reg = regions.DEFAULT_REGIONS
S_LO, S_HI = reg.bounds(regions.SENSORY)
P_LO, P_HI = reg.bounds(regions.PALLIUM)


def _stab(a):
    """Direction stability of one (T, D) trajectory: mean cosine to its own mean."""
    a = np.asarray(a)
    keep = np.linalg.norm(a, axis=1) > 1e-8
    a = a[keep]
    if len(a) == 0:
        return float("nan")
    m = a.mean(0)
    m /= np.linalg.norm(m) + 1e-12
    return float(((a @ m) / (np.linalg.norm(a, axis=1) + 1e-12)).mean())


def pooled(a):
    """What every experiment from E100 on actually computed: hens flattened in with
    time before the mean direction is taken."""
    a = np.asarray(a)
    return _stab(a.reshape(-1, a.shape[-1]))


def per_hen(a):
    """Each hen scored against her OWN mean direction, then averaged."""
    a = np.asarray(a)
    return float(np.mean([_stab(a[:, h]) for h in range(a.shape[1])]))


def between_hen(a):
    """How much of the pooled number is hens differing from each other: cosine between
    each hen's mean direction and the grand mean direction."""
    a = np.asarray(a)
    means = np.stack([a[:, h].mean(0) for h in range(a.shape[1])])
    means /= np.linalg.norm(means, axis=1, keepdims=True) + 1e-12
    g = means.mean(0)
    g /= np.linalg.norm(g) + 1e-12
    return float(np.mean(means @ g))


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
            obs, r[:, S_LO:S_HI], r[:, P_LO:P_HI], r[:, -n_motor:],
            d.cortical, d.reflex, motor)
    return jax.lax.scan(step, (w, x, key), None, length=n)[1]


STAGES = ("observation", "sensory stub", "pallium", "motor stub",
          "CORTICAL", "reflex", "motor out")
t0 = time.perf_counter()
print(f"E107 verification -- pooled vs per-hen direction stability. {SEEDS} seeds\n")

for rule_label, pc in (("untrained", None),
                       ("reared, hebbian", plasticity.PlasticConfig(
                           enabled=True, hebbian_readout=True,
                           readout_scaling_strength=0.3)),
                       ("reared, instrumental", plasticity.PlasticConfig(
                           enabled=True))):
    acc = {s: ([], [], []) for s in STAGES}
    for s in range(SEEDS):
        k = jax.random.key(s)
        p0 = connectome.build(jax.random.fold_in(k, 1), reg, n_hens=16)
        p2 = p0
        if pc is not None:
            w = world.reset(k, BASE)
            x = brain.initial_state(p0, 16)
            st = plasticity.initial_state(p0, 16, pc)
            _w, _x, p2, _ps, _k, _t = simulate.rollout(
                w, x, p0, jax.random.fold_in(k, 2), BASE, REAR, pc=pc, ps=st)
        out = probe(world.reset(k, BASE), brain.initial_state(p2, 16), p2,
                    jax.random.fold_in(k, 5), BASE, PROBE)
        for name, a in zip(STAGES, out):
            acc[name][0].append(pooled(a))
            acc[name][1].append(per_hen(a))
            acc[name][2].append(between_hen(a))
    print(f"[{rule_label}]")
    print(f"{'stage':>15}{'POOLED':>10}{'PER-HEN':>10}{'between-hen':>13}")
    for name in STAGES:
        p_, h_, b_ = (np.mean(v) for v in acc[name])
        print(f"{name:>15}{p_:>10.4f}{h_:>10.4f}{b_:>13.4f}")
    print()

print("E100/E103 published (pooled): cortical 0.6193 untrained -> 0.9587 hebbian,")
print("0.9133 instrumental; motor stub 0.9930/0.9925; observation 0.6375/0.6573.")
print("\nreading it: if PER-HEN cortical is ~0.99 both untrained and reared, E100's")
print("'training makes the pathway less state-dependent' is a statement about hens")
print("differing from each other, and E101-E106's primary falsifiers were miscalibrated.")
print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
