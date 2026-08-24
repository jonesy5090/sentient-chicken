"""E106: does a pooled interneuron in the pallium and motor stub expose the signal?

E106a established the signal is there: de-meaning the populations post hoc takes the
motor stub from 0.9925 to 0.7443 and the reared readout's output from 0.9587 to 0.7699.
This puts the interneuron in the loop and asks whether the running model finds it -- and
whether the flock survives having one, which section 4 names as the likelier failure.
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

ARMS = (("A baseline", 0.0, 0.0, False),
        ("B balanced_ei", 0.0, 0.0, True),
        ("C interneuron 0.5", 0.5, 0.0, False),
        ("D interneuron 1.0", 1.0, 0.0, False),
        ("E + sensory 1.0", 1.0, 1.0, False))


def stability(a):
    a = np.asarray(a).reshape(-1, a.shape[-1])
    a = a[np.linalg.norm(a, axis=1) > 1e-8]
    if len(a) == 0:
        return float("nan")
    m = a.mean(0)
    m /= np.linalg.norm(m) + 1e-12
    return float(((a @ m) / (np.linalg.norm(a, axis=1) + 1e-12)).mean())


@partial(jax.jit, static_argnames=("cfg", "n"))
def probe(w, x, p, key, cfg, n):
    """Silent replay: no noise, no plasticity. Reads what the READOUT reads, which
    under `recurrent_lateral` is the pooled rate rather than the raw one -- measuring
    the raw rate here would be the fifth instance in this project of reading a quantity
    in a different regime from the one it is used in."""
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
            seen[:, P_LO:P_HI], seen[:, -n_motor:], d.cortical,
            jnp.mean(raw[:, P_LO:P_HI]), jnp.mean(motor, axis=(0, 1)))
    return jax.lax.scan(step, (w, x, key), None, length=n)[1]


t0 = time.perf_counter()
print(f"E106 -- an interneuron in the pallium and the motor stub. {SEEDS} seeds\n")
print(f"{'arm':>19}{'pallium':>10}{'motor stub':>12}{'CORTICAL':>10}"
      f"{'|cort|':>9}{'pal rate':>10}{'motor out':>11}")

results = {}
for label, rec, sens, bal in ARMS:
    cfg = BASE._replace(recurrent_lateral=rec, sensory_lateral=sens)
    pal, stb, crt, mag, rate_, mot = [], [], [], [], [], []
    for s in range(SEEDS):
        k = jax.random.key(s)
        p0 = connectome.build(jax.random.fold_in(k, 1), reg, n_hens=16,
                              balanced_ei=bal)
        w = world.reset(k, cfg)
        x = brain.initial_state(p0, 16)
        st = plasticity.initial_state(p0, 16, PC)
        _w, _x, p2, _ps, _k, _t = simulate.rollout(
            w, x, p0, jax.random.fold_in(k, 2), cfg, REAR, pc=PC, ps=st)
        w3 = world.reset(k, cfg)
        x3 = brain.initial_state(p2, 16)
        a, b, c, r_, m_ = probe(w3, x3, p2, jax.random.fold_in(k, 5), cfg, PROBE)
        pal.append(stability(a))
        stb.append(stability(b))
        crt.append(stability(c))
        mag.append(float(np.mean(np.abs(np.asarray(c)))))
        rate_.append(float(np.mean(np.asarray(r_))))
        mot.append(float(np.mean(np.asarray(m_))))
    results[label] = (np.mean(pal), np.mean(stb), np.mean(crt),
                      np.mean(mag), np.mean(rate_), np.mean(mot))
    print(f"{label:>19}{np.mean(pal):>10.4f}{np.mean(stb):>12.4f}{np.mean(crt):>10.4f}"
          f"{np.mean(mag):>9.4f}{np.mean(rate_):>10.4f}{np.mean(mot):>11.4f}")

base = results["A baseline"]
print("\nE106a post-hoc ceiling: pallium 0.7164, motor stub 0.7443, cortical 0.7699")
print("E105 reference: motor stub 0.9925, cortical 0.9587")
print("\n--- pre-registered falsifiers (E106 section 4) ---")
worst = min(v[1] for v in results.values())
print(f"primary     motor-stub stability >= 0.90 at EVERY strength "
      f"(best achieved {worst:.4f}) -> option A closed")
for label, _, _, _ in ARMS:
    p_, s_, c_, m_, r_, o_ = results[label]
    flags = []
    if s_ < 0.85 and c_ >= 0.90:
        flags.append("CEILING falsifier: representation moved, readout did not")
    if r_ < 0.15 or r_ > 0.95:
        flags.append(f"DEGENERACY: pallial rate {r_:.3f} outside [0.15, 0.95]")
    if m_ < 0.5 * base[3]:
        flags.append(f"DEGENERACY: |cortical| {m_:.3f} vs baseline {base[3]:.3f}")
    if flags:
        print(f"  {label}: " + "; ".join(flags))
print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
