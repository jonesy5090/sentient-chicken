"""E105: does decorrelating the readout stop the collapse, and does the input have to vary?

2x2: {current rule, decorrelating rule} x {no temporal adaptation, adaptation}.
Deliberately not stacked -- E089's lesson, and E104's direct evidence that the spatial
half of the sensory fix improved the representation while leaving the readout worse.
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
DECORR = 1.0
ADAPT_TAU = 2.0     # seconds; slow against a 0.01 s step, fast against a 30 min rearing
reg = regions.DEFAULT_REGIONS
S_LO, S_HI = reg.bounds(regions.SENSORY)


def stability(a):
    """Mean cosine to the arm's own mean direction. 1.0 = one fixed pattern."""
    a = np.asarray(a).reshape(-1, a.shape[-1])
    a = a[np.linalg.norm(a, axis=1) > 1e-8]
    if len(a) == 0:
        return float("nan")
    m = a.mean(0)
    m /= np.linalg.norm(m) + 1e-12
    return float(((a @ m) / (np.linalg.norm(a, axis=1) + 1e-12)).mean())


def dc_share(a):
    a = np.asarray(a).reshape(-1, a.shape[-1])
    return float(np.linalg.norm(a.mean(0))
                 / max(np.linalg.norm(a, axis=1).mean(), 1e-12))


def rank_stats(w_out):
    """Top-1 energy share and effective rank (exp of the spectral entropy)."""
    w = np.asarray(w_out)
    t1, er = [], []
    for h in range(w.shape[0]):
        s = np.linalg.svd(w[h], compute_uv=False) ** 2
        q = s / max(s.sum(), 1e-30)
        t1.append(q[0])
        er.append(np.exp(-(q * np.log(q + 1e-12)).sum()))
    return float(np.mean(t1)), float(np.mean(er))


@partial(jax.jit, static_argnames=("cfg", "n"))
def probe(w, x, p, adapt_bar, key, cfg, n):
    """Silent replay: no noise, no plasticity. Adaptation state still advances --
    it is state, not learning, and freezing it would measure the pathway in a
    different regime from the one it was reared in (E098)."""
    def step(c, _):
        w, x, adapt_bar, key = c
        key, kw = jax.random.split(key)
        obs = sensing.observe(w, cfg)
        ab = adapt_bar if cfg.sensory_adapt_tau_s is not None else None
        x, motor, d = brain.step(x, obs, p, cfg.dt,
                                 sensory_lateral=cfg.sensory_lateral,
                                 adapt_bar=ab)
        if cfg.sensory_adapt_tau_s is not None:
            a_a = cfg.dt / cfg.sensory_adapt_tau_s
            adapt_bar = adapt_bar + a_a * (d.current - adapt_bar)
        r = neurons.rate(x)
        return (world.step(w, motor, kw, cfg), x, adapt_bar, key), (
            obs, r[:, S_LO:S_HI], d.cortical, d.reflex)
    return jax.lax.scan(step, (w, x, adapt_bar, key), None, length=n)[1]


t0 = time.perf_counter()
print(f"E105 -- decorrelating readout x temporal adaptation. {SEEDS} seeds, "
      f"{REAR * BASE.dt / 60:.0f} min rearing\n")
print(f"{'rule':>14}{'adapt':>8}{'top-1':>9}{'eff rank':>10}{'cort stab':>11}"
      f"{'|cort|':>9}{'stub stab':>11}{'stub DC%':>10}{'obs stab':>10}")

untrained_ref = None
for decorr in (0.0, DECORR):
    for adapt in (None, ADAPT_TAU):
        cfg = BASE._replace(sensory_lateral=(1.0 if adapt is not None else 0.0),
                            sensory_adapt_tau_s=adapt)
        pc = PC._replace(readout_decorrelate=decorr)
        t1s, ers, cs, cm, ss, dc, os_ = [], [], [], [], [], [], []
        u_t1s, u_ers = [], []
        for s in range(SEEDS):
            k = jax.random.key(s)
            p0 = connectome.build(jax.random.fold_in(k, 1), reg, n_hens=16)
            w = world.reset(k, cfg)
            x = brain.initial_state(p0, 16)
            ps = plasticity.initial_state(p0, 16, pc)
            a, b = rank_stats(p0.W_out)
            u_t1s.append(a)
            u_ers.append(b)
            _w, _x, p2, ps2, _k, _t = simulate.rollout(
                w, x, p0, jax.random.fold_in(k, 2), cfg, REAR, pc=pc, ps=ps)
            w3 = world.reset(k, cfg)
            x3 = brain.initial_state(p2, 16)
            obs, stub, cort, refl = probe(
                w3, x3, p2, ps2.adapt_bar, jax.random.fold_in(k, 5), cfg, PROBE)
            a, b = rank_stats(p2.W_out)
            t1s.append(a)
            ers.append(b)
            cs.append(stability(cort))
            cm.append(float(np.mean(np.abs(np.asarray(cort)))))
            ss.append(stability(stub))
            dc.append(dc_share(stub))
            os_.append(stability(obs))
        if untrained_ref is None:
            untrained_ref = (np.mean(u_t1s), np.mean(u_ers))
        label = "decorrelating" if decorr else "current"
        print(f"{label:>14}{('on' if adapt else 'off'):>8}"
              f"{100 * np.mean(t1s):>8.1f}%{np.mean(ers):>10.2f}"
              f"{np.mean(cs):>11.4f}{np.mean(cm):>9.4f}"
              f"{np.mean(ss):>11.4f}{100 * np.mean(dc):>9.1f}%{np.mean(os_):>10.4f}")

print(f"\nuntrained reference (this seed block): top-1 {100*untrained_ref[0]:.1f}%, "
      f"eff rank {untrained_ref[1]:.2f}")
print("E100/E104 reference: cortical stability reared 0.9587 (lateral 0), 0.9846 (lateral 1)")
print("\n--- pre-registered falsifiers (E105 section 4) ---")
print("primary     cortical stability >= 0.90 in EVERY arm -> the collapse is not")
print("            attributable to the rule either; record it as architectural")
print("rank        eff rank does not exceed 2.5 under decorrelation -> mechanism inert")
print("degeneracy  rank rises but mean |cortical| falls >50% vs the current-rule arm")
print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
