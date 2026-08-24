"""E106c: fresh seeds, and the control the welfare result needs.

Two jobs.

**Replication.** E021's rule: no status changes on one seed block. E106's representation
numbers and E106b's hunger result both came off seeds 0-7; this re-measures them on
seeds 8-15 and 4-7 respectively.

**The control that matters.** Under the interneuron the cortical pathway's magnitude
collapses 99% (|cortical| 1.606 -> 0.020) and the mean motor output falls 0.52 -> 0.32.
A hen who does less might simply burn less energy, which would make the hunger
improvement a fact about vigour rather than about the representation -- the same
mundane mechanism that made E101's gate a degenerate win. So: a hen with **no cortical
pathway at all** (`readout_scale=0`, `eta_out=0`), which is the limit of "the learned
pathway went quiet". If she feeds as well as the interneuron hen, the benefit is
silence. If the interneuron beats her, something is being read.
"""
import time
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, neurons, plasticity, regions
from run import simulate

BASE = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=60.0)
REAR, PROBE = int(30 * 60 / BASE.dt), int(2 * 60 / BASE.dt)
PC = plasticity.PlasticConfig(enabled=True, hebbian_readout=True,
                              readout_scaling_strength=0.3)
reg = regions.DEFAULT_REGIONS
P_LO, P_HI = reg.bounds(regions.PALLIUM)


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
    n_motor = p.W_out.shape[-1]

    def step(c, _):
        w, x, key = c
        key, kw = jax.random.split(key)
        obs = sensing.observe(w, cfg)
        x, motor, d = brain.step(x, obs, p, cfg.dt,
                                 recurrent_lateral=cfg.recurrent_lateral)
        raw = neurons.rate(x)
        seen = (neurons.pooled(raw, p.region_pools, cfg.recurrent_lateral)
                if cfg.recurrent_lateral else raw)
        return (world.step(w, motor, kw, cfg), x, key), (
            seen[:, P_LO:P_HI], seen[:, -n_motor:], d.cortical)
    return jax.lax.scan(step, (w, x, key), None, length=n)[1]


def rear(seed, cfg, pc, readout_scale=0.05):
    k = jax.random.key(seed)
    p0 = connectome.build(jax.random.fold_in(k, 1), reg, n_hens=16,
                          readout_scale=readout_scale)
    w = world.reset(k, cfg)
    x = brain.initial_state(p0, 16)
    st = plasticity.initial_state(p0, 16, pc)
    return k, p0, w, x, st


t0 = time.perf_counter()
print("E106c -- fresh seeds, and the control the welfare result needs\n")

# --- representation, fresh block (seeds 4-7) --------------------------------
print("representation, seeds 4-7 (E106 measured seeds 0-3)")
print(f"{'arm':>18}{'pallium':>10}{'motor stub':>12}{'cortical':>10}{'|cort|':>9}")
for label, rec in (("baseline", 0.0), ("interneuron 1.0", 1.0)):
    cfg = BASE._replace(recurrent_lateral=rec)
    pal, stb, crt, mag = [], [], [], []
    for s in range(4, 8):
        k, p0, w, x, st = rear(s, cfg, PC)
        _w, _x, p2, _ps, _k, _t = simulate.rollout(
            w, x, p0, jax.random.fold_in(k, 2), cfg, REAR, pc=PC, ps=st)
        a, b, c = probe(world.reset(k, cfg), brain.initial_state(p2, 16), p2,
                        jax.random.fold_in(k, 5), cfg, PROBE)
        pal.append(stability(a))
        stb.append(stability(b))
        crt.append(stability(c))
        mag.append(float(np.mean(np.abs(np.asarray(c)))))
    print(f"{label:>18}{np.mean(pal):>10.4f}{np.mean(stb):>12.4f}"
          f"{np.mean(crt):>10.4f}{np.mean(mag):>9.4f}")
print("  seeds 0-3 gave: baseline 0.9927 / 0.9925 / 0.9587 / 1.606,"
      " interneuron 0.7105 / 0.7400 / 0.8428 / 0.020")

# --- welfare, fresh block (seeds 8-15), with the silence control ------------
print(f"\nwelfare, seeds 8-15 (E106b measured seeds 0-7)")
print(f"{'arm':>22}{'hunger':>10}{'caught/dive':>13}{'vigour':>11}")
ARMS = (("baseline", 0.0, 0.05, PC),
        ("interneuron 1.0", 1.0, 0.05, PC),
        ("no cortical pathway", 0.0, 0.0, PC._replace(eta_out=0.0)))
hung, caught = {}, {}
for label, rec, rs, pc in ARMS:
    cfg = BASE._replace(recurrent_lateral=rec)
    h, c, mo = [], [], []
    for s in range(8, 16):
        k, p0, w, x, st = rear(s, cfg, pc, readout_scale=rs)
        w2, _x, _p, _ps, _k = simulate.rollout_quiet(
            w, x, p0, jax.random.fold_in(k, 2), cfg, REAR, st, pc)
        h.append(float(np.mean(np.asarray(w2.hunger))))
        c.append(float(np.sum(np.asarray(w2.n_caught_any))
                       / max(float(np.sum(np.asarray(w2.n_dives))), 1.0)))
        # `vigour` is VOCAL energy, not motor output -- 1.0 is rested, 0.0 is a hen
        # who has been calling flat out. Reported because the difference turned out to
        # be large, not because it was planned.
        mo.append(float(np.mean(np.asarray(w2.vigour))))
    hung[label], caught[label] = np.array(h), np.array(c)
    print(f"{label:>22}{np.mean(h):>10.4f}{np.mean(c):>13.4f}{np.mean(mo):>11.4f}")


def paired(a, b, name):
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / (se + 1e-12)
    print(f"    {name:<44}{d.mean():+.4f} +/- {se:.4f}  t={t:+.2f}"
          f"{'  SIGNIFICANT' if abs(t) > 2.365 else ''}")


print("\n  paired, df=7, crit 2.365 (lower hunger is better):")
paired(hung["interneuron 1.0"], hung["baseline"], "interneuron vs baseline (hunger)")
paired(hung["no cortical pathway"], hung["baseline"],
       "silence control vs baseline (hunger)")
paired(hung["interneuron 1.0"], hung["no cortical pathway"],
       "interneuron vs SILENCE CONTROL (hunger)  <-- the test")
paired(caught["interneuron 1.0"], caught["baseline"],
       "interneuron vs baseline (caught/dive)")
print("\n  seeds 0-7 gave: hunger -0.1009 (t=-3.45), caught/dive -0.0092 (t=-0.32)")
print("  reading it: if the interneuron does not beat the silence control, the")
print("  hunger benefit is 'the cortical pathway went quiet' and not the representation.")
print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
