"""E101 verification: is the predation drop real, and is it LEARNED?

Two checks, both of which could kill it.

  FRESH SEEDS  -- 4 seeds is one block. E021 is this project's standing warning: a
                  t=3.84 evaporated to t=0.01 on seeds 12-23 instead of 0-11. Seeds 8-11.

  UNTRAINED GATE -- the gate wired but never reared. W_gate starts at zero, so the gate
                  sits at sigmoid(4.0)=0.982 and should do nothing. If predation drops
                  anyway, the effect is the mechanism's presence, not what it learned.
"""
import time
import jax, jax.numpy as jnp, numpy as np
from coop import spec, world
from hen import brain, connectome, plasticity, regions
from run import simulate

CFG = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=60.0)
REAR, TEST = int(30*60/CFG.dt), int(10*60/CFG.dt)
BASE = dict(enabled=True, hebbian_readout=True, readout_scaling_strength=0.3)
OFF = plasticity.PlasticConfig(**BASE)
GATE = plasticity.PlasticConfig(**BASE, reflex_gate=True)


def measure(pc, seed, rear=True):
    k = jax.random.key(seed)
    p0 = connectome.build(jax.random.fold_in(k, 1), regions.DEFAULT_REGIONS, n_hens=16)
    ps = plasticity.initial_state(p0, 16, pc)
    p2, ps2 = p0, ps
    if rear:
        w = world.reset(k, CFG); x = brain.initial_state(p0, 16)
        _w, _x, p2, ps2, _k, _t = simulate.rollout(
            w, x, p0, jax.random.fold_in(k, 2), CFG, REAR, pc=pc, ps=ps)
    pc_off = plasticity.PlasticConfig(enabled=False, reflex_gate=pc.reflex_gate)
    w3 = world.reset(k, CFG); x3 = brain.initial_state(p2, 16)
    wf, *_ = simulate.rollout(w3, x3, p2, jax.random.fold_in(k, 7), CFG, TEST,
                              pc=pc_off, ps=ps2)
    d = float(jnp.sum(wf.n_dives)); c = float(jnp.sum(wf.n_caught_any))
    return c / max(d, 1), float(jnp.mean(wf.hunger))


t0 = time.perf_counter()
print("FRESH SEED BLOCK (8-11) -- E021's standing warning\n")
print(f"{'arm':>22}{'caught/dive':>13}{'hunger':>9}{'vs off':>11}")
res = {}
for name, pc, rear in (("off (reared)", OFF, True),
                       ("gate (reared)", GATE, True),
                       ("gate, NEVER reared", GATE, False)):
    cd, hu = [], []
    for s in range(8, 12):
        a, b = measure(pc, s, rear)
        cd.append(a); hu.append(b)
    res[name] = (float(np.mean(cd)), float(np.mean(hu)), cd)
    base = res["off (reared)"][0]
    delta = "" if name.startswith("off") else f"{100*(np.mean(cd)-base)/base:+.1f}%"
    print(f"{name:>22}{np.mean(cd):>13.4f}{np.mean(hu):>9.3f}{delta:>11}")

o, g = res["off (reared)"][2], res["gate (reared)"][2]
d = np.array(g) - np.array(o)
se = d.std(ddof=1) / np.sqrt(len(d))
print(f"\npaired (reared gate - off): {d.mean():+.4f} +/- {se:.4f}, "
      f"t={d.mean()/max(se,1e-12):+.2f} vs t(3)=3.182")
print(f"per-seed deltas: {[round(float(v),4) for v in d]}")
print(f"\nwall clock: {time.perf_counter()-t0:.0f} s")
