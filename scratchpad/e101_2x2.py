"""E101: the 2x2 my own control was missing.

The 'untrained gate' arm compared an UNTRAINED connectome against REARED ones, so it
conflated gate-presence with trained-vs-untrained. The cell it never measured is
untrained-and-no-gate, which is the only honest baseline.

2 x 2: {untrained, reared} x {no gate, gate}. 8 seeds, matched.
"""
import time
import jax, jax.numpy as jnp, numpy as np
from coop import spec, world
from hen import brain, connectome, plasticity, regions
from run import simulate

CFG = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=60.0)
import os
REAR, TEST = int(30*60/CFG.dt), int(10*60/CFG.dt)
SEED0 = int(os.environ.get('E101_SEED0', '0'))
SEEDS = 8
BASE = dict(enabled=True, hebbian_readout=True, readout_scaling_strength=0.3)


def cell(reared, gate, seed):
    pc = plasticity.PlasticConfig(**BASE, reflex_gate=gate)
    k = jax.random.key(seed)
    p = connectome.build(jax.random.fold_in(k, 1), regions.DEFAULT_REGIONS, n_hens=16)
    ps = plasticity.initial_state(p, 16, pc)
    if reared:
        w = world.reset(k, CFG); x = brain.initial_state(p, 16)
        _w, _x, p, ps, _k, _t = simulate.rollout(
            w, x, p, jax.random.fold_in(k, 2), CFG, REAR, pc=pc, ps=ps)
    pc_off = plasticity.PlasticConfig(enabled=False, reflex_gate=gate)
    w3 = world.reset(k, CFG); x3 = brain.initial_state(p, 16)
    wf, *_ = simulate.rollout(w3, x3, p, jax.random.fold_in(k, 7), CFG, TEST,
                              pc=pc_off, ps=ps)
    d = float(jnp.sum(wf.n_dives)); c = float(jnp.sum(wf.n_caught_any))
    return c / max(d, 1), float(jnp.mean(wf.hunger))


t0 = time.perf_counter()
R = {}
print(f"seeds {SEED0}-{SEED0+SEEDS-1}\n")
print(f"{'cell':>22}{'caught/dive':>13}{'hunger':>9}")
for reared in (False, True):
    for gate in (False, True):
        cd, hu = [], []
        for s in range(SEED0, SEED0 + SEEDS):
            a, b = cell(reared, gate, s)
            cd.append(a); hu.append(b)
        name = f"{'reared' if reared else 'untrained'}, {'gate' if gate else 'no gate'}"
        R[(reared, gate)] = np.array(cd)
        print(f"{name:>22}{np.mean(cd):>13.4f}{np.mean(hu):>9.3f}")


def paired(a, b, label, crit=2.365):
    d = b - a
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / max(se, 1e-12)
    print(f"  {label:<38}{d.mean():+.4f} +/- {se:.4f}  t={t:+.2f}  "
          f"{'SIGNIFICANT' if abs(t) > crit else 'not significant'}")


print(f"\npaired contrasts, df=7, t crit 2.365:")
paired(R[(False, False)], R[(True, False)], "rearing effect (no gate)")
paired(R[(False, False)], R[(False, True)], "gate effect, untrained brain")
paired(R[(True, False)], R[(True, True)], "gate effect, reared brain")
paired(R[(False, False)], R[(True, True)], "reared+gate vs untrained baseline")
print(f"\nwall clock: {time.perf_counter()-t0:.0f} s")
