"""E114c: which channels carry the gate's benefit? Asked without touching the world.

E114 and E114b both foundered on the same thing: `head_down` blinding dominates predation
so completely that weakening it floors the metric. E114b's sweep shows a 25% reduction in
blinding more than halves baseline predation (0.1955 -> 0.0875) and a 50% reduction
essentially abolishes it (0.0094). Only two settings are usable and the baseline moves
between them, so the absolute gate effect cannot be compared cleanly.

The better question is answerable inside the brain, with the world left alone. E102's
claim is about WHICH channels the gate closes. So open them selectively:

  * PECK only   -- the one closed channel that does blind her
  * TURNS only  -- the two closed channels that do not
  * SCRATCH     -- the blinding channel the gate SPARES; closing it is what E102's own
                   story predicts she should have learned to do, and she did not

If the benefit tracks PECK, vigilance is the mechanism and E102 is right for one channel
out of three. If it tracks the turns, it is something else entirely.
"""
import os
import time
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from coop import spec, world
from hen import brain, connectome, plasticity, regions
from run import simulate

CFG = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=60.0)
REAR, TEST = int(30 * 60 / CFG.dt), int(10 * 60 / CFG.dt)
SEED0, SEEDS = int(os.environ.get("E114C_SEED0", "0")), 8
BG = plasticity.PlasticConfig(enabled=True, hebbian_readout=True,
                              readout_scaling_strength=0.3, bg_gate=True)
OFF = plasticity.PlasticConfig(enabled=False, bg_gate=True)
reg = regions.DEFAULT_REGIONS
TURNS = (spec.M_TURN_L, spec.M_TURN_R)

# Each arm names the channels whose learned gate rows are KEPT; every other row is
# zeroed, which opens that channel to sigmoid(GATE_OPEN_BIAS) = 0.982.
ARMS = {
    "learned (all)": None,
    "PECK only": (spec.M_PECK,),
    "TURNS only": TURNS,
    "PECK+TURNS": (spec.M_PECK,) + TURNS,
    "SCRATCH too": (spec.M_PECK, spec.M_SCRATCH) + TURNS,
    "none": (),
}


def keep(w_str, channels):
    if channels is None:
        return w_str
    mask = np.zeros(w_str.shape[1], dtype=np.float32)
    for c in channels:
        mask[c] = 1.0
    return w_str * jnp.asarray(mask)[None, :, None]


@partial(jax.jit, static_argnames=("cfg", "pc", "n"))
def test(w, x, p, ps, key, cfg, pc, n):
    def step(c, _):
        c, _o = simulate._one_step(c, None, cfg, pc)
        return c, jnp.mean(c[0].head_down)
    (w, x, p, ps, key), hd = jax.lax.scan(step, (w, x, p, ps, key), None, length=n)
    return w, jnp.mean(hd)


t0 = time.perf_counter()
print(f"E114c -- which channels carry the benefit? seeds {SEED0}-{SEED0+SEEDS-1}\n")
print(f"HEAD_DOWN_ACTIONS = (PECK, SCRATCH). The learned gate closes PECK and both "
      f"TURNS,\nand spares SCRATCH. World untouched in every arm.\n")

res = {k: {"cd": [], "hd": []} for k in ARMS}
for s in range(SEED0, SEED0 + SEEDS):
    k = jax.random.key(s)
    p0 = connectome.build(jax.random.fold_in(k, 1), reg, n_hens=16)
    _w, _x, p, ps, _k, _t = simulate.rollout(
        world.reset(k, CFG), brain.initial_state(p0, 16), p0,
        jax.random.fold_in(k, 2), CFG, REAR, pc=BG,
        ps=plasticity.initial_state(p0, 16, BG))
    for name, chans in ARMS.items():
        pp = p._replace(W_str=keep(p.W_str, chans))
        wf, h = test(world.reset(k, CFG), brain.initial_state(pp, 16), pp, ps,
                     jax.random.fold_in(k, 7), CFG, OFF, TEST)
        d = float(jnp.sum(wf.n_dives))
        res[name]["cd"].append(float(jnp.sum(wf.n_caught_any)) / max(d, 1))
        res[name]["hd"].append(float(h))

print(f"{'arm':>16}{'caught/dive':>13}{'head-down':>11}{'vs none':>10}{'t':>8}")
none = np.array(res["none"]["cd"])
for name in ARMS:
    a = np.array(res[name]["cd"])
    d = a - none
    se = d.std(ddof=1) / np.sqrt(len(d)) if name != "none" else 0.0
    t = d.mean() / (se + 1e-12) if name != "none" else 0.0
    print(f"{name:>16}{a.mean():>13.4f}{np.mean(res[name]['hd']):>11.3f}"
          f"{d.mean():>+10.4f}{t:>8.2f}")

full = np.array(res["learned (all)"]["cd"]) - none
print(f"\n  share of the full gate effect ({full.mean():+.4f}) recovered by each subset:")
for name in ("PECK only", "TURNS only", "PECK+TURNS", "SCRATCH too"):
    d = (np.array(res[name]["cd"]) - none).mean()
    print(f"    {name:>14}  {100*d/full.mean():>5.0f}%")
print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
