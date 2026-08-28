"""E114b: the graded version, because the pre-registered design hit a floor.

E114 set `head_down_blinds` to 0.0 and predation went to EXACTLY 0.0000 in both gate
conditions -- removing the vigilance gate removes all predation, so there was no room
for a gate effect to appear in. The primary falsifier fired and printed "vigilance IS the
mechanism", which is not a finding; it is a floored dependent variable. CLAUDE.md's rule
5, walked into with the falsifier written and the check omitted.

The fix is a graded manipulation. Sweep `head_down_blinds` and watch whether the gate's
advantage scales with it. If the gate works through vigilance, its benefit should vanish
as blinding is removed -- and at intermediate settings predation is still off the floor,
so that is observable.

Also restructured: rear ONCE per seed and test every condition on the same brain, which
is both faster and a better-controlled comparison than E114's per-cell rearing.
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
SEED0, SEEDS = int(os.environ.get("E114B_SEED0", "0")), 8
BLINDS = (1.0, 0.75, 0.5, 0.25, 0.0)
BG = plasticity.PlasticConfig(enabled=True, hebbian_readout=True,
                              readout_scaling_strength=0.3, bg_gate=True)
OFF = plasticity.PlasticConfig(enabled=False, bg_gate=True)
reg = regions.DEFAULT_REGIONS


@partial(jax.jit, static_argnames=("cfg", "pc", "n"))
def test(w, x, p, ps, key, cfg, pc, n):
    """Scalars only -- E114 emitted per-step arrays and spent 33 minutes on it."""
    def step(c, _):
        c, _o = simulate._one_step(c, None, cfg, pc)
        return c, jnp.mean(c[0].head_down)
    (w, x, p, ps, key), hd = jax.lax.scan(step, (w, x, p, ps, key), None, length=n)
    return w, jnp.mean(hd)


t0 = time.perf_counter()
print(f"E114b -- graded blinding. seeds {SEED0}-{SEED0+SEEDS-1}\n")
print("E114 set blinds=0.0 and caught/dive went to 0.0000 in BOTH arms: a floor, not a")
print("result. This sweeps it instead.\n")

cd = {(b, g): [] for b in BLINDS for g in (True, False)}
hd = {(b, g): [] for b in BLINDS for g in (True, False)}
for s in range(SEED0, SEED0 + SEEDS):
    k = jax.random.key(s)
    p0 = connectome.build(jax.random.fold_in(k, 1), reg, n_hens=16)
    # One rearing per seed, at the default world. Every cell below tests the SAME
    # learned gate, so the only thing varying is the world.
    _w, _x, p, ps, _k, _t = simulate.rollout(
        world.reset(k, CFG), brain.initial_state(p0, 16), p0,
        jax.random.fold_in(k, 2), CFG, REAR, pc=BG,
        ps=plasticity.initial_state(p0, 16, BG))
    p_off = p._replace(W_str=jnp.zeros_like(p.W_str))
    for b in BLINDS:
        cfg = CFG._replace(head_down_blinds=b)
        for gate_on in (True, False):
            pp = p if gate_on else p_off
            wf, h = test(world.reset(k, cfg), brain.initial_state(pp, 16), pp, ps,
                         jax.random.fold_in(k, 7), cfg, OFF, TEST)
            d = float(jnp.sum(wf.n_dives))
            cd[(b, gate_on)].append(float(jnp.sum(wf.n_caught_any)) / max(d, 1))
            hd[(b, gate_on)].append(float(h))

print(f"{'blinds':>8}{'caught/dive on':>16}{'off':>9}{'gate effect':>13}"
      f"{'t':>8}{'head-down on/off':>20}")
effects = {}
for b in BLINDS:
    a = np.array(cd[(b, True)])
    c = np.array(cd[(b, False)])
    d = a - c
    se = d.std(ddof=1) / np.sqrt(len(d))
    effects[b] = (d.mean(), se)
    print(f"{b:>8.2f}{a.mean():>16.4f}{c.mean():>9.4f}{d.mean():>+13.4f}"
          f"{d.mean()/(se+1e-12):>8.2f}"
          f"{np.mean(hd[(b,True)]):>11.3f}/{np.mean(hd[(b,False)]):.3f}")

print("\n--- reading it ---")
floors = [b for b in BLINDS if np.mean(cd[(b, False)]) < 0.02]
print(f"floored cells (no-gate caught/dive < 0.02): "
      f"{floors if floors else 'none'} -- these cannot show a gate effect either way")
usable = [b for b in BLINDS if b not in floors]
print(f"usable settings: {usable}")
if len(usable) >= 2:
    hi, lo = max(usable), min(usable)
    e_hi, e_lo = effects[hi][0], effects[lo][0]
    print(f"gate effect at blinds={hi:.2f}: {e_hi:+.4f}   "
          f"at blinds={lo:.2f}: {e_lo:+.4f}")
    print(f"retained at the weakest usable blinding: "
          f"{100*e_lo/e_hi if e_hi else float('nan'):.0f}%")
    print(f"-> {'vigilance IS the mechanism' if abs(e_lo) < 0.5*abs(e_hi) else 'vigilance is NOT the whole mechanism'}")
print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
