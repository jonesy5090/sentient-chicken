"""Design-informing diagnostic for E085. Not a pre-registered experiment.

E084 recommended choosing the target feeder per seed from an independent baseline run.
That assumes clump location is reproducible across runs. Two things to establish before
pre-registering anything:

1. Is clump location driven by the WORLD key (initial positions) or the RUN key
   (dynamics RNG)? If a target selected from an independent run is not where the flock
   goes in the test run, the repair does not work.

2. Occupancy-fraction inherits clump location. `mean dwell per visit` does not -- it
   conditions on her having arrived, and it measures leaving directly, which is what the
   hypothesis is actually about. Does it have usable spread and a sane distribution?
"""
import sys, time
sys.path.insert(0, 'scratchpad')
import jax, jax.numpy as jnp, numpy as np
from functools import partial
import e083_leaving_anchor as E
from coop import world
from hen import brain, connectome, plasticity, regions
from run import simulate

CEN, CFG, HENS, STEPS = E.CEN, E.CFG, E.HENS, E.STEPS
CENJ = jnp.asarray(CEN, dtype=jnp.float32)
R = 3.33


@partial(jax.jit, static_argnames=("cfg", "pc", "n"))
def run(w, x, p, ps, key, cfg, pc, n):
    def step(carry, _):
        carry, _o = simulate._one_step(carry, None, cfg=cfg, pc=pc)
        pos = carry[0].pos
        d = jnp.linalg.norm(pos[:, None, :] - CENJ[None, :, :], axis=-1)   # (H,25)
        return carry, (d < R)                                             # (H,25) bool
    return jax.lax.scan(step, (w, x, p, ps, key), None, length=n)[1]


def dwell_stats(inside, cell):
    """Mean steps per contiguous visit to `cell`, pooled over hens."""
    a = np.asarray(inside[:, :, cell])            # (steps, hens)
    runs = []
    for h in range(a.shape[1]):
        col = a[:, h].astype(np.int8)
        d = np.diff(np.concatenate([[0], col, [0]]))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        runs.extend((ends - starts).tolist())
    runs = np.array(runs) if runs else np.array([0])
    return len(runs), runs.mean() * CFG.dt, runs.sum() / a.size


pc = plasticity.PlasticConfig(**E.FROZEN, pred_gain=0.0)
t0 = time.perf_counter()
print("design diagnostic -- is clump location reproducible, and does dwell behave?\n")
print("occupancy vector correlation between run pairs (25 cells, Pearson):")
print(f"{'seed':>5}{'same world, diff run':>23}{'diff world, diff run':>23}")
occ_store = {}
for s in range(4):
    k = jax.random.key(s)
    p = connectome.build(jax.random.fold_in(k, 1), regions.DEFAULT_REGIONS, n_hens=HENS,
                         gakel_scaffold=True, shared_place_map=True)
    outs = {}
    for tag, wk, rk in (("A", k, jax.random.fold_in(k, 2)),
                        ("B", k, jax.random.fold_in(k, 9)),          # same world, new run
                        ("C", jax.random.fold_in(k, 6), jax.random.fold_in(k, 7))):
        w = world.reset(wk, CFG)
        w = w._replace(food_pos=jnp.asarray(np.stack([CEN[E.P], CEN[E.P2]]),
                                            dtype=jnp.float32))
        x = brain.initial_state(p, HENS); ps = plasticity.initial_state(p, HENS, pc)
        outs[tag] = np.asarray(run(w, x, p, ps, rk, CFG, pc, STEPS))
    occ = {t: outs[t].mean(axis=(0, 1)) for t in outs}
    occ_store[s] = (occ, outs)
    print(f"{s:>5}{np.corrcoef(occ['A'], occ['B'])[0,1]:>23.3f}"
          f"{np.corrcoef(occ['A'], occ['C'])[0,1]:>23.3f}")

print(f"\ntarget = most-occupied cell in run A; measured in run B (same world, new run)")
print(f"{'seed':>5}{'target':>8}{'occ in A':>10}{'occ in B':>10}{'visits B':>10}"
      f"{'mean dwell B (s)':>18}")
for s in range(4):
    occ, outs = occ_store[s]
    tgt = int(np.argmax(occ['A']))
    n_v, dwell, frac = dwell_stats(outs['B'], tgt)
    print(f"{s:>5}{tgt:>8}{occ['A'][tgt]:>10.3f}{occ['B'][tgt]:>10.3f}{n_v:>10d}"
          f"{dwell:>18.2f}")
print(f"\nwall clock: {time.perf_counter()-t0:.0f} s")
print("repair works if occ in B stays high at a target picked from A, and if visits B")
print("is large enough that mean dwell is estimated from many visits rather than one.")
