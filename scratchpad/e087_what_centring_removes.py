"""E087 follow-up: if the centring's cost is not a timescale, what is it?

E087 falsified the timescale story: lengthening `pred_bar_tau_s` does not recover
decodability, it degrades it, and selectivity degrades with it. So the ~20-point gap
between uncentred `z_lag` (90.0%) and centred (73.7%) is not the baseline tracking dwell.

`z_lag_bar` is per-hen and per-unit. In a flock that AGGREGATES -- E084 measured spread
1.66-7.21 m in a 20 m arena, and E085 measured occupancy at a single cell running
0.09-0.97 across seeds -- much of "where is she" is a BETWEEN-HEN variable: at any moment
different hens sit in different places, and each hen's own running mean encodes where she
personally tends to be. Subtracting it removes exactly that.

Tests it directly by removing the between-hen component by hand, at zero timescale:
  raw            uncentred z_lag                            (E086's 90.0%)
  per-hen mean   z_lag minus each hen's own full-run mean   (between-hen removed)
  global mean    z_lag minus the across-hen mean            (between-hen KEPT)
If per-hen collapses to ~73% and global does not, the cost is structural, not temporal,
and no value of pred_bar_tau_s can fix it.
"""
import time, sys
from functools import partial
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, 'scratchpad')
import e083_leaving_anchor as E
import e085_repaired_instrument as M
from coop import world
from hen import brain, connectome, plasticity, regions
from run import simulate

CEN, CFG, HENS, STEPS, R = M.CEN, M.CFG, M.HENS, M.STEPS, M.R
CENJ = jnp.asarray(CEN, dtype=jnp.float32)
SEEDS, EVERY = 8, M.SAMPLE_EVERY
reg = regions.DEFAULT_REGIONS
H_LO, H_HI = reg.bounds(regions.HIPPOCAMPUS)
HIPP = np.arange(H_LO, H_HI)


@partial(jax.jit, static_argnames=("cfg", "pc", "blocks", "every"))
def run(w, x, p, ps, key, cfg, pc, blocks, every):
    def inner(carry, _):
        carry, _o = simulate._one_step(carry, None, cfg=cfg, pc=pc)
        d = jnp.linalg.norm(carry[0].pos[:, None, :] - CENJ[None, :, :], axis=-1)
        return carry, (d < R)
    def outer(carry, _):
        carry, ins = jax.lax.scan(inner, carry, None, length=every)
        d = jnp.linalg.norm(carry[0].pos[:, None, :] - CENJ[None, :, :], axis=-1)
        return carry, (ins, d, carry[3].z_lag)          # UNcentred
    return jax.lax.scan(outer, (w, x, p, ps, key), None, length=blocks)[1]


def go(p, pc, wk, rk):
    w = world.reset(wk, CFG)
    w = w._replace(food_pos=jnp.asarray(np.stack([CEN[E.P], CEN[E.P2]]), dtype=jnp.float32))
    x = brain.initial_state(p, HENS); ps = plasticity.initial_state(p, HENS, pc)
    ins, d, z = run(w, x, p, ps, rk, CFG, pc, STEPS // EVERY, EVERY)
    ins = np.asarray(ins)
    return ins.reshape(-1, *ins.shape[2:]), np.asarray(d), np.asarray(z)


def variants(z):
    """(blocks, H, N) -> dict of (blocks*H, N), each a different mean removed."""
    return {"raw": z,
            "per-hen": z - z.mean(axis=0, keepdims=True),       # each hen's own mean
            "global": z - z.mean(axis=(0, 1), keepdims=True)}   # one mean for all hens


pc = plasticity.PlasticConfig(**E.FROZEN, pred_gain=0.0)
t0 = time.perf_counter()
print(f"E087 follow-up -- what does centring remove? {SEEDS} seeds\n")
rows = {k: [] for k in ("raw", "per-hen", "global")}
occ = []
for s in range(SEEDS):
    k = jax.random.key(s)
    p = connectome.build(jax.random.fold_in(k, 1), reg, n_hens=HENS,
                         gakel_scaffold=True, shared_place_map=True,
                         place_to_hippocampus=True)
    ins_s, dst_s, z_s = go(p, pc, k, jax.random.fold_in(k, 2))
    tgt = int(np.argmax(ins_s.mean(axis=(0, 1))))
    ins_t, dst_t, z_t = go(p, pc, k, jax.random.fold_in(k, 9))
    occ.append(float(ins_t.mean(axis=(0, 1))[tgt]))
    Dtr, Dte = M.flat(dst_s[:, :, tgt]), M.flat(dst_t[:, :, tgt])
    Vtr, Vte = variants(z_s), variants(z_t)
    line = []
    for name in rows:
        w_, thr = M.fit(M.flat(Vtr[name])[:, HIPP], Dtr, R)
        a, _r, _p = M.evaluate(w_, thr, M.flat(Vte[name])[:, HIPP], Dte, R)
        rows[name].append(a); line.append(f"{name} {a:.1%}")
    print(f"  seed {s}: " + "  ".join(line))

occ = np.array(occ); bal = np.abs(occ - 0.5) < 0.25
print(f"\n{'variant':>10}{'all seeds':>12}{'balanced':>11}   (balanced = "
      f"{int(bal.sum())} of {SEEDS})")
for name in ("raw", "global", "per-hen"):
    a = np.array(rows[name])
    print(f"{name:>10}{a.mean():>12.1%}{a[bal].mean():>11.1%}")
print(f"\nE086 reference: centred (per-hen, EMA tau 20s) 73.7% balanced")
print(f"wall clock: {time.perf_counter()-t0:.0f} s")
print("\nIf per-hen ~= 73.7% and global ~= raw, the loss is structural: `z_lag_bar` is a")
print("per-hen baseline, and in an aggregating flock position is largely a between-hen")
print("variable. No value of pred_bar_tau_s recovers it -- E087 Part A already showed that.")
