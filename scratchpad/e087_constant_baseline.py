"""E087 follow-up 2: does a CONSTANT centring baseline give both selectivity and signal?

Two mechanisms for the centring's ~20-point cost have now been proposed and falsified:
timescale (E087 Part A -- longer tau is worse, not better) and between-hen structure
(follow-up 1 -- removing each hen's own constant mean costs nothing, 89.8% vs 90.0%).

What survives is narrow and testable. The runtime uses a CAUSAL, TIME-VARYING EMA
(`z_lag_bar`), which is a high-pass. A CONSTANT baseline is a pure DC removal. Follow-up 1
showed constant removal keeps decodability at ~90%. It did NOT measure selectivity, which
is the only reason the centring exists -- E070 measured raw `z_lag` predicting 1.0000 at
its own place and 0.9637 at another, a ratio of 1.04.

So: measure E070's selectivity ratio under three baselines, on identical settled states.
  none      raw z_lag                        -- expect ~1.0, E070's failure
  constant  z_lag minus the across-place mean -- the candidate fix
  EMA       z_lag - z_lag_bar at tau 20 s     -- the runtime, E087 Part A measured 32.0
"""
import time, sys
import jax, numpy as np
sys.path.insert(0, 'scratchpad')
import e083_leaving_anchor as E
from hen import connectome, plasticity, regions

reg = regions.DEFAULT_REGIONS
CELLS = [E.P] + list(E.OTHERS_FOR_DISC)   # the target plus the tour's other cells;
                                          # E.P is not itself in E.TOUR
SEEDS = 4


def raw_state(p, cell, pc):
    """Settled `z_lag`, uncentred -- E._centred returns z_lag - z_lag_bar."""
    from coop import world as _w
    from hen import brain as _b
    ps = plasticity.initial_state(p, E.HENS, pc)
    w = _w.reset(jax.random.key(0), E.CFG)
    x = _b.initial_state(p, E.HENS)
    ps = E._settle(w, x, ps, p, E._positions(cell), E.CFG, pc)
    return np.asarray(ps.z_lag[0]), np.asarray((ps.z_lag - ps.z_lag_bar)[0])


pc = plasticity.PlasticConfig(**E.FROZEN, pred_gain=1.0)
t0 = time.perf_counter()
print(f"E087 follow-up 2 -- selectivity under three baselines, {SEEDS} seeds")
print(f"E070's failure (raw): 1.0000 at P vs 0.9637 elsewhere, ratio 1.04\n")

out = {k: [] for k in ("none", "constant", "EMA")}
for s in range(SEEDS):
    k = jax.random.key(s)
    p = connectome.build(jax.random.fold_in(k, 1), reg, n_hens=E.HENS,
                         gakel_scaffold=True, shared_place_map=True,
                         place_to_hippocampus=True)
    src = np.asarray(p.pred_src)
    raw = {c: raw_state(p, c, pc) for c in CELLS}
    R = {c: raw[c][0] for c in CELLS}          # uncentred z_lag
    Ema = {c: raw[c][1] for c in CELLS}        # z_lag - z_lag_bar
    b = np.mean([R[c] for c in CELLS], axis=0)  # ONE constant, the across-place mean
    Con = {c: R[c] - b for c in CELLS}

    for name, S in (("none", R), ("constant", Con), ("EMA", Ema)):
        others = [c for c in CELLS if c != E.P]
        sP = S[E.P]; sO = np.mean([S[c] for c in others], axis=0)
        disc = (sP - sO) * src
        disc = disc / (float(disc @ sP) + 1e-9)
        at_P = float(disc @ sP)
        at_other = float(np.mean([disc @ S[c] for c in others]))
        out[name].append((at_P, at_other, at_P / (abs(at_other) + 1e-9)))

print(f"{'baseline':>10}{'pred @ P':>11}{'pred elsewhere':>17}{'ratio':>9}")
for name in ("none", "constant", "EMA"):
    a = np.array(out[name])
    print(f"{name:>10}{a[:,0].mean():>11.4f}{a[:,1].mean():>17.4f}{a[:,2].mean():>9.2f}")

print(f"\nwall clock: {time.perf_counter()-t0:.0f} s")
print("decodability under movement, from follow-up 1: raw 90.0%, constant 89.8%,")
print("runtime EMA 73.7%. If `constant` also holds selectivity here, it wins on both")
print("axes and the fix is a fixed baseline rather than a tracking one.")
