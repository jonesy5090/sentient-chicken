"""E086: does routing place cells to the hippocampus make position legible while moving?

Reuses E085's Part C machinery unchanged -- same estimator (difference-of-means), same
held-out protocol (fit on the selection run, evaluate on the test run of the same world),
same per-seed target selection, same 8 seeds, same radius. The ONLY thing that differs
between the two arms is `place_to_hippocampus`, so any difference is attributable to the
routing change.

See docs/experiments/E086-place-to-hippocampus.md sections 1-5, written first.
"""
import time, sys
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, 'scratchpad')
import e083_leaving_anchor as E
import e085_repaired_instrument as M
from coop import sensing, spec, world
from hen import brain, connectome, neurons, plasticity, regions

SEEDS, R = M.SEEDS, M.R
CEN, CFG, HENS = M.CEN, M.CFG, M.HENS
T5, T7 = 2.571, 2.365


def arm(flag, pc0):
    """One condition. Returns per-seed (accuracy, ratio, profile, occupancy)."""
    acc, rat, prof, occ = [], [], [], []
    for s in range(SEEDS):
        k = jax.random.key(s)
        p = connectome.build(jax.random.fold_in(k, 1), regions.DEFAULT_REGIONS,
                             n_hens=HENS, gakel_scaffold=True, shared_place_map=True,
                             place_to_hippocampus=flag)
        ins_s, dst_s, z_s = M.go(p, pc0, k, jax.random.fold_in(k, 2))     # selection
        tgt = int(np.argmax(ins_s.mean(axis=(0, 1))))
        ins_t, dst_t, z_t = M.go(p, pc0, k, jax.random.fold_in(k, 9))     # test
        Ztr, Dtr = M.flat(z_s), M.flat(dst_s[:, :, tgt])
        Zte, Dte = M.flat(z_t), M.flat(dst_t[:, :, tgt])
        w, thr = M.fit(Ztr, Dtr, R)
        a, r, pr = M.evaluate(w, thr, Zte, Dte, R)
        acc.append(a); rat.append(r); prof.append(pr)
        occ.append(float(ins_t.mean(axis=(0, 1))[tgt]))
    return np.array(acc), np.array(rat), np.array(prof), np.array(occ)


# --- prediction 2: parked decodability must not fall (E081's protocol) ------
P_LO, P_HI = regions.DEFAULT_REGIONS.bounds(regions.PALLIUM)
H_LO, H_HI = regions.DEFAULT_REGIONS.bounds(regions.HIPPOCAMPUS)
CELLS, NSAMP, JIT = [2, 6, 12, 18, 22], 24, 0.35


def parked(flag):
    """E081's measurement, read from whatever `pred_src` covers in this condition."""
    accs = []
    for s in range(4):
        p = connectome.build(jax.random.key(s), regions.DEFAULT_REGIONS, n_hens=2,
                             shared_place_map=True, place_to_hippocampus=flag)
        src = np.asarray(p.pred_src)
        rng = np.random.default_rng(s)
        cfg = spec.DEFAULT_COOP._replace(n_hens=2, food_deplete_rate=0.0,
                                         place_cells_enabled=True)

        def settle(cell):
            w = world.reset(jax.random.key(0), cfg)
            pos = CEN[cell] + rng.normal(0, JIT, 2)
            w = w._replace(pos=jnp.broadcast_to(jnp.asarray(pos, dtype=jnp.float32), (2, 2)),
                           heading=jnp.zeros((2,)))
            x = brain.initial_state(p, 2); o = sensing.observe(w, cfg)
            for _ in range(200):
                x, _m, _d = brain.step(x, o, p, cfg.dt)
            return np.asarray(neurons.rate(x)[0])[src]

        S = {c: np.stack([settle(c) for _ in range(NSAMP)]) for c in CELLS}
        tr, te = slice(0, NSAMP // 2), slice(NSAMP // 2, NSAMP)
        tgt, others = CELLS[0], CELLS[1:]
        OT_tr = np.concatenate([S[o][tr] for o in others])
        OT_te = np.concatenate([S[o][te] for o in others])
        w = S[tgt][tr].mean(0) - OT_tr.mean(0)
        w = w / (np.linalg.norm(w) + 1e-9)
        thr = 0.5 * ((S[tgt][tr] @ w).mean() + (OT_tr @ w).mean())
        accs.append(0.5 * (((S[tgt][te] @ w) > thr).mean() + ((OT_te @ w) <= thr).mean()))
    return float(np.mean(accs))


pc0 = plasticity.PlasticConfig(**E.FROZEN, pred_gain=0.0)
t0 = time.perf_counter()
print(f"E086 -- place -> hippocampus, {SEEDS} seeds, {E.MINUTES:.0f} min/run, "
      f"radius {R:.2f} m")
print(f"pallium {P_HI-P_LO} units, hippocampus {H_HI-H_LO} units\n")

res = {}
for flag in (False, True):
    res[flag] = arm(flag, pc0)
    a, r, _pr, o = res[flag]
    print(f"place_to_hippocampus={str(flag):5}  held-out acc {a.mean():.1%}  "
          f"ratio {r.mean():.2f}  " + " ".join(f"{v:.1%}" for v in a))

(a0, r0, p0, o0), (a1, r1, p1, o1) = res[False], res[True]
bal = np.abs(o0 - 0.5) < 0.25          # E085's balanced-split criterion, on the OFF arm
print(f"\nbalanced-split seeds (from the off arm, so the subset is not chosen on the "
      f"treatment): {int(bal.sum())} of {SEEDS}")
print(f"  off {a0[bal].mean():.1%}   on {a1[bal].mean():.1%}")

d = (a1 - a0)[bal]
se = d.std(ddof=1) / np.sqrt(bal.sum())
tc = T5 if bal.sum() == 6 else T7
print(f"  paired change {100*d.mean():+.1f} +/- {100*se:.1f} pts  "
      f"t={d.mean()/se:+.2f} vs t={tc}  "
      f"{'SIGNIFICANT' if abs(d.mean()/se) > tc else 'not significant'}")

print(f"\n{'bin (m)':>14}{'off':>9}{'on':>9}")
PL0, PL1 = np.nanmean(p0, 0), np.nanmean(p1, 0)
for i in range(M.NB):
    print(f"{f'{M.EDGES[i]:.1f}-{M.EDGES[i+1]:.1f}':>14}{PL0[i]:>9.3f}{PL1[i]:>9.3f}")

pk0, pk1 = parked(False), parked(True)
print(f"\nparked decodability (E081 protocol): off {pk0:.1%}  on {pk1:.1%}")

acc_on = a1[bal].mean()
dec = bool(PL1[0] > PL1[1] > PL1[2])
print(f"\nwall clock: {time.perf_counter()-t0:.0f} s")
print("--- pre-registered falsifiers (E086 section 4) ---")
print(f"primary    balanced-split acc {acc_on:.1%} (fires if <60%, predicted >=65%, "
      f"E085 baseline 54.3%) -> {'FIRES' if acc_on < 0.60 else 'clear'}")
print(f"regression parked {pk1:.1%} (fires if <75%) -> "
      f"{'FIRES' if pk1 < 0.75 else 'clear'}")
print(f"profile    decreasing over first three bins = {dec} "
      f"(prediction 4; not a falsifier)")
