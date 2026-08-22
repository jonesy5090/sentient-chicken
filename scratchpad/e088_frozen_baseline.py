"""E088: does a frozen centring baseline deliver both selectivity and signal?

E087 showed the centring's cost is the baseline TRACKING, and that a constant baseline
sits better on the trade-off -- selectivity 5.00 at 89.8% decodability, against the
runtime's 23.28 at 73.7%. But E087's constant was the mean across settled place states,
which needs the places known in advance: a diagnostic, not a mechanism.

This tests the causal version: track for a calibration window, then freeze.

Part A sweeps the freeze time and measures held-out place decodability, reporting the
baseline's convergence AT the moment of freezing so a baseline frozen before it
represented anything is visible rather than reported as a result.
Part B measures selectivity at EVERY freeze time -- not only the best -- so "both axes on
one configuration" is checkable rather than asserted. That is E087's lesson.

See docs/experiments/E088-frozen-centring-baseline.md sections 1-5, written first.
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
FREEZES = (10.0, 20.0, 40.0, 60.0, 120.0, None)
GATE_ACC, GATE_RATIO = 0.85, 2.0


@partial(jax.jit, static_argnames=("cfg", "pc", "blocks", "every"))
def run(w, x, p, ps, key, cfg, pc, blocks, every):
    def inner(carry, _):
        carry, _o = simulate._one_step(carry, None, cfg=cfg, pc=pc)
        d = jnp.linalg.norm(carry[0].pos[:, None, :] - CENJ[None, :, :], axis=-1)
        return carry, (d < R)
    def outer(carry, _):
        carry, ins = jax.lax.scan(inner, carry, None, length=every)
        d = jnp.linalg.norm(carry[0].pos[:, None, :] - CENJ[None, :, :], axis=-1)
        q = carry[3]
        conv = jnp.mean(jnp.abs(q.z_lag_bar)) / (jnp.mean(jnp.abs(q.z_lag)) + 1e-9)
        return carry, (ins, d, q.z_lag - q.z_lag_bar, conv)
    return jax.lax.scan(outer, (w, x, p, ps, key), None, length=blocks)[1]


def go(p, pc, wk, rk, freeze_s):
    w = world.reset(wk, CFG)
    w = w._replace(food_pos=jnp.asarray(np.stack([CEN[E.P], CEN[E.P2]]), dtype=jnp.float32))
    x = brain.initial_state(p, HENS); ps = plasticity.initial_state(p, HENS, pc)
    ins, d, z, conv = run(w, x, p, ps, rk, CFG, pc, STEPS // EVERY, EVERY)
    ins = np.asarray(ins); conv = np.asarray(conv)
    # convergence at the freeze moment, not at the end -- that is what the frozen value is
    idx = (min(int(freeze_s / (CFG.dt * EVERY)), len(conv) - 1)
           if freeze_s is not None else len(conv) - 1)
    return (ins.reshape(-1, *ins.shape[2:]), np.asarray(d), np.asarray(z),
            float(conv[idx]))


def build(k):
    return connectome.build(jax.random.fold_in(k, 1), reg, n_hens=HENS,
                            gakel_scaffold=True, shared_place_map=True,
                            place_to_hippocampus=True)


print(f"E088 -- frozen centring baseline, {SEEDS} seeds, {E.MINUTES:.0f} min/run")
print("references: runtime EMA 73.7%, E087's idealised constant 89.8%, raw 90.0%")
print("            selectivity -- raw 1.04 (E070's failure), constant 5.00, EMA 23.28\n")
t0 = time.perf_counter()

print(f"{'freeze (s)':>11}{'held-out':>10}{'balanced':>10}{'conv @ freeze':>15}")
occ_ref, resA = None, {}
for fz in FREEZES:
    pc = plasticity.PlasticConfig(**E.FROZEN, pred_gain=0.0, pred_bar_freeze_s=fz)
    accs, convs, occs = [], [], []
    for s in range(SEEDS):
        k = jax.random.key(s); p = build(k)
        ins_s, dst_s, z_s, _c = go(p, pc, k, jax.random.fold_in(k, 2), fz)
        tgt = int(np.argmax(ins_s.mean(axis=(0, 1))))
        ins_t, dst_t, z_t, c = go(p, pc, k, jax.random.fold_in(k, 9), fz)
        w_, thr = M.fit(M.flat(z_s)[:, HIPP], M.flat(dst_s[:, :, tgt]), R)
        a, _r, _pr = M.evaluate(w_, thr, M.flat(z_t)[:, HIPP], M.flat(dst_t[:, :, tgt]), R)
        accs.append(a); convs.append(c); occs.append(float(ins_t.mean(axis=(0, 1))[tgt]))
    if occ_ref is None:
        occ_ref = np.array(occs)                 # subset fixed on the FIRST condition
    bal = np.abs(occ_ref - 0.5) < 0.25
    a = np.array(accs)
    resA[fz] = (a.mean(), a[bal].mean(), float(np.mean(convs)))
    print(f"{str(fz):>11}{a.mean():>10.1%}{a[bal].mean():>10.1%}{np.mean(convs):>15.3f}")

print(f"\n{'freeze (s)':>11}{'pred @ P':>11}{'pred elsewhere':>17}{'ratio':>9}")
resB = {}
for fz in FREEZES:
    pc = plasticity.PlasticConfig(**E.FROZEN, pred_gain=1.0, pred_bar_freeze_s=fz)
    aP, aO = [], []
    for s in range(4):
        p = build(jax.random.key(s))
        cells = [E.P] + list(E.OTHERS_FOR_DISC)
        S = {c: E._centred(p, c, pc) for c in cells}
        others = [c for c in cells if c != E.P]
        sP = S[E.P]; sO = np.mean([S[c] for c in others], axis=0)
        disc = (sP - sO) * np.asarray(p.pred_src)
        disc = disc / (float(disc @ sP) + 1e-9)
        aP.append(float(disc @ sP))
        aO.append(float(np.mean([disc @ S[c] for c in others])))
    mP, mO = float(np.mean(aP)), float(np.mean(aO))
    resB[fz] = (mP, mO, mP / (abs(mO) + 1e-9))
    print(f"{str(fz):>11}{mP:>11.4f}{mO:>17.4f}{resB[fz][2]:>9.2f}")

cand = [f for f in FREEZES if f is not None]
best = max(cand, key=lambda f: resA[f][1])
acc, ratio = resA[best][1], resB[best][2]
accs_only = [resA[f][1] for f in cand]
spread = 100 * (max(accs_only) - min(accs_only))
print(f"\nwall clock: {time.perf_counter()-t0:.0f} s")
print("--- pre-registered falsifiers (E088 section 4) ---")
print(f"primary     best freeze {best:.0f}s -> balanced acc {acc:.1%} "
      f"(fires if <{GATE_ACC:.0%}) -> {'FIRES' if acc < GATE_ACC else 'clear'}")
print(f"selectivity ratio at that freeze {ratio:.2f} (fires if <{GATE_RATIO}) -> "
      f"{'FIRES' if ratio < GATE_RATIO else 'clear'}")
print(f"calibration spread across freeze times {spread:.1f} pts "
      f"(fires if >15 with no trend) -> {'CHECK' if spread > 15 else 'clear'}")
print(f"control     never-freeze arm {resA[None][1]:.1%} "
      f"(should reproduce E087's 73.7%)")
