"""E087: does decoupling the prediction-centring timescale recover decodability without
bringing back E070's selectivity failure?

Part A sweeps `pred_bar_tau_s` and measures held-out decodability from the hippocampus,
reporting each condition's convergence ratio so an unconverged baseline cannot be read as
a result (the E071 error, repeated by E082).

Part B measures selectivity the way E070 measured it -- plant a place association, read
its prediction at its own place and at a different one. E070's failure was 1.0000 vs
0.9637, a ratio of 1.04. This is the falsifier that matters: it is the entire reason the
centring exists, and E086's diagnostic never tested it.

See docs/experiments/E087-prediction-centring-timescale.md sections 1-5, written first.
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
TAUS = (20.0, 60.0, 150.0, 300.0, 600.0)
GATE_ACC, GATE_RATIO = 0.80, 2.0


@partial(jax.jit, static_argnames=("cfg", "pc", "blocks", "every"))
def run(w, x, p, ps, key, cfg, pc, blocks, every):
    def inner(carry, _):
        carry, _o = simulate._one_step(carry, None, cfg=cfg, pc=pc)
        d = jnp.linalg.norm(carry[0].pos[:, None, :] - CENJ[None, :, :], axis=-1)
        return carry, (d < R)
    def outer(carry, _):
        carry, ins = jax.lax.scan(inner, carry, None, length=every)
        d = jnp.linalg.norm(carry[0].pos[:, None, :] - CENJ[None, :, :], axis=-1)
        ps_ = carry[3]
        conv = jnp.mean(jnp.abs(ps_.z_lag_bar)) / (jnp.mean(jnp.abs(ps_.z_lag)) + 1e-9)
        return carry, (ins, d, ps_.z_lag - ps_.z_lag_bar, conv)
    return jax.lax.scan(outer, (w, x, p, ps, key), None, length=blocks)[1]


def go(p, pc, wk, rk):
    w = world.reset(wk, CFG)
    w = w._replace(food_pos=jnp.asarray(np.stack([CEN[E.P], CEN[E.P2]]), dtype=jnp.float32))
    x = brain.initial_state(p, HENS); ps = plasticity.initial_state(p, HENS, pc)
    ins, d, z, conv = run(w, x, p, ps, rk, CFG, pc, STEPS // EVERY, EVERY)
    ins = np.asarray(ins)
    return (ins.reshape(-1, *ins.shape[2:]), np.asarray(d), np.asarray(z),
            float(np.asarray(conv)[-1]))


def build(k):
    return connectome.build(jax.random.fold_in(k, 1), reg, n_hens=HENS,
                            gakel_scaffold=True, shared_place_map=True,
                            place_to_hippocampus=True)


print(f"E087 -- prediction-centring timescale, {SEEDS} seeds, {E.MINUTES:.0f} min/run")
print(f"reference lines from E086: centred@20s 73.7%, z_lag uncentred 90.0%, "
      f"raw rate 90.7% (balanced-split)\n")
t0 = time.perf_counter()

# --- Part A ---------------------------------------------------------------
print(f"{'tau (s)':>9}{'held-out acc':>14}{'balanced':>11}{'convergence':>13}")
occ_ref, resA = None, {}
for tau in TAUS:
    pc = plasticity.PlasticConfig(**E.FROZEN, pred_gain=0.0, pred_bar_tau_s=tau)
    accs, convs, occs = [], [], []
    for s in range(SEEDS):
        k = jax.random.key(s); p = build(k)
        ins_s, dst_s, z_s, _c = go(p, pc, k, jax.random.fold_in(k, 2))
        tgt = int(np.argmax(ins_s.mean(axis=(0, 1))))
        ins_t, dst_t, z_t, c = go(p, pc, k, jax.random.fold_in(k, 9))
        w_, thr = M.fit(M.flat(z_s)[:, HIPP], M.flat(dst_s[:, :, tgt]), R)
        a, _r, _pr = M.evaluate(w_, thr, M.flat(z_t)[:, HIPP], M.flat(dst_t[:, :, tgt]), R)
        accs.append(a); convs.append(c); occs.append(float(ins_t.mean(axis=(0, 1))[tgt]))
    if occ_ref is None:
        occ_ref = np.array(occs)                    # subset fixed on the FIRST condition
    bal = np.abs(occ_ref - 0.5) < 0.25
    a = np.array(accs)
    resA[tau] = (a.mean(), a[bal].mean(), float(np.mean(convs)))
    print(f"{tau:>9.0f}{a.mean():>14.1%}{a[bal].mean():>11.1%}{np.mean(convs):>13.3f}")

# --- Part B: selectivity, E070's measurement ------------------------------
print(f"\n{'tau (s)':>9}{'pred @ P':>11}{'pred @ P2':>12}{'ratio':>9}")
resB = {}
for tau in TAUS:
    pc = plasticity.PlasticConfig(**E.FROZEN, pred_gain=1.0, pred_bar_tau_s=tau)
    aP, aO = [], []
    for s in range(4):
        k = jax.random.key(s); p = build(k)
        sP = E._centred(p, E.P, pc)
        sO = E._centred(p, E.OTHERS_FOR_DISC[0], pc)
        disc = (sP - np.mean([E._centred(p, c, pc)
                              for c in E.OTHERS_FOR_DISC], axis=0)) * np.asarray(p.pred_src)
        disc = disc / (float(disc @ sP) + 1e-9)
        aP.append(float(disc @ sP)); aO.append(float(disc @ sO))
    mP, mO = float(np.mean(aP)), float(np.mean(aO))
    resB[tau] = (mP, mO, mP / (abs(mO) + 1e-9))
    print(f"{tau:>9.0f}{mP:>11.4f}{mO:>12.4f}{resB[tau][2]:>9.2f}")

best = max(TAUS, key=lambda t: resA[t][1])
acc, ratio = resA[best][1], resB[best][2]
print(f"\nwall clock: {time.perf_counter()-t0:.0f} s")
print("--- pre-registered falsifiers (E087 section 4) ---")
print(f"primary     best tau {best:.0f}s -> balanced acc {acc:.1%} "
      f"(fires if <{GATE_ACC:.0%}, predicted >=85% at 300s) -> "
      f"{'FIRES' if acc < GATE_ACC else 'clear'}")
print(f"selectivity ratio at best tau {ratio:.2f} (fires if <{GATE_RATIO}; "
      f"E070's failure was 1.04) -> {'FIRES' if ratio < GATE_RATIO else 'clear'}")
unconv = [t for t in TAUS if resA[t][2] < 0.5]
print(f"convergence unconverged taus (ratio <0.5): {unconv or 'none'} -> "
      f"{'EXCLUDE those from interpretation' if unconv else 'clear'}")
