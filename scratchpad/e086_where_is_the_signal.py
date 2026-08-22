"""E086 follow-up diagnostic: where in `pred_src` does the place signal live?

E086 routed place to the hippocampus and got parked decodability 84.6% -> 99.5% with a
decreasing distance profile for the first time -- the mechanism plainly works -- while
decodability *while moving* rose only 54.3% -> 58.9% (ns).

`pred_src` is now pallium (256 units) + hippocampus (80). If the place signal is
concentrated in the 80 and the 256 carry mostly non-spatial variance, a difference-of-
means readout over all 336 dilutes it. Fits the same estimator on three unit subsets, on
identical data.

Also settles whether the lagged trace is to blame, empirically rather than by the
back-of-envelope argument (0.45 m of travel per tau_lag against a 6.67 m disc), by
decoding from instantaneous rate on the same samples.
"""
import time, sys
from functools import partial
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, 'scratchpad')
import e083_leaving_anchor as E
import e085_repaired_instrument as M
from coop import world
from hen import brain, connectome, neurons, plasticity, regions
from run import simulate

CEN, CFG, HENS, STEPS, R = M.CEN, M.CFG, M.HENS, M.STEPS, M.R
CENJ = jnp.asarray(CEN, dtype=jnp.float32)
SEEDS, EVERY = 8, M.SAMPLE_EVERY
reg = regions.DEFAULT_REGIONS
P_LO, P_HI = reg.bounds(regions.PALLIUM)
H_LO, H_HI = reg.bounds(regions.HIPPOCAMPUS)


@partial(jax.jit, static_argnames=("cfg", "pc", "blocks", "every"))
def run(w, x, p, ps, key, cfg, pc, blocks, every):
    def inner(carry, _):
        carry, _o = simulate._one_step(carry, None, cfg=cfg, pc=pc)
        d = jnp.linalg.norm(carry[0].pos[:, None, :] - CENJ[None, :, :], axis=-1)
        return carry, (d < R)
    def outer(carry, _):
        carry, ins = jax.lax.scan(inner, carry, None, length=every)
        d = jnp.linalg.norm(carry[0].pos[:, None, :] - CENJ[None, :, :], axis=-1)
        lag = carry[3].z_lag - carry[3].z_lag_bar          # what the runtime reads
        rate = neurons.rate(carry[1])                       # instantaneous
        raw = carry[3].z_lag                                # lagged but NOT centred
        return carry, (ins, d, lag, rate, raw)
    return jax.lax.scan(outer, (w, x, p, ps, key), None, length=blocks)[1]


def go(p, pc, wk, rk):
    w = world.reset(wk, CFG)
    w = w._replace(food_pos=jnp.asarray(np.stack([CEN[E.P], CEN[E.P2]]), dtype=jnp.float32))
    x = brain.initial_state(p, HENS); ps = plasticity.initial_state(p, HENS, pc)
    ins, d, lag, rate, raw = run(w, x, p, ps, rk, CFG, pc, STEPS // EVERY, EVERY)
    ins = np.asarray(ins)
    return (ins.reshape(-1, *ins.shape[2:]), np.asarray(d), np.asarray(lag),
            np.asarray(rate), np.asarray(raw))


def score(Ztr, Dtr, Zte, Dte, cols):
    w, thr = M.fit(Ztr[:, cols], Dtr, R)
    a, r, _p = M.evaluate(w, thr, Zte[:, cols], Dte, R)
    return a, r


pc0 = plasticity.PlasticConfig(**E.FROZEN, pred_gain=0.0)
src_full = np.where(np.asarray(connectome.build(
    jax.random.key(0), reg, n_hens=2, shared_place_map=True,
    place_to_hippocampus=True).pred_src))[0]
hipp = np.arange(H_LO, H_HI)
pall = np.arange(P_LO, P_HI)
print(f"pred_src {len(src_full)} units = pallium {len(pall)} + hippocampus {len(hipp)}\n")

t0 = time.perf_counter()
rows = {k: [] for k in ("full", "hipp", "pall", "hipp_rate", "hipp_uncentred")}
occ = []
for s in range(SEEDS):
    k = jax.random.key(s)
    p = connectome.build(jax.random.fold_in(k, 1), reg, n_hens=HENS,
                         gakel_scaffold=True, shared_place_map=True,
                         place_to_hippocampus=True)
    ins_s, dst_s, lag_s, rt_s, raw_s = go(p, pc0, k, jax.random.fold_in(k, 2))
    tgt = int(np.argmax(ins_s.mean(axis=(0, 1))))
    ins_t, dst_t, lag_t, rt_t, raw_t = go(p, pc0, k, jax.random.fold_in(k, 9))
    occ.append(float(ins_t.mean(axis=(0, 1))[tgt]))
    Dtr, Dte = M.flat(dst_s[:, :, tgt]), M.flat(dst_t[:, :, tgt])
    L_tr, L_te = M.flat(lag_s), M.flat(lag_t)
    R_tr, R_te = M.flat(rt_s), M.flat(rt_t)
    U_tr, U_te = M.flat(raw_s), M.flat(raw_t)
    rows["full"].append(score(L_tr, Dtr, L_te, Dte, src_full)[0])
    rows["hipp"].append(score(L_tr, Dtr, L_te, Dte, hipp)[0])
    rows["pall"].append(score(L_tr, Dtr, L_te, Dte, pall)[0])
    rows["hipp_rate"].append(score(R_tr, Dtr, R_te, Dte, hipp)[0])
    rows["hipp_uncentred"].append(score(U_tr, Dtr, U_te, Dte, hipp)[0])
    print(f"  seed {s}: full {rows['full'][-1]:.1%}  hipp {rows['hipp'][-1]:.1%}  "
          f"pall {rows['pall'][-1]:.1%}  hipp/rate {rows['hipp_rate'][-1]:.1%}  "
          f"hipp/uncentred {rows['hipp_uncentred'][-1]:.1%}")

occ = np.array(occ); bal = np.abs(occ - 0.5) < 0.25
print(f"\n{'subset':>12}{'all seeds':>12}{'balanced':>11}   (balanced = "
      f"{int(bal.sum())} of {SEEDS})")
for name, label in (("full", "pred_src 336"), ("hipp", "hippocampus 80"),
                    ("pall", "pallium 256"), ("hipp_uncentred", "hipp, z_lag only"),
                    ("hipp_rate", "hipp, raw rate")):
    a = np.array(rows[name])
    print(f"{label:>12}{a.mean():>12.1%}{a[bal].mean():>11.1%}")
print(f"\nwall clock: {time.perf_counter()-t0:.0f} s")
print("dilution => hippocampus alone beats the pooled 336.")
print("Splitting the trace: `z_lag only` isolates the low-pass (tau_lag 1.5 s) from the")
print("centring high-pass (`- z_lag_bar`, tau 20 s). Dwell times are 17-75 s (E085), so a")
print("20 s baseline tracks and subtracts exactly the slowly-varying signal position is.")
