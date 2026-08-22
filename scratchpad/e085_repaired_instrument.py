"""E085: repair T2's instrument and measure what it can resolve.

Three parts, no behavioural claim.
  A  does per-seed target selection hold up across runs of the same world
  B  what the metric resolves -- MDE for occupancy and for mean dwell per visit,
     from a NULL-NULL contrast that does not consume the treatment
  C  is position linearly decodable during free movement (E084's unanswered Part A),
     with the label radius centred on the per-seed target so a degenerate split
     cannot occur by construction

See docs/experiments/E085-repaired-instrument.md sections 1-5, written first.
"""
import time, sys
from functools import partial
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, 'scratchpad')
import e083_leaving_anchor as E
from coop import world
from hen import brain, connectome, plasticity, regions
from run import simulate

CEN, CFG, HENS, STEPS = E.CEN, E.CFG, E.HENS, E.STEPS
CENJ = jnp.asarray(CEN, dtype=jnp.float32)
SEEDS = 8
R = E.SPACING                                  # 3.33 m, matched to E082/E083
SAMPLE_EVERY = 100
EDGES = np.array([0.0, 1.0, 2.0, 3.33, 5.0, 7.0, 10.0, 99.0])
NB = len(EDGES) - 1
T_CRIT = {4: 3.182, 8: 2.365}
GATE_ACC, GATE_RATIO = 0.70, 2.0
SEL_MIN, MDE_MAX = 0.15, 0.25


# Nested scan: the inner loop emits only the cheap occupancy bitmap every step (dwell
# needs every step), the outer loop emits the 512-wide pallial state once per block.
# A flat scan emitting `z` every step allocates 120000 x 16 x 512 x 4 B = 3.9 GB per
# run and the process is OOM-killed -- which, behind a pipe, loses Python's block
# buffer and presents as exit 0 with an empty output file. See E085 section 6.
@partial(jax.jit, static_argnames=("cfg", "pc", "blocks", "every"))
def run(w, x, p, ps, key, cfg, pc, blocks, every):
    def inner(carry, _):
        carry, _o = simulate._one_step(carry, None, cfg=cfg, pc=pc)
        d = jnp.linalg.norm(carry[0].pos[:, None, :] - CENJ[None, :, :], axis=-1)
        return carry, (d < R)
    def outer(carry, _):
        carry, ins = jax.lax.scan(inner, carry, None, length=every)
        d = jnp.linalg.norm(carry[0].pos[:, None, :] - CENJ[None, :, :], axis=-1)
        z = (carry[3].z_lag - carry[3].z_lag_bar) * p.pred_src[None, :]
        return carry, (ins, d, z)
    return jax.lax.scan(outer, (w, x, p, ps, key), None, length=blocks)[1]


def go(p, pc, world_key, run_key):
    w = world.reset(world_key, CFG)
    w = w._replace(food_pos=jnp.asarray(np.stack([CEN[E.P], CEN[E.P2]]), dtype=jnp.float32))
    x = brain.initial_state(p, HENS)
    ps = plasticity.initial_state(p, HENS, pc)
    ins, dist, z = run(w, x, p, ps, run_key, CFG, pc, STEPS // SAMPLE_EVERY, SAMPLE_EVERY)
    ins = np.asarray(ins)                       # (blocks, every, H, 25)
    return (ins.reshape(-1, *ins.shape[2:]), np.asarray(dist), np.asarray(z))


def dwell(inside, cell):
    """Mean seconds per contiguous visit to `cell`, pooled over hens."""
    a = inside[:, :, cell]
    runs = []
    for h in range(a.shape[1]):
        d = np.diff(np.concatenate([[0], a[:, h].astype(np.int8), [0]]))
        runs.extend((np.where(d == -1)[0] - np.where(d == 1)[0]).tolist())
    return (np.mean(runs) * CFG.dt if runs else 0.0), len(runs)


def fit(Z, D, radius):
    at = D < radius
    w = Z[at].mean(0) - Z[~at].mean(0)
    w = w / (np.linalg.norm(w) + 1e-9)
    return w, 0.5 * ((Z[at] @ w).mean() + (Z[~at] @ w).mean())


def evaluate(w, thr, Z, D, radius):
    at = D < radius
    assert at.sum() and (~at).sum(), f"degenerate split: {at.sum()} in / {(~at).sum()} out"
    s = Z @ w
    acc = 0.5 * ((s[at] > thr).mean() + (s[~at] <= thr).mean())
    g = np.maximum(s, 0.0)
    b = np.clip(np.searchsorted(EDGES, D) - 1, 0, NB - 1)
    prof = np.array([g[b == i].mean() if (b == i).any() else np.nan for i in range(NB)])
    return acc, g[at].mean() / max(g[~at].mean(), 1e-9), prof / max(np.nanmean(prof), 1e-9)


def flat(x):
    """(blocks, H, ...) -> (blocks*H, ...); hens are independent draws from one policy."""
    return x.reshape(-1, x.shape[-1]) if x.ndim == 3 else x.reshape(-1)


# Driver in a function so E086 can import `go`, `fit`, `evaluate` and `flat`
# without re-running the 18-minute measurement. Same reason as
# e083_leaving_anchor.py -- see E085 section 6c.
def _main():
    pc0 = plasticity.PlasticConfig(**E.FROZEN, pred_gain=0.0)
    pc2 = plasticity.PlasticConfig(**E.FROZEN, pred_gain=2.0)
    t0 = time.perf_counter()
    print(f"E085 -- repaired instrument, {SEEDS} seeds, {E.MINUTES:.0f} min/run, radius {R:.2f} m\n")

    A_sel, A_test, D_test, D_twin, O_test, O_twin = [], [], [], [], [], []
    acc_l, acc_lt, acc_p, rat_l, rat_p, prof_l, prof_p = [], [], [], [], [], [], []
    det = []
    print(f"{'seed':>5}{'target':>8}{'ctrl':>6}{'occ sel':>9}{'occ test':>10}"
          f"{'dwell test':>12}{'visits':>8}{'live acc':>10}{'park acc':>10}")
    for s in range(SEEDS):
        k = jax.random.key(s)
        p0 = connectome.build(jax.random.fold_in(k, 1), regions.DEFAULT_REGIONS, n_hens=HENS,
                              gakel_scaffold=True, shared_place_map=True)
        p, _ = E.plant(p0, E._PLANT_CFG)          # parked-fit plant, for Part C comparison

        ins_s, dst_s, z_s = go(p, pc0, k, jax.random.fold_in(k, 2))          # selection
        occ_s = ins_s.mean(axis=(0, 1))
        order = np.argsort(-occ_s)
        tgt, ctl = int(order[0]), int(order[1])

        ins_t, dst_t, z_t = go(p, pc0, k, jax.random.fold_in(k, 9))          # test
        ins_n, _d, _z = go(p, pc0, k, jax.random.fold_in(k, 11))             # null twin

        A_sel.append(occ_s[tgt]); A_test.append(ins_t.mean(axis=(0, 1))[tgt])
        O_test.append(ins_t.mean(axis=(0, 1))[tgt]); O_twin.append(ins_n.mean(axis=(0, 1))[tgt])
        d_t, n_t = dwell(ins_t, tgt); d_n, _ = dwell(ins_n, tgt)
        D_test.append(d_t); D_twin.append(d_n)

        # Part C, labels centred on this seed's own target
        Ztr, Dtr = flat(z_s), flat(dst_s[:, :, tgt])
        Zte, Dte = flat(z_t), flat(dst_t[:, :, tgt])
        w_l, thr_l = fit(Ztr, Dtr, R)
        a_l, r_l, pr_l = evaluate(w_l, thr_l, Zte, Dte, R)
        a_lt, _, _ = evaluate(w_l, thr_l, Ztr, Dtr, R)
        w_p = np.asarray(p.W_pred)[0, E.GAKEL_CH, :]
        w_p = w_p / (np.linalg.norm(w_p) + 1e-9)
        at = Dtr < R
        thr_p = 0.5 * ((Ztr[at] @ w_p).mean() + (Ztr[~at] @ w_p).mean())
        a_p, r_p, pr_p = evaluate(w_p, thr_p, Zte, Dte, R)
        acc_l.append(a_l); acc_lt.append(a_lt); acc_p.append(a_p)
        rat_l.append(r_l); rat_p.append(r_p); prof_l.append(pr_l); prof_p.append(pr_p)

        if s == 0:                                  # determinism check
            ins_g, _d2, _z2 = go(p, pc2, k, jax.random.fold_in(k, 9))
            det.append(not np.array_equal(ins_g, ins_t))

        print(f"{s:>5}{tgt:>8}{ctl:>6}{occ_s[tgt]:>9.3f}{A_test[-1]:>10.3f}"
              f"{d_t:>12.2f}{n_t:>8d}{a_l:>10.1%}{a_p:>10.1%}")

    A_sel, A_test = np.array(A_sel), np.array(A_test)
    drift = (A_test - A_sel) / A_sel
    print(f"\n--- Part A: selection validity ---")
    print(f"occ at target: selection {A_sel.mean():.3f}, test {A_test.mean():.3f}, "
          f"min test {A_test.min():.3f}")
    print(f"drift mean {100*drift.mean():+.1f}%, per seed "
          + ", ".join(f"{100*d:+.0f}%" for d in drift))

    print(f"\n--- Part B: what the metric resolves (null-null, no treatment) ---")
    print(f"{'metric':>12}{'baseline':>11}{'diff sd':>10}{'MDE n=4':>11}{'MDE n=8':>11}")
    mdes = {}
    for name, a, b in (("occupancy", np.array(O_test), np.array(O_twin)),
                       ("dwell (s)", np.array(D_test), np.array(D_twin))):
        d = b - a; base = a.mean(); sd = d.std(ddof=1)
        m4, m8 = T_CRIT[4]*sd/2, T_CRIT[8]*sd/np.sqrt(8)
        mdes[name] = m8 / base
        print(f"{name:>12}{base:>11.4f}{sd:>10.4f}"
              f"{100*m4/base:>10.1f}%{100*m8/base:>10.1f}%")

    print(f"\n--- Part C: decodability during free movement ---")
    print(f"live-fit  held-out {np.mean(acc_l):.1%}  train {np.mean(acc_lt):.1%}  "
          f"ratio {np.mean(rat_l):.2f}")
    print(f"parked-fit held-out {np.mean(acc_p):.1%}  ratio {np.mean(rat_p):.2f}")
    PL, PP = np.nanmean(prof_l, 0), np.nanmean(prof_p, 0)
    print(f"\n{'bin (m)':>14}{'live-fit':>11}{'parked-fit':>13}")
    for i in range(NB):
        print(f"{f'{EDGES[i]:.1f}-{EDGES[i+1]:.1f}':>14}{PL[i]:>11.3f}{PP[i]:>13.3f}")

    dec = bool(PL[0] > PL[1] > PL[2])
    print(f"\nwall clock: {time.perf_counter()-t0:.0f} s")
    print("--- pre-registered falsifiers (E085 section 4) ---")
    print(f"selection  min occ {A_test.min():.3f} (need >{SEL_MIN}), drift {100*drift.mean():+.1f}% "
          f"(fires if <-20%) -> "
          f"{'FIRES' if (A_test.min() <= SEL_MIN or drift.mean() < -0.20) else 'clear'}")
    print(f"resolution dwell MDE n=8 {100*mdes['dwell (s)']:.1f}% (fires if >{100*MDE_MAX:.0f}%), "
          f"vs occupancy {100*mdes['occupancy']:.1f}% -> "
          f"{'FIRES' if mdes['dwell (s)'] > MDE_MAX else 'clear'}")
    print(f"gate       acc {np.mean(acc_l):.1%} (need >={GATE_ACC:.0%}), ratio {np.mean(rat_l):.2f} "
          f"(need >={GATE_RATIO}), profile decreasing={dec} -> "
          f"{'PASS' if (np.mean(acc_l) >= GATE_ACC and np.mean(rat_l) >= GATE_RATIO and dec) else 'FIRES'}")
    print(f"diagnosis  parked-fit {np.mean(acc_p):.1%} (E083 wrong if >=70%) -> "
          f"{'FIRES' if np.mean(acc_p) >= 0.70 else 'clear'}")
    print(f"leakage    train {np.mean(acc_lt):.1%} vs held-out {np.mean(acc_l):.1%} = "
          f"{100*(np.mean(acc_lt)-np.mean(acc_l)):+.1f} pts (fires if >15) -> "
          f"{'FIRES' if (np.mean(acc_lt)-np.mean(acc_l)) > 0.15 else 'clear'}")
    print(f"determinism gain changes trajectory on seed 0: {det[0]} -> "
          f"{'clear' if det[0] else 'FIRES'}")



if __name__ == "__main__":
    _main()
