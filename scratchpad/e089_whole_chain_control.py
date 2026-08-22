"""E089: the whole-chain positive control, on a stack where every link is measured.

Backlog staging step 3, redone. E082 and E083 both attempted it and neither was valid --
anti-selective plant (E083), under-resolved metric (E084), unreadable place signal (E085),
readout removing most of it (E086/E087). All fixed: hippocampal routing (E086), frozen
centring baseline (E088), per-seed target selection resolving 5.1% at n=8 (E085), and an
anchor that suppresses pecking without freezing locomotion (E083).

The plant is fitted on LIVE hippocampal states and GATED on selectivity before the
behavioural ladder runs. E082 and E083 each passed an amplitude-only pre-flight while the
plant was useless; that gate is replaced, not supplemented, and it aborts.

See docs/experiments/E089-whole-chain-control-on-the-repaired-stack.md sections 1-5.
"""
import time, sys
from functools import partial
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, 'scratchpad')
import e083_leaving_anchor as E
import e085_repaired_instrument as M
from coop import spec, world
from hen import brain, connectome, plasticity, regions
from run import simulate

CEN, CFG, HENS, STEPS, R = M.CEN, M.CFG, M.HENS, M.STEPS, M.R
CENJ = jnp.asarray(CEN, dtype=jnp.float32)
SEEDS, EVERY = 8, M.SAMPLE_EVERY
reg = regions.DEFAULT_REGIONS
H_LO, H_HI = reg.bounds(regions.HIPPOCAMPUS)
HIPP = np.arange(H_LO, H_HI)
FREEZE = 60.0
GAINS = (0.0, 0.5, 1.0, 2.0)
GATE_ACC, GATE_RATIO = 0.80, 2.0


def build(k):
    return connectome.build(jax.random.fold_in(k, 1), reg, n_hens=HENS,
                            gakel_scaffold=True, shared_place_map=True,
                            place_to_hippocampus=True)


@partial(jax.jit, static_argnames=("cfg", "pc", "blocks", "every"))
def collect(w, x, p, ps, key, cfg, pc, blocks, every):
    def inner(carry, _):
        carry, _o = simulate._one_step(carry, None, cfg=cfg, pc=pc)
        d = jnp.linalg.norm(carry[0].pos[:, None, :] - CENJ[None, :, :], axis=-1)
        return carry, (d < R)
    def outer(carry, _):
        carry, ins = jax.lax.scan(inner, carry, None, length=every)
        d = jnp.linalg.norm(carry[0].pos[:, None, :] - CENJ[None, :, :], axis=-1)
        q = carry[3]
        return carry, (ins, d, q.z_lag - q.z_lag_bar)
    return jax.lax.scan(outer, (w, x, p, ps, key), None, length=blocks)[1]


def go(p, pc, wk, rk, food):
    w = world.reset(wk, CFG)
    w = w._replace(food_pos=jnp.asarray(food, dtype=jnp.float32))
    x = brain.initial_state(p, HENS); ps = plasticity.initial_state(p, HENS, pc)
    ins, d, z = collect(w, x, p, ps, rk, CFG, pc, STEPS // EVERY, EVERY)
    ins = np.asarray(ins)
    return ins.reshape(-1, *ins.shape[2:]), np.asarray(d), np.asarray(z)


@partial(jax.jit, static_argnames=("cfg", "pc", "blocks", "every", "tgt", "ctl"))
def ladder(w, x, p, ps, key, cfg, pc, blocks, every, tgt, ctl):
    def inner(carry, _):
        carry, (motor, obs, _r, _m) = simulate._one_step(carry, None, cfg=cfg, pc=pc)
        wl, q = carry[0], carry[3]
        dT = jnp.linalg.norm(wl.pos - CENJ[tgt], axis=-1)
        dC = jnp.linalg.norm(wl.pos - CENJ[ctl], axis=-1)
        pred = jnp.einsum("hon,hn->ho", p.W_pred,
                          (q.z_lag - q.z_lag_bar) * p.pred_src[None, :])
        g = jax.nn.relu(pred[:, E.GAKEL_CH])
        at = dT < R
        return carry, (at, dC < R, wl.hunger, motor[:, spec.M_FORWARD],
                       motor[:, spec.M_PECK] * at, at, g * at, g * ~at, ~at)
    return jax.lax.scan(inner, (w, x, p, ps, key), None, length=blocks * every)[1]


def fit_and_gate(p, pc, k):
    """Fit on the selection run, gate on the held-out test run. Aborts on failure."""
    food = np.stack([CEN[E.P], CEN[E.P2]])
    ins_s, dst_s, z_s = go(p, pc, k, jax.random.fold_in(k, 2), food)
    occ = ins_s.mean(axis=(0, 1))
    order = np.argsort(-occ)
    tgt, ctl = int(order[0]), int(order[1])
    # food at the chosen cells, so "declining to eat here" is a real option
    food = np.stack([CEN[tgt], CEN[ctl]])
    ins_s, dst_s, z_s = go(p, pc, k, jax.random.fold_in(k, 2), food)
    ins_t, dst_t, z_t = go(p, pc, k, jax.random.fold_in(k, 9), food)

    Ztr, Dtr = M.flat(z_s)[:, HIPP], M.flat(dst_s[:, :, tgt])
    Zte, Dte = M.flat(z_t)[:, HIPP], M.flat(dst_t[:, :, tgt])
    w_h, thr = M.fit(Ztr, Dtr, R)
    acc, ratio, prof = M.evaluate(w_h, thr, Zte, Dte, R)
    dec = bool(prof[0] > prof[1] > prof[2])

    full = np.zeros(np.asarray(p.pred_src).shape[0], dtype=np.float32)
    full[HIPP] = w_h
    full = full * np.asarray(p.pred_src)
    # Normalise so the prediction reads ~1.0 at the target, matching E082/E083's
    # convention so the gain ladder is comparable. `sP` is already restricted to the
    # hippocampal columns, since `Zte` is.
    sP = Zte[Dte < R].mean(0)
    scale = float(w_h @ sP)
    scale = scale if abs(scale) > 1e-9 else 1.0
    w_pred = np.zeros_like(np.asarray(p.W_pred))
    w_pred[:, E.GAKEL_CH, :] = full / (scale + 1e-9)
    return (p._replace(W_pred=jnp.asarray(w_pred)), tgt, ctl, food,
            float(acc), float(ratio), dec)


# Driver in a function so diagnostics can import `build` and `fit_and_gate`
# without re-running the 30-minute experiment. Same reason as
# e083_leaving_anchor.py -- see E085 section 6c.
def _main():
    print(f"E089 -- whole-chain control on the repaired stack, {SEEDS} seeds, "
          f"freeze {FREEZE:.0f}s")
    print("gate: held-out acc >=80%, selectivity >=2.0, decreasing profile -- aborts\n")
    t0 = time.perf_counter()
    pc_fit = plasticity.PlasticConfig(**E.FROZEN, pred_gain=0.0, pred_bar_freeze_s=FREEZE)
    PLANT = {}
    print(f"{'seed':>5}{'target':>8}{'ctrl':>6}{'acc':>9}{'ratio':>8}{'decreasing':>12}")
    for s in range(SEEDS):
        k = jax.random.key(s)
        PLANT[s] = fit_and_gate(build(k), pc_fit, k)
        _p, tgt, ctl, _f, acc, ratio, dec = PLANT[s]
        print(f"{s:>5}{tgt:>8}{ctl:>6}{acc:>9.1%}{ratio:>8.2f}{str(dec):>12}")
    accs = [v[4] for v in PLANT.values()]; ratios = [v[5] for v in PLANT.values()]
    decs = [v[6] for v in PLANT.values()]
    print(f"\nmean acc {np.mean(accs):.1%}  mean ratio {np.mean(ratios):.2f}  "
          f"decreasing {sum(decs)}/{SEEDS}")
    assert np.mean(accs) >= GATE_ACC, f"PLANT GATE FAILED: acc {np.mean(accs):.3f}"
    assert np.mean(ratios) >= GATE_RATIO, f"PLANT GATE FAILED: ratio {np.mean(ratios):.2f}"
    assert sum(decs) > SEEDS // 2, "PLANT GATE FAILED: profile not decreasing on most seeds"
    print("PLANT GATE PASSED -- the behavioural ladder is interpretable\n")

    print(f"{'gain':>6}{'occ target':>12}{'occ ctrl':>10}{'hunger':>8}{'fwd':>7}"
          f"{'peck@T':>8}{'pred@T':>8}{'pred elsewhere':>16}")
    ROWS = {}
    for gain in GAINS:
        pc = plasticity.PlasticConfig(**E.FROZEN, pred_gain=gain, pred_bar_freeze_s=FREEZE)
        acc = {n: [] for n in ("oT", "oC", "hu", "fw", "pk", "gT", "gO")}
        for s in range(SEEDS):
            p, tgt, ctl, food, *_ = PLANT[s]
            k = jax.random.key(s)
            w = world.reset(k, CFG)._replace(
                food_pos=jnp.asarray(food, dtype=jnp.float32))
            x = brain.initial_state(p, HENS); ps = plasticity.initial_state(p, HENS, pc)
            at, atC, hu, fw, pk, cnt, gT, gO, cntO = ladder(
                w, x, p, ps, jax.random.fold_in(k, 9), CFG, pc,
                STEPS // EVERY, EVERY, tgt, ctl)
            n_at = float(jnp.sum(cnt)); n_out = float(jnp.sum(cntO))
            acc["oT"].append(float(jnp.mean(at))); acc["oC"].append(float(jnp.mean(atC)))
            acc["hu"].append(float(jnp.mean(hu))); acc["fw"].append(float(jnp.mean(fw)))
            acc["pk"].append(float(jnp.sum(pk)) / max(n_at, 1))
            acc["gT"].append(float(jnp.sum(gT)) / max(n_at, 1))
            acc["gO"].append(float(jnp.sum(gO)) / max(n_out, 1))
        ROWS[gain] = {n: float(np.mean(v)) for n, v in acc.items()}
        r = ROWS[gain]
        print(f"{gain:>6.1f}{r['oT']:>12.4f}{r['oC']:>10.4f}{r['hu']:>8.3f}{r['fw']:>7.3f}"
              f"{r['pk']:>8.3f}{r['gT']:>8.3f}{r['gO']:>16.3f}")

    t0_, t2_ = ROWS[0.0]["oT"], ROWS[2.0]["oT"]
    c0_, c2_ = ROWS[0.0]["oC"], ROWS[2.0]["oC"]
    mono = all(ROWS[a]["oT"] >= ROWS[b]["oT"] for a, b in zip(GAINS, GAINS[1:]))
    print(f"\nwall clock: {time.perf_counter()-t0:.0f} s")
    print("--- pre-registered falsifiers (E089 section 4) ---")
    print(f"primary   occ target {t0_:.4f} -> {t2_:.4f} = {100*(t2_-t0_)/t0_:+.1f}% "
          f"(need <=-10%, predicted <=-15%, monotonic={mono}; MDE 5.1% at n=8) -> "
          f"{'clear' if ((t2_-t0_)/t0_ <= -0.10 and mono) else 'FIRES -- this is the STOP condition'}")
    print(f"agitation occ ctrl {c0_:.4f} -> {c2_:.4f} = {100*(c2_-c0_)/c0_:+.1f}% "
          f"(fires if <=-10%) -> {'FIRES' if (c2_-c0_)/c0_ <= -0.10 else 'clear'}")
    print(f"starve    hunger {ROWS[2.0]['hu']:.3f} (fires if >0.60) -> "
          f"{'FIRES' if ROWS[2.0]['hu'] > 0.60 else 'clear'}")



if __name__ == "__main__":
    _main()
