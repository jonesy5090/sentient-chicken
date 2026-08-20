"""E084 Part A: is 'where she is' linearly decodable from pallial state while MOVING?

E083 found E082/E083's plant anti-selective in the live run and diagnosed a regime
mismatch: the discriminant is fitted on a hen parked at a grid centre and read back on
a hen moving. This fits the SAME estimator on live trajectory states and evaluates it
on a held-out run -- same connectome, different world key -- so any difference is
attributable to regime and not to a better fitting method.

Also evaluates the parked-fit plant on that same held-out data, which tests E083's
diagnosis on data that did not generate it.

The sampled state is `z_lag - z_lag_bar` masked by `pred_src` -- what the runtime
readout actually consumes, not raw rate(x).
"""
import time, sys
from functools import partial
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, 'scratchpad')
import e083_leaving_anchor as E
from coop import world
from hen import brain, connectome, plasticity, regions
from run import simulate

CEN, P, P2, CFG, HENS, STEPS = E.CEN, E.P, E.P2, E.CFG, E.HENS, E.STEPS
SAMPLE_EVERY = 100                       # 1 s at dt=0.01 -> 1200 samples/run
RADII = (3.33, 1.5)
EDGES = np.array([0.0, 1.0, 2.0, 3.33, 5.0, 7.0, 10.0, 99.0])
NB = len(EDGES) - 1
GATE_ACC, GATE_RATIO = 0.70, 2.0


@partial(jax.jit, static_argnames=("cfg", "pc", "n", "every"))
def collect(w, x, p, ps, key, cfg, pc, n, every):
    """Free-running, no plant installed. Returns sampled (state, distance-to-P)."""
    def step(carry, i):
        carry, _out = simulate._one_step(carry, None, cfg=cfg, pc=pc)
        z = (carry[3].z_lag - carry[3].z_lag_bar) * p.pred_src[None, :]
        dP = jnp.linalg.norm(carry[0].pos - jnp.asarray(CEN[P], dtype=jnp.float32), axis=-1)
        keep = (i % every) == 0
        return carry, (z, dP, keep)
    _c, (z, d, keep) = jax.lax.scan(step, (w, x, p, ps, key), jnp.arange(n))
    return z, d, keep


def run_one(p, pc, world_key, run_key):
    w = world.reset(world_key, CFG)
    w = w._replace(food_pos=jnp.asarray(np.stack([CEN[P], CEN[P2]]), dtype=jnp.float32))
    x = brain.initial_state(p, HENS)
    ps = plasticity.initial_state(p, HENS, pc)
    z, d, keep = collect(w, x, p, ps, run_key, CFG, pc, STEPS, SAMPLE_EVERY)
    m = np.asarray(keep)
    # (steps, hens, N) -> (samples, N); hens are independent draws from the same policy
    Z = np.asarray(z)[m].reshape(-1, np.asarray(z).shape[-1])
    D = np.asarray(d)[m].reshape(-1)
    return Z, D


def fit(Z, D, radius):
    """The estimator E082 used, unchanged. Only the data it sees is different."""
    at = D < radius
    w = Z[at].mean(0) - Z[~at].mean(0)
    w = w / (np.linalg.norm(w) + 1e-9)
    thr = 0.5 * ((Z[at] @ w).mean() + (Z[~at] @ w).mean())
    return w, thr, at.mean()


def evaluate(w, thr, Z, D, radius):
    at = D < radius
    n_at, n_out = int(at.sum()), int((~at).sum())
    if n_at == 0 or n_out == 0:
        raise ValueError(f"degenerate split at radius {radius}: {n_at} in / {n_out} out")
    if not np.isfinite(Z).all():
        raise ValueError(f"non-finite states: {int((~np.isfinite(Z)).sum())} entries")
    s = Z @ w
    acc = 0.5 * ((s[at] > thr).mean() + (s[~at] <= thr).mean())
    g = np.maximum(s, 0.0)
    ratio = g[at].mean() / max(g[~at].mean(), 1e-9)
    b = np.clip(np.searchsorted(EDGES, D) - 1, 0, NB - 1)
    prof = np.array([g[b == i].mean() if (b == i).any() else np.nan for i in range(NB)])
    return acc, ratio, prof / max(np.nanmean(prof), 1e-9)


if __name__ != "__main__":
    raise SystemExit

pc = plasticity.PlasticConfig(**E.FROZEN, pred_gain=0.0)
t0 = time.perf_counter()
print(f"E084 Part A -- live-fitted vs parked-fitted place discriminant")
print(f"P = cell {P} {CEN[P].round(1)}, {E.SEEDS} seeds, {E.MINUTES:.0f} min/run, "
      f"sample every {SAMPLE_EVERY} steps\n")

res = {r: {"live_te": [], "live_tr": [], "park_te": [],
           "rl": [], "rp": [], "pl": [], "pp": []} for r in RADII}
base = []
for s in range(E.SEEDS):
    k = jax.random.key(s)
    p = connectome.build(jax.random.fold_in(k, 1), regions.DEFAULT_REGIONS, n_hens=HENS,
                         gakel_scaffold=True, shared_place_map=True)
    Ztr, Dtr = run_one(p, pc, k, jax.random.fold_in(k, 2))                 # train run
    Zte, Dte = run_one(p, pc, jax.random.fold_in(k, 6), jax.random.fold_in(k, 7))
    base.append((Dtr < RADII[0]).mean())

    # the parked-fit plant E082/E083 used, as a directly comparable vector
    p_pl, _ = E.plant(p, E._PLANT_CFG)
    w_park = np.asarray(p_pl.W_pred)[0, E.GAKEL_CH, :]
    w_park = w_park / (np.linalg.norm(w_park) + 1e-9)

    for r in RADII:
        w_live, thr, _ = fit(Ztr, Dtr, r)
        a_te, ra_te, pr_te = evaluate(w_live, thr, Zte, Dte, r)
        a_tr, _, _ = evaluate(w_live, thr, Ztr, Dtr, r)
        at = Dtr < r
        thr_p = 0.5 * ((Ztr[at] @ w_park).mean() + (Ztr[~at] @ w_park).mean())
        a_pk, ra_pk, pr_pk = evaluate(w_park, thr_p, Zte, Dte, r)
        res[r]["live_te"].append(a_te); res[r]["live_tr"].append(a_tr)
        res[r]["park_te"].append(a_pk)
        res[r]["rl"].append(ra_te); res[r]["rp"].append(ra_pk)
        res[r]["pl"].append(pr_te); res[r]["pp"].append(pr_pk)
        print(f"  seed {s} r={r:.2f}: live held-out {a_te:.1%} (train {a_tr:.1%}, "
              f"ratio {ra_te:.2f}) | parked held-out {a_pk:.1%} (ratio {ra_pk:.2f})")

print(f"base rate 'at feeder' (radius {RADII[0]} m): {np.mean(base):.3f}\n")
print(f"{'radius':>8}{'fit':>8}{'held-out acc':>14}{'train acc':>11}{'ratio':>8}")
for r in RADII:
    d = res[r]
    print(f"{r:>8.2f}{'live':>8}{np.mean(d['live_te']):>14.1%}"
          f"{np.mean(d['live_tr']):>11.1%}{np.mean(d['rl']):>8.2f}")
    print(f"{'':>8}{'parked':>8}{np.mean(d['park_te']):>14.1%}{'--':>11}"
          f"{np.mean(d['rp']):>8.2f}")

print(f"\ndistance profile on held-out data (relative pred, radius {RADII[0]} m fit)")
print(f"{'bin (m)':>14}{'live-fit':>11}{'parked-fit':>13}")
PL, PP = np.nanmean(res[RADII[0]]["pl"], 0), np.nanmean(res[RADII[0]]["pp"], 0)
for i in range(NB):
    print(f"{f'{EDGES[i]:.1f}-{EDGES[i+1]:.1f}':>14}{PL[i]:>11.3f}{PP[i]:>13.3f}")

acc = np.mean(res[RADII[0]]["live_te"]); ratio = np.mean(res[RADII[0]]["rl"])
tr = np.mean(res[RADII[0]]["live_tr"])
dec = bool(PL[0] > PL[1] > PL[2])
print(f"\nwall clock: {time.perf_counter()-t0:.0f} s")
print("--- pre-registered falsifiers (E084 section 4) ---")
print(f"gate      acc {acc:.1%} (need >={GATE_ACC:.0%}), ratio {ratio:.2f} "
      f"(need >={GATE_RATIO}), profile decreasing={dec} -> "
      f"{'PASS' if (acc >= GATE_ACC and ratio >= GATE_RATIO and dec) else 'FIRES'}")
print(f"diagnosis parked-fit held-out acc {np.mean(res[RADII[0]]['park_te']):.1%} "
      f"(E083 wrong if >=70%) -> "
      f"{'FIRES' if np.mean(res[RADII[0]]['park_te']) >= 0.70 else 'clear'}")
print(f"leakage   train {tr:.1%} vs held-out {acc:.1%} = {100*(tr-acc):+.1f} pts "
      f"(fires if >15) -> {'FIRES' if (tr - acc) > 0.15 else 'clear'}")
