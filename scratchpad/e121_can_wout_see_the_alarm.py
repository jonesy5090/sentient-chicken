"""E121 -- can the evolvable pathway even see the alarm?

`brain.step` computes `cortical = einsum(W_out, motor_stub)` where `motor_stub` is the
MOTOR region's rates. So `W_out` -- one of the two things evolution can touch -- reads the
motor stub and nothing else. E120's null has a leading explanation: the alarm never gets
there. This measures it, then tries to plant a comprehension where it should live.

Part A: decode "an aerial alarm is audible" from sensory stub / pallium / motor stub,
        among hen-steps where she CANNOT see a hawk herself.
Part B: write the motor-stub discriminant into W_out's crouch row and see if it saves hens.

Run:  PYTHONPATH=. python scratchpad/e121_can_wout_see_the_alarm.py [partA|partB]
"""

import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

from coop import spec, world
from hen import brain, neurons, plasticity, regions
from run import evolve, simulate
from scratchpad.e120_ladder_checks import AERIAL_CH, founders

HAWK = 20.0
N_SEEDS = 8
TRACE_S = 240.0
CHUNK = 10
CACHE = "scratchpad/e121_cache.json"
REG = regions.DEFAULT_REGIONS


def cfg():
    return spec.DEFAULT_COOP._replace(hawk_period_s=HAWK, channel_mode="intact",
                                      call_log_steps=spec.YOKE_LOG_STEPS)


def auc(scores, labels):
    """Rank-based AUC. 0.5 = chance."""
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos < 5 or n_neg < 5:
        return np.nan
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def lda_direction(X, y):
    """Diagonal-covariance linear discriminant: (mu_pos - mu_neg) / var. Returns (D,).

    Diagonal rather than full covariance deliberately -- with a few hundred units and a
    few thousand samples a full covariance inverse overfits, and E081 already found a
    decodability number inflated that way.
    """
    y = np.asarray(y).astype(bool)
    mu1, mu0 = X[y].mean(0), X[~y].mean(0)
    var = X.var(0) + 1e-6
    return (mu1 - mu0) / var


def collect(seed):
    """Chunked replay recording obs and internal rates at every chunk boundary."""
    c = cfg()
    pc = evolve.NO_LEARNING
    key = jax.random.key(6000 + seed)
    p = founders(jax.random.fold_in(key, 0), c)
    w = world.reset(jax.random.fold_in(key, 1), c)
    x = brain.initial_state(p, c.n_hens)
    ps = plasticity.initial_state(p, c.n_hens, pc)
    k = jax.random.fold_in(key, 2)

    from coop import sensing
    obs_l, rate_l = [], []
    for _ in range(int(TRACE_S / c.dt) // CHUNK):
        obs_l.append(np.asarray(sensing.observe(w, c)))
        rate_l.append(np.asarray(neurons.rate(x)))
        w, x, _, ps, k = simulate.rollout_quiet(w, x, p, k, c, CHUNK, ps, pc)
    return np.concatenate(obs_l), np.concatenate(rate_l), p


def part_a():
    s_lo, s_hi = REG.bounds(regions.SENSORY)
    p_lo, p_hi = REG.bounds(regions.PALLIUM)
    m_lo, m_hi = REG.bounds(regions.MOTOR)
    bands = [("sensory stub", s_lo, s_hi), ("pallium", p_lo, p_hi),
             ("motor stub", m_lo, m_hi)]

    out = {n: {"single": [], "lda": []} for n, _, _ in bands}
    frac_kept, n_pos = [], []
    for g in range(N_SEEDS):
        obs, rates, _ = collect(g)
        heard = obs[:, AERIAL_CH]
        seen = obs[:, spec.IDX_AERIAL]
        # She must not be able to see a hawk herself, or "alarm audible" is just
        # "hawk overhead" and the decoder reads her own eyes.
        blind = seen <= np.percentile(seen, 25)
        lab = heard[blind] > np.percentile(heard[blind], 75)
        frac_kept.append(float(blind.mean()))
        n_pos.append(int(lab.sum()))
        if lab.sum() < 20 or (~lab).sum() < 20:
            print(f"    seed {g}: degenerate labels, skipped")
            continue
        for name, lo, hi in bands:
            X = rates[blind, lo:hi]
            singles = [auc(X[:, j], lab) for j in range(X.shape[1])]
            best = float(np.nanmax([max(a, 1 - a) for a in singles]))
            w = lda_direction(X, lab)
            proj = X @ w
            a = auc(proj, lab)
            out[name]["single"].append(best)
            out[name]["lda"].append(float(max(a, 1 - a)))
        print(f"    seed {g}: motor-stub LDA AUC={out['motor stub']['lda'][-1]:.4f} "
              f"(blind {frac_kept[-1]:.0%}, {n_pos[-1]} positives)", flush=True)

    res = {n: {k: (float(np.mean(v)), float(np.std(v) / np.sqrt(max(len(v), 1))))
               for k, v in d.items()} for n, d in out.items()}
    print("\n" + "=" * 78)
    print("A. Where does the alarm signal die? (AUC for 'alarm audible', hen blind)")
    print("=" * 78)
    print(f"{'region':>14} {'best single unit':>22} {'pooled (LDA)':>22}")
    for n, _, _ in bands:
        s, se = res[n]["single"]
        l, lse = res[n]["lda"]
        print(f"{n:>14} {s:.4f} +- {se:.4f}      {l:.4f} +- {lse:.4f}")
    print("\n0.5 = chance. `W_out` reads the motor stub and nothing else, so the last")
    print("row is the ceiling on any comprehension evolution could ever express.")
    return res


def plant_direction(rates, obs, n_hens, m_lo, m_hi, rng):
    """Per-hen crouch-on-alarm direction over the motor stub, and a matched scramble.

    Returned directions are scaled so that the projection difference between
    "alarm audible" and "not" is exactly 1.0, which makes the sweep's `gain` read
    directly as *extra crouch drive when she hears an alarm* -- comparable to the reflex
    arc's own magnitudes (E019 records hearing an alarm driving crouch to
    sigmoid(1.5-2.5)=0.269, so drives of order 1-3 are what matter here).

    Each direction is also made orthogonal to the mean motor-stub rate, so the plant adds
    ~no constant drive to crouch and responds only to deviations. Without that, a plant
    would raise baseline crouching and "fewer catches" would mean "crouches all the time"
    -- E117's `readout_1.00` failure. The scramble permutes the direction across units,
    keeping the norm, the channel and the DC treatment identical while destroying the
    alignment with alarm-coding units.
    """
    T = rates.shape[0] // n_hens
    R = rates.reshape(T, n_hens, -1)[:, :, m_lo:m_hi]
    O = obs.reshape(T, n_hens, -1)
    real, scram = [], []
    for h in range(n_hens):
        heard, seen = O[:, h, AERIAL_CH], O[:, h, spec.IDX_AERIAL]
        blind = seen <= np.percentile(seen, 25)
        X = R[blind, h]
        if X.shape[0] < 40:
            real.append(np.zeros(m_hi - m_lo))
            scram.append(np.zeros(m_hi - m_lo))
            continue
        lab = heard[blind] > np.percentile(heard[blind], 75)
        if lab.sum() < 10 or (~lab).sum() < 10:
            real.append(np.zeros(m_hi - m_lo))
            scram.append(np.zeros(m_hi - m_lo))
            continue
        w = lda_direction(X, lab)
        rbar = X.mean(0)
        w = w - (w @ rbar) / (rbar @ rbar + 1e-9) * rbar      # no constant drive
        # Scale so "alarm audible" projects exactly 1.0 higher than "not", making the
        # sweep's `gain` read as extra crouch drive per heard alarm.
        sep = float((X[lab] @ w).mean() - (X[~lab] @ w).mean())
        w = w / (sep + 1e-9)
        real.append(w)
        ws = w[rng.permutation(len(w))]
        ws = ws - (ws @ rbar) / (rbar @ rbar + 1e-9) * rbar
        scram.append(ws)
    return np.array(real), np.array(scram)


def part_b(gains=(0.01, 0.02, 0.05, 0.1, 0.25)):
    """Gains chosen after a first sweep at 0.5-4.0 landed entirely in a degenerate regime.

    There, crouch sat at 0.84-0.89 and only 8-14% of unplanted catches remained, so both
    arms were on the floor with no room to differ -- CLAUDE.md check 5, "does the dependent
    variable have room to move in both directions". The first sweep is kept in the cache
    and reported, because the scramble beating unplanted almost as much as the plant did
    is itself the evidence that the regime was degenerate.
    """
    c = cfg()
    pc = evolve.NO_LEARNING
    m_lo, m_hi = REG.bounds(regions.MOTOR)
    rows = {}
    for g in range(N_SEEDS):
        obs, rates, p = collect(g)
        rng = np.random.default_rng(1234 + g)
        real, scram = plant_direction(obs=obs, rates=rates, n_hens=c.n_hens,
                                      m_lo=m_lo, m_hi=m_hi, rng=rng)
        key = jax.random.key(6000 + g)
        for arm, W in (("plant", real), ("scram", scram)):
            for gain in gains:
                wo = np.asarray(p.W_out).copy()
                wo[:, spec.M_CROUCH, :] += gain * W
                pp = p._replace(W_out=jnp.asarray(wo))
                w0 = world.reset(jax.random.fold_in(key, 1), c)
                x0 = brain.initial_state(pp, c.n_hens)
                ps = plasticity.initial_state(pp, c.n_hens, pc)
                w_end, *_ = simulate.rollout_quiet(
                    w0, x0, pp, jax.random.fold_in(key, 9), c,
                    int(600.0 / c.dt), ps, pc)
                _, _, _, _, _, tr = simulate.rollout(
                    world.reset(jax.random.fold_in(key, 1), c),
                    brain.initial_state(pp, c.n_hens), pp,
                    jax.random.fold_in(key, 9), c, int(60.0 / c.dt),
                    plasticity.initial_state(pp, c.n_hens, pc), pc)
                rows.setdefault((arm, gain), []).append(dict(
                    caught=float(jnp.mean(w_end.n_caught_any)),
                    hunger=float(jnp.mean(w_end.hunger)),
                    crouch=float(jnp.mean(tr.motor[:, :, spec.M_CROUCH]))))
        # Unplanted reference, once per seed -- including its crouch rate, which the
        # first sweep omitted and which is the baseline the degeneracy check needs.
        w0 = world.reset(jax.random.fold_in(key, 1), c)
        w_end, *_ = simulate.rollout_quiet(
            w0, brain.initial_state(p, c.n_hens), p, jax.random.fold_in(key, 9), c,
            int(600.0 / c.dt), plasticity.initial_state(p, c.n_hens, pc), pc)
        _, _, _, _, _, tr0 = simulate.rollout(
            world.reset(jax.random.fold_in(key, 1), c),
            brain.initial_state(p, c.n_hens), p, jax.random.fold_in(key, 9), c,
            int(60.0 / c.dt), plasticity.initial_state(p, c.n_hens, pc), pc)
        rows.setdefault(("none", 0.0), []).append(dict(
            caught=float(jnp.mean(w_end.n_caught_any)),
            hunger=float(jnp.mean(w_end.hunger)),
            crouch=float(jnp.mean(tr0.motor[:, :, spec.M_CROUCH]))))
        print(f"    seed {g} done", flush=True)

    print("\n" + "=" * 88)
    print("B. Plant the motor-stub alarm direction into W_out's crouch row")
    print("=" * 88)
    base = np.array([r["caught"] for r in rows[("none", 0.0)]])
    print(f"{'arm':>7} {'gain':>5} {'caught/hen':>22} {'vs unplanted':>22} "
          f"{'crouch':>7} {'hunger':>7}")
    print(f"{'none':>7} {0.0:>5.2f} {base.mean():>22.3f} {'--':>22} "
          f"{np.mean([r['crouch'] for r in rows[('none',0.0)]]):>7.3f} "
          f"{np.mean([r['hunger'] for r in rows[('none',0.0)]]):>7.4f}")
    for arm in ("plant", "scram"):
        for gain in gains:
            v = rows[(arm, gain)]
            cg = np.array([r["caught"] for r in v])
            d = cg - base
            se = d.std(ddof=1) / np.sqrt(len(d))
            print(f"{arm:>7} {gain:>5.2f} {cg.mean():>22.3f} "
                  f"{d.mean():+.3f}+-{se:.3f} t={d.mean()/se:+5.2f} "
                  f"{np.mean([r['crouch'] for r in v]):>7.3f} "
                  f"{np.mean([r['hunger'] for r in v]):>7.4f}")
    print("\nReference: the reflex scaffold is worth -1.195 catches/hen (E120 6c).")
    print("`plant` must beat `scram` at matched gain, or the benefit is extra crouching")
    print("rather than comprehension. Crouch rate is the degeneracy check.")
    return {f"{a}_{g}": v for (a, g), v in rows.items()}


def main():
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    what = sys.argv[1] if len(sys.argv) > 1 else "partA"
    if what in ("partA", "all"):
        print(f"E121 part A: hawk_period={HAWK:.0f}, {N_SEEDS} seeds, "
              f"{TRACE_S:.0f}s traces\n")
        cache["partA"] = part_a()
        json.dump(cache, open(CACHE, "w"), indent=1)
    if what in ("partB", "all"):
        print(f"\nE121 part B: planting into W_out, {N_SEEDS} seeds\n")
        cache["partB"] = part_b()
        json.dump(cache, open(CACHE, "w"), indent=1)


if __name__ == "__main__":
    main()
