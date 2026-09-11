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


def main():
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    what = sys.argv[1] if len(sys.argv) > 1 else "partA"
    if what in ("partA", "all"):
        print(f"E121 part A: hawk_period={HAWK:.0f}, {N_SEEDS} seeds, "
              f"{TRACE_S:.0f}s traces\n")
        cache["partA"] = part_a()
        json.dump(cache, open(CACHE, "w"), indent=1)


if __name__ == "__main__":
    main()
