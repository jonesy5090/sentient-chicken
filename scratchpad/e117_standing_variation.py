"""E117 -- where does standing variation come from?

E116 fixed a flock's zero fitness repeatability by mutating founders at sigma=2.0, an
amplification chosen because it worked. This asks which *construction choice* in
`connectome.build` is responsible, across six arms, with a real error bar (8 genome
seeds, Fisher-z averaged) instead of E116's single n=16 correlations.

Run:  PYTHONPATH=. python scratchpad/e117_standing_variation.py
"""

import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from coop import spec, world
from hen import brain, connectome, plasticity, regions
from run import evolve, simulate

LIFETIME_S = 600.0
N_SEEDS = 8
HAWK_PERIOD = 50.0
CFG = spec.DEFAULT_COOP._replace(hawk_period_s=HAWK_PERIOD)
STEPS = int(LIFETIME_S / CFG.dt)
REG = regions.DEFAULT_REGIONS
EVO = evolve.EvoConfig()
GAIN = 0.95           # `connectome.build`'s default, re-stated so the per-hen-mask arm
                      # rebuilds `W` on exactly the same scale as every other arm.

# Block 1 is genome seeds 1000-1007. `E117_BLOCK=2` moves to 1008-1015 -- a *fresh*
# block, because E021 turned a t=3.84 into a t=0.01 by re-measuring the same contrast on
# different seeds, and this project does not move a status on one block.
BLOCK = int(os.environ.get("E117_BLOCK", "1"))
SEED_BASE = 1000 + (BLOCK - 1) * N_SEEDS
CACHE = f"scratchpad/e117_cache{'' if BLOCK == 1 else f'_b{BLOCK}'}.json"


# --------------------------------------------------------------------------- arms

def _per_hen_mask(p, key, n_hens):
    """Give each hen her own connectivity draw at the same density.

    `p.mask` is read only by structural growth (off here) and by reporting, never by
    `brain.step`, so applying an independent per-hen mask directly to `W` is a genuine
    per-hen anatomy. The density is matched to the shared mask's so this arm varies
    *which* synapses exist, not how many -- the confound E035 was built out of.
    """
    n = p.mask.shape[0]
    # Sample per-hen masks from the *same* region-pair probability matrix the shared mask
    # came from, so this arm varies which synapses exist and not how many, and every
    # region pair keeps its own connectivity. Anything coarser (a single global density)
    # would flatten the architecture and confound "different wiring" with "different
    # brain".
    rid = connectome._region_of(REG)
    p_reg = np.asarray(regions.REGION_CONNECTIVITY, dtype=np.float32)
    p_ij = jnp.asarray(p_reg[rid[None, :], rid[:, None]])          # (N, N)
    m = jax.random.uniform(key, (n_hens, n, n)) < p_ij[None]
    m = m & ~jnp.eye(n, dtype=bool)[None]
    # Each hen's weights are renormalised by her own fan-in, exactly as `build` does it,
    # so a different mask does not also mean a different total drive -- the confound
    # E035 found in the modality-segregation figures.
    fan_hen = jnp.maximum(jnp.sum(m, axis=2, keepdims=True), 1.0)
    mag = jnp.abs(jax.random.normal(jax.random.fold_in(key, 3), p.W.shape))
    w = mag * (GAIN / jnp.sqrt(fan_hen)) * m * p.dale[None, None, :]
    return p._replace(W=w)


def build_arm(arm: str, key):
    """Returns (params, plastic_config)."""
    pc = evolve.NO_LEARNING
    kw = dict(n_hens=CFG.n_hens)

    if arm == "readout_0.25":
        p = connectome.build(key, REG, readout_scale=0.25, **kw)
    elif arm == "readout_1.00":
        p = connectome.build(key, REG, readout_scale=1.00, **kw)
    else:
        p = connectome.build(key, REG, **kw)

    if arm == "founder2":
        p = evolve._mutate(p, jax.random.fold_in(key, 77), 2.0)
    elif arm == "shared_shift":
        # The control that separates the two things every "successful" arm changes at
        # once. `readout_0.25` and `founder2` both raised repeatability, and both also
        # moved the flock's motor output a long way toward saturation (crouch 0.50 and
        # 0.33 against a default of 0.16) -- the same axis whose far end, `readout_1.00`,
        # is a bird doing everything flat out.
        #
        # So: shift the operating point by exactly as much as `readout_0.25` does, while
        # adding **no between-hen variation at all**. Adding 4x hen 0's own readout to
        # every hen multiplies the mean magnitude by 5, matching `readout_scale=0.25`,
        # and leaves the between-hen spread of `W_out` untouched at 0.0445, because a
        # constant added to every hen cancels out of every difference between them.
        #
        # If repeatability rises anyway, it was never about individual differences.
        p = p._replace(W_out=p.W_out + 4.0 * p.W_out[0][None])
    elif arm == "per_hen_mask":
        p = _per_hen_mask(p, jax.random.fold_in(key, 91), CFG.n_hens)
    elif arm == "explore_0.1":
        pc = pc._replace(explore_sigma=0.1)

    return p, pc


ARMS = ["default", "founder2", "readout_0.25", "readout_1.00", "shared_shift",
        "per_hen_mask", "explore_0.1"]


# ------------------------------------------------------------------- measurement

def one_run(p, pc, k_world, k_run):
    w0 = world.reset(k_world, CFG)
    x0 = brain.initial_state(p, CFG.n_hens)
    ps = plasticity.initial_state(p, CFG.n_hens, pc)
    w_end, *_ = simulate.rollout_quiet(w0, x0, p, k_run, CFG, STEPS, ps, pc)
    drives = w_end.hunger + w_end.cold + w_end.thirst
    caught = w_end.n_caught_any.astype(jnp.float32)
    return (np.asarray(evolve.fitness(w_end, CFG, EVO)),
            np.asarray(-drives),
            np.asarray(-EVO.caught_weight * caught),
            float(jnp.mean(w_end.hunger)))


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def fisher_mean(rs):
    """Mean correlation via Fisher z, plus SE and t on the z scale (df = n-1)."""
    z = np.arctanh(np.clip(np.asarray(rs), -0.999, 0.999))
    m, se = z.mean(), z.std(ddof=1) / np.sqrt(len(z))
    return float(np.tanh(m)), float(se), float(m / se if se > 1e-12 else 0.0)


def measure(arm):
    rs_f, rs_d, rs_c, hungers, fits = [], [], [], [], []
    wout_sd, seeds_used = [], []
    for g in range(N_SEEDS):
        k = jax.random.key(SEED_BASE + g)
        k_gen, k_world, k_run = jax.random.split(k, 3)
        p, pc = build_arm(arm, k_gen)
        # between-hen spread of the readout, CLAUDE.md check 1
        wout_sd.append(float(jnp.mean(jnp.std(p.W_out, axis=0))))
        fA, dA, cA, hA = one_run(p, pc, k_world, jax.random.fold_in(k_run, 0))
        fB, dB, cB, hB = one_run(p, pc, k_world, jax.random.fold_in(k_run, 1))
        rs_f.append(pearson(fA, fB))
        rs_d.append(pearson(dA, dB))
        rs_c.append(pearson(cA, cB))
        hungers.append(0.5 * (hA + hB))
        fits.append(0.5 * (fA.mean() + fB.mean()))
        seeds_used.append(g)
        print(f"    seed {g}: r_fit={rs_f[-1]:+.3f} r_drives={rs_d[-1]:+.3f} "
              f"r_caught={rs_c[-1]:+.3f} hunger={hungers[-1]:.4f}", flush=True)

    r, se, t = fisher_mean(rs_f)
    rd, sed, td = fisher_mean(rs_d)
    rc, sec, tc = fisher_mean(rs_c)
    return dict(arm=arm, r_fit=r, se_fit=se, t_fit=t,
                r_drives=rd, se_drives=sed, t_drives=td,
                r_caught=rc, se_caught=sec, t_caught=tc,
                mean_hunger=float(np.mean(hungers)),
                mean_fitness=float(np.mean(fits)),
                wout_sd=float(np.mean(wout_sd)),
                per_seed_fit=rs_f, per_seed_drives=rs_d, per_seed_caught=rs_c)


def main():
    only = sys.argv[1:] or ARMS
    cache = {}
    if os.path.exists(CACHE):
        cache = json.load(open(CACHE))

    print(f"E117: lifetime={LIFETIME_S:.0f}s hawk_period={HAWK_PERIOD:.0f}s "
          f"hens={CFG.n_hens} seeds={N_SEEDS}\n")

    for arm in only:
        if arm in cache:
            print(f"[{arm}] cached")
            continue
        print(f"[{arm}]", flush=True)
        t0 = time.time()
        cache[arm] = measure(arm)
        cache[arm]["secs"] = time.time() - t0
        json.dump(cache, open(CACHE, "w"), indent=1)
        print(f"  -> r_fit={cache[arm]['r_fit']:+.3f} "
              f"({time.time()-t0:.0f}s)\n", flush=True)

    print("\n" + "=" * 92)
    print(f"{'arm':>14} {'r(fitness)':>18} {'r(-drives)':>18} {'r(predation)':>18} "
          f"{'hunger':>8} {'fitness':>9}")
    print("=" * 92)
    for arm in ARMS:
        if arm not in cache:
            continue
        c = cache[arm]
        print(f"{arm:>14} "
              f"{c['r_fit']:+.3f}+-{c['se_fit']:.3f} t={c['t_fit']:+5.2f} "
              f"{c['r_drives']:+.3f}+-{c['se_drives']:.3f} t={c['t_drives']:+5.2f} "
              f"{c['r_caught']:+.3f}+-{c['se_caught']:.3f} t={c['t_caught']:+5.2f} "
              f"{c['mean_hunger']:8.4f} {c['mean_fitness']:9.4f}")
    print("=" * 92)
    print("threshold |t| = 2.365 (df=7, two-sided 0.05)")
    print("\nW_out between-hen spread (did the manipulation move anything):")
    for arm in ARMS:
        if arm in cache:
            print(f"  {arm:>14}: {cache[arm]['wout_sd']:.5f}")


if __name__ == "__main__":
    main()
