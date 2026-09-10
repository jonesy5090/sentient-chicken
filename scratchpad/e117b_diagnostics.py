"""E117 diagnostics -- the three things the main run owes an explanation for.

(a) `readout_1.00` scored repeatability +0.871 with predation variance *exactly* zero and
    hunger up 0.46 -> 0.61. Section 4 pre-registered that an arm can raise repeatability
    by breaking hens rather than differentiating them. This checks which happened.
(b) `per_hen_mask` moved nothing. Did the manipulated variable actually vary?
    (CLAUDE.md check 1 -- the one this project has been burned by most.)
(c) `explore_0.1` gave repeatability *significantly negative* (-0.118, t=-3.63). A hen
    who does well in one run does slightly worse in the replicate. The obvious mechanism
    is competition: `food_deplete_rate=2e-2` makes food a shared, exhaustible resource,
    so one hen's meal is taken from the flock's total.

Run:  PYTHONPATH=. python scratchpad/e117b_diagnostics.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from coop import spec, world
from hen import brain, connectome, plasticity, regions
from run import evolve, simulate
from scratchpad.e117_standing_variation import (CFG, GAIN, REG, STEPS, build_arm,
                                                fisher_mean, one_run, pearson)

MOTOR = ["forward", "turn_L", "turn_R", "peck", "scratch", "crouch", "flee",
         "c_contact", "c_food", "c_aerial", "c_ground", "c_gakel"]


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


# ---------------------------------------------------------------- (a) is it broken?

def diag_readout():
    section("(a) readout_1.00 -- differentiated, or catatonic?")
    print(f"{'arm':>13} {'hunger':>8} {'caught/hen':>11} {'dives':>7} "
          + " ".join(f"{m:>9}" for m in ("forward", "peck", "crouch", "flee")))
    for arm in ("default", "readout_0.25", "readout_1.00", "founder2"):
        k = jax.random.key(1000)
        k_gen, k_world, k_run = jax.random.split(k, 3)
        p, pc = build_arm(arm, k_gen)
        w0 = world.reset(k_world, CFG)
        x0 = brain.initial_state(p, CFG.n_hens)
        ps = plasticity.initial_state(p, CFG.n_hens, pc)
        w_end, *_ = simulate.rollout_quiet(w0, x0, p, jax.random.fold_in(k_run, 0),
                                           CFG, STEPS, ps, pc)
        # A short traced run for the motor profile -- 60 s is plenty for a steady state.
        w0b = world.reset(k_world, CFG)
        _, _, _, _, _, tr = simulate.rollout(
            w0b, brain.initial_state(p, CFG.n_hens), p,
            jax.random.fold_in(k_run, 0), CFG, int(60.0 / CFG.dt),
            plasticity.initial_state(p, CFG.n_hens, pc), pc)
        m = np.asarray(jnp.mean(tr.motor, axis=(0, 1)))
        c = np.asarray(w_end.n_caught_any)
        print(f"{arm:>13} {float(jnp.mean(w_end.hunger)):8.4f} "
              f"{c.mean():11.2f} {float(jnp.sum(w_end.n_dives)):7.0f} "
              + " ".join(f"{m[i]:9.4f}" for i in
                         (spec.M_FORWARD, spec.M_PECK, spec.M_CROUCH, spec.M_FLEE)))
        print(f"{'':>13} per-hen caught: {np.array2string(c.astype(int), max_line_width=200)}")


# ------------------------------------------------------- (b) did the mask actually vary?

def diag_mask():
    section("(b) per_hen_mask -- did the manipulated variable vary?")
    k = jax.random.key(1000)
    k_gen, _, _ = jax.random.split(k, 3)
    p_def, _ = build_arm("default", k_gen)
    p_var, _ = build_arm("per_hen_mask", k_gen)

    live_def = (p_def.W != 0.0)
    live_var = (p_var.W != 0.0)
    for name, live in (("default", live_def), ("per_hen_mask", live_var)):
        a, b = live[0], live[1]
        differ = float(jnp.mean(a != b))
        dens = float(jnp.mean(live))
        # between-hen spread of the weights themselves
        sd = float(jnp.mean(jnp.std(live.astype(jnp.float32), axis=0)))
        print(f"  {name:>13}: density={dens:.4f}  hen0-vs-hen1 synapses differing="
              f"{differ*100:6.2f}%   between-hen sd of connectivity={sd:.4f}")
    print("\n  If per_hen_mask's 'differing' is ~0 the arm never varied anything and its")
    print("  null is about the harness, not about anatomy.")


# ------------------------------------------------- (c) is the flock's fitness zero-sum?

def diag_competition(n_seeds=8, deplete_rates=(0.0,),
                     arms=("default", "explore_0.1", "founder2")):
    section("(c) competition -- is a hen's gain taken from her flockmates?")
    print("Repeatability with food depletion OFF. The depletion-on numbers are the main")
    print("run's table. If depletion couples the hens, switching it off removes the")
    print("negative r -- and tells us whether fitness here is partly zero-sum.")
    import scratchpad.e117_standing_variation as E

    rows = []
    for deplete in deplete_rates:
        E.CFG = spec.DEFAULT_COOP._replace(hawk_period_s=50.0,
                                           food_deplete_rate=deplete)
        E.STEPS = int(600.0 / E.CFG.dt)
        for arm in arms:
            rf, rd, spreads = [], [], []
            for g in range(n_seeds):
                k = jax.random.key(1000 + g)
                k_gen, k_world, k_run = jax.random.split(k, 3)
                p, pc = E.build_arm(arm, k_gen)
                fA, dA, _, _ = E.one_run(p, pc, k_world, jax.random.fold_in(k_run, 0))
                fB, dB, _, _ = E.one_run(p, pc, k_world, jax.random.fold_in(k_run, 1))
                rf.append(E.pearson(fA, fB))
                rd.append(E.pearson(dA, dB))
                spreads.append(float(np.std(dA)))
            r, se, t = E.fisher_mean(rf)
            r2, se2, t2 = E.fisher_mean(rd)
            rows.append((deplete, arm, r, se, t, r2, se2, t2, float(np.mean(spreads))))
            print(f"  deplete={deplete:<7} {arm:>9}: r_fit={r:+.3f}+-{se:.3f} "
                  f"t={t:+5.2f}   r_drives={r2:+.3f}+-{se2:.3f} t={t2:+5.2f}   "
                  f"between-hen sd(-drives)={np.mean(spreads):.4f}", flush=True)
    return rows


if __name__ == "__main__":
    diag_readout()
    diag_mask()
    diag_competition()
