"""E121c -- is crouching free? Measure it, do not read it.

`coop/actuation.py` says, in a comment directly above the code:

    Crouching is freezing: it suppresses locomotion entirely. That is the trade the
    hen makes -- invisible to the hawk, but not foraging and not going anywhere.

I suspected that comment was wrong. `crouch` scales `mobility` only, while `coop/world.py`
computes `fed = at_food_any & pecking` -- feeding needs pecking, not moving -- so a hen
already standing on a patch looked like she could crouch and eat at once.

**She cannot, and the comment is approximately right.** Measured: a crouching hen feeds at
**52.7%** of an upright hen's rate (-0.3264 +/- 0.0195, t=-16.71). Reading the code was not
enough, which is the reason for the rule this file was written to honour:

    CLAUDE.md: "A plausible sentence sitting next to the code is not evidence. Prose is a
    claim; only a measurement is a fact."

The finding that survives is subtler. The trade-off is real *locally* and weak *globally*:
a hen is at a food patch only **4.5%** of the time, so halving her feeding while crouching
costs almost nothing in aggregate -- doubling the crouch rate cost 0.024 hunger while
saving 2.4 catches. Unconditional vigilance still wins, for a different reason than I
guessed.

Run:  PYTHONPATH=. python scratchpad/e121c_is_crouching_free.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from coop import sensing, spec, world
from hen import brain, neurons, plasticity
from run import evolve, simulate
from scratchpad.e121_can_wout_see_the_alarm import CHUNK, cfg, collect  # noqa: F401
from scratchpad.e120_ladder_checks import founders

N_SEEDS = 6
TRACE_S = 240.0


def main():
    c = cfg()
    pc = evolve.NO_LEARNING
    rows = []
    for g in range(N_SEEDS):
        key = jax.random.key(7700 + g)
        p = founders(jax.random.fold_in(key, 0), c)
        w = world.reset(jax.random.fold_in(key, 1), c)
        x = brain.initial_state(p, c.n_hens)
        ps = plasticity.initial_state(p, c.n_hens, pc)
        k = jax.random.fold_in(key, 2)

        at_food, pecking, crouching, fed = [], [], [], []
        for _ in range(int(TRACE_S / c.dt) // CHUNK):
            d = jnp.linalg.norm(w.pos[:, None, :] - w.food_pos[None, :, :], axis=-1)
            af = np.asarray(jnp.any((d < c.peck_radius)
                                    & (w.food_amount[None, :] > 0.01), axis=-1))
            # one step to read this instant's motor output
            _, _, _, _, _, tr = simulate.rollout(
                w, x, p, k, c, 1, ps, pc)
            m = np.asarray(tr.motor[0])
            at_food.append(af)
            pecking.append(m[:, spec.M_PECK] > 0.5)
            crouching.append(m[:, spec.M_CROUCH] > 0.5)
            fed.append(af & (m[:, spec.M_PECK] > 0.5))
            w, x, _, ps, k = simulate.rollout_quiet(w, x, p, k, c, CHUNK, ps, pc)

        af = np.concatenate(at_food)
        pk = np.concatenate(pecking)
        cr = np.concatenate(crouching)
        fd = np.concatenate(fed)
        both = af & cr
        if both.sum() < 20 or (af & ~cr).sum() < 20:
            print(f"  seed {g}: too few samples ({both.sum()} crouching-at-food)")
            continue
        rows.append(dict(
            p_fed_crouch=float(fd[both].mean()),
            p_fed_upright=float(fd[af & ~cr].mean()),
            p_peck_crouch=float(pk[both].mean()),
            p_peck_upright=float(pk[af & ~cr].mean()),
            frac_crouch=float(cr.mean()),
            # The number that reconciles a real local trade-off with a small aggregate
            # cost: if she is rarely on a patch, halving her feeding while crouching
            # costs little overall.
            frac_at_food=float(af.mean()),
            p_fed_overall=float(fd.mean()),
            n_both=int(both.sum())))
        print(f"  seed {g}: P(fed | at food, crouching)={rows[-1]['p_fed_crouch']:.4f}  "
              f"P(fed | at food, upright)={rows[-1]['p_fed_upright']:.4f}  "
              f"(n={rows[-1]['n_both']})", flush=True)

    print("\n" + "=" * 76)
    print("Is crouching free? P(fed) while crouching vs upright, both AT a food patch")
    print("=" * 76)
    a = np.array([r["p_fed_crouch"] for r in rows])
    b = np.array([r["p_fed_upright"] for r in rows])
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"  P(fed | crouching) {a.mean():.4f}   P(fed | upright) {b.mean():.4f}")
    print(f"  difference {d.mean():+.4f} +- {se:.4f}  t={d.mean()/se:+.2f} "
          f"(df={len(d)-1})")
    print(f"  ratio: a crouching hen feeds at {a.mean()/max(b.mean(),1e-9)*100:.1f}% "
          f"of an upright hen's rate")
    ff = np.mean([r["frac_at_food"] for r in rows])
    fc = np.mean([r["frac_crouch"] for r in rows])
    print(f"\n  BUT she is at a food patch only {ff*100:.1f}% of the time, and crouches "
          f"{fc*100:.1f}% of it.")
    print(f"  So the aggregate foraging cost of crouching more is bounded by ~{ff*100:.1f}% "
          f"of hen-steps,")
    print("  which is why doubling the crouch rate cost only 0.024 hunger while saving")
    print("  2.4 catches. The trade-off is real locally and weak globally.")
    print("\n  So `actuation.py`'s comment stands: crouching really does cost foraging.")
    print("  But the cost lands on a small slice of her life, and E121 part B measured")
    print("  the net incentive for crouching MORE at +0.233 fitness with no information")
    print("  involved. An alarm call pays only when crouching is expensive enough that")
    print("  you want to do it selectively. Here it is not.")


if __name__ == "__main__":
    main()
