"""E122c -- why crouching more does not cost intake: crouching IS camping.

E122b found that an information-free increase in crouching leaves cumulative feeding
(`n_fed`) essentially unchanged, and at low bias *raises* it. That should not happen if
crouching suppresses feeding, which E121c measured it doing locally (P(fed | at food,
crouching) 0.363 against 0.690 upright).

The resolution is in `coop/actuation.py`: `mobility = 1 - crouch`. Crouching zeroes
locomotion, so a crouching hen **stays where she is**. If she is on a patch, she keeps
working it instead of wandering off -- and [E112](../docs/experiments/E112-repair-the-peck-reflex.md)
named exactly that as the behaviour the hens were missing:

    "What the camped oracle does that the repaired hen still cannot is **stay on its
    patch**."

So the anti-predator response and the optimal foraging policy are the *same action*. If
that is right, raising the crouch bias should raise time-at-patch, and the two effects --
less feeding per second on a patch, more seconds on the patch -- cancel.

Run:  PYTHONPATH=. python scratchpad/e122c_crouch_is_camping.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from coop import spec, world
from hen import brain, plasticity
from run import evolve, simulate
from scratchpad.e120_ladder_checks import founders

N_SEEDS = 4
TRACE_S = 240.0
CHUNK = 10
BIASES = (0.0, 0.75, 1.5)


def main():
    c = spec.DEFAULT_COOP._replace(hawk_period_s=20.0, channel_mode="intact",
                                   call_log_steps=spec.YOKE_LOG_STEPS)
    pc = evolve.NO_LEARNING
    out = {}
    for bias in BIASES:
        at_food_frac, dwell, speed, fed_frac = [], [], [], []
        for s in range(N_SEEDS):
            key = jax.random.key(8800 + s)
            p = founders(jax.random.fold_in(key, 0), c)
            if bias:
                bm = np.asarray(p.b_motor).copy()
                bm[spec.M_CROUCH] += bias
                p = p._replace(b_motor=jnp.asarray(bm))
            w = world.reset(jax.random.fold_in(key, 1), c)
            x = brain.initial_state(p, c.n_hens)
            ps = plasticity.initial_state(p, c.n_hens, pc)
            k = jax.random.fold_in(key, 2)
            af_l, pos_l = [], []
            for _ in range(int(TRACE_S / c.dt) // CHUNK):
                d = jnp.linalg.norm(w.pos[:, None, :] - w.food_pos[None, :, :], axis=-1)
                af_l.append(np.asarray(jnp.any((d < c.peck_radius)
                                               & (w.food_amount[None, :] > 0.01),
                                               axis=-1)))
                pos_l.append(np.asarray(w.pos))
                w, x, _, ps, k = simulate.rollout_quiet(w, x, p, k, c, CHUNK, ps, pc)
            af = np.array(af_l)                      # (T, H)
            pos = np.array(pos_l)                    # (T, H, 2)
            at_food_frac.append(float(af.mean()))
            speed.append(float(np.linalg.norm(np.diff(pos, axis=0), axis=-1).mean()))
            # mean run-length of consecutive at-food steps, per hen
            runs = []
            for h in range(af.shape[1]):
                v = af[:, h].astype(int)
                if v.sum() == 0:
                    continue
                edges = np.diff(np.concatenate([[0], v, [0]]))
                starts = np.where(edges == 1)[0]
                ends = np.where(edges == -1)[0]
                runs.extend((ends - starts).tolist())
            dwell.append(float(np.mean(runs)) if runs else float("nan"))
        out[bias] = dict(at_food=np.mean(at_food_frac), dwell=np.nanmean(dwell),
                         speed=np.mean(speed))
        print(f"  bias {bias}: at_food {out[bias]['at_food']*100:.2f}%  "
              f"dwell {out[bias]['dwell']:.1f} chunks  "
              f"step {out[bias]['speed']:.4f} m", flush=True)

    print("\n" + "=" * 80)
    print("Crouching more: does she spend longer on patches?")
    print("=" * 80)
    print(f"{'crouch bias':>12} {'time at food':>14} {'patch dwell':>14} "
          f"{'distance/step':>15}")
    for bias in BIASES:
        r = out[bias]
        print(f"{bias:>12.2f} {r['at_food']*100:>13.2f}% {r['dwell']:>11.1f} ch "
              f"{r['speed']:>14.4f} m")
    b0, b1 = out[BIASES[0]], out[BIASES[-1]]
    print(f"\n  time at food {b0['at_food']*100:.2f}% -> {b1['at_food']*100:.2f}% "
          f"({(b1['at_food']/b0['at_food']-1)*100:+.0f}%)")
    print(f"  patch dwell  {b0['dwell']:.1f} -> {b1['dwell']:.1f} chunks "
          f"({(b1['dwell']/b0['dwell']-1)*100:+.0f}%)")
    print(f"  locomotion   {b0['speed']:.4f} -> {b1['speed']:.4f} m/step "
          f"({(b1['speed']/b0['speed']-1)*100:+.0f}%)")
    print("\n  If time-at-food and dwell RISE while locomotion falls, crouching is")
    print("  camping: the anti-predator action and the best foraging action are the")
    print("  same action, there is no vigilance/foraging trade-off to calibrate, and")
    print("  an alarm call cannot pay however the coop's parameters are set.")


if __name__ == "__main__":
    main()
