"""E111b: a better oracle, because E111's was demonstrably not optimal.

E111's instrument falsifier half-fired. The greedy nearest-patch oracle fed more than the
reflex hen (3.7% vs 2.6%) but was **at a patch less often** (3.7% vs 6.4%) -- it spends
its time travelling, because sixteen hens all chase the same nearest patch, strip it, and
then all travel together to the next one.

That matters for the claim. "No headroom" rests on the ceiling being genuinely a ceiling,
and a policy that is at a patch less than the reflex hen is not one. So: assign each hen
a home patch and let her camp on it. That removes the herding entirely and should push
utilisation toward the resource limit.

If the camped oracle also fails to beat the reflex hen, the ceiling claim is much harder
to argue with. If it succeeds, E111's headline is withdrawn and the greedy oracle was the
problem.
"""
import os
import time
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from coop import spec, world
from hen import brain, connectome, plasticity, regions
from run import simulate

BASE = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=60.0)
STEPS, SEEDS = int(30 * 60 / BASE.dt), 8
OFFSET = int(os.environ.get('E111_SEED_OFFSET', '0'))  # E033's lesson
reg = regions.DEFAULT_REGIONS
FROZEN = plasticity.PlasticConfig(enabled=True, eta_out=0.0)
HOME = jnp.arange(BASE.n_hens) % BASE.n_food     # hen i camps on patch i mod n_food


def camped_motor(w, cfg):
    """Each hen owns a patch and stays on it. No herding, no shared travel.

    She pecks whenever she is in range and the patch has food -- including waiting on an
    empty patch, which is correct here: regrowth is `(1-amount)/food_regrow_s`, so
    standing on a stripped patch and taking each unit as it appears is exactly how a
    resource-limited forager should behave.
    """
    target = w.food_pos[HOME]
    delta = target - w.pos
    dist = jnp.linalg.norm(delta, axis=-1)
    want = jnp.arctan2(delta[:, 1], delta[:, 0])
    err = (want - w.heading + jnp.pi) % (2 * jnp.pi) - jnp.pi

    at_patch = dist < cfg.peck_radius
    motor = jnp.zeros((w.pos.shape[0], spec.MOTOR_DIM))
    motor = motor.at[:, spec.M_PECK].set(at_patch.astype(jnp.float32))
    turn = jnp.clip(err * 2.0, -1.0, 1.0)
    motor = motor.at[:, spec.M_TURN_L].set(
        jnp.where(at_patch, 0.0, jnp.maximum(turn, 0.0)))
    motor = motor.at[:, spec.M_TURN_R].set(
        jnp.where(at_patch, 0.0, jnp.maximum(-turn, 0.0)))
    motor = motor.at[:, spec.M_FORWARD].set(
        jnp.where(at_patch | (jnp.abs(err) > 0.6), 0.0, 1.0))
    return motor


@partial(jax.jit, static_argnames=("cfg", "n"))
def run_camped(w, key, cfg, n):
    def step(c, _):
        w, key = c
        key, kw = jax.random.split(key)
        motor = camped_motor(w, cfg)
        d = jnp.linalg.norm(w.pos[:, None, :] - w.food_pos[None, :, :], axis=-1)
        at = jnp.any((d < cfg.peck_radius) & (w.food_amount[None, :] > 0.01), axis=-1)
        return (world.step(w, motor, kw, cfg), key), (
            at, at & (motor[:, spec.M_PECK] > 0.5), w.food_amount.mean())
    (w, _k), out = jax.lax.scan(step, (w, key), None, length=n)
    return w, out


@partial(jax.jit, static_argnames=("cfg", "pc", "n"))
def run_brain(w, x, p, ps, key, cfg, pc, n):
    def step(c, _):
        w = c[0]
        d = jnp.linalg.norm(w.pos[:, None, :] - w.food_pos[None, :, :], axis=-1)
        at = jnp.any((d < cfg.peck_radius) & (w.food_amount[None, :] > 0.01), axis=-1)
        amount = w.food_amount.mean()
        c, out = simulate._one_step(c, None, cfg, pc)
        return c, (at, at & (out[0][:, spec.M_PECK] > 0.5), amount)
    (w, x, p, ps, key), out = jax.lax.scan(step, (w, x, p, ps, key), None, length=n)
    return w, out


t0 = time.perf_counter()
floor_h = ((BASE.n_hens / BASE.hunger_fill_s)
           / ((BASE.n_food / BASE.food_regrow_s)
              * (BASE.peck_food_rate / BASE.food_deplete_rate)))
print(f"E111b -- a camped oracle. seeds {OFFSET}-{OFFSET+SEEDS-1}\n")
print(f"analytic resource floor: {floor_h:.4f}   "
      f"E111 gave: greedy oracle 0.5999, reflex 0.6332\n")

res = {}
for arm in ("camped", "reflex"):
    h, c, at_f, fed_f, amt = [], [], [], [], []
    for s in range(OFFSET, OFFSET + SEEDS):
        k = jax.random.key(s)
        w0 = world.reset(k, BASE)
        if arm == "reflex":
            p = connectome.build(jax.random.fold_in(k, 1), reg, n_hens=BASE.n_hens)
            w2, out = run_brain(w0, brain.initial_state(p, BASE.n_hens), p,
                                plasticity.initial_state(p, BASE.n_hens, FROZEN),
                                jax.random.fold_in(k, 2), BASE, FROZEN, STEPS)
        else:
            w2, out = run_camped(w0, jax.random.fold_in(k, 2), BASE, STEPS)
        at, fed, amount = (np.asarray(a) for a in out)
        h.append(float(np.mean(np.asarray(w2.hunger))))
        c.append(float(np.sum(np.asarray(w2.n_caught_any))
                       / max(float(np.sum(np.asarray(w2.n_dives))), 1.0)))
        at_f.append(float(at.mean()))
        fed_f.append(float(fed.mean()))
        amt.append(float(amount.mean()))
    res[arm] = dict(h=np.array(h), c=np.array(c), at=np.mean(at_f),
                    fed=np.mean(fed_f), amt=np.mean(amt))
    print(f"{arm:>8}  hunger {np.mean(h):.4f}   at a patch {100*np.mean(at_f):>5.1f}%   "
          f"feeding {100*np.mean(fed_f):>5.1f}%   mean food_amount {np.mean(amt):.4f}   "
          f"caught/dive {np.mean(c):.4f}")

d = res["camped"]["h"] - res["reflex"]["h"]
se = d.std(ddof=1) / np.sqrt(len(d))
print(f"\n  camped vs reflex (hunger): {d.mean():+.4f} +/- {se:.4f}  "
      f"t={d.mean()/(se+1e-12):+.2f}   (df={SEEDS-1}, crit 2.365)")
print(f"\n  headroom {abs(d.mean()):.4f} vs 2x SE {2*se:.4f} -> "
      f"{'THERE IS HEADROOM' if abs(d.mean()) > 2*se else 'NO HEADROOM'}")
print(f"  instrument: camped at a patch {100*res['camped']['at']:.1f}% vs reflex "
      f"{100*res['reflex']['at']:.1f}%, feeding {100*res['camped']['fed']:.1f}% vs "
      f"{100*res['reflex']['fed']:.1f}%")
print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
