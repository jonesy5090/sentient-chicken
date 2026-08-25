"""E111: what does the best possible forager achieve, and is the gap detectable?

The oracle bypasses the brain entirely -- a motor vector computed from world state.
Nothing in `hen/` is touched, so nothing here can affect any other result.
"""
import time
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from coop import spec, world
from hen import brain, connectome, plasticity, regions
from run import simulate

BASE = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=60.0)
STEPS, SEEDS = int(30 * 60 / BASE.dt), 8
reg = regions.DEFAULT_REGIONS
FROZEN = plasticity.PlasticConfig(enabled=True, eta_out=0.0)


def analytic_floor(cfg, n_hens):
    """The resource-limited equilibrium: no policy can hold hunger below this."""
    gain = n_hens / cfg.hunger_fill_s
    supply = cfg.n_food / cfg.food_regrow_s
    per_unit = cfg.peck_food_rate / cfg.food_deplete_rate
    return gain / (supply * per_unit)


def oracle_motor(w, cfg):
    """Turn toward the nearest non-empty patch, walk, peck on arrival.

    Ignores hawks, water, cold and flockmates. A FORAGING ceiling, which is the right
    ceiling for a reward that is ~83% hunger (E107).
    """
    d = jnp.linalg.norm(w.pos[:, None, :] - w.food_pos[None, :, :], axis=-1)
    # An empty patch is not a target: `world.step` requires food_amount > 0.01 to feed.
    d = jnp.where(w.food_amount[None, :] > 0.01, d, jnp.inf)
    nearest = jnp.argmin(d, axis=-1)
    target = w.food_pos[nearest]
    dist = jnp.min(d, axis=-1)
    delta = target - w.pos
    want = jnp.arctan2(delta[:, 1], delta[:, 0])
    err = (want - w.heading + jnp.pi) % (2 * jnp.pi) - jnp.pi

    at_patch = dist < cfg.peck_radius
    motor = jnp.zeros((w.pos.shape[0], spec.MOTOR_DIM))
    motor = motor.at[:, spec.M_PECK].set(at_patch.astype(jnp.float32))
    # Turn proportionally, saturating: `turn = M_TURN_L - M_TURN_R` in actuation.py.
    turn = jnp.clip(err * 2.0, -1.0, 1.0)
    motor = motor.at[:, spec.M_TURN_L].set(jnp.where(at_patch, 0.0,
                                                     jnp.maximum(turn, 0.0)))
    motor = motor.at[:, spec.M_TURN_R].set(jnp.where(at_patch, 0.0,
                                                     jnp.maximum(-turn, 0.0)))
    # Walk only when roughly facing the target; a hen spinning on the spot gets nowhere.
    motor = motor.at[:, spec.M_FORWARD].set(
        jnp.where(at_patch | (jnp.abs(err) > 0.6), 0.0, 1.0))
    # No target reachable anywhere: wander forward rather than freeze.
    motor = motor.at[:, spec.M_FORWARD].set(
        jnp.where(jnp.isinf(dist), 1.0, motor[:, spec.M_FORWARD]))
    return motor


@partial(jax.jit, static_argnames=("cfg", "mode", "n"))
def run_scripted(w, key, cfg, mode, n):
    """`oracle` or `floor` -- no brain in the loop at all."""
    def step(c, _):
        w, key = c
        key, kw = jax.random.split(key)
        motor = (oracle_motor(w, cfg) if mode == "oracle"
                 else jnp.zeros((w.pos.shape[0], spec.MOTOR_DIM)))
        d = jnp.linalg.norm(w.pos[:, None, :] - w.food_pos[None, :, :], axis=-1)
        at = jnp.any((d < cfg.peck_radius) & (w.food_amount[None, :] > 0.01), axis=-1)
        return (world.step(w, motor, kw, cfg), key), (
            at, at & (motor[:, spec.M_PECK] > 0.5), w.food_amount.mean())
    (w, _k), out = jax.lax.scan(step, (w, key), None, length=n)
    return w, out


@partial(jax.jit, static_argnames=("cfg", "pc", "n"))
def run_brain(w, x, p, ps, key, cfg, pc, n):
    """The shipped hen. Same emitted statistics, so the arms are compared like for like."""
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
floor_h = analytic_floor(BASE, BASE.n_hens)
print(f"E111 -- headroom. {SEEDS} seeds, {STEPS*BASE.dt/60:.0f} min, "
      f"{BASE.n_hens} hens, {BASE.n_food} patches\n")
print(f"analytic resource floor (section 2): mean hunger cannot go below "
      f"{floor_h:.4f}\n")

res = {}
for arm in ("oracle", "reflex", "floor"):
    h, c, at_f, fed_f, amt = [], [], [], [], []
    for s in range(SEEDS):
        k = jax.random.key(s)
        w0 = world.reset(k, BASE)
        if arm == "reflex":
            p = connectome.build(jax.random.fold_in(k, 1), reg, n_hens=BASE.n_hens)
            w2, out = run_brain(w0, brain.initial_state(p, BASE.n_hens), p,
                                plasticity.initial_state(p, BASE.n_hens, FROZEN),
                                jax.random.fold_in(k, 2), BASE, FROZEN, STEPS)
        else:
            w2, out = run_scripted(w0, jax.random.fold_in(k, 2), BASE, arm, STEPS)
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


def paired(a, b, name):
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / (se + 1e-12)
    print(f"    {name:<40}{d.mean():+.4f} +/- {se:.4f}  t={t:+.2f}")
    return abs(d.mean()), se


print(f"\n  paired, df={SEEDS-1} (lower hunger is better):")
gap, se = paired(res["oracle"]["h"], res["reflex"]["h"], "oracle vs reflex (hunger)")
paired(res["floor"]["h"], res["reflex"]["h"], "floor vs reflex (hunger)")
paired(res["oracle"]["c"], res["reflex"]["c"], "oracle vs reflex (caught/dive)")

print("\n--- pre-registered falsifiers (E111 section 4) ---")
print(f"primary      headroom {gap:.4f} vs 2x SE {2*se:.4f} -> "
      f"{'THERE IS HEADROOM' if gap > 2*se else 'NO HEADROOM: H2 unanswerable here'}")
print(f"instrument   oracle must forage more: at a patch "
      f"{100*res['oracle']['at']:.1f}% vs reflex {100*res['reflex']['at']:.1f}%, "
      f"feeding {100*res['oracle']['fed']:.1f}% vs {100*res['reflex']['fed']:.1f}%")
print(f"model        oracle hunger {np.mean(res['oracle']['h']):.4f} must NOT be below "
      f"the analytic floor {floor_h:.4f}"
      f"{'  <-- FIRES, section 2 withdrawn' if np.mean(res['oracle']['h']) < floor_h else ''}")
print(f"triviality   floor arm {np.mean(res['floor']['h']):.4f} must be clearly worse "
      f"than reflex {np.mean(res['reflex']['h']):.4f}")
print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
