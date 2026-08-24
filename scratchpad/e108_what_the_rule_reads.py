"""E108: decode the reward's own event from the quantity the rule reads, when it reads it.

The trajectory comes from `simulate._one_step` itself. Re-implementing the step here
would measure a rule that is not the one that runs, which is the mistake this project
keeps finding in its own history.
"""
import time
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, neurons, plasticity, regions
from run import simulate

BASE = spec.DEFAULT_COOP._replace(n_hens=16)
PC = plasticity.PlasticConfig(enabled=True)
REAR, SEEDS = int(30 * 60 / BASE.dt), 4
INTERVAL = PC.interval
WINDOWS = int(300 / BASE.dt) // INTERVAL          # 300 s of probe
reg = regions.DEFAULT_REGIONS


def auc(scores, labels):
    labels = np.asarray(labels).astype(bool)
    if labels.all() or not labels.any():
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n1 = labels.sum()
    n0 = len(labels) - n1
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ridge_auc(x, y, lam=1.0):
    """Train on the first half of the windows, score the held-out second half."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y)
    n = len(y)
    tr, te = slice(0, n // 2), slice(n // 2, n)
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-8
    a = np.hstack([(x[tr] - mu) / sd, np.ones((n // 2, 1))])
    t = y[tr].astype(np.float64) * 2 - 1
    w = np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ t)
    b = np.hstack([(x[te] - mu) / sd, np.ones((n - n // 2, 1))])
    return auc(b @ w, y[te])


@partial(jax.jit, static_argnames=("cfg", "pc", "n_windows"))
def windows(carry, cfg, pc, n_windows):
    """Outer loop over consolidation windows, inner loop over the steps in one.

    Emits once per window, at the boundary -- the instant `consolidate` is called.
    """
    n_motor = carry[2].W_out.shape[-1]

    def inner(c, _):
        # `m_acc` read BEFORE the step, so at the boundary it has not yet been reset.
        # One-step offset on a 50-step window, declared in section 5.
        m_pre = c[3].m_acc
        hunger = c[0].hunger
        n_fed = c[0].n_fed
        at_food = jnp.min(jnp.linalg.norm(
            c[0].pos[:, None, :] - c[0].food_pos[None, :, :], axis=-1), axis=-1) < cfg.peck_radius
        c, _out = simulate._one_step(c, None, cfg, pc)
        return c, (m_pre, hunger, n_fed, at_food)

    def outer(c, _):
        c, (m_pre, hunger, n_fed, at_food) = jax.lax.scan(
            inner, c, None, length=INTERVAL)
        w, x, p, ps, key = c
        r = neurons.rate(x)
        third = INTERVAL // 3
        return c, (
            (ps.z_slow - ps.z_slow_bar)[:, -n_motor:],   # what the rule reads
            ps.z_slow[:, -n_motor:],                     # uncentred
            r[:, -n_motor:],                             # instantaneous rate
            sensing.observe(w, cfg),                     # instrument check
            m_pre[-1],                                   # m over the window
            hunger[0] - w.hunger,                        # hunger DROP over the window
            n_fed[-1] - n_fed[0] + (w.n_fed - n_fed[-1]),
            at_food[:third].mean(0), at_food[third:2 * third].mean(0),
            at_food[2 * third:].mean(0))
    return jax.lax.scan(outer, carry, None, length=n_windows)[1]


def collect(seed, reared):
    k = jax.random.key(seed)
    p = connectome.build(jax.random.fold_in(k, 1), reg, n_hens=16)
    w = world.reset(k, BASE)
    x = brain.initial_state(p, 16)
    ps = plasticity.initial_state(p, 16, PC)
    if reared:
        w, x, p, ps, _key = simulate.rollout_quiet(
            w, x, p, jax.random.fold_in(k, 2), BASE, REAR, ps, PC)
    carry = (w, x, p, ps, jax.random.fold_in(k, 5))
    return [np.asarray(a) for a in windows(carry, BASE, PC, WINDOWS)]


t0 = time.perf_counter()
print(f"E108 -- what the rule reads. {SEEDS} seeds, {WINDOWS} windows of "
      f"{INTERVAL} steps ({INTERVAL*BASE.dt:.2f} s), tau_slow={PC.tau_slow} s\n")

for label, reared in (("untrained", False), ("reared 30 min", True)):
    rows = {k: [] for k in ("dz_slow (THE RULE)", "z_slow uncentred",
                            "rate at boundary", "observation (instrument)")}
    m_auc, base_rate, thirds = [], [], {0: [], 1: [], 2: []}
    for s in range(SEEDS):
        dz, zs, rate_, obs, m, dh, fed, t0f, t1f, t2f = collect(s, reared)
        # Label: median split on hunger drop, so the base rate is 50% by construction.
        flat = lambda a: a.reshape(-1, a.shape[-1])
        y = dh.ravel()
        y = y > np.median(y)
        for name, a in (("dz_slow (THE RULE)", dz), ("z_slow uncentred", zs),
                        ("rate at boundary", rate_), ("observation (instrument)", obs)):
            rows[name].append(ridge_auc(flat(a), y))
        m_auc.append(auc(m.ravel(), y))
        base_rate.append(float((fed.ravel() > 0).mean()))
        # Prediction 3: does WHERE in the window she fed change what survives?
        for i, tf in enumerate((t0f, t1f, t2f)):
            sel = tf.ravel() > 0
            if sel.sum() > 30 and (~sel).sum() > 30:
                thirds[i].append(ridge_auc(flat(dz), sel))

    print(f"[{label}]  windows in which she fed at all: "
          f"{100*np.mean(base_rate):.1f}%  (triviality falsifier: outside 5-95%)")
    print(f"{'source':>28}{'AUC':>8}")
    for name, v in rows.items():
        print(f"{name:>28}{np.nanmean(v):>8.3f}")
    print(f"{'m (teaching signal)':>28}{np.nanmean(m_auc):>8.3f}")
    print(f"  decoding 'fed in the FIRST/MIDDLE/LAST third' from dz_slow: " +
          " / ".join(f"{np.nanmean(thirds[i]):.3f}" if thirds[i] else "n/a"
                     for i in range(3)))
    print()

print("--- pre-registered falsifiers (E108 section 4) ---")
print("primary      dz_slow AUC >= 0.70 AND m AUC >= 0.70 -> the rule has what it")
print("             needs, my hypothesis is wrong, and the failure is downstream")
print("instrument   observation AUC must be > 0.80 or nothing here means anything")
print("confound     if 'rate at boundary' ~ dz_slow, the loss is the representation")
print("             E107 already measured and this adds nothing")
print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
