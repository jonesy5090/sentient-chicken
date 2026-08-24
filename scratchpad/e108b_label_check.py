"""E108b: is E108's label actually about feeding?

E108's triviality falsifier fired, marginally: only 4.3-5.5% of windows contain any
feeding, against a pre-registered floor of 5%. The decode label was a median split on
hunger drop, which is 50/50 by construction -- so the falsifier fired for a quantity
that was not the label, and I mis-specified it.

But the near-miss points at something real. If 95% of windows contain no feeding, a
median split on hunger drop is mostly separating "hunger rose slightly less" from
"hunger rose slightly more" in windows where she never ate -- which could be driven by
movement cost rather than by food. That would make E108's headline a decode of vigour
wearing a feeding label.

So: re-run the decode against labels that cannot be confounded that way.
"""
import time
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, neurons, plasticity, regions
from run import simulate

BASE = spec.DEFAULT_COOP._replace(n_hens=16)
PC = plasticity.PlasticConfig(enabled=True)
INTERVAL = PC.interval
WINDOWS, SEEDS = int(300 / BASE.dt) // INTERVAL, 4
reg = regions.DEFAULT_REGIONS


def auc(scores, labels):
    labels = np.asarray(labels).astype(bool)
    if labels.all() or not labels.any():
        return float("nan")
    o = np.argsort(scores)
    r = np.empty(len(scores))
    r[o] = np.arange(1, len(scores) + 1)
    n1 = labels.sum()
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(labels) - n1)))


def ridge_auc(x, y, lam=1.0):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y)
    n = len(y)
    tr, te = slice(0, n // 2), slice(n // 2, n)
    if y[tr].all() or not y[tr].any() or y[te].all() or not y[te].any():
        return float("nan")
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-8
    a = np.hstack([(x[tr] - mu) / sd, np.ones((n // 2, 1))])
    t = y[tr].astype(np.float64) * 2 - 1
    w = np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ t)
    b = np.hstack([(x[te] - mu) / sd, np.ones((n - n // 2, 1))])
    return auc(b @ w, y[te])


@partial(jax.jit, static_argnames=("cfg", "pc", "n_windows"))
def windows(carry, cfg, pc, n_windows):
    n_motor = carry[2].W_out.shape[-1]

    def inner(c, _):
        w = c[0]
        at_food = jnp.min(jnp.linalg.norm(
            w.pos[:, None, :] - w.food_pos[None, :, :], axis=-1),
            axis=-1) < cfg.peck_radius
        speed = jnp.linalg.norm(w.vel, axis=-1) if hasattr(w, "vel") else jnp.zeros_like(w.hunger)
        c, _o = simulate._one_step(c, None, cfg, pc)
        return c, (at_food, w.hunger, w.n_fed, speed)

    def outer(c, _):
        c, (at_food, hunger, n_fed, speed) = jax.lax.scan(inner, c, None,
                                                          length=INTERVAL)
        w, x, p, ps, key = c
        return c, ((ps.z_slow - ps.z_slow_bar)[:, -n_motor:],
                   neurons.rate(x)[:, -n_motor:],
                   sensing.observe(w, cfg),
                   at_food.mean(0),                 # fraction of window at a feeder
                   hunger[0] - w.hunger,            # hunger drop
                   w.n_fed - n_fed[0],              # feeding steps in the window
                   speed.mean(0))                   # movement, the confound
    return jax.lax.scan(outer, carry, None, length=n_windows)[1]


t0 = time.perf_counter()
print(f"E108b -- is the label about feeding? {SEEDS} seeds\n")
res = {}
for s in range(SEEDS):
    k = jax.random.key(s)
    p = connectome.build(jax.random.fold_in(k, 1), reg, n_hens=16)
    carry = (world.reset(k, BASE), brain.initial_state(p, 16), p,
             plasticity.initial_state(p, 16, PC), jax.random.fold_in(k, 5))
    dz, rate_, obs, atf, dh, fed, spd = [np.asarray(a) for a in
                                          windows(carry, BASE, PC, WINDOWS)]
    flat = lambda a: a.reshape(-1, a.shape[-1])
    labels = {
        "median split on hunger drop (E108's label)": dh.ravel() > np.median(dh),
        "fed at all in the window": fed.ravel() > 0,
        "at a feeder for >half the window": atf.ravel() > 0.5,
        "median split, EXCLUDING feeding windows": None,   # handled below
        "moved more than the median (the confound)": spd.ravel() > np.median(spd),
    }
    for name, y in labels.items():
        if name.startswith("median split, EXCLUDING"):
            keep = fed.ravel() == 0
            sub_dh = dh.ravel()[keep]
            y = sub_dh > np.median(sub_dh)
            for src, a in (("dz_slow", dz), ("observation", obs)):
                res.setdefault((name, src), []).append(
                    ridge_auc(flat(a)[keep], y))
            continue
        for src, a in (("dz_slow", dz), ("observation", obs)):
            res.setdefault((name, src), []).append(ridge_auc(flat(a), y))
    res.setdefault(("__overlap__", ""), []).append(
        float(np.mean((dh.ravel() > np.median(dh)) == (fed.ravel() > 0))))
    res.setdefault(("__fedrate__", ""), []).append(float((fed.ravel() > 0).mean()))

print(f"windows containing any feeding: {100*np.mean(res[('__fedrate__','')]):.1f}%")
print(f"agreement between E108's median-split label and 'fed at all': "
      f"{100*np.mean(res[('__overlap__','')]):.1f}%\n")
print(f"{'label':>44}{'dz_slow':>10}{'observation':>13}")
for name in ("median split on hunger drop (E108's label)",
             "fed at all in the window",
             "at a feeder for >half the window",
             "median split, EXCLUDING feeding windows",
             "moved more than the median (the confound)"):
    d = np.nanmean(res[(name, "dz_slow")])
    o = np.nanmean(res[(name, "observation")])
    print(f"{name:>44}{d:>10.3f}{o:>13.3f}")

print("\nreading it: if 'fed at all' and 'at a feeder' decode as well as the median")
print("split, E108's headline is about feeding and stands. If the median split decodes")
print("well but the direct feeding labels do not, E108 measured movement.")
print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
