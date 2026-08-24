"""E107 verification 2: is the reward's dominant contingency invisible to the rule?

The review's second finding, in three parts, each measured here from scratch:

  (a) `reward()` is ~91% "hunger just fell", so the teaching signal is essentially
      "did you eat".
  (b) The observation carries "am I at a feeder" on essentially ONE channel,
      `IDX_FOOD_ARRIVAL`, because the vision channels are saturated at this coop size.
  (c) That channel does not survive the afferent projection, so the presynaptic factor
      the readout learns on -- the motor stub's trace -- is uninformative about the
      event the reward measures.

If (c) holds, no readout-side or representation-side repair can bind reward to feeding,
which would explain every H2 null upstream of everything E100-E106 tried.

Decoding is a plain ridge classifier, trained on the first half of the trajectory and
scored by AUC on the held-out second half. A linear decoder is a LOWER bound on what is
present -- so a high number is proof of presence and a low one is only suggestive.
"""
import time
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, neurons, plasticity, regions
from run import simulate

BASE = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=20.0)
STEPS, SEEDS = int(300 / BASE.dt), 3
PC = plasticity.PlasticConfig(enabled=True)
reg = regions.DEFAULT_REGIONS
S_LO, S_HI = reg.bounds(regions.SENSORY)
P_LO, P_HI = reg.bounds(regions.PALLIUM)


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
    """Train on the first half, score the second. Standardised, intercept included."""
    n = len(y)
    tr, te = slice(0, n // 2), slice(n // 2, n)
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-8
    a = (x[tr] - mu) / sd
    a = np.hstack([a, np.ones((len(a), 1))])
    t = y[tr].astype(np.float64) * 2 - 1
    wgt = np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ t)
    b = np.hstack([(x[te] - mu) / sd, np.ones((n - n // 2, 1))])
    if y[te].all() or not y[te].any():
        return float("nan")
    return auc(b @ wgt, y[te])


@partial(jax.jit, static_argnames=("cfg", "n"))
def probe(w, x, p, key, cfg, n):
    n_motor = p.W_out.shape[-1]

    def step(c, _):
        w, x, key = c
        key, kw = jax.random.split(key)
        obs = sensing.observe(w, cfg)
        x, motor, d = brain.step(x, obs, p, cfg.dt)
        r = neurons.rate(x)
        w2 = world.step(w, motor, kw, cfg)
        # "at a feeder" -- distance from this hen to the nearest food patch.
        dist = jnp.linalg.norm(w.pos[:, None, :] - w.food_pos[None, :, :], axis=-1)
        at_food = jnp.min(dist, axis=-1) < cfg.peck_radius
        # `hawk_pos` is (2,) -- one hawk, not a set. And it only means anything while
        # `hawk_on`; between dives it holds its last position.
        near_hawk = ((jnp.linalg.norm(w.pos - w.hawk_pos[None, :], axis=-1) < 3.0)
                     & (w.hawk_on > 0.0))
        return (w2, x, key), (obs, r[:, S_LO:S_HI], r[:, P_LO:P_HI],
                              r[:, -n_motor:], d.cortical, d.reflex,
                              at_food, near_hawk,
                              w.hunger, w2.hunger, w.cold, w2.cold,
                              w.thirst, w2.thirst,
                              w2.n_strike_events - w.n_strike_events)
    return jax.lax.scan(step, (w, x, key), None, length=n)[1]


t0 = time.perf_counter()
print(f"E107 verification 2 -- routing. {SEEDS} seeds, hawk every {BASE.hawk_period_s}s\n")

rw = {k: [] for k in ("hunger", "thirst", "cold", "struck")}
stage_auc = {s: {"at food": [], "hawk near": []} for s in
             ("observation", "sensory stub", "pallium", "motor stub",
              "cortical", "reflex")}
chan = {"IDX_FOOD_ARRIVAL": [], "CLS_FOOD vision": [], "whole observation": []}
sat = {"at food": [], "away": []}

for s in range(SEEDS):
    k = jax.random.key(s)
    p0 = connectome.build(jax.random.fold_in(k, 1), reg, n_hens=16)
    out = probe(world.reset(k, BASE), brain.initial_state(p0, 16), p0,
                jax.random.fold_in(k, 5), BASE, STEPS)
    (obs, sens, pal, stub, cort, refl, at_food, near_hawk,
     h0, h1, c0, c1, t0_, t1_, struck) = (np.asarray(a) for a in out)

    # (a) reward decomposition. Each term as it enters `reward()`, in its own units,
    # scored by its covariance with the total -- the share of the teaching signal's
    # variance it explains.
    dt, rs = BASE.dt, PC.reward_scale
    terms = {"hunger": (h0 - h1) / dt * rs, "thirst": (t0_ - t1_) / dt * rs,
             "cold": (c0 - c1) / dt * rs, "struck": -struck * PC.strike_penalty}
    total = sum(terms.values())
    var = total.var()
    for name, v in terms.items():
        rw[name].append(float(np.cov(v.ravel(), total.ravel())[0, 1] / (var + 1e-12)))

    # (b) and (c). Flatten hens into samples; each stage decoded separately.
    flat = lambda a: a.reshape(-1, a.shape[-1])
    y_food, y_hawk = at_food.ravel(), near_hawk.ravel()
    for name, a in (("observation", obs), ("sensory stub", sens), ("pallium", pal),
                    ("motor stub", stub), ("cortical", cort), ("reflex", refl)):
        stage_auc[name]["at food"].append(ridge_auc(flat(a), y_food))
        stage_auc[name]["hawk near"].append(ridge_auc(flat(a), y_hawk))

    fa = obs[:, :, spec.IDX_FOOD_ARRIVAL].ravel()
    chan["IDX_FOOD_ARRIVAL"].append(auc(fa, y_food))
    vis = flat(obs[:, :, spec.CLS_FOOD * spec.N_BINS:
                   (spec.CLS_FOOD + 1) * spec.N_BINS])
    chan["CLS_FOOD vision"].append(ridge_auc(vis, y_food))
    chan["whole observation"].append(ridge_auc(flat(obs), y_food))
    v = vis.mean(-1)
    sat["at food"].append(float(v[y_food].mean()))
    sat["away"].append(float(v[~y_food].mean()))

print("(a) reward variance decomposition -- share of the teaching signal")
for name, v in rw.items():
    print(f"    {name:>10}{100*np.mean(v):>8.1f}%")
print(f"    CLAUDE.md line 23 asserts 'the reward is 87% n_struck'.")

print("\n(b) can she perceive that she is at a feeder?")
for name, v in chan.items():
    print(f"    {name:>20}  AUC {np.mean(v):.3f}")
print(f"    CLS_FOOD vision mean: {np.mean(sat['at food']):.3f} at a feeder vs "
      f"{np.mean(sat['away']):.3f} away  (vision_range "
      f"{BASE.vision_range} m, peck_radius {BASE.peck_radius} m)")

print("\n(c) does it survive into the brain?")
print(f"{'stage':>15}{'AUC at food':>14}{'AUC hawk near':>15}")
for name, d in stage_auc.items():
    print(f"{name:>15}{np.mean(d['at food']):>14.3f}{np.mean(d['hawk near']):>15.3f}")

print("\nreading it: the readout's presynaptic factor is the MOTOR STUB's trace")
print("(plasticity.py, dz_slow[-n_motor:]). If its 'at food' AUC is at chance while the")
print("reward is ~90% hunger, then at the moment the teaching signal fires, nothing the")
print("rule is looking at is about the event that caused it.")
print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
