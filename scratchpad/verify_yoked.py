"""Independent check of the review's two load-bearing empirical claims.

A. A yoked (time-shifted) channel destroys the information a permuted one preserves.
B. The channel's benefit runs through the head-raise, not through the crouch response.

Both rebuilt offline from ONE intact trajectory, so the counterfactual channels differ
from the real one in routing alone -- no separate rollout, no divergent dynamics.
"""
from functools import partial

import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, regions

HENS, STEPS = 16, 18_000
AER = spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)


@partial(jax.jit, static_argnames=("cfg",))
def record(w, x, p, key, cfg):
    def step(carry, _):
        w, x, key = carry
        key, k = jax.random.split(key)
        obs = sensing.observe(w, cfg)
        x, motor, _ = brain.step(x, obs, p, cfg.dt)
        d = jnp.linalg.norm(w.pos - w.hawk_pos[None, :], axis=-1)
        near = (d < cfg.hawk_strike_radius) & (w.hawk_on > 0.5)
        w = world.step(w, motor, k, cfg)
        return (w, x, key), (w.calls, w.pos, near)
    return jax.lax.scan(step, (w, x, key), None, length=STEPS)[1]


def audio_from(calls, pos, cfg, route):
    """Recompute what each hen hears, given a routing rule. (T, H)"""
    d = np.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)
    at = np.clip(1.0 - d / cfg.hear_range, 0.0, 1.0)
    for t in range(at.shape[0]):
        np.fill_diagonal(at[t], 0.0)
    c = calls[:, :, AER]                                   # (T, H)
    if route == "intact":
        src = c
    elif route == "permuted":                              # what sensing.py does now
        rng = np.random.default_rng(0)
        src = c
        at = np.stack([at[t][rng.permutation(HENS)] for t in range(at.shape[0])])
    elif route == "yoked":                                 # time-shifted per hen
        rng = np.random.default_rng(1)
        lag = rng.integers(2000, STEPS - 2000, size=HENS)   # >= 20 s, > a 12 s dive
        src = np.stack([np.roll(c[:, j], lag[j]) for j in range(HENS)], axis=1)
    return np.sqrt((at ** 2 * src[:, None, :] ** 2).sum(-1))


print("A. does the routing destroy information about MY hawk?\n")
print(f"{'routing':<12}{'corr(heard, hawk on me)':>26}{'heard|hawk':>12}"
      f"{'heard|no hawk':>15}{'ratio':>8}")
rows = {}
for seed in range(3):
    cfg = spec.DEFAULT_COOP._replace(n_hens=HENS, hawk_period_s=60.0)
    w = world.reset(jax.random.key(seed), cfg)
    p = connectome.build(jax.random.fold_in(jax.random.key(seed), 1),
                         regions.DEFAULT_REGIONS.with_pallium(1.5),
                         n_hens=HENS, auditory_scaffold=True)
    x = brain.initial_state(p, HENS)
    calls, pos, near = (np.asarray(a) for a in record(w, x, p, jax.random.key(9), cfg))
    for route in ("intact", "permuted", "yoked"):
        h = audio_from(calls, pos, cfg, route).ravel()
        n = near.ravel()
        rows.setdefault(route, []).append(
            (np.corrcoef(h, n.astype(float))[0, 1], h[n].mean(), h[~n].mean()))
for route, v in rows.items():
    c, a, b = np.mean([r[0] for r in v]), np.mean([r[1] for r in v]), \
              np.mean([r[2] for r in v])
    print(f"{route:<12}{c:>26.4f}{a:>12.4f}{b:>15.4f}{a/max(b,1e-9):>8.2f}")
