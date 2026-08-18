"""How much does the wall-escape reflex actually reduce time spent pinned at a wall,
over a realistic (10-minute, viz-length) window? Ablation against the same connectome,
matching the E025/E048 pattern.
"""
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, regions

HENS, STEPS = 16, 60_000   # 10 min


def strip_wall_escape(p):
    r = np.asarray(p.reflex).copy()
    r[:, spec.IDX_WALL_ESCAPE_L] = 0.0
    r[:, spec.IDX_WALL_ESCAPE_R] = 0.0
    return p._replace(reflex=jnp.asarray(r))


@partial(jax.jit, static_argnames=("cfg",))
def run(w, x, p, key, cfg):
    def step(carry, _):
        w, x, key = carry
        key, k = jax.random.split(key)
        obs = sensing.observe(w, cfg)
        x, motor, _ = brain.step(x, obs, p, cfg.dt)
        wall = obs[:, spec.IDX_WALL]
        w = world.step(w, motor, k, cfg)
        return (w, x, key), wall
    return jax.lax.scan(step, (w, x, key), None, length=STEPS)[1]


cfg = spec.DEFAULT_COOP._replace(n_hens=HENS)
for label, strip in (("pre-fix (no wall escape)", True), ("fixed", False)):
    dwell, max_dwell = [], []
    for seed in range(3):
        w = world.reset(jax.random.key(seed), cfg)
        p = connectome.build(jax.random.fold_in(jax.random.key(seed), 1),
                             regions.DEFAULT_REGIONS, n_hens=HENS)
        if strip:
            p = strip_wall_escape(p)
        x = brain.initial_state(p, HENS)
        wall = run(w, x, p, jax.random.key(99), cfg)
        wall = np.asarray(wall)   # (STEPS, H)
        dwell.append((wall > 0.0).mean())
        # longest consecutive run any single hen spends with wall > 0
        per_hen_runs = []
        for h in range(HENS):
            on = wall[:, h] > 0.0
            if on.any():
                changes = np.diff(np.concatenate([[0], on.astype(int), [0]]))
                starts = np.where(changes == 1)[0]
                ends = np.where(changes == -1)[0]
                per_hen_runs.append((ends - starts).max())
        max_dwell.append(max(per_hen_runs) if per_hen_runs else 0)
    print(f"{label:<28} mean hen-steps near a wall: {100*np.mean(dwell):.2f}%  "
          f"longest single dwell: {np.mean(max_dwell):.0f} steps "
          f"({np.mean(max_dwell)*cfg.dt:.1f}s)")
