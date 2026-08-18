"""Does the E025 personal-space fix actually disperse the flock? (E048)

`what_clumps_them.py` found gregariousness's attraction-only wiring is what clumps the
flock (21.9% -> 6.8% inside one strike radius when stripped entirely), and diagnosed the
fix: a crowding channel so the reflex can turn *away* at close range instead of only
toward. That channel (`CLS_CROWDING`) is now built. This reruns the same diagnostic,
same metrics, same world, to check the targeted fix gets some of stripping's dispersal
without stripping gregariousness's function (huddling, flock cohesion) entirely.

Conditions:
  - pre-fix (E025 baseline): CLS_CROWDING wired but zeroed out, i.e. exactly what the
    reflex arc did before this session -- attraction only.
  - fixed (E025 personal space): the arc as it ships now.
  - no gregariousness (E025 ablation, for reference): attraction stripped entirely too.
"""
from functools import partial

import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, innate, regions

HENS, STEPS = 16, 36_000


@partial(jax.jit, static_argnames=("cfg",))
def run(w, x, p, key, cfg):
    def step(carry, _):
        w, x, key = carry
        key, k = jax.random.split(key)
        obs = sensing.observe(w, cfg)
        x, motor, _ = brain.step(x, obs, p, cfg.dt)
        d = jnp.linalg.norm(w.pos - w.hawk_pos[None, :], axis=-1)
        near = ((d < cfg.hawk_strike_radius) & (w.hawk_on > 0.5)).astype(jnp.float32)
        dh = jnp.linalg.norm(w.pos[:, None] - w.pos[None, :], axis=-1)
        nn = jnp.min(dh + jnp.eye(HENS) * 1e6, axis=1)
        w = world.step(w, motor, k, cfg)
        return (w, x, key), (near.mean(), w.hawk_on, nn.mean(), dh.max(),
                             w.cold.mean(), jnp.mean(w.n_fed))
    return jax.lax.scan(step, (w, x, key), None, length=STEPS)[1]


def strip_channel(p, cls):
    """Zero a vision class's turn-motor weights in the innate arc."""
    r = np.asarray(p.reflex).copy()
    for b in range(spec.N_BINS):
        r[:, spec.vis_index(b, cls)] = 0.0
    return p._replace(reflex=jnp.asarray(r))


print(f"{'condition':<28}{'nn dist':>9}{'spread':>9}{'in strike radius':>18}"
      f"{'cold':>8}{'fed/hen':>9}")
for label, strip_crowding, strip_flockmate in (
        ("pre-fix (E025 baseline)", True, False),
        ("fixed (E025 personal space)", False, False),
        ("no gregariousness (ref.)", False, True)):
    nns, sps, fr, cds, fds = [], [], [], [], []
    for seed in range(3):
        cfg = spec.DEFAULT_COOP._replace(n_hens=HENS, hawk_period_s=60.0)
        w = world.reset(jax.random.key(seed), cfg)
        p = connectome.build(jax.random.fold_in(jax.random.key(seed), 1),
                             regions.DEFAULT_REGIONS.with_pallium(1.5),
                             n_hens=HENS, auditory_scaffold=True)
        if strip_crowding:
            p = strip_channel(p, spec.CLS_CROWDING)
        if strip_flockmate:
            p = strip_channel(p, spec.CLS_FLOCKMATE)
        x = brain.initial_state(p, HENS)
        near, on, nn, spread, cold, fed = run(w, x, p, jax.random.key(99), cfg)
        near, on = np.asarray(near), np.asarray(on) > 0.5
        nns.append(np.asarray(nn).mean()); sps.append(np.asarray(spread).mean())
        cds.append(np.asarray(cold).mean()); fds.append(float(np.asarray(fed)[-1]))
        if on.sum():
            fr.append(near[on].mean())
    print(f"{label:<28}{np.mean(nns):>9.2f}{np.mean(sps):>9.2f}"
          f"{100*np.mean(fr):>17.1f}%{np.mean(cds):>8.3f}{np.mean(fds):>9.0f}")

print("\n'pre-fix' should reproduce E025's baseline row (nn 0.39, strike 21.9%).")
print("'fixed' dispersal, relative to pre-fix and to full removal, is the finding.")
print("'cold' rising toward the no-gregariousness row would mean huddling broke too.")
