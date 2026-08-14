"""How strong should the flockmate-approach reflex be? (E025)

The ablation found gregariousness, not thermotaxis, is what clumps the flock:
zeroing it took the strike-radius overlap from 21.9% to 6.8% *and raised* feeding
34%. But zeroing it is not an option -- hens are gregarious, that is documented
behaviour and H1's ethogram tests for it.

There is a specific defect in how it is wired. Vision is proximity-graded
(`prox = 1 - d/vision_range`), so the turn-toward drive is *proportional to
closeness*: the nearer a flockmate already is, the harder the hen turns toward her.
A cohesion force should fall off as birds converge, not intensify. Real fowl have a
documented individual distance -- attraction at range, repulsion when crowded -- and
this model has attraction only.

Modelling repulsion properly needs a crowding channel, which changes OBS_DIM and so
invalidates every prior comparison. This sweeps the weight instead, to find the
mildest reduction that buys a usable control, and the departure gets documented.
"""
from functools import partial

import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, regions
from run import probes

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


def scale_gregariousness(p, factor):
    r = np.asarray(p.reflex).copy()
    for b in range(spec.N_BINS):
        r[:, spec.vis_index(b, spec.CLS_FLOCKMATE)] *= factor
    return p._replace(reflex=jnp.asarray(r))


print(f"{'weight':>8}{'nn dist':>9}{'spread':>9}{'in strike radius':>18}"
      f"{'cold':>8}{'fed/hen':>9}")
for factor in (1.0, 0.5, 0.25, 0.1, 0.0):
    nns, sps, fr, cds, fds = [], [], [], [], []
    for seed in range(3):
        cfg = spec.DEFAULT_COOP._replace(n_hens=HENS, hawk_period_s=60.0)
        w = world.reset(jax.random.key(seed), cfg)
        p = connectome.build(jax.random.fold_in(jax.random.key(seed), 1),
                             regions.DEFAULT_REGIONS.with_pallium(1.5),
                             n_hens=HENS, auditory_scaffold=True)
        p = scale_gregariousness(p, factor)
        x = brain.initial_state(p, HENS)
        near, on, nn, spread, cold, fed = run(w, x, p, jax.random.key(99), cfg)
        near, on = np.asarray(near), np.asarray(on) > 0.5
        nns.append(np.asarray(nn).mean()); sps.append(np.asarray(spread).mean())
        cds.append(np.asarray(cold).mean()); fds.append(float(np.asarray(fed)[-1]))
        if on.sum():
            fr.append(near[on].mean())
    print(f"{1.2*factor:>8.2f}{np.mean(nns):>9.2f}{np.mean(sps):>9.2f}"
          f"{100*np.mean(fr):>17.1f}%{np.mean(cds):>8.3f}{np.mean(fds):>9.0f}")

print("\nE024's control needs this well under 20%. Watch cold: an ablation that")
print("disperses them by leaving them freezing trades one broken world for another.")
