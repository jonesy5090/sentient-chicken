"""Does the shuffled channel carry information about a hen's OWN hawk? (jitted)

E024's control depends on 'no'. The flock clumps, and if a randomly reassigned sender
stands in the same place then her alarm reports the same hawk, making L = C? an
identity rather than a result.
"""
from functools import partial

import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, regions

HENS, STEPS = 16, 18_000          # 3 min of chicken time
CH = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)


@partial(jax.jit, static_argnames=("cfg",))
def trace(w, x, p, key, cfg):
    def step(carry, _):
        w, x, key = carry
        key, k = jax.random.split(key)
        obs = sensing.observe(w, cfg)
        x, motor, _ = brain.step(x, obs, p, cfg.dt)
        d = jnp.linalg.norm(w.pos - w.hawk_pos[None, :], axis=-1)
        near = (d < cfg.hawk_strike_radius) & (w.hawk_on > 0.5)
        w = world.step(w, motor, k, cfg)
        return (w, x, key), (obs[:, CH], near, w.hawk_on)
    _c, out = jax.lax.scan(step, (w, x, key), None, length=STEPS)
    return out


print(f"{'mode':<10}{'corr(heard, hawk on me)':>26}{'heard|hawk':>12}"
      f"{'heard|no hawk':>15}{'ratio':>8}")
for mode in ("intact", "shuffled"):
    cs, wh, wo = [], [], []
    for seed in range(3):
        cfg = spec.DEFAULT_COOP._replace(n_hens=HENS, hawk_period_s=60.0,
                                         channel_mode=mode)
        w = world.reset(jax.random.key(seed), cfg)
        p = connectome.build(jax.random.fold_in(jax.random.key(seed), 1),
                             regions.DEFAULT_REGIONS.with_pallium(1.5),
                             n_hens=HENS, auditory_scaffold=True)
        x = brain.initial_state(p, HENS)
        heard, near, _on = trace(w, x, p, jax.random.key(99), cfg)
        h = np.asarray(heard).ravel()
        n = np.asarray(near).ravel()
        if n.sum() and h.std() > 0:
            cs.append(np.corrcoef(h, n.astype(float))[0, 1])
            wh.append(h[n].mean()); wo.append(h[~n].mean())
    a, b = np.mean(wh), np.mean(wo)
    print(f"{mode:<10}{np.mean(cs):>26.4f}{a:>12.4f}{b:>15.4f}{a/max(b,1e-9):>8.2f}")

# Geometry: when a hawk is live, how much of the flock is inside its strike radius?
cfg = spec.DEFAULT_COOP._replace(n_hens=HENS, hawk_period_s=60.0)
w = world.reset(jax.random.key(0), cfg)
p = connectome.build(jax.random.key(1), regions.DEFAULT_REGIONS.with_pallium(1.5),
                     n_hens=HENS, auditory_scaffold=True)
x = brain.initial_state(p, HENS)
_h, near, on = trace(w, x, p, jax.random.key(99), cfg)
near, on = np.asarray(near), np.asarray(on)
live = on > 0.5
frac = near[live].sum(axis=1) / HENS if live.sum() else np.array([0.0])
print(f"\nwhen a hawk is live: mean {100*frac.mean():.1f}% of the flock is inside its "
      f"strike radius (max {100*frac.max():.0f}%)")
print("If that fraction is high, every hen shares one hawk and a reassigned sender")
print("carries the same news -- which would make C? a copy of L, not a control.")
