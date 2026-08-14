"""Which force actually holds the flock together? (E025)

Food depletion was the predicted fix and it did essentially nothing: 23.0% -> 21.9% of
the flock inside one strike radius, with patches genuinely drawn down to 0.70. So the
binding force is not foraging. Rather than guess again, ablate each candidate.

Candidates, both in the innate arc:
  - thermotaxis: huddling is the ONLY heat source in the coop (`huddle_warm_rate`), and
    cold accumulates over `cold_fill_s`. A hen who does not huddle gets steadily colder,
    so proximity is not a preference, it is a requirement.
  - gregariousness: flockmate-in-bin drives turn-toward at weight 1.2.
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


def strip_gregariousness(p):
    """Zero the flockmate -> turn-toward weights in the innate arc."""
    r = np.asarray(p.reflex).copy()
    for b in range(spec.N_BINS):
        r[:, spec.vis_index(b, spec.CLS_FLOCKMATE)] = 0.0
    return p._replace(reflex=jnp.asarray(r))


print(f"{'ablation':<28}{'nn dist':>9}{'spread':>9}{'in strike radius':>18}"
      f"{'cold':>8}{'fed/hen':>9}")
for label, patch, strip in (
        ("none (E025 baseline)", {}, False),
        ("no thermal pressure", {"cold_fill_s": 1e9, "huddle_warm_rate": 0.0}, False),
        ("no gregariousness", {}, True),
        ("neither", {"cold_fill_s": 1e9, "huddle_warm_rate": 0.0}, True)):
    nns, sps, fr, cds, fds = [], [], [], [], []
    for seed in range(3):
        cfg = spec.DEFAULT_COOP._replace(n_hens=HENS, hawk_period_s=60.0, **patch)
        w = world.reset(jax.random.key(seed), cfg)
        p = connectome.build(jax.random.fold_in(jax.random.key(seed), 1),
                             regions.DEFAULT_REGIONS.with_pallium(1.5),
                             n_hens=HENS, auditory_scaffold=True)
        if strip:
            p = strip_gregariousness(p)
        x = brain.initial_state(p, HENS)
        near, on, nn, spread, cold, fed = run(w, x, p, jax.random.key(99), cfg)
        near, on = np.asarray(near), np.asarray(on) > 0.5
        nns.append(np.asarray(nn).mean()); sps.append(np.asarray(spread).mean())
        cds.append(np.asarray(cold).mean()); fds.append(float(np.asarray(fed)[-1]))
        if on.sum():
            fr.append(near[on].mean())
    print(f"{label:<28}{np.mean(nns):>9.2f}{np.mean(sps):>9.2f}"
          f"{100*np.mean(fr):>17.1f}%{np.mean(cds):>8.3f}{np.mean(fds):>9.0f}")

print("\nlower 'in strike radius' is what a sender-scrambling control needs.")
print("'cold' near 1.0 means the ablation left them freezing -- not a usable fix.")
