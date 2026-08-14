"""Which gregariousness weight makes the shuffled control actually uninformative?

E024's mistake was inferring the control was sound from an argument about realism
instead of measuring it. So the weight is chosen on the quantity the control depends
on -- how much of the intact channel's information about *your own* hawk survives
being scrambled -- and not on the strike-radius overlap, which is only a proxy for it.

Also runs the ethogram at each weight, because `approach flockmates when cold` is an
H1 assay and H1 is SUPPORTED. Dispersing the flock by breaking a proven status is not
a fix.
"""
from functools import partial

import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, regions
from run import probes

HENS, STEPS = 16, 24_000
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
        return (w, x, key), (obs[:, CH], near)
    return jax.lax.scan(step, (w, x, key), None, length=STEPS)[1]


def scaled(p, factor):
    r = np.asarray(p.reflex).copy()
    for b in range(spec.N_BINS):
        r[:, spec.vis_index(b, spec.CLS_FLOCKMATE)] *= factor
    return p._replace(reflex=jnp.asarray(r))


print(f"{'weight':>8}{'corr intact':>13}{'corr shuffled':>15}{'info kept':>11}")
for factor in (1.0, 0.25, 0.1, 0.0):
    got = {}
    for mode in ("intact", "shuffled"):
        cs = []
        for seed in range(3):
            cfg = spec.DEFAULT_COOP._replace(n_hens=HENS, hawk_period_s=60.0,
                                             channel_mode=mode)
            w = world.reset(jax.random.key(seed), cfg)
            p = connectome.build(jax.random.fold_in(jax.random.key(seed), 1),
                                 regions.DEFAULT_REGIONS.with_pallium(1.5),
                                 n_hens=HENS, auditory_scaffold=True)
            p = scaled(p, factor)
            x = brain.initial_state(p, HENS)
            heard, near = trace(w, x, p, jax.random.key(99), cfg)
            h, n = np.asarray(heard).ravel(), np.asarray(near).ravel()
            if n.sum() and h.std() > 0:
                cs.append(np.corrcoef(h, n.astype(float))[0, 1])
        got[mode] = np.mean(cs) if cs else float("nan")
    kept = got["shuffled"] / max(got["intact"], 1e-9)
    print(f"{1.2*factor:>8.2f}{got['intact']:>13.4f}{got['shuffled']:>15.4f}"
          f"{100*kept:>10.0f}%")

print("\nE024 ran at weight 1.20 and kept 90% of the information -- no control at all.")
print("\nethogram at each weight (H1 is SUPPORTED on 7/7):")
import hen.innate as innate
_orig = innate.reflex_matrix
for factor in (0.25, 0.1):
    def patched(auditory_scaffold=False, _f=factor):
        r = _orig(auditory_scaffold)
        for b in range(spec.N_BINS):
            r[:, spec.vis_index(b, spec.CLS_FLOCKMATE)] *= _f
        return r
    innate.reflex_matrix = patched
    import importlib
    from hen import connectome as _c
    importlib.reload(_c)
    res = probes.run_all()
    ok = sum(1 for pr in res if pr.passed)
    fails = [pr.name for pr in res if not pr.passed]
    print(f"  weight {1.2*factor:.2f}: {ok}/{len(res)} pass"
          + (f"   FAILED: {', '.join(fails)}" if fails else ""))
innate.reflex_matrix = _orig
