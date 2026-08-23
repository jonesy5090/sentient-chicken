"""E096: is H2f's audience effect carried by the audience, or by what the audience says?

Red-team finding 1, re-measured independently. `run/audience.py:_staged` puts 15
flockmates at 2 m for audience=True and parks them at ABSENT otherwise, with the staged
hawk at HAWK_DISTANCE=7.0 -- which is 5-9 m from the audience, inside vision_range. They
see it and alarm-call, so "audience present" also drives the focal hen's aerial channel.

Measures the E057 contrast twice on the SAME reared brains: once as published, once with
the audio channel muted at TEST time only. The audience is still there, still seen, still
counted by IDX_ISOLATION and CLS_FLOCKMATE -- only what they say is removed.

If the effect is about an audience, muting should barely touch it.
"""
import sys, time
sys.path.insert(0, 'scratchpad')
import jax, jax.numpy as jnp, numpy as np
from coop import spec
from hen.plasticity import PlasticConfig
from run import audience

SEEDS, MINUTES = 3, 30.0
CFG = spec.DEFAULT_COOP._replace(n_hens=16, food_deplete_rate=0.0)
PC = PlasticConfig(enabled=True, hebbian_readout=True, readout_scaling_strength=0.3)


def observed_audio(p, cfg, n_hens, audience_on):
    """What the focal hen actually hears on the aerial channel in the staged assay."""
    from hen import brain
    from run import simulate
    w = audience._staged(cfg, n_hens, audience=audience_on, hawk=True, food=False)
    x = brain.initial_state(p, n_hens)
    *_, tr = simulate.rollout(w, x, p, jax.random.key(11),
                              cfg._replace(n_hens=n_hens), 300)
    return float(jnp.mean(tr.obs[:, 0, spec.IDX_AERIAL]))


print(f"E096 -- H2f audience effect, audio intact vs muted at test. {SEEDS} seeds, "
      f"{MINUTES:.0f} min rearing")
print("E057 published: general elevation +0.123, audience-specific DiD +0.232\n")
t0 = time.perf_counter()
rows = []
for s in range(SEEDS):
    p, _ps = audience.rear_and_assay(s, CFG, MINUTES * 60, PC, return_brain=True) \
        if False else (None, None)
    # rear once via the module's own path, then assay twice
    from run import lifetime  # noqa
    res_intact, brain_p = None, None
    # rear_and_assay does rearing + assay together; re-implement minimally so the SAME
    # reared brain is assayed under both channel modes.
    from hen import connectome, regions, brain as B
    from run import simulate
    from coop import world
    k = jax.random.key(s)
    p = connectome.build(jax.random.fold_in(k, 1), regions.DEFAULT_REGIONS,
                         n_hens=CFG.n_hens, auditory_scaffold=True)
    w = world.reset(k, CFG)
    x = B.initial_state(p, CFG.n_hens)
    from hen import plasticity as PL
    ps = PL.initial_state(p, CFG.n_hens, PC)
    w, x, p, ps, _k, _tr = simulate.rollout(
        w, x, p, jax.random.fold_in(k, 2), CFG, int(MINUTES * 60 / CFG.dt), pc=PC, ps=ps)

    intact = audience.assay(p, CFG, CFG.n_hens)
    muted = audience.assay(p, CFG._replace(channel_mode="none"), CFG.n_hens)
    aud_on = observed_audio(p, CFG, CFG.n_hens, True)
    aud_off = observed_audio(p, CFG, CFG.n_hens, False)
    rows.append((intact, muted, aud_on, aud_off))
    print(f"  seed {s}: aerial channel heard -- audience {aud_on:.4f}, alone {aud_off:.4f}")

def did(r):
    return (r.alarm_audience - r.alarm_alone) - (r.food_audience - r.food_alone)

di = np.mean([did(r[0]) for r in rows])
dm = np.mean([did(r[1]) for r in rows])
gi = np.mean([r[0].alarm_alone for r in rows])
print(f"\n{'condition':>22}{'alarm alone':>13}{'alarm aud':>11}{'DiD':>9}")
for name, i in (("audio intact", 0), ("audio muted at test", 1)):
    a = np.mean([r[i].alarm_alone for r in rows])
    b = np.mean([r[i].alarm_audience for r in rows])
    print(f"{name:>22}{a:>13.4f}{b:>11.4f}{(dm if i else di):>9.4f}")
print(f"\naudience-specific DiD: intact {di:+.4f}, muted {dm:+.4f} "
      f"-> {100*dm/max(di,1e-9):.0f}% survives muting")
print(f"wall clock: {time.perf_counter()-t0:.0f} s")
