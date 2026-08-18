"""E055 diagnostic: is the significant audience-effect result targeted (calls more
specifically when an audience is present) or general dysregulation (calls/moves more
across the board, of which the audience contrast is a side effect)?

The hunger jump in the main result (0.39 -> 0.73) is the tell to check: rear a few
seeds under the hebbian condition and look at the FULL breakdown (alone vs audience,
alarm vs food, reflex vs cortical drive magnitude), not just the alarm_effect scalar.
"""
import jax, jax.numpy as jnp
from coop import spec, world
from hen import brain, connectome, regions
from hen.plasticity import PlasticConfig
from run import simulate
from run.audience import assay, comprehension

cfg = spec.DEFAULT_COOP._replace(n_hens=16, food_deplete_rate=0.0)
LEARN_HEBBIAN = PlasticConfig(enabled=True, growth_enabled=False, kin_audible=True,
                              explore_sigma=0.6, hebbian_readout=True)
seconds = 30 * 60.0

print(f"{'seed':>5}{'alarm alone':>13}{'alarm aud.':>12}{'food alone':>12}"
      f"{'food aud.':>10}{'|reflex|':>10}{'|cortical|':>11}")
for seed in range(3):
    key = jax.random.key(seed)
    w = world.reset(key, cfg)
    p = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS,
                         n_hens=cfg.n_hens, auditory_scaffold=True)
    x = brain.initial_state(p, cfg.n_hens)
    _w, x_end, p_end, *_ = simulate.simulate(
        w, x, p, jax.random.fold_in(key, 2), cfg, seconds, 60.0, LEARN_HEBBIAN)

    r = assay(p_end, cfg, cfg.n_hens)

    # Reflex vs cortical drive magnitude, on the flock at end of rearing.
    obs = jax.vmap(lambda *_: None)  # placeholder, unused
    from coop import sensing
    obs_live = sensing.observe(_w, cfg)
    _x2, _motor, drives = brain.step(x_end, obs_live, p_end, cfg.dt)
    reflex_mag = float(jnp.mean(jnp.abs(drives.reflex)))
    cortical_mag = float(jnp.mean(jnp.abs(drives.cortical)))

    print(f"{seed:>5}{r.alarm_alone:>13.4f}{r.alarm_audience:>12.4f}"
          f"{r.food_alone:>12.4f}{r.food_audience:>10.4f}"
          f"{reflex_mag:>10.4f}{cortical_mag:>11.4f}")

print("\nIf alarm_alone rose about as much as alarm_audience, the 'audience effect' is")
print("a side effect of calling more overall, not a targeted contingency.")
print("If |cortical| >> |reflex|, the readout is overwhelming the reflex arc --")
print("the documented failure mode where behaviour gets worse, not better.")
