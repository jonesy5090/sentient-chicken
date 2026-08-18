"""E056 mandatory diagnostic (pre-registered): is the t=45.59 audience effect targeted
or another artifact? Same breakdown as e055b, pointed at the scaled hebbian condition.
"""
import jax, jax.numpy as jnp
from coop import spec, world, sensing
from hen import brain, connectome, regions
from hen.plasticity import PlasticConfig
from run import simulate
from run.audience import assay

cfg = spec.DEFAULT_COOP._replace(n_hens=16, food_deplete_rate=0.0)
LEARN_HEBBIAN_SCALED = PlasticConfig(enabled=True, growth_enabled=False, kin_audible=True,
                                     explore_sigma=0.6, hebbian_readout=True,
                                     readout_scaling_strength=0.3)
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
        w, x, p, jax.random.fold_in(key, 2), cfg, seconds, 60.0, LEARN_HEBBIAN_SCALED)

    r = assay(p_end, cfg, cfg.n_hens)

    obs_live = sensing.observe(_w, cfg)
    _x2, _motor, drives = brain.step(x_end, obs_live, p_end, cfg.dt)
    reflex_mag = float(jnp.mean(jnp.abs(drives.reflex)))
    cortical_mag = float(jnp.mean(jnp.abs(drives.cortical)))

    print(f"{seed:>5}{r.alarm_alone:>13.4f}{r.alarm_audience:>12.4f}"
          f"{r.food_alone:>12.4f}{r.food_audience:>10.4f}"
          f"{reflex_mag:>10.4f}{cortical_mag:>11.4f}")

print("\nGenuine targeted effect: alarm_alone flat/low, alarm_audience specifically up,")
print("|cortical| well under |reflex|. Anything else is still an artifact.")
