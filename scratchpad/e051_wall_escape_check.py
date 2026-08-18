"""Does the wall-escape reflex actually get a cornered hen away from the boundary?

Places a single hen 0.1 m from a wall, facing straight into it, and rolls forward a
short window with no other drives active (food/water/flockmates/threats all absent) to
isolate the reflex being tested.
"""
import jax, jax.numpy as jnp, numpy as np
from coop import actuation, sensing, spec, world
from hen import brain, connectome, regions

cfg = spec.DEFAULT_COOP._replace(n_hens=1, n_food=1, n_water=1, hawk_period_s=1e9,
                                 ground_pred_period_s=1e9)
w = world.reset(jax.random.key(0), cfg)
# Isolate: hen 0.1 m from the left wall, heading pointed directly into it (angle pi,
# i.e. facing -x). Food/water pushed far away so foraging drive doesn't interfere.
w = w._replace(pos=jnp.array([[0.1, cfg.size / 2]]),
               heading=jnp.array([jnp.pi]),
               food_pos=jnp.array([[cfg.size - 1, cfg.size - 1]]),
               water_pos=jnp.array([[cfg.size - 1, cfg.size - 1]]))
p = connectome.build(jax.random.key(1), regions.DEFAULT_REGIONS, n_hens=1)
x = brain.initial_state(p, 1)

print(f"{'step':>6}{'x':>8}{'heading (deg)':>16}")
for step in range(400):
    obs = sensing.observe(w, cfg)
    x, motor, _ = brain.step(x, obs, p, cfg.dt)
    kin = actuation.apply_motor(w, motor, cfg)
    w = w._replace(pos=kin.pos, heading=kin.heading, speed=kin.speed,
                   head_down=kin.head_down)
    if step % 40 == 0:
        print(f"{step:>6}{float(w.pos[0,0]):>8.3f}{np.degrees(float(w.heading[0])):>16.1f}")

print(f"\nfinal x = {float(w.pos[0,0]):.3f} m (started at 0.1 m, wall at x=0)")
print("success if x has grown well past the wall's ~1m influence radius")
