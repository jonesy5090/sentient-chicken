"""Is the zero-strike test failure a single-seed RNG artefact of OBS_DIM growing, or a
real behavioural effect of the wall-escape reflex? Sweep seeds, jitted once.
"""
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, regions

CFG = spec.DEFAULT_COOP._replace(hawk_period_s=20.0)
STEPS = 10_000


@partial(jax.jit, static_argnames=("cfg",))
def run(w, x, p, cfg):
    def step(carry, t):
        w, x = carry
        obs = sensing.observe(w, cfg)
        x, motor, _ = brain.step(x, obs, p, cfg.dt)
        wn = world.step(w, motor, jax.random.fold_in(jax.random.key(4), t), cfg)
        events = jnp.sum(wn.n_strike_events - w.n_strike_events)
        return (wn, x), events
    (w, x), events = jax.lax.scan(step, (w, x), jnp.arange(STEPS))
    return jnp.sum(events)


for seed in range(8):
    w = world.reset(jax.random.key(seed), CFG)
    p = connectome.build(jax.random.key(seed + 1), regions.DEFAULT_REGIONS,
                         n_hens=CFG.n_hens)
    x = brain.initial_state(p, CFG.n_hens)
    events = float(run(w, x, p, CFG))
    print(f"seed {seed}: events = {events}")
