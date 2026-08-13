import time, jax, jax.numpy as jnp
from coop import spec
from run import experiment
from run.experiment import Condition, run_condition
from hen.plasticity import PlasticConfig

cfg = spec.DEFAULT_COOP
fixed = Condition("fixed", PlasticConfig(enabled=False, explore_sigma=0.0))
t0=time.perf_counter()
r = run_condition(fixed, 0, cfg, 60.0, 60.0)
print("1 min sim (incl compile):", time.perf_counter()-t0, r)
t0=time.perf_counter()
r = run_condition(fixed, 1, cfg, 60.0, 60.0)
print("1 min sim (cached):", time.perf_counter()-t0, r)
t0=time.perf_counter()
r = run_condition(fixed, 2, cfg, 300.0, 60.0)
print("5 min sim:", time.perf_counter()-t0, r)
