"""Did E067's fix actually make sickness_penalty reach the weights it gates?

The |W_out| diagnostic in E065/E066/E068 CANNOT answer this: under
hebbian_readout=True, W_out's update uses m_out=ones_like(m) -- it is not
reward-gated at all. The fix affects `m`, which gates W (the recurrent weights).
So the right measurement is |W| drift, and specifically whether it responds to
sickness_penalty being on vs off.
"""
from functools import partial
import jax, jax.numpy as jnp
from coop import spec, world
from hen import brain, connectome, plasticity, regions
from hen.plasticity import PlasticConfig
from run import simulate

HENS = 16
CFG = spec.DEFAULT_COOP._replace(n_hens=HENS, food_deplete_rate=0.0)
BASE = dict(enabled=True, growth_enabled=False, kin_audible=True,
            explore_sigma=0.6, hebbian_readout=True, readout_scaling_strength=0.3)

for label, sp in (("sickness_penalty=0.0", 0.0), ("sickness_penalty=1.0", 1.0)):
    pc = PlasticConfig(**BASE, sickness_penalty=sp)
    drifts = []
    for seed in range(2):
        key = jax.random.key(seed)
        w = world.reset(key, CFG)
        p0 = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS,
                              n_hens=HENS)
        x = brain.initial_state(p0, HENS)
        ps = plasticity.initial_state(p0, HENS, pc)
        w2, x2, p2, ps2, k2, summ = simulate.simulate(
            w, x, p0, jax.random.fold_in(key, 2), CFG, seconds=600.0, chunk_s=60.0, pc=pc)
        dW = float(jnp.mean(jnp.abs(p2.W - p0.W)))
        drifts.append(dW)
    print(f"{label:<24} mean |W - W_innate| after 10 min: "
          f"{sum(drifts)/len(drifts):.6e}   (per seed: {[f'{d:.4e}' for d in drifts]})")
