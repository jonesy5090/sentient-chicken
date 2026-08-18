"""Same measurement as the earlier informal check (user question about food-call
saturation), re-run after E053's discovery-pulse fix. 16 hens, 1 minute, no predators.
"""
import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, regions

CFG = spec.DEFAULT_COOP._replace(hawk_period_s=1e9, ground_pred_period_s=1e9)
w = world.reset(jax.random.key(0), CFG)
p = connectome.build(jax.random.key(1), regions.DEFAULT_REGIONS, n_hens=CFG.n_hens)
x = brain.initial_state(p, CFG.n_hens)

food_call = []
for t in range(6_000):   # 1 minute
    obs = sensing.observe(w, CFG)
    x, motor, _ = brain.step(x, obs, p, CFG.dt)
    food_call.append(np.asarray(motor[:, spec.M_CALL_FOOD]) > 0.5)
    w = world.step(w, motor, jax.random.fold_in(jax.random.key(4), t), CFG)

food_call = np.array(food_call)   # (T, H)
print(f"hens calling FOOD on >50% of steps: "
      f"{(food_call.mean(axis=0) > 0.5).sum()} / {CFG.n_hens}")
print(f"mean fraction of hen-steps with a food call active: {food_call.mean():.3f}")
print("pre-fix (E052): 4/16 hens >50%, 0.428 mean fraction")
