"""Why doesn't sickness_penalty move the weights, even with E067's fix?
Compare the magnitude of one sickness event's contribution to `m` against the
ambient reward the same window carries, and how often such an event occurs.
"""
from functools import partial
import jax, jax.numpy as jnp
from coop import spec, world
from hen import brain, connectome, plasticity, regions
from hen.plasticity import PlasticConfig
from run import simulate

HENS = 16
CFG = spec.DEFAULT_COOP._replace(n_hens=HENS, food_deplete_rate=0.0)
pc = PlasticConfig(enabled=True, growth_enabled=False, kin_audible=True,
                   explore_sigma=0.6, hebbian_readout=True,
                   readout_scaling_strength=0.3, sickness_penalty=1.0)

key = jax.random.key(0)
w = world.reset(key, CFG)
p = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS, n_hens=HENS)
x = brain.initial_state(p, HENS)
ps = plasticity.initial_state(p, HENS, pc)

@partial(jax.jit, static_argnames=("cfg", "pc", "n"))
def trace(w, x, p, ps, key, cfg, pc, n):
    def step(carry, _):
        w_prev = carry[0]
        carry, _o = simulate._one_step(carry, None, cfg=cfg, pc=pc)
        w_next = carry[0]
        rew = plasticity.reward(w_prev, w_next, cfg, pc)
        onset = (w_next.sick_on & ~w_prev.sick_on).sum()
        return carry, (jnp.abs(rew).mean(), onset)
    return jax.lax.scan(step, (w, x, p, ps, key), None, length=n)[1]

N = 60_000   # 10 min
abs_rew, onsets = trace(w, x, p, ps, jax.random.fold_in(key, 2), CFG, pc, N)
mean_abs_rew = float(jnp.mean(abs_rew))
total_onsets = float(jnp.sum(onsets))

sick_contrib = pc.sickness_penalty / pc.interval   # one event, averaged over its window
n_windows = N / pc.interval
windows_with_event = total_onsets / HENS           # per hen

print(f"ambient mean |reward| per step (per hen):      {mean_abs_rew:.4f}")
print(f"one sickness event's contribution to that")
print(f"  window's mean m (penalty/interval):          {sick_contrib:.4f}")
print(f"  -> ratio to ambient:                         {sick_contrib/mean_abs_rew:.2f}x")
print()
print(f"consolidation windows in 10 min:               {n_windows:.0f}")
print(f"sickness onsets (flock total / per hen):       {total_onsets:.0f} / {windows_with_event:.2f}")
print(f"-> fraction of a hen's windows carrying one:   {100*windows_with_event/n_windows:.4f}%")
