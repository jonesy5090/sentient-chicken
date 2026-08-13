"""During a real hawk event, does the alarm channel carry WHO/HOW-MUCH, or just 1.0?"""
import jax, jax.numpy as jnp
from coop import spec, world
from hen import brain, connectome, regions, plasticity
from hen.plasticity import PlasticConfig
from run import simulate

cfg = spec.DEFAULT_COOP; n = cfg.n_hens
pc = PlasticConfig(enabled=False, explore_sigma=0.0)
AER = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)

def body(carry, _):
    carry, (motor, obs, _r, _m) = simulate._one_step(carry, None, cfg, pc)
    w = carry[0]
    return carry, (w.hawk_on, obs[:, AER], w.calls[:, spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)])

@jax.jit
def go(w,x,p,ps,k): return jax.lax.scan(body,(w,x,p,ps,k),None,length=30000)

for seed in (3, 11):
    key = jax.random.key(seed)
    w = world.reset(key, cfg)
    p = connectome.build(jax.random.fold_in(key,1), regions.DEFAULT_REGIONS, n_hens=n)
    x = brain.initial_state(p,n); ps = plasticity.initial_state(p,n,pc)
    _c,(hon, heard, emit) = go(w,x,p,ps,jax.random.fold_in(key,2))
    on = hon > 0.5
    h_on, e_on = heard[on], emit[on]
    ncall = jnp.sum(e_on > 0.05, axis=-1)
    print(f"seed {seed}: hawk steps={int(jnp.sum(on))}")
    print(f"  emitted alarm amplitude: mean={float(jnp.mean(e_on)):.3f} max={float(jnp.max(e_on)):.3f}"
          f"  callers/step mean={float(jnp.mean(ncall)):.1f}/16")
    print(f"  HEARD channel during hawk: mean={float(jnp.mean(h_on)):.3f} "
          f"frac of hen-steps at >0.99 (saturated)={float(jnp.mean(h_on>0.99)):.2f} "
          f"sd across hens within a step={float(jnp.mean(jnp.std(h_on,axis=-1))):.4f}")
    print(f"  HEARD channel with no hawk: mean={float(jnp.mean(heard[~on])):.4f}")
