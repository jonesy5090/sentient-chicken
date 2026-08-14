"""H4 precondition, with the threshold that matters: crouch fires at aerial > 0.3125
(reflex weight 8.0 vs REST_BIAS 2.5). 'Blind' = her own eye cannot trigger the reflex."""
import jax, jax.numpy as jnp
from coop import spec, world
from hen import brain, connectome, regions, plasticity
from hen.plasticity import PlasticConfig
from run import simulate
cfg = spec.DEFAULT_COOP; n = cfg.n_hens
pc = PlasticConfig(enabled=False, explore_sigma=0.0)
AER = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)
THR = 2.5/8.0
def body(carry,_):
    carry,(motor,obs,_r,_m) = simulate._one_step(carry,None,cfg,pc)
    w = carry[0]
    return carry,(w.hawk_on, obs[:,spec.IDX_AERIAL], obs[:,AER],
                  jnp.linalg.norm(w.pos-w.hawk_pos[None,:],axis=-1))
@jax.jit
def go(w,x,p,ps,k): return jax.lax.scan(body,(w,x,p,ps,k),None,length=60000)
for seed in (3, 11):
    key = jax.random.key(seed)
    w = world.reset(key,cfg); p = connectome.build(jax.random.fold_in(key,1),regions.DEFAULT_REGIONS,n_hens=n)
    x = brain.initial_state(p,n); ps = plasticity.initial_state(p,n,pc)
    _c,(hon,aer,heard,dh) = go(w,x,p,ps,jax.random.fold_in(key,2))
    # steps where a hawk is close enough that a head-up hen WOULD crouch
    could = (hon>0.5)[:,None] & ((1.0 - dh/cfg.vision_range) > THR)
    sees = aer > THR
    blind = could & ~sees
    someone = jnp.any(could & sees, axis=-1, keepdims=True)
    opp = blind & someone
    tot = int(jnp.sum(could))
    print(f"seed {seed}: hen-steps where a head-up hen would crouch: {tot}")
    if tot:
        print(f"   she sees it herself: {100*float(jnp.sum(could&sees))/tot:.1f}%   "
              f"blind: {100*float(jnp.sum(blind))/tot:.1f}%   "
              f"blind AND a flockmate sees it (the case a call helps): {100*float(jnp.sum(opp))/tot:.1f}%")
        a = heard[opp]; b = heard[~could.any(axis=-1,keepdims=True) & jnp.ones_like(heard,dtype=bool)]
        if a.size: print(f"   heard alarm in that case: {float(jnp.mean(a)):.3f}  vs baseline {float(jnp.mean(heard[~could])):.3f}")
