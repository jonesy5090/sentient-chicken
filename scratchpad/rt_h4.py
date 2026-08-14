"""The precondition for H4: does a blind hen's ear tell her anything her eye cannot?"""
import jax, jax.numpy as jnp
from coop import spec, world
from hen import brain, connectome, regions, plasticity
from hen.plasticity import PlasticConfig
from run import simulate
cfg = spec.DEFAULT_COOP; n = cfg.n_hens
pc = PlasticConfig(enabled=False, explore_sigma=0.0)
AER = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)

def body(carry,_):
    carry,(motor,obs,_r,_m) = simulate._one_step(carry,None,cfg,pc)
    w = carry[0]
    return carry,(w.hawk_on, obs[:,spec.IDX_AERIAL], obs[:,AER],
                  jnp.max(motor[:,jnp.array(list(spec.HEAD_DOWN_ACTIONS))],axis=-1),
                  jnp.linalg.norm(w.pos-w.hawk_pos[None,:],axis=-1))
@jax.jit
def go(w,x,p,ps,k): return jax.lax.scan(body,(w,x,p,ps,k),None,length=60000)

for seed in (3, 11):
    key = jax.random.key(seed)
    w = world.reset(key,cfg); p = connectome.build(jax.random.fold_in(key,1),regions.DEFAULT_REGIONS,n_hens=n)
    x = brain.initial_state(p,n); ps = plasticity.initial_state(p,n,pc)
    _c,(hon,aer,heard,hd,dhawk) = go(w,x,p,ps,jax.random.fold_in(key,2))
    inrange = dhawk < cfg.vision_range
    hawk = (hon>0.5)[:,None] & inrange       # a hawk she COULD see if head-up
    blind = aer < 1e-6                        # her own eye tells her nothing
    # among blind hen-steps: heard alarm when a visible-to-someone hawk is present vs not
    a = heard[blind & hawk]; b = heard[blind & ~hawk]
    # the opportunity: blind while >=1 flockmate is not blind
    someone_sees = jnp.any(aer > 1e-6, axis=-1, keepdims=True)
    opp = blind & hawk & someone_sees
    print(f"seed {seed}: hawk-in-range hen-steps={int(jnp.sum(hawk))}/{hawk.size}")
    print(f"   blind hen-steps with a hawk overhead: {int(jnp.sum(blind&hawk))}"
          f"   of which >=1 flockmate can see it: {int(jnp.sum(opp))}"
          f"  ({100*float(jnp.sum(opp))/max(int(jnp.sum(hawk)),1):.1f}% of hawk hen-steps)")
    if a.size and b.size:
        print(f"   heard-alarm level when blind: hawk present {float(jnp.mean(a)):.3f} "
              f"(sd {float(jnp.std(a)):.3f})  vs absent {float(jnp.mean(b)):.4f} "
              f"(sd {float(jnp.std(b)):.4f})   d'={(float(jnp.mean(a))-float(jnp.mean(b)))/((float(jnp.std(a))+float(jnp.std(b)))/2+1e-9):.2f}")
