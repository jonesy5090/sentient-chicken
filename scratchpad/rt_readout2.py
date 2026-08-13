import sys, jax, jax.numpy as jnp
from coop import spec, world
from hen import brain, connectome, regions, plasticity
from hen.plasticity import PlasticConfig
MODE = sys.argv[1]
orig = plasticity.consolidate
if MODE == "uncentred":
    def uncentred(p, ps, m, pc):
        z = ps._replace(z_fast_bar=jnp.zeros_like(ps.z_fast_bar),
                        z_slow_bar=jnp.zeros_like(ps.z_slow_bar),
                        z_motor_bar=jnp.zeros_like(ps.z_motor_bar))
        return orig(p, z, m, pc)
    plasticity.consolidate = uncentred
from run import simulate           # import AFTER patching, so nothing is cached
cfg = spec.DEFAULT_COOP; n = cfg.n_hens; MIN = 10.0
pc = PlasticConfig(enabled=True, growth_enabled=False, explore_sigma=0.6)
for seed in (0, 5):
    key = jax.random.key(seed)
    w = world.reset(key, cfg)
    p0 = connectome.build(jax.random.fold_in(key,1), regions.DEFAULT_REGIONS, n_hens=n)
    x = brain.initial_state(p0, n)
    w_end,_x,p1,_ps,_k,s = simulate.simulate(w,x,p0,jax.random.fold_in(key,2),cfg,MIN*60,60.0,pc)
    third = max(1,len(s.hunger)//3)
    print(f"{MODE:10s} seed {seed}: |dW_out|/|W_out|={float(jnp.linalg.norm(p1.W_out-p0.W_out)/jnp.linalg.norm(p0.W_out)):.4f}"
          f"  reflex/cortical={float(s.reflex_drive[-1]):.3f}/{float(s.cortical_drive[-1]):.3f}"
          f"  hunger {float(jnp.mean(s.hunger[-third:])-jnp.mean(s.hunger[:third])):+.4f}"
          f"  fed {100*float(jnp.sum(w_end.n_fed))/(n*MIN*60/cfg.dt):.2f}%"
          f"  syn {float(jnp.mean(jnp.sum(p1.W!=0,axis=(1,2)))):.0f}", flush=True)
