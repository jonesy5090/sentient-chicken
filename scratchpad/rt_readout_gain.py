"""How much does W_out actually move now, vs under the pre-E019 (uncentred) rule?"""
import jax, jax.numpy as jnp
from coop import spec, world
from hen import brain, connectome, regions, plasticity
from hen.plasticity import PlasticConfig
from run import simulate

cfg = spec.DEFAULT_COOP; n = cfg.n_hens
pc = PlasticConfig(enabled=True, growth_enabled=False, explore_sigma=0.6)
MIN = 10.0

orig = plasticity.consolidate
def uncentred(p, ps, m, pc):
    """The pre-E019 rule: raw traces, no centring."""
    zero = ps._replace(z_fast_bar=jnp.zeros_like(ps.z_fast_bar),
                       z_slow_bar=jnp.zeros_like(ps.z_slow_bar),
                       z_motor_bar=jnp.zeros_like(ps.z_motor_bar))
    return orig(p, zero, m, pc)

for label, fn in (("centred (current)", orig), ("uncentred (pre-E019)", uncentred)):
    plasticity.consolidate = fn
    import importlib; importlib.reload(simulate)
    for seed in (0, 5):
        key = jax.random.key(seed)
        w = world.reset(key, cfg)
        p0 = connectome.build(jax.random.fold_in(key,1), regions.DEFAULT_REGIONS, n_hens=n)
        x = brain.initial_state(p0, n)
        w_end,_x,p1,_ps,_k,s = simulate.simulate(w,x,p0,jax.random.fold_in(key,2),cfg,MIN*60,60.0,pc)
        dW = p1.W_out - p0.W_out
        rel = float(jnp.linalg.norm(dW)/jnp.linalg.norm(p0.W_out))
        third = max(1,len(s.hunger)//3)
        print(f"{label:22s} seed {seed}: |dW_out|/|W_out| = {rel:.4f}"
              f"  |W_out| {float(jnp.mean(jnp.abs(p0.W_out))):.4f}->{float(jnp.mean(jnp.abs(p1.W_out))):.4f}"
              f"  reflex/cortical drive = {float(s.reflex_drive[-1]):.3f}/{float(s.cortical_drive[-1]):.3f}"
              f"  hunger change {float(jnp.mean(s.hunger[-third:])-jnp.mean(s.hunger[:third])):+.4f}"
              f"  fed {100*float(jnp.sum(w_end.n_fed))/(n*MIN*60/cfg.dt):.2f}%"
              f"  syn {float(jnp.mean(jnp.sum(p1.W!=0,axis=(1,2)))):.0f}", flush=True)
plasticity.consolidate = orig
