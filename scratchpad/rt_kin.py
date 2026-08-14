"""Is the audibility-weighted kin term (E006) different from the flat flock mean?"""
import jax, jax.numpy as jnp
from coop import spec, world
from hen import brain, connectome, regions, plasticity
from hen.plasticity import PlasticConfig
from run import simulate
cfg = spec.DEFAULT_COOP; n = cfg.n_hens
pc = PlasticConfig(enabled=False, explore_sigma=0.0)
for seed in (0, 3, 7):
    key = jax.random.key(seed)
    w = world.reset(key, cfg)
    p = connectome.build(jax.random.fold_in(key,1), regions.DEFAULT_REGIONS, n_hens=n)
    x = brain.initial_state(p, n)
    w2,*_ = simulate.rollout_quiet(w, x, p, jax.random.fold_in(key,2), cfg, 18000)
    d = jnp.linalg.norm(w2.pos[:,None,:]-w2.pos[None,:,:],axis=-1) + jnp.eye(n)*1e6
    aud = jnp.clip(1.0 - d/cfg.hear_range, 0.0, 1.0)
    own = jax.random.normal(jax.random.key(99),(n,))    # arbitrary per-hen welfare
    heard = (aud @ own)/(jnp.sum(aud,axis=-1)+1e-6)     # kin_audible=True
    flat  = (jnp.sum(own)-own)/(n-1)                    # kin_audible=False
    off = aud[~jnp.eye(n,dtype=bool)]
    print(f"seed {seed}: pairwise audibility weights min={float(jnp.min(off)):.3f} "
          f"max={float(jnp.max(off)):.3f}  flock diameter={float(jnp.max(d[d<1e5])):.2f} m")
    print(f"   corr(audible-kin, flat-kin) = {float(jnp.corrcoef(heard, flat)[0,1]):.5f}"
          f"   max|difference| = {float(jnp.max(jnp.abs(heard-flat))):.4f}"
          f"   sd(flat)={float(jnp.std(flat)):.4f}")
