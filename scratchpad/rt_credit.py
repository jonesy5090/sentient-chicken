import functools, jax, jax.numpy as jnp
from coop import spec, world
from hen import brain, connectome, regions, plasticity
from hen.plasticity import PlasticConfig
from run import simulate

cfg = spec.DEFAULT_COOP; n = cfg.n_hens; STEPS = 30000
pc = PlasticConfig(enabled=False, explore_sigma=0.0)

def body(carry, _):
    carry, (motor, obs, _r, _m) = simulate._one_step(carry, None, cfg, pc)
    w = carry[0]
    return carry, (w.pos, motor[:, spec.M_PECK], w.hunger, w.cold, w.n_struck)

@functools.partial(jax.jit, static_argnames=())
def go(w, x, p, ps, k):
    return jax.lax.scan(body, (w, x, p, ps, k), None, length=STEPS)

for seed in (0, 3, 7):
    key = jax.random.key(seed)
    w = world.reset(key, cfg)
    p = connectome.build(jax.random.fold_in(key,1), regions.DEFAULT_REGIONS, n_hens=n)
    x = brain.initial_state(p, n); ps = plasticity.initial_state(p, n, pc)
    (w_end,*_), (pos, peck, hung, cold_, struck) = go(w, x, p, ps, jax.random.fold_in(key,2))
    d = jnp.min(jnp.linalg.norm(pos[:, :, None, :] - w_end.food_pos[None, None, :, :], axis=-1), axis=-1)
    at = d < cfg.peck_radius; near = (d < 1.0) & ~at; far = d >= 1.0
    f = lambda m: float(jnp.sum((peck > 0.5) & m) / jnp.maximum(jnp.sum(m), 1))
    frac_at = jnp.mean(at, axis=0)
    print(f"seed {seed}: P(peck|on food)={f(at):.3f} P(peck|<1m)={f(near):.3f} P(peck|>1m)={f(far):.3f}")
    print(f"   time-on-food per hen: min={float(jnp.min(frac_at)):.3f} med={float(jnp.median(frac_at)):.3f} "
          f"max={float(jnp.max(frac_at)):.3f}; hens never reaching food {int(jnp.sum(frac_at==0))}/{n}")
    dh = -jnp.diff(hung, axis=0)/cfg.dt*60.0
    dc = -jnp.diff(cold_, axis=0)/cfg.dt*60.0
    dsr = -jnp.diff(struck, axis=0)*1.0
    tot = dh+dc+dsr
    print("   reward variance share:", {k2: round(float(jnp.var(v)/(jnp.var(tot)+1e-12)),3)
          for k2,v in (("hunger",dh),("cold",dc),("strike",dsr))},
          f"  mean|reward|={float(jnp.mean(jnp.abs(tot))):.3f}")
    # how long is a hen continuously on food, and how long between visits
    a = jnp.asarray(at, dtype=jnp.int32)
    ch = jnp.abs(jnp.diff(a, axis=0)).sum(axis=0)
    print(f"   on/off-food transitions per hen over {STEPS*cfg.dt:.0f}s: "
          f"mean={float(jnp.mean(ch)):.1f}; mean bout length={float(jnp.sum(a)/jnp.maximum(jnp.sum(ch)/2,1))*cfg.dt:.2f}s")
