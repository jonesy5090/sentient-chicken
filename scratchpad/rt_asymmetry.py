"""H1a / H4 precondition: is vigilance actually PRIVATE, or is the flock synchronised?"""
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
    hd = jnp.max(motor[:, jnp.array(list(spec.HEAD_DOWN_ACTIONS))], axis=-1)
    return carry, (hd, w.hawk_on, obs[:, spec.IDX_AERIAL], w.pos)

@jax.jit
def go(w, x, p, ps, k):
    return jax.lax.scan(body, (w, x, p, ps, k), None, length=STEPS)

for seed in (0, 3, 7, 11):
    key = jax.random.key(seed)
    w = world.reset(key, cfg)
    p = connectome.build(jax.random.fold_in(key,1), regions.DEFAULT_REGIONS, n_hens=n)
    x = brain.initial_state(p, n); ps = plasticity.initial_state(p, n, pc)
    _c, (hd, hawk_on, aerial, pos) = go(w, x, p, ps, jax.random.fold_in(key,2))
    down = hd > 0.5
    frac = float(jnp.mean(down))
    # cross-hen correlation of head-down
    d = down.astype(jnp.float32)
    dm = d - jnp.mean(d, axis=0, keepdims=True)
    c = (dm.T @ dm) / STEPS
    sd = jnp.sqrt(jnp.diag(c)) + 1e-9
    corr = c / (sd[:, None] * sd[None, :])
    off = corr[~jnp.eye(n, dtype=bool)]
    # nearest-neighbour distance
    dd = jnp.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1) + jnp.eye(n)*1e6
    nn = jnp.min(dd, axis=-1)
    # during hawk events: how many hens can see it
    on = hawk_on > 0.5
    if bool(jnp.any(on)):
        nseen = jnp.sum(aerial[on] > 0.01, axis=-1)
        print(f"seed {seed}: head-down {frac*100:.0f}%  mean cross-hen corr(head-down)={float(jnp.mean(off)):+.3f}"
              f"  NN dist median={float(jnp.median(nn)):.2f} m")
        print(f"   hawk on for {int(jnp.sum(on))} steps; hens with aerial>0: "
              f"mean={float(jnp.mean(nseen)):.1f}/16  frac steps where 0 see it="
              f"{float(jnp.mean(nseen==0)):.2f}  where all 16 see it={float(jnp.mean(nseen==n)):.2f}"
              f"  where 1..15 see it={float(jnp.mean((nseen>0)&(nseen<n))):.2f}")
    else:
        print(f"seed {seed}: head-down {frac*100:.0f}%  corr={float(jnp.mean(off)):+.3f}"
              f"  NN median={float(jnp.median(nn)):.2f} m  -- no hawk in 300 s")
