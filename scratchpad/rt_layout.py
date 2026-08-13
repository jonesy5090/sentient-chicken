"""How much of the between-seed variance is the random food layout?
Same protocol as the harness, but the coop's resources are held fixed across seeds;
only the genome, the hens' start positions and the predator stream vary."""
import jax, jax.numpy as jnp, statistics as st
from coop import spec, world
from hen import brain, connectome, regions
from hen.plasticity import PlasticConfig
from run import simulate
cfg = spec.DEFAULT_COOP; MIN=10.0
pc = PlasticConfig(enabled=False, explore_sigma=0.0)
ref = world.reset(jax.random.key(0), cfg)      # one canonical resource layout
def run(seed, fix_layout):
    key = jax.random.key(seed)
    w = world.reset(key, cfg)
    if fix_layout:
        w = w._replace(food_pos=ref.food_pos, water_pos=ref.water_pos)
    p = connectome.build(jax.random.fold_in(key,1), regions.DEFAULT_REGIONS, n_hens=cfg.n_hens)
    x = brain.initial_state(p, cfg.n_hens)
    w_end,_x,_p,_ps,_k,s = simulate.simulate(w,x,p,jax.random.fold_in(key,2),cfg,MIN*60,60.0,pc)
    third=max(1,len(s.hunger)//3)
    return (float(jnp.mean(s.hunger[-third:])-jnp.mean(s.hunger[:third])),
            100*float(jnp.sum(w_end.n_fed))/(cfg.n_hens*MIN*60/cfg.dt))
for fix in (True,):
    out=[run(s, fix) for s in range(10)]
    ch=[o[0] for o in out]; fed=[o[1] for o in out]
    print(f"food layout FIXED across seeds, n=10:")
    print(f"  hunger change: mean {st.mean(ch):+.4f}  sd {st.stdev(ch):.4f}   per-seed "
          + " ".join(f"{c:+.3f}" for c in ch))
    print(f"  fed %:         mean {st.mean(fed):.2f}   sd {st.stdev(fed):.2f}  range {min(fed):.2f}-{max(fed):.2f}")
print("  (for comparison, random layout, n=27: hunger-change sd 0.0701, fed% 1.31-14.87)")
