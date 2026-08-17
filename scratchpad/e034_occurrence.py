"""E034 part B: does the hawk-vs-alarm-call contrast H2d is built on actually occur
during live coop operation, now that E019 made calls audible?

Samples a full per-step trace at the standard H4 configuration (16 hens, hawk every
20 s -- the config nearly every H4-lineage experiment treats as "the coop") and reads
off, per hen-step:
  - the visual aerial channel (obs[IDX_AERIAL]) -- already gated by head-down in
    coop/sensing.py, so nonzero means "currently seeing the hawk"
  - the auditory aerial-call channel (obs[AUDIO_LO+2]) -- nonzero means "currently
    hearing a flockmate's aerial alarm"
"""
import argparse

import jax, jax.numpy as jnp, numpy as np
from coop import spec
from hen import brain, connectome, regions
from run import simulate

ap = argparse.ArgumentParser()
ap.add_argument("--minutes", type=float, default=5.0)
ap.add_argument("--hawk-period", type=float, default=20.0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

cfg = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=a.hawk_period)
key = jax.random.key(a.seed)
p = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS, n_hens=16)
w = None
from coop import world
w = world.reset(jax.random.fold_in(key, 0), cfg)
x = brain.initial_state(p, 16)

n_steps = int(a.minutes * 60.0 / cfg.dt)
print(f"rolling out {a.minutes:.0f} min ({n_steps} steps) at hawk_period_s={a.hawk_period} ...")
w, x, p, ps, k, trace = simulate.rollout(w, x, p, jax.random.fold_in(key, 2), cfg, n_steps)

obs = np.asarray(trace.obs)          # (T, H, OBS_DIM)
vis_aerial = obs[:, :, spec.IDX_AERIAL]
aud_aerial = obs[:, :, spec.AUDIO_LO + 2]

THRESH = 0.05   # "on" -- well above float noise, well below a full-amplitude signal

sees = vis_aerial > THRESH
hears = aud_aerial > THRESH
blind_and_hears = (~sees) & hears

n = vis_aerial.size
print(f"\nhen-steps total: {n}")
print(f"visual aerial channel 'on' (sees hawk)      : {sees.mean()*100:.3f}%")
print(f"auditory aerial channel 'on' (hears alarm)   : {hears.mean()*100:.3f}%")
print(f"blind to hawk AND hearing an alarm           : {blind_and_hears.mean()*100:.3f}%")
print(f"  -- of hen-steps where a hawk is audible somewhere, "
      f"{100*blind_and_hears.sum()/max(hears.sum(),1):.1f}% are also blind")

print(f"\nauditory aerial channel stats: min={aud_aerial.min():.4f} "
      f"max={aud_aerial.max():.4f} mean={aud_aerial.mean():.4f} std={aud_aerial.std():.4f}")
print(f"visual aerial channel stats:   min={vis_aerial.min():.4f} "
      f"max={vis_aerial.max():.4f} mean={vis_aerial.mean():.4f} std={vis_aerial.std():.4f}")

# how many distinct call "events" -- rising edges of hears, any hen
any_hears = hears.any(axis=1)
edges = int(np.sum(np.diff(any_hears.astype(np.int8)) == 1))
print(f"\ndistinct flock-wide alarm-audible episodes (rising edges): {edges}")
any_sees = sees.any(axis=1)
edges_v = int(np.sum(np.diff(any_sees.astype(np.int8)) == 1))
print(f"distinct flock-wide hawk-visible episodes (rising edges): {edges_v}")
