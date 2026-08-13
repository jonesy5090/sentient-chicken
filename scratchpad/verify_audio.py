"""Independent check of the reviewer's finding 3: is the audio channel constant?"""
import jax, jax.numpy as jnp, numpy as np
from coop import spec, sensing, world
from hen import brain, connectome, regions
from run import simulate

for n in (2, 4, 8, 16):
    cfg = spec.DEFAULT_COOP._replace(n_hens=n)
    w = world.reset(jax.random.key(0), cfg)
    p = connectome.build(jax.random.key(1), regions.DEFAULT_REGIONS, n_hens=n)
    x = brain.initial_state(p, n)
    # let the flock settle into whatever configuration it actually adopts
    w, x, *_ = simulate.rollout_quiet(w, x, p, jax.random.key(5), cfg, 12_000)
    obs = sensing.observe(w, cfg)
    aud = np.asarray(obs[:, spec.AUDIO_LO:spec.AUDIO_HI])
    calls = np.asarray(w.calls)
    d = np.asarray(jnp.linalg.norm(w.pos[:, None] - w.pos[None, :], axis=-1))
    nn = np.min(d + np.eye(n) * 1e6, axis=1)
    print(f"n_hens={n:>3}  audio min={aud.min():.4f} mean={aud.mean():.4f} "
          f"max={aud.max():.4f} | raw call floor={calls.min():.4f} "
          f"| nearest-neighbour {nn.mean():.2f} m  spread {d.max():.2f} m")

# Does a genuine full-amplitude alarm move the channel at n=16?
cfg = spec.DEFAULT_COOP._replace(n_hens=16)
w = world.reset(jax.random.key(0), cfg)
p = connectome.build(jax.random.key(1), regions.DEFAULT_REGIONS, n_hens=16)
x = brain.initial_state(p, 16)
w, x, *_ = simulate.rollout_quiet(w, x, p, jax.random.key(5), cfg, 12_000)
idx = spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)
base = float(sensing.observe(w, cfg)[0, spec.AUDIO_LO + idx])
loud = w._replace(calls=w.calls.at[1, idx].set(1.0))
print(f"\nn=16: aerial audio heard by hen 0, baseline {base:.4f} -> "
      f"{float(sensing.observe(loud, cfg)[0, spec.AUDIO_LO + idx]):.4f} "
      f"when hen 1 calls at FULL amplitude")
