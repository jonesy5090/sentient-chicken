"""Re-run E019's three measurements against the fixed code."""
import jax, jax.numpy as jnp, numpy as np
from coop import spec, sensing, world
from hen import brain, connectome, plasticity, regions
from hen.plasticity import PlasticConfig
from run import simulate

print("=== FIX 1: can a call be heard? ===")
for n in (4, 8, 16, 32):
    cfg = spec.DEFAULT_COOP._replace(n_hens=n)
    w = world.reset(jax.random.key(0), cfg)
    p = connectome.build(jax.random.key(1), regions.DEFAULT_REGIONS, n_hens=n)
    x = brain.initial_state(p, n)
    w, x, *_ = simulate.rollout_quiet(w, x, p, jax.random.key(5), cfg, 12_000)
    aud = np.asarray(sensing.observe(w, cfg)[:, spec.AUDIO_LO:spec.AUDIO_HI])
    i = spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)
    base = float(sensing.observe(w, cfg)[0, spec.AUDIO_LO + i])
    loud = w._replace(calls=w.calls.at[1, i].set(1.0))
    after = float(sensing.observe(loud, cfg)[0, spec.AUDIO_LO + i])
    print(f"n={n:>3}  resting audio mean={aud.mean():.4f} max={aud.max():.4f} | "
          f"hen 0 hears aerial {base:.4f} -> {after:.4f} when a neighbour calls "
          f"(delta {after - base:+.4f})")

print("\n=== FIX 2: can the readout learn more than a constant? ===")
cfg = spec.DEFAULT_COOP._replace(n_hens=16)
pc = PlasticConfig(enabled=True, growth_enabled=False, explore_sigma=0.6)
w0 = world.reset(jax.random.key(0), cfg)
p0 = connectome.build(jax.random.key(1), regions.DEFAULT_REGIONS, n_hens=16)
x0 = brain.initial_state(p0, 16)
w, x, p1, *_ = simulate.simulate(w0, x0, p0, jax.random.key(2), cfg, 200.0, 60.0, pc)

dW = np.asarray(p1.W_out - p0.W_out)
shares = [np.linalg.svd(dW[h], compute_uv=False) for h in range(16)]
shares = [s[0] ** 2 / (s ** 2).sum() for s in shares]
print(f"top-1 singular value share of dW_out : {np.mean(shares):.4f}  "
      f"(was 0.9981; 1.0 = rank one, a pure constant offset)")

cort = []
for t in range(300):
    obs = sensing.observe(w, cfg)
    x, motor, drives = brain.step(x, obs, p1, cfg.dt)
    w = world.step(w, motor, jax.random.fold_in(jax.random.key(7), t), cfg)
    cort.append(np.asarray(drives.cortical))
cort = np.array(cort)
mag, sd = np.abs(cort).mean(), cort.std(axis=0).mean()
print(f"cortical drive sd/|mean|             : {sd / mag:.4f}  "
      f"(was 0.0070; 0 = no state dependence at all)")

print("\n=== FIX 3: what is the reward made of? ===")
w = world.reset(jax.random.key(0), cfg)
p = connectome.build(jax.random.key(1), regions.DEFAULT_REGIONS, n_hens=16)
x = brain.initial_state(p, 16)
comps = {k: [] for k in ("hunger", "thirst", "cold", "vigour", "strike")}
for t in range(3000):
    obs = sensing.observe(w, cfg)
    x, motor, _ = brain.step(x, obs, p, cfg.dt)
    wn = world.step(w, motor, jax.random.fold_in(jax.random.key(8), t), cfg)
    for k in ("hunger", "thirst", "cold"):
        comps[k].append(np.asarray(getattr(w, k) - getattr(wn, k)) / cfg.dt
                        * pc.reward_scale)
    comps["vigour"].append(np.asarray(wn.vigour - w.vigour) / cfg.dt * pc.reward_scale)
    comps["strike"].append(-np.asarray(wn.n_struck - w.n_struck) * pc.strike_penalty)
    w = wn

# vigour is measured but no longer entered into reward(); show what reward now weighs
in_reward = ("hunger", "thirst", "cold", "strike")
tot = sum(np.array(comps[k]).var() for k in in_reward)
print(f"{'component':<10}{'in reward?':>12}{'sd':>10}{'share of reward var':>22}")
for k, v in comps.items():
    v = np.array(v)
    inr = k in in_reward
    share = f"{100 * v.var() / max(tot, 1e-12):>21.1f}%" if inr else f"{'--':>22}"
    print(f"{k:<10}{('yes' if inr else 'NO'):>12}{v.std():>10.4f}{share}")
