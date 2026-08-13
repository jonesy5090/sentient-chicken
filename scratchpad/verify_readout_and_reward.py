"""Independent check of the reviewer's findings 1 and 2.

1. Is dW_out rank-1 -- i.e. can the learned readout only apply a constant offset?
2. Does the vigour (call-cost) term dominate the reward signal?
"""
import jax, jax.numpy as jnp, numpy as np
from coop import spec, sensing, world
from hen import brain, connectome, plasticity, regions
from hen.plasticity import PlasticConfig
from run import simulate

cfg = spec.DEFAULT_COOP._replace(n_hens=16)
pc = PlasticConfig(enabled=True, growth_enabled=False, explore_sigma=0.6)
w0 = world.reset(jax.random.key(0), cfg)
p0 = connectome.build(jax.random.key(1), regions.DEFAULT_REGIONS, n_hens=16)
x0 = brain.initial_state(p0, 16)

w, x, p1, ps, key, _summary = simulate.simulate(
    w0, x0, p0, jax.random.key(2), cfg, 200.0, 60.0, pc)

# --- 1. rank of the learned readout change ---
dW = np.asarray(p1.W_out - p0.W_out)          # (H, MOTOR_DIM, n_motor)
shares, flat = [], []
for h in range(dW.shape[0]):
    s = np.linalg.svd(dW[h], compute_uv=False)
    shares.append(s[0] ** 2 / (s ** 2).sum())
    flat.append(dW[h].ravel() / (np.linalg.norm(dW[h]) + 1e-12))
flat = np.array(flat)
cos = flat @ flat.T
off = cos[~np.eye(16, dtype=bool)]
print("--- finding 1: can the readout learn a policy, or only an offset? ---")
print(f"top-1 singular value share of dW_out : mean {np.mean(shares):.4f} "
      f"min {np.min(shares):.4f}   (1.0 = exactly rank 1)")
print(f"mean pairwise cosine across the 16 hens: {off.mean():.3f} "
      f"(1.0 = every hen learned the identical thing)")

# how much does the cortical drive actually VARY as behaviour unfolds?
xx = x
cort = []
for _ in range(300):
    obs = sensing.observe(w, cfg)
    xx, motor, drives = brain.step(xx, obs, p1, cfg.dt)
    w = world.step(w, motor, jax.random.fold_in(jax.random.key(7), _), cfg)
    cort.append(np.asarray(drives.cortical))
cort = np.array(cort)                                   # (T, H, MOTOR_DIM)
mag = np.abs(cort).mean()
sd = cort.std(axis=0).mean()
print(f"cortical drive: |mean| {mag:.3f}, sd over time {sd:.4f}, "
      f"sd/|mean| = {sd / mag:.3f}   (0 = a pure constant, no state dependence)")

names = ("forward", "turn_L", "turn_R", "peck", "scratch", "crouch", "flee",
         "c_contact", "c_food", "c_aerial", "c_ground")
c0 = np.asarray(jnp.einsum("hmn,hn->hm", p0.W_out,
                           jnp.ones((16, p0.W_out.shape[-1])) * 0.27)).mean(0)
c1 = np.asarray(jnp.einsum("hmn,hn->hm", p1.W_out,
                           jnp.ones((16, p1.W_out.shape[-1])) * 0.27)).mean(0)
print("\nlearned constant offset per motor channel (hatch -> 200 s):")
for i in np.argsort(c1 - c0)[:5]:
    print(f"  {names[i]:<10} {c0[i]:+.2f} -> {c1[i]:+.2f}   ({c1[i]-c0[i]:+.2f})")

# --- 2. what is the reward signal actually made of? ---
print("\n--- finding 2: what dominates the reward? ---")
w = world.reset(jax.random.key(0), cfg)
p = connectome.build(jax.random.key(1), regions.DEFAULT_REGIONS, n_hens=16)
xx = brain.initial_state(p, 16)
comps = {k: [] for k in ("hunger", "thirst", "cold", "vigour", "strike")}
rews, fed_mask, food_vis = [], [], []
for t in range(900):
    obs = sensing.observe(w, cfg)
    xx, motor, _ = brain.step(xx, obs, p, cfg.dt)
    wn = world.step(w, motor, jax.random.fold_in(jax.random.key(8), t), cfg)
    for k in ("hunger", "thirst", "cold"):
        comps[k].append(np.asarray(getattr(w, k) - getattr(wn, k)) / cfg.dt
                        * pc.reward_scale)
    comps["vigour"].append(np.asarray(wn.vigour - w.vigour) / cfg.dt
                           * pc.reward_scale)
    comps["strike"].append(-np.asarray(wn.n_struck - w.n_struck)
                           * pc.strike_penalty)
    rews.append(np.asarray(plasticity.reward(w, wn, cfg, pc)))
    fed_mask.append(np.asarray(w.hunger - wn.hunger) > 1e-7)
    front = [spec.vis_index(b, spec.CLS_FOOD) for b in (5, 6)]
    food_vis.append(np.asarray(obs[:, front]).max(axis=1) > 0.01)
    w = wn

rews = np.array(rews)
tot_var = sum(np.array(v).var() for v in comps.values())
print(f"{'component':<10}{'mean':>10}{'sd':>10}{'share of variance':>20}")
for k, v in comps.items():
    v = np.array(v)
    print(f"{k:<10}{v.mean():>+10.4f}{v.std():>10.4f}"
          f"{100 * v.var() / max(tot_var, 1e-12):>19.1f}%")

fv, fm = np.array(food_vis), np.array(fed_mask)
print(f"\nmean reward, food NOT in front  : {rews[~fv].mean():+.4f}  (n={(~fv).sum()})")
print(f"mean reward, food in front      : {rews[fv].mean():+.4f}  (n={fv.sum()})")
print(f"mean reward, actually fed        : {rews[fm].mean():+.4f}  (n={fm.sum()})")
print(f"mean reward, saw food, did NOT eat: {rews[fv & ~fm].mean():+.4f}  "
      f"(n={(fv & ~fm).sum()})")
