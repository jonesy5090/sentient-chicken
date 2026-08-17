"""How far back does the reward's cause actually sit? (E031 diagnostic)

`hen/plasticity.py:34-37` states the rule's effective credit window is `tau_slow` = 0.2 s
and that "anything that has to bridge a longer gap than that is not learnable by this rule
as written". That has been cited for four experiments as the likely reason H2 is a null.

**It has never been measured.** Before proposing a fix, measure whether the foraging
task's reward actually requires a long bridge. Two outcomes, both useful:

  - reward is predicted by motor activity at short lag -> the window is ADEQUATE, the
    credit-window story is wrong, and H2's null needs a different explanation
  - the predictive information sits seconds back -> the window is the blocker and it
    earns a hypothesis node

Method: run a flock, record per-step reward and motor output, then correlate reward at
time t against each motor channel at t - lag, sweeping lag from 0 to 10 s. Also report
the raw event structure -- how often reward arrives at all, and in what size lumps --
because a rule cannot learn from a signal that is silent most of the time either.
"""
import argparse

import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, plasticity, regions
from hen.plasticity import PlasticConfig

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=120.0)
ap.add_argument("--hens", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

cfg = spec.DEFAULT_COOP._replace(n_hens=args.hens)
pc = PlasticConfig()
steps = int(args.seconds / cfg.dt)

w = world.reset(jax.random.key(args.seed), cfg)
p = connectome.build(jax.random.key(args.seed + 1), regions.DEFAULT_REGIONS,
                     n_hens=cfg.n_hens)
x = brain.initial_state(p, cfg.n_hens)

R, M, FED = [], [], []
for t in range(steps):
    obs = sensing.observe(w, cfg)
    x, motor, _ = brain.step(x, obs, p, cfg.dt)
    wn = world.step(w, motor, jax.random.fold_in(jax.random.key(7), t), cfg)
    R.append(np.asarray(plasticity.reward(w, wn, cfg, pc)))
    M.append(np.asarray(motor))
    FED.append(np.asarray(wn.n_fed - w.n_fed))
    w = wn

R = np.stack(R)          # (T, H)
M = np.stack(M)          # (T, H, MOTOR_DIM)
FED = np.stack(FED)      # (T, H)
T = R.shape[0]

print(f"{args.seconds:.0f} s, {args.hens} hens, dt={cfg.dt}\n")

# --- Is there a signal at all? -------------------------------------------------
per_hen_events = FED.sum(0)
frac_rewarded = float((np.abs(R) > 1e-6).mean())
print("reward signal structure")
print(f"  steps with any reward movement : {100*frac_rewarded:5.1f}%")
print(f"  feeding events per hen         : {per_hen_events.mean():.1f} "
      f"(min {per_hen_events.min():.0f}, max {per_hen_events.max():.0f})")
print(f"  mean gap between feeds         : "
      f"{args.seconds / max(per_hen_events.mean(), 1e-9):.1f} s")
print(f"  reward sd                      : {R.std():.4f}\n")

# --- Where does the predictive information sit? --------------------------------
# Correlate reward(t) against motor(t - lag). A rule whose eligibility trace decays
# with time constant tau can only exploit structure inside roughly that window.
LAGS_S = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
names = {spec.M_PECK: "peck", spec.M_FORWARD: "forward",
         spec.M_SCRATCH: "scratch", spec.M_TURN_L: "turn_L",
         spec.M_CROUCH: "crouch"}

print("corr( reward(t), motor(t - lag) ), pooled over hens")
hdr = "  lag (s)  " + "".join(f"{n:>10}" for n in names.values())
print(hdr); print("  " + "-" * (len(hdr) - 2))
best = {}
for lag_s in LAGS_S:
    k = int(round(lag_s / cfg.dt))
    row = []
    for ch in names:
        a = R[k:].reshape(-1)
        b = M[:T - k if k else T, :, ch].reshape(-1)
        c = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
        row.append(c)
        best[ch] = max(best.get(ch, (0.0, 0.0)), (abs(c), lag_s), key=lambda z: z[0])
    print(f"  {lag_s:>7.2f}  " + "".join(f"{v:>10.4f}" for v in row))

print("\n  peak |correlation| by channel:")
for ch, n in names.items():
    mag, lag = best[ch]
    print(f"    {n:<10}{mag:.4f} at lag {lag:.2f} s"
          f"{'   <- INSIDE the 0.2 s window' if lag <= 0.2 else ''}")

print(f"\n  the rule's credit window is tau_slow = {pc.tau_slow} s.")
print("  if the peaks sit at or below that, the window is not what blocks H2.")
