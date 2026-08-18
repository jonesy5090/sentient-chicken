"""E044: structural read of W_pred after rearing -- is the growth E043 found
(max reaching 30-40% of cap while the mean stayed flat) concentrated on the entries a
real association should strengthen, or diffuse/arbitrary? See docs/experiments/E044.

For each reared connectome:
  1. mean|W_pred| per target observation channel, averaged over pallial sources --
     is IDX_AERIAL specifically elevated over the other 58 channels?
  2. of the pallial neurons with the strongest weight onto IDX_AERIAL, do they
     overlap with the neurons that respond most to *hearing the alarm call*
     (measured directly, via the same settle-and-separate probe as E017/E034/E041)?
"""
import argparse, json, os, time

import jax, jax.numpy as jnp, numpy as np
from coop import spec, world
from hen import brain, connectome, neurons, regions
from hen.plasticity import PlasticConfig
from run import simulate

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=6)
ap.add_argument("--minutes", type=float, default=20.0)
ap.add_argument("--hawk-period", type=float, default=10.0)
ap.add_argument("--density", type=float, default=1.0)
ap.add_argument("--cache", default="scratchpad/e044_cache.npz")
a = ap.parse_args()

cfg = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=a.hawk_period,
                                 food_deplete_rate=0.0)
ASSOC = PlasticConfig(enabled=True, growth_enabled=False, explore_sigma=0.6,
                      pred_enabled=True)

reg = regions.DEFAULT_REGIONS
p_lo, p_hi = reg.bounds(regions.PALLIUM)
DT, HOLD = 0.01, 200
o_call = np.zeros(spec.OBS_DIM, np.float32); o_call[spec.AUDIO_LO + 2] = 1.0
o_rest = np.zeros(spec.OBS_DIM, np.float32)


def call_response_hen0(p, n_hens):
    """Per-pallial-neuron activation under the alarm call, relative to rest, for hen 0
    specifically -- p's weight arrays carry all n_hens, so the settle must be batched
    to match (brain.step multiplies against p.W etc., which are (n_hens, N, N)) and
    hen 0 extracted afterward, not settled as a standalone 1-hen params object."""
    def settle(obs):
        x = brain.initial_state(p, n_hens)
        o = jnp.tile(jnp.asarray(obs)[None, :], (n_hens, 1))
        for _ in range(HOLD):
            x, _, _ = brain.step(x, o, p, DT)
        return np.asarray(neurons.rate(x))[0]      # hen 0 only
    return settle(o_call)[p_lo:p_hi] - settle(o_rest)[p_lo:p_hi]


def run_one(seed):
    key = jax.random.key(seed)
    w = world.reset(key, cfg)
    p = connectome.build(jax.random.fold_in(key, 1), reg, n_hens=16,
                         sensory_pallium_density=a.density)
    x = brain.initial_state(p, 16)
    _w, _x, p_end, *_ = simulate.simulate(
        w, x, p, jax.random.fold_in(key, 2), cfg, a.minutes * 60.0, 60.0, ASSOC)
    # W_pred: (H, OBS_DIM, N). Per-channel summary averages |weight| over hens *and*
    # pallial sources -- a coarse, flock-level "did learning concentrate on IDX_AERIAL"
    # read. The correlation analysis below needs weight and response paired for the
    # *same* hen (the shared mask still lets per-hen weight draws differ), so it uses
    # hen 0 specifically on both sides rather than an average.
    w_pred_mean = np.asarray(jnp.mean(jnp.abs(p_end.W_pred), axis=0))[:, p_lo:p_hi]
    w_pred_hen0 = np.asarray(jnp.abs(p_end.W_pred[0]))[:, p_lo:p_hi]
    resp = call_response_hen0(p_end, 16)
    return w_pred_mean, w_pred_hen0, resp


t0 = time.perf_counter()
all_w_mean, all_w_hen0, all_resp = [], [], []
for s in range(a.seeds):
    w_mean, w_hen0, resp = run_one(s)
    all_w_mean.append(w_mean)
    all_w_hen0.append(w_hen0)
    all_resp.append(resp)
    print(f"seed {s} done ({time.perf_counter()-t0:.0f}s)")

W = np.stack(all_w_mean)      # (seeds, OBS_DIM, PALLIUM) -- flock mean, for §1
W0 = np.stack(all_w_hen0)     # (seeds, OBS_DIM, PALLIUM) -- hen 0 only, paired with R
R = np.stack(all_resp)        # (seeds, PALLIUM) -- hen 0's call response

print(f"\nE044 -- structural read, {a.seeds} seeds, density={a.density}, "
      f"hawk_period={a.hawk_period}s\n")

# 1. mean |W_pred| per target channel, averaged over pallial sources and seeds
per_channel = W.mean(axis=(0, 2))   # (OBS_DIM,)
order = np.argsort(-per_channel)
print("top 8 target channels by mean|W_pred| (averaged over pallial sources):")
names = {spec.IDX_AERIAL: "IDX_AERIAL (visual, THE target)"}
for i in range(spec.AUDIO_LO, spec.AUDIO_HI):
    names[i] = f"AUDIO[{i - spec.AUDIO_LO}]"
for ch in order[:8]:
    tag = names.get(int(ch), "")
    print(f"  channel {ch:>3}  mean|W_pred|={per_channel[ch]:.6f}  {tag}")
aerial_rank = int(np.where(order == spec.IDX_AERIAL)[0][0]) + 1
print(f"\nIDX_AERIAL rank: {aerial_rank} of {spec.OBS_DIM}  "
      f"(mean|W_pred|={per_channel[spec.IDX_AERIAL]:.6f}, "
      f"overall mean={per_channel.mean():.6f}, "
      f"{'ABOVE' if per_channel[spec.IDX_AERIAL] > per_channel.mean() else 'at/below'} average)")

# 2. do the pallial neurons with strongest weight onto IDX_AERIAL overlap with the
# neurons that respond most to hearing the call? Hen 0's weights paired with hen 0's
# own response -- the shared mask still lets per-hen weight draws differ, so this must
# not be averaged across hens on one side and not the other.
w_aerial = W0[:, spec.IDX_AERIAL, :]        # (seeds, PALLIUM)
corrs = [float(np.corrcoef(w_aerial[s], R[s])[0, 1]) for s in range(a.seeds)]
print(f"\ncorrelation, per seed, between |W_pred[->IDX_AERIAL]| and call-responsiveness:")
for s, c in enumerate(corrs):
    print(f"  seed {s}: r={c:+.3f}")
print(f"  mean r = {np.mean(corrs):+.3f} +/- {np.std(corrs):.3f}")
print("  positive & consistent -> targeted association (call-responsive neurons drive")
print("  the aerial prediction). near zero / inconsistent -> unstructured growth.")
