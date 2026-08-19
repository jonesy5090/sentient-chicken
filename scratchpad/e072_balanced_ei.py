"""E072: does balancing E/I in W -- removing the sensory->pallium DC that E027 already
removed from W_out -- improve H2d's separability?

Paired per genome from the start (E035: unpaired ratio-of-means on 6 genomes gave both
a false positive and a false negative for a structural question of this shape). Same
settle-and-separate probe as E009/E017/E023/E034/E035/E041, so numbers are comparable.

The load-bearing comparison is balanced vs a GAIN-MATCHED control, not vs baseline:
gain reduction scales common-mode and differential down together, balancing removes
common-mode while preserving differential.
"""
import jax, jax.numpy as jnp, numpy as np
from coop import spec
from hen import brain, connectome, neurons, regions
from run.experiment import _t_critical

reg = regions.DEFAULT_REGIONS
DT, HOLD, N = 0.01, 200, 12
p_lo, p_hi = reg.bounds(regions.PALLIUM)

o_hawk = np.zeros(spec.OBS_DIM, np.float32); o_hawk[spec.IDX_AERIAL] = 1.0
o_call = np.zeros(spec.OBS_DIM, np.float32); o_call[spec.AUDIO_LO + 2] = 1.0
o_rest = np.zeros(spec.OBS_DIM, np.float32)


def settle(p, obs):
    x = brain.initial_state(p, 1)
    o = jnp.asarray(obs)[None, :]
    for _ in range(HOLD):
        x, _m, _d = brain.step(x, o, p, DT)
    return np.asarray(neurons.rate(x))[0]


def sep_and_rate(p):
    h, c, z = (settle(p, o)[p_lo:p_hi] for o in (o_hawk, o_call, o_rest))
    rest = float(np.mean(np.abs(z)))
    return float(np.sqrt(np.mean((h - c) ** 2)) / (rest + 1e-9)), rest


def build(seed, **kw):
    return connectome.build(jax.random.key(seed), reg, n_hens=1, **kw)


# --- match the control's gain to balanced's mean pallial rate --------------
bal_rate = float(np.mean([sep_and_rate(build(s, balanced_ei=True))[1] for s in range(4)]))
base_rate = float(np.mean([sep_and_rate(build(s))[1] for s in range(4)]))
lo, hi = 0.05, 0.95
for _ in range(18):
    mid = 0.5 * (lo + hi)
    r = float(np.mean([sep_and_rate(build(s, gain=mid))[1] for s in range(4)]))
    lo, hi = (mid, hi) if r < bal_rate else (lo, mid)
gain_matched = 0.5 * (lo + hi)
print(f"mean pallial rate -- baseline {base_rate:.4f}, balanced {bal_rate:.4f}")
print(f"gain-matched control uses gain={gain_matched:.4f}\n")

CONDS = {
    "baseline": dict(),
    "balanced_ei": dict(balanced_ei=True),
    f"gain-matched ({gain_matched:.2f})": dict(gain=gain_matched),
}

res = {}
for name, kw in CONDS.items():
    seps, rates = zip(*(sep_and_rate(build(s, **kw)) for s in range(N)))
    res[name] = np.array(seps)
    print(f"{name:<24} separability {np.mean(seps):.4f} "
          f"(mean rate {np.mean(rates):.4f})")

crit = _t_critical(N - 1)
print(f"\npaired contrasts, {N} genomes, threshold t={crit:.3f}")
base = res["baseline"]
gm = res[f"gain-matched ({gain_matched:.2f})"]
for label, a, b in (("balanced vs baseline", res["balanced_ei"], base),
                    ("gain-matched vs baseline", gm, base),
                    ("balanced vs GAIN-MATCHED  <-- the test", res["balanced_ei"], gm)):
    d = a - b
    se = d.std(ddof=1) / np.sqrt(N)
    t = abs(d.mean()) / (se + 1e-12)
    print(f"  {label:<40}{d.mean():+.4f} +/- {se:.4f}  t={t:.2f}  "
          f"{'SIGNIFICANT' if t > crit else 'not significant'}  "
          f"({a.mean()/b.mean():.2f}x)")
