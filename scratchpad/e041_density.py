"""E041: does sparser sensory->pallium connectivity improve H2d's separability?

Paired per-genome, from the start -- E035 found unpaired ratio-of-means on 6 genomes
gave a false positive (and, on one first pass, a false negative) for a structural
question of exactly this kind. Same settle-and-separate probe as E009/E017/E023/E034/
E035.
"""
import argparse

import jax, jax.numpy as jnp, numpy as np
from coop import spec
from hen import brain, connectome, neurons, regions
from run.experiment import _t_critical

ap = argparse.ArgumentParser()
ap.add_argument("--genomes", type=int, default=12)
a = ap.parse_args()

reg = regions.DEFAULT_REGIONS
DT, HOLD = 0.01, 200
p_lo, p_hi = reg.bounds(regions.PALLIUM)

o_hawk = np.zeros(spec.OBS_DIM, np.float32); o_hawk[spec.IDX_AERIAL] = 1.0
o_call = np.zeros(spec.OBS_DIM, np.float32); o_call[spec.AUDIO_LO + 2] = 1.0
o_rest = np.zeros(spec.OBS_DIM, np.float32)


def settle(p, obs):
    x = brain.initial_state(p, 1)
    o = jnp.asarray(obs)[None, :]
    for _ in range(HOLD):
        x, _, _ = brain.step(x, o, p, DT)
    return np.asarray(neurons.rate(x))[0]


def pallial_sep(p):
    h, c, z = (settle(p, o)[p_lo:p_hi] for o in (o_hawk, o_call, o_rest))
    return float(np.sqrt(np.mean((h - c) ** 2)) / (np.mean(np.abs(z)) + 1e-9))


def pallial_mean_rate(p):
    return float(np.mean(np.abs(settle(p, o_rest)[p_lo:p_hi])))


DENSITIES = [0.30, 0.15, 0.08, 0.04, 0.02]
N = a.genomes

print(f"E041 -- sensory->pallium density sweep, {N} genomes, paired per genome\n")
print(f"{'density':<10}{'mean rate':>12}{'separability':>16}{'vs 0.30':>10}")

rows = {}
for d in DENSITIES:
    seps, rates = [], []
    for s in range(N):
        p = connectome.build(jax.random.key(s), reg, n_hens=1, sensory_pallium_density=d)
        seps.append(pallial_sep(p))
        rates.append(pallial_mean_rate(p))
    rows[d] = np.array(seps)
    base = rows[DENSITIES[0]].mean() if d != DENSITIES[0] else None
    ratio = f"{np.mean(seps)/rows[DENSITIES[0]].mean():.2f}x" if d != DENSITIES[0] else "1.00x"
    print(f"{d:<10.2f}{np.mean(rates):>12.4f}{np.mean(seps):>10.4f}+-{np.std(seps):<5.4f}{ratio:>10}")

print("\npaired contrast vs density=0.30 (matched genomes, threshold at df=N-1):")
base = rows[DENSITIES[0]]
crit = _t_critical(N - 1)
for d in DENSITIES[1:]:
    diff = rows[d] - base
    m, se = diff.mean(), diff.std(ddof=1) / N ** 0.5
    t = abs(m) / (se + 1e-12)
    print(f"  {d:.2f} - 0.30: {m:+.4f} +/- {se:.4f}  t={t:.2f}  threshold={crit:.3f}  -> "
          f"{'SIGNIFICANT' if t > crit else 'not significant'}")
