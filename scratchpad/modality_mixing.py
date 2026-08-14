"""Is the audio channel anatomically distinguishable from vision at the first synapse?

Measures, on the innate connectome as built today:
  1. how much of the sensory stub's afferent drive is auditory vs visual
  2. how much the "saw a hawk" and "heard an aerial alarm" input vectors overlap
  3. separability of those two percepts at the input stage, before any recurrence
"""
import jax, jax.numpy as jnp, numpy as np
from coop import spec
from hen import connectome, regions

reg = regions.DEFAULT_REGIONS
seeds = range(6)

frac_audio, cos_in, sep_in, sep_pall, shared = [], [], [], [], []

for s in seeds:
    p = connectome.build(jax.random.key(s), reg, n_hens=2)
    W_in = np.asarray(p.W_in)
    s_lo, s_hi = reg.bounds(regions.SENSORY)
    stub = W_in[s_lo:s_hi]                       # (64, OBS_DIM)

    aud = slice(spec.AUDIO_LO, spec.AUDIO_HI)
    vis = slice(spec.VIS_LO, spec.VIS_HI)

    # 1. share of afferent weight
    total = stub.sum()
    frac_audio.append(stub[:, aud].sum() / total)

    # 2/3. two percepts, matched in magnitude: a hawk seen overhead at proximity 1,
    # vs an aerial alarm call heard at amplitude 1. Nothing else differs.
    o_hawk = np.zeros(spec.OBS_DIM, np.float32); o_hawk[spec.IDX_AERIAL] = 1.0
    o_call = np.zeros(spec.OBS_DIM, np.float32); o_call[spec.AUDIO_LO + 2] = 1.0

    u_hawk, u_call = stub @ o_hawk, stub @ o_call
    cos = float(u_hawk @ u_call / (np.linalg.norm(u_hawk) * np.linalg.norm(u_call) + 1e-9))
    cos_in.append(cos)
    # fraction of stub units driven by BOTH
    shared.append(float(np.mean((u_hawk > 0) & (u_call > 0))))

    # separability at the input, as % of mean drive (same metric as E009's gain sweep)
    drive = np.concatenate([u_hawk, u_call])
    sep_in.append(float(np.linalg.norm(u_hawk - u_call) / (np.abs(drive).mean() * np.sqrt(len(u_hawk)))))

f = lambda a: f"{np.mean(a):.3f} +- {np.std(a):.3f}"
print(f"sensory stub afferent weight that is auditory : {f(frac_audio)}"
      f"   ({spec.AUDIO_HI - spec.AUDIO_LO}/{spec.OBS_DIM - 6} extero channels"
      f" = {(spec.AUDIO_HI-spec.AUDIO_LO)/(spec.OBS_DIM-6):.3f} by count)")
print(f"stub units driven by hawk AND by alarm call   : {f(shared)}")
print(f"cosine(hawk drive, call drive) on the stub    : {f(cos_in)}")
print(f"input-stage separability (x mean drive)       : {f(sep_in)}")
