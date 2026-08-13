"""Re-baseline the recurrent gain against mixed E/I.

The 0.70 default was chosen in E010 for a pallium that was 100% excitatory. That pool
sat on a saddle-node, which is why E010 had to record a measured optimum at 0.75 while
refusing to use it ("a value that has to be held to two decimal places is not a value
to build on"). With inhibition present the operating point should be stable over a much
wider band. This measures whether it is, and picks the new default.

Same probe as E009/E017: settle on "saw a hawk" vs "heard an aerial alarm", read
pallial separability relative to mean activity under a null observation.
"""
import jax, jax.numpy as jnp, numpy as np
from coop import spec
from hen import brain, connectome, neurons, regions

reg = regions.DEFAULT_REGIONS
DT, HOLD = 0.01, 200
p_lo, p_hi = reg.bounds(regions.PALLIUM)

o_hawk = np.zeros(spec.OBS_DIM, np.float32); o_hawk[spec.IDX_AERIAL] = 1.0
o_call = np.zeros(spec.OBS_DIM, np.float32)
o_call[spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)] = 1.0
o_rest = np.zeros(spec.OBS_DIM, np.float32)


def settle(p, obs):
    x = brain.initial_state(p, 1)
    o = jnp.asarray(obs)[None, :]
    for _ in range(HOLD):
        x, _, _ = brain.step(x, o, p, DT)
    return np.asarray(neurons.rate(x))[0]


print(f"{'gain':>6}{'mean pallial rate':>20}{'sd across genomes':>20}"
      f"{'separability':>15}{'sd':>9}")
for gain in (0.60, 0.70, 0.75, 0.78, 0.90, 1.10, 1.40):
    rates, seps = [], []
    for s in range(6):
        p = connectome.build(jax.random.key(s), reg, n_hens=1, gain=gain)
        h, c, z = (settle(p, o)[p_lo:p_hi] for o in (o_hawk, o_call, o_rest))
        rates.append(float(np.mean(z)))
        seps.append(float(np.sqrt(np.mean((h - c) ** 2)) / (np.mean(np.abs(z)) + 1e-9)))
    print(f"{gain:>6.2f}{np.mean(rates):>20.3f}{np.std(rates):>20.3f}"
          f"{np.mean(seps):>15.4f}{np.std(seps):>9.4f}")

print("\nE010 measured, on the 100%-excitatory pallium:")
print("  0.60 -> rate 0.212, sep 0.033 | 0.70 -> 0.271, 0.075 | 0.75 -> 0.349, 0.142")
print("  0.78 -> 0.497, 0.062 | 0.90 -> 0.830, 0.009   (collapse past 0.75)")
