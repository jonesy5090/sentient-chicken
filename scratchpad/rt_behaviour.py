"""Did the Dale re-baselining change what the bird does? (E028)

The reward decomposition came back sane, but strike-steps at the H4 configuration went
from 1000/3000 to 0/3000 -- so either the hens have stopped being caught, or something
about the world stopped working. Measuring the behaviour directly rather than inferring
it from the reward, per the rule that cost this project twenty-six experiments.

`--lesion` sets W_out to zero, which isolates whether any change is the readout's doing.
"""
import argparse

import jax, jax.numpy as jnp, numpy as np
from coop import spec, world
from hen import brain, connectome, regions
from run import simulate

ap = argparse.ArgumentParser()
ap.add_argument("--minutes", type=float, default=2.0)
ap.add_argument("--seeds", type=int, default=3)
args = ap.parse_args()

print(f"behaviour at the H4 configuration, {args.seeds} seeds x {args.minutes:.0f} min\n")
hdr = (f"{'genome':<22}{'crouch':>9}{'peck':>9}{'head_dn':>9}"
       f"{'fed %':>8}{'strikes':>9}{'blind risks':>13}")
print(hdr); print("-" * len(hdr))

for label, lesion in (("as built", False), ("W_out lesioned", True)):
    acc = []
    for s in range(args.seeds):
        cfg = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=20.0)
        w = world.reset(jax.random.key(s), cfg)
        p = connectome.build(jax.random.fold_in(jax.random.key(s), 1),
                             regions.DEFAULT_REGIONS.with_pallium(1.5),
                             n_hens=16, auditory_scaffold=True)
        if lesion:
            p = p._replace(W_out=jnp.zeros_like(p.W_out))
        x = brain.initial_state(p, 16)
        w_end, _x, _p, _ps, _k, summary = simulate.simulate(
            w, x, p, jax.random.fold_in(jax.random.key(s), 2), cfg,
            args.minutes * 60.0, 60.0, simulate.NO_PLASTICITY)
        m = np.asarray(summary.motor)          # (chunks, MOTOR_DIM), already flock-mean
        steps = args.minutes * 60.0 / cfg.dt * 16
        acc.append((
            float(m[:, spec.M_CROUCH].mean()),
            float(m[:, spec.M_PECK].mean()),
            float(np.maximum(m[:, spec.M_PECK], m[:, spec.M_SCRATCH]).mean()),
            float(jnp.sum(w_end.n_fed)) / steps * 100,
            float(jnp.sum(w_end.n_strike_events)),
            float(jnp.sum(w_end.n_blind_risk)),
        ))
    a = np.array(acc).mean(0)
    print(f"{label:<22}{a[0]:>9.3f}{a[1]:>9.3f}{a[2]:>9.3f}"
          f"{a[3]:>8.2f}{a[4]:>9.1f}{a[5]:>13.1f}")

print("\ncrouch near 1.0 means she is permanently hiding -- safe, blind and starving.")
