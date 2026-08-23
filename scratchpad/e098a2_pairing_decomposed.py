"""E098 Part A, addendum -- is the audible aerial call actually about hawks?

Part A's raw bout count came back at ~1600 per 30-minute run at `hawk_period_s=900`,
a world containing zero to one hawk. A count that large in a world with no predator is
not measuring the contingency section 2(c) asks about; it is measuring chatter. Arm P
rears at `explore_sigma=0.6`, so every hen's aerial-call motor carries exploration noise
above `spec.CALL_FLOOR`, and sixteen of those sum into one channel.

So decompose it. The quantity section 2(c) actually needs is "flockmates in view precedes
aerial alarm audio", i.e. how many *hawk events* the focal hen hears a flockmate's alarm
during. That is bounded above by the number of hawks, which is the point.

Reported per condition:
  - hawk events, and the fraction of steps a hawk is overhead
  - the received aerial channel's mean with a hawk on vs off, and their difference
  - point-biserial correlation of the channel with hawk presence
  - PAIRED EVENTS: hawk events during which the focal hen's received aerial channel
    rises `delta` above its own hawk-off median -- the learnable pairing count
  - the same, restricted to events where she could not see the hawk herself

    PYTHONPATH=. .venv/bin/python scratchpad/e098a2_pairing_decomposed.py
"""
import argparse
import time
from functools import partial

import jax
import numpy as np

from coop import spec, world
from hen import brain, connectome, plasticity, regions
from hen.plasticity import PlasticConfig
from run import simulate

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--minutes", type=float, default=30.0)
ap.add_argument("--periods", default="900,300,60")
ap.add_argument("--delta", type=float, default=0.05,
                help="rise above the hawk-off median that counts as hearing an alarm")
a = ap.parse_args()

BASE = spec.DEFAULT_COOP._replace(n_hens=16, food_deplete_rate=0.0)
STEPS = int(a.minutes * 60.0 / BASE.dt)
AER_AUDIO = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)
PC = PlasticConfig(enabled=True, pred_enabled=True, pred_gain=1.0,
                   pred_centred=True, pred_bar_freeze_s=60.0)


@partial(jax.jit, static_argnames=("cfg", "pc", "n_steps"))
def _listen(w, x, p, ps, key, cfg, pc, n_steps):
    def body(carry, _):
        carry, (motor, obs, _r, _m) = simulate._one_step(carry, None, cfg, pc)
        # received aerial audio (all hens), focal's own sight of the hawk, focal's
        # own emitted aerial call, and whether a hawk is overhead.
        return carry, (obs[:, AER_AUDIO], obs[0, spec.IDX_AERIAL],
                       motor[0, spec.M_CALL_AERIAL], carry[0].hawk_on)
    _carry, out = jax.lax.scan(body, (w, x, p, ps, key), None, length=n_steps)
    return out


def events(on: np.ndarray) -> list:
    """(start, stop) index pairs for each contiguous run of hawk presence."""
    on = on > 0.5
    d = np.diff(np.concatenate([[False], on, [False]]).astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


print(f"E098 Part A addendum -- decomposing the aerial-alarm channel.\n"
      f"{a.seeds} seeds, {a.minutes:.0f} min, {BASE.n_hens} hens, arm P's config "
      f"(explore_sigma={PC.explore_sigma}).\n", flush=True)

t0 = time.perf_counter()
rows = {}
for period in [float(x) for x in a.periods.split(",")]:
    cfg = BASE._replace(hawk_period_s=period)
    rows[period] = []
    for seed in range(a.seeds):
        k = jax.random.key(seed)
        p = connectome.build(jax.random.fold_in(k, 1), regions.DEFAULT_REGIONS,
                             n_hens=cfg.n_hens, auditory_scaffold=True)
        w = world.reset(k, cfg)
        x = brain.initial_state(p, cfg.n_hens)
        ps = plasticity.initial_state(p, cfg.n_hens, PC)
        audio, seen, own, hawk = _listen(w, x, p, ps, jax.random.fold_in(k, 2),
                                         cfg, PC, STEPS)
        audio = np.asarray(audio)[:, 0]        # focal hen's received channel
        seen = np.asarray(seen)
        own = np.asarray(own)
        hawk = np.asarray(hawk)

        on = hawk > 0.5
        ev = events(hawk)
        base_med = float(np.median(audio[~on])) if (~on).any() else 0.0
        thr = base_med + a.delta
        paired = [(s, e) for s, e in ev if audio[s:e].max() >= thr]
        # Of those, the ones where she never saw the hawk herself.
        blind = [(s, e) for s, e in paired if seen[s:e].max() < 0.05]
        sd = audio.std()
        r = (float(np.corrcoef(audio, on.astype(float))[0, 1])
             if sd > 1e-9 and 0 < on.sum() < len(on) else float("nan"))
        rec = {
            "seed": seed, "n_hawk": len(ev), "frac_on": float(on.mean()),
            "mean_off": float(audio[~on].mean()) if (~on).any() else float("nan"),
            "mean_on": float(audio[on].mean()) if on.any() else float("nan"),
            "med_off": base_med, "r": r,
            "paired": len(paired), "blind": len(blind),
            "own_off": float(own[~on].mean()) if (~on).any() else float("nan"),
            "own_on": float(own[on].mean()) if on.any() else float("nan"),
        }
        rows[period].append(rec)
        print(f"  period {period:>5.0f} s  seed {seed}: {rec['n_hawk']:>3d} hawk "
              f"events, {rec['paired']:>3d} paired, audio off/on "
              f"{rec['mean_off']:.3f}/{rec['mean_on']:.3f}  "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)

print("\n--- the received aerial channel, focal hen, split by hawk presence -----")
hdr = (f"{'period':>8}{'hawks':>7}{'%on':>7}{'audio off':>11}{'audio on':>10}"
       f"{'delta':>9}{'corr':>8}{'own off':>9}{'own on':>8}")
print(hdr)
print("-" * len(hdr))
for period, rs in rows.items():
    g = lambda k: float(np.mean([r[k] for r in rs]))
    print(f"{period:>8.0f}{g('n_hawk'):>7.1f}{100 * g('frac_on'):>7.1f}"
          f"{g('mean_off'):>11.4f}{g('mean_on'):>10.4f}"
          f"{g('mean_on') - g('mean_off'):>+9.4f}{g('r'):>8.3f}"
          f"{g('own_off'):>9.4f}{g('own_on'):>8.4f}")
print("'own' = the focal hen's OWN emitted aerial call. If that barely moves with the")
print("hawk either, the call is not hawk-contingent in this world at all.")

print("\n--- PAIRED EVENTS: hawk events she hears a flockmate's alarm during -----")
print(f"threshold = the run's own hawk-off median + {a.delta:.2f}")
hdr2 = (f"{'period':>8}{'hawk events':>13}{'paired':>9}{'% paired':>10}"
        f"{'paired & blind':>16}")
print(hdr2)
print("-" * len(hdr2))
for period, rs in rows.items():
    g = lambda k: float(np.mean([r[k] for r in rs]))
    pct = 100.0 * g("paired") / g("n_hawk") if g("n_hawk") else float("nan")
    print(f"{period:>8.0f}{g('n_hawk'):>13.1f}{g('paired'):>9.1f}{pct:>10.0f}"
          f"{g('blind'):>16.1f}")

print("\nper-seed paired counts:")
for period, rs in rows.items():
    print(f"  {period:>5.0f} s: " + ", ".join(f"{r['paired']}/{r['n_hawk']}"
                                              for r in rs))

print("\nsection 3 prediction 3 asked for <5 at 900 s and >20 at 60 s.")
print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
