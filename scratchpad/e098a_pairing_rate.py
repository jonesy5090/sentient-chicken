"""E098 Part A -- the pairing rate (docs/experiments/E098 section 5).

"Count, per rearing run, how often a flockmate's aerial call is audible to the focal
hen, at `hawk_period_s` in {900, 300, 60}. This decides Part B's world."

The quantity is the *received* aerial-alarm auditory channel, `obs[:, AUDIO_LO + 2]`.
`coop/sensing.py` builds it from `atten @ calls` with the self-distance set to 1e6, so
it contains flockmates' voices only -- a hen never hears herself on it. An "event" is a
bout, counted with hysteresis (rise above `hi`, and not counted again until it has
fallen below `lo`), because a chorus fluctuates and a raw threshold-crossing count
would report one bout many times.

NOT `spec.IDX_AERIAL`: that channel is the hen's own *sight* of a hawk overhead
(`sensing.observe` writes it from hawk geometry alone, and no call ever touches it), so
it cannot report what a flockmate said. The audio channel above is the one that carries
another hen's voice. Both are recorded, and bouts are additionally split by whether the
focal hen could see the hawk herself at onset -- the case where the call tells her
something she does not already have is the one the contingency has to be learnable from.

Reared under arm P's config -- the `W_pred` rule whose learnability section 2(c)
questions -- so the count describes the run that has to contain the contingency.

    PYTHONPATH=. .venv/bin/python scratchpad/e098a_pairing_rate.py
"""
import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from coop import spec, world
from hen import brain, connectome, plasticity, regions
from hen.plasticity import PlasticConfig
from run import simulate

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--minutes", type=float, default=30.0)
ap.add_argument("--periods", default="900,300,60")
a = ap.parse_args()

BASE = spec.DEFAULT_COOP._replace(n_hens=16, food_deplete_rate=0.0)
STEPS = int(a.minutes * 60.0 / BASE.dt)
AER_AUDIO = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)

# Arm P, verbatim from scratchpad/e097_wpred_audience.py.
PC = PlasticConfig(enabled=True, pred_enabled=True, pred_gain=1.0,
                   pred_centred=True, pred_bar_freeze_s=60.0)

# Bout thresholds. `hi` is the primary; the others are reported so the choice of
# threshold is visible rather than assumed.
HI_LEVELS = (0.1, 0.2, 0.3, 0.5)
PRIMARY_HI = 0.2


@partial(jax.jit, static_argnames=("cfg", "pc", "n_steps"))
def _listen(w, x, p, ps, key, cfg, pc, n_steps):
    """Rear, retaining only the received aerial-alarm channel and hawk presence."""
    def body(carry, _):
        carry, (_motor, obs, _r, _m) = simulate._one_step(carry, None, cfg, pc)
        return carry, (obs[:, AER_AUDIO], obs[0, spec.IDX_AERIAL], carry[0].hawk_on)
    _carry, out = jax.lax.scan(body, (w, x, p, ps, key), None, length=n_steps)
    return out


def bout_onsets(sig: np.ndarray, hi: float, lo: float) -> list:
    """Indices of rising bouts: a rise above `hi` having previously fallen below `lo`."""
    above_hi = sig >= hi
    below_lo = sig <= lo
    out, armed = [], True
    for i, (h, l) in enumerate(zip(above_hi.tolist(), below_lo.tolist())):
        if armed and h:
            out.append(i)
            armed = False
        elif l:
            armed = True
    return out


def bouts(sig: np.ndarray, hi: float, lo: float) -> int:
    return len(bout_onsets(sig, hi, lo))


print(f"E098 Part A -- how often is a flockmate's aerial alarm call audible to the "
      f"focal hen?\n{a.seeds} seeds, {a.minutes:.0f} min rearing, {BASE.n_hens} hens, "
      f"arm P's plastic config.\nChannel: obs[:, {AER_AUDIO}] (AUDIO_LO + aerial); "
      f"flockmates only, self excluded by sensing.py.\n", flush=True)

t0 = time.perf_counter()
res = {}
for period in [float(x) for x in a.periods.split(",")]:
    cfg = BASE._replace(hawk_period_s=period)
    res[period] = []
    for seed in range(a.seeds):
        k = jax.random.key(seed)
        p = connectome.build(jax.random.fold_in(k, 1), regions.DEFAULT_REGIONS,
                             n_hens=cfg.n_hens, auditory_scaffold=True)
        w = world.reset(k, cfg)
        x = brain.initial_state(p, cfg.n_hens)
        ps = plasticity.initial_state(p, cfg.n_hens, PC)
        audio, seen, hawk_on = _listen(w, x, p, ps, jax.random.fold_in(k, 2),
                                       cfg, PC, STEPS)
        audio = np.asarray(audio)                      # (T, H)
        seen = np.asarray(seen)                        # (T,) focal's own IDX_AERIAL
        hawk_on = np.asarray(hawk_on)                  # (T,)
        focal = audio[:, 0]
        onsets = {h: bout_onsets(focal, h, h / 2) for h in HI_LEVELS}
        rec = {
            "seed": seed,
            "hawk_bouts": bouts(hawk_on, 0.5, 0.5),
            "q": [float(np.quantile(focal, q))
                  for q in (0.5, 0.9, 0.95, 0.99, 1.0)],
            "focal": {h: len(onsets[h]) for h in HI_LEVELS},
            # Of those, the ones where she could not see the hawk herself at onset --
            # the subset where the call carries something her own eyes did not.
            "focal_blind": {h: int(np.sum(seen[onsets[h]] < 0.05)) if onsets[h] else 0
                            for h in HI_LEVELS},
            "flock_mean": {h: float(np.mean([bouts(audio[:, i], h, h / 2)
                                             for i in range(audio.shape[1])]))
                           for h in HI_LEVELS},
        }
        res[period].append(rec)
        print(f"  period {period:>5.0f} s  seed {seed}: hawk bouts "
              f"{rec['hawk_bouts']:>3d}   focal audible bouts "
              f"{rec['focal'][PRIMARY_HI]:>4d} (hi={PRIMARY_HI})   "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)

# ------------------------------------------------------------------ report
print("\n--- received aerial-alarm channel, focal hen, quantiles ----------------")
print(f"{'period':>8}{'median':>10}{'p90':>10}{'p95':>10}{'p99':>10}{'max':>10}")
for period, rs in res.items():
    q = np.mean([r["q"] for r in rs], axis=0)
    print(f"{period:>8.0f}" + "".join(f"{v:>10.4f}" for v in q))

print("\n--- audible flockmate aerial-call bouts per 30 min rearing run ---------")
print("focal hen (hen 0); one bout = channel rises above `hi` having fallen below hi/2")
hdr = f"{'period':>8}{'hawks':>8}" + "".join(f"{'hi=' + str(h):>10}" for h in HI_LEVELS)
print(hdr)
print("-" * len(hdr))
for period, rs in res.items():
    hw = np.mean([r["hawk_bouts"] for r in rs])
    print(f"{period:>8.0f}{hw:>8.1f}"
          + "".join(f"{np.mean([r['focal'][h] for r in rs]):>10.1f}"
                   for h in HI_LEVELS))

print("\nsame, averaged over all 16 hens rather than the focal one:")
print(hdr)
print("-" * len(hdr))
for period, rs in res.items():
    hw = np.mean([r["hawk_bouts"] for r in rs])
    print(f"{period:>8.0f}{hw:>8.1f}"
          + "".join(f"{np.mean([r['flock_mean'][h] for r in rs]):>10.1f}"
                   for h in HI_LEVELS))

print("\nof those focal bouts, the ones where she could NOT see the hawk herself at")
print("onset (own IDX_AERIAL < 0.05) -- the call is then telling her something new:")
print(hdr)
print("-" * len(hdr))
for period, rs in res.items():
    hw = np.mean([r["hawk_bouts"] for r in rs])
    print(f"{period:>8.0f}{hw:>8.1f}"
          + "".join(f"{np.mean([r['focal_blind'][h] for r in rs]):>10.1f}"
                   for h in HI_LEVELS))

print("\nper-seed focal counts at the primary threshold "
      f"(hi={PRIMARY_HI}, lo={PRIMARY_HI / 2}):")
for period, rs in res.items():
    print(f"  {period:>5.0f} s: " + ", ".join(str(r["focal"][PRIMARY_HI]) for r in rs)
          + "   (of which blind: "
          + ", ".join(str(r["focal_blind"][PRIMARY_HI]) for r in rs) + ")")

print("\nsection 3 prediction 3: fewer than 5 per run at 900 s, more than 20 at 60 s.")
print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
