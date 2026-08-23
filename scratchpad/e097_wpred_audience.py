"""E097: `W_pred` on the audience task -- the rule H2f's falsifier actually named.

Implements docs/experiments/E097-wpred-on-the-audience-task.md section 5 exactly.

Four arms, matched seeds 0-7, 30 min rearing at E057's configuration (16 hens,
food_deplete_rate=0.0, auditory scaffold on):

    S   plasticity off                                            baseline
    H   hebbian_readout, readout_scaling_strength=0.3             E096 reproduction
    P0  pred_enabled, pred_gain=0.0                               wired, cannot reach
    P   pred_enabled, pred_gain=1.0, centred, freeze 60 s         the falsifier's rule

Every arm is reared ONCE and assayed TWICE on that same reared brain -- audio intact,
and audio muted at test (`channel_mode="none"`, so the audience is still present, still
seen, still counted; only what they say is removed). The muted contrast is primary.

DiD (audience-specific) = (alarm_audience - alarm_alone) - (food_audience - food_alone),
matching E096's definition.

    PYTHONPATH=. .venv/bin/python scratchpad/e097_wpred_audience.py
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from coop import spec, world
from hen import brain, connectome, plasticity, regions
from hen.plasticity import PlasticConfig
from run import audience, simulate
from run.experiment import _t_critical

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--minutes", type=float, default=30.0)
ap.add_argument("--cache", default="scratchpad/e097_cache.json")
ap.add_argument("--arms", default="S,H,P0,P")
a = ap.parse_args()

CFG = spec.DEFAULT_COOP._replace(n_hens=16, food_deplete_rate=0.0)
MUTED_CFG = CFG._replace(channel_mode="none")
SEEDS = list(range(a.seeds))
STEPS = int(a.minutes * 60.0 / CFG.dt)

# Section 5's table. The prose under it -- "E088's adopted centring configuration is used
# for the `W_pred` arms" -- is plural, and the gain falsifier compares P against P0, so P0
# is P with pred_gain=0.0 and nothing else changed.
ARMS = {
    "S":  PlasticConfig(enabled=False, explore_sigma=0.0),
    "H":  PlasticConfig(enabled=True, hebbian_readout=True,
                        readout_scaling_strength=0.3),
    "P0": PlasticConfig(enabled=True, pred_enabled=True, pred_gain=0.0,
                        pred_centred=True, pred_bar_freeze_s=60.0),
    "P":  PlasticConfig(enabled=True, pred_enabled=True, pred_gain=1.0,
                        pred_centred=True, pred_bar_freeze_s=60.0),
}
ORDER = [n for n in a.arms.split(",") if n in ARMS]

E096_MUTED_DID = 0.0577          # the reproduction falsifier's target


def rear(seed: int, pc: PlasticConfig):
    """One reared brain. Key derivation identical to E057/E096."""
    k = jax.random.key(seed)
    p = connectome.build(jax.random.fold_in(k, 1), regions.DEFAULT_REGIONS,
                         n_hens=CFG.n_hens, auditory_scaffold=True)
    w = world.reset(k, CFG)
    x = brain.initial_state(p, CFG.n_hens)
    ps = plasticity.initial_state(p, CFG.n_hens, pc)
    # rollout_quiet, not rollout: identical dynamics (same `_one_step`, same carry, same
    # key stream), but a 180k-step trace is 1.7 GB and this box has 9 GB.
    _w, _x, p_end, *_ = simulate.rollout_quiet(
        w, x, p, jax.random.fold_in(k, 2), CFG, STEPS, pc=pc, ps=ps)
    return p_end


def assay_at_gain(p, cfg, gain: float, steps: int = 300):
    """`audience.assay` with the test-time top-down gain set explicitly.

    DIAGNOSTIC ONLY, not part of section 5. `run/audience.py:assay` leaves `pred_gain` at
    `NO_PLASTICITY`'s default of 1.0, so a learned `W_pred` writes onto perception during
    the assay in every arm. This lets that be switched off so its contribution is visible.
    """
    pc = PlasticConfig(enabled=False, explore_sigma=0.0, pred_gain=gain)

    def call_rate(aud, hawk, food, channel):
        w = audience._staged(cfg, CFG.n_hens, audience=aud, hawk=hawk, food=food)
        x = brain.initial_state(p, CFG.n_hens)
        *_, tr = simulate.rollout(w, x, p, jax.random.key(11),
                                  cfg._replace(n_hens=CFG.n_hens), steps, pc=pc)
        return float(jnp.mean(tr.motor[:, 0, channel]))

    return audience.AudienceResult(
        alarm_alone=call_rate(False, True, False, spec.M_CALL_AERIAL),
        alarm_audience=call_rate(True, True, False, spec.M_CALL_AERIAL),
        food_alone=call_rate(False, False, True, spec.M_CALL_FOOD),
        food_audience=call_rate(True, False, True, spec.M_CALL_FOOD),
    )


def run_cell(seed, arm):
    p = rear(seed, ARMS[arm])
    intact = audience.assay(p, CFG, CFG.n_hens)
    muted = audience.assay(p, MUTED_CFG, CFG.n_hens)
    out = {"intact": list(intact), "muted": list(muted)}
    if arm in ("P0", "P"):
        out["intact_g0"] = list(assay_at_gain(p, CFG, 0.0))
        out["muted_g0"] = list(assay_at_gain(p, MUTED_CFG, 0.0))
    return out


# --------------------------------------------------------------------------- run
cache = json.load(open(a.cache)) if os.path.exists(a.cache) else {}
t0 = time.perf_counter()
for arm in ORDER:
    for s in SEEDS:
        ck = f"{arm}|{s}|{a.minutes}"
        if ck not in cache:
            cache[ck] = run_cell(s, arm)
            json.dump(cache, open(a.cache, "w"))
            print(f"  reared+assayed {arm} seed {s} "
                  f"({time.perf_counter() - t0:.0f} s elapsed)", flush=True)

# --------------------------------------------------------------------------- report
def cell(arm, seed, which):
    return audience.AudienceResult(*cache[f"{arm}|{seed}|{a.minutes}"][which])


def did(r):
    return (r.alarm_audience - r.alarm_alone) - (r.food_audience - r.food_alone)


def per_seed(arm, which, f):
    return np.array([f(cell(arm, s, which)) for s in SEEDS])


def mean_se(v):
    m = float(np.mean(v))
    if len(v) < 2:
        return m, 0.0
    return m, float(np.std(v, ddof=1)) / (len(v) ** 0.5)


print(f"\nE097 -- `W_pred` on the audience task. {len(SEEDS)} matched seeds "
      f"(0-{SEEDS[-1]}), {a.minutes:.0f} min rearing, {CFG.n_hens} hens, "
      f"food_deplete_rate={CFG.food_deplete_rate}")
print("Every arm reared once; the SAME reared brain assayed twice -- audio intact, and")
print('audio muted at test (channel_mode="none"). The muted contrast is the primary')
print("measure. DiD = (alarm_aud - alarm_alone) - (food_aud - food_alone).\n")

hdr = (f"{'arm':<4}{'audio':<8}{'alarm_alone':>12}{'alarm_aud':>11}"
       f"{'food_alone':>12}{'food_aud':>10}{'alarm_eff':>11}{'food_eff':>10}{'DiD':>9}")
print(hdr)
print("-" * len(hdr))
for arm in ORDER:
    for which, label in (("intact", "intact"), ("muted", "MUTED")):
        m = lambda f: float(np.mean(per_seed(arm, which, f)))
        print(f"{arm:<4}{label:<8}"
              f"{m(lambda r: r.alarm_alone):>12.4f}"
              f"{m(lambda r: r.alarm_audience):>11.4f}"
              f"{m(lambda r: r.food_alone):>12.4f}"
              f"{m(lambda r: r.food_audience):>10.4f}"
              f"{m(lambda r: r.alarm_effect):>+11.4f}"
              f"{m(lambda r: r.food_effect):>+10.4f}"
              f"{m(did):>+9.4f}")
    print()

print(f"{'arm':<4}{'DiD intact':>12}{'+/- SE':>9}{'DiD muted':>12}{'+/- SE':>9}"
      f"{'survives':>10}")
print("-" * 56)
summary = {}
for arm in ORDER:
    di, sei = mean_se(per_seed(arm, "intact", did))
    dm, sem = mean_se(per_seed(arm, "muted", did))
    summary[arm] = (di, sei, dm, sem)
    frac = (100.0 * dm / di) if abs(di) > 1e-9 else float("nan")
    print(f"{arm:<4}{di:>+12.4f}{sei:>9.4f}{dm:>+12.4f}{sem:>9.4f}{frac:>9.0f}%")

crit = _t_critical(len(SEEDS) - 1)
print(f"\n(paired t critical at df={len(SEEDS) - 1}: {crit:.3f})")
for arm in ORDER:
    v = per_seed(arm, "muted", did)
    m, se = mean_se(v)
    t = abs(m) / (se + 1e-12)
    print(f"  {arm:<4} muted DiD {m:>+8.4f} +/- {se:.4f} SE   t={t:>7.2f}   "
          f"{'SIGNIFICANT' if t > crit else 'not significant'}")

# Diagnostic: how much of each W_pred arm's assay depends on the top-down projection
# writing onto perception AT TEST. See `assay_at_gain`.
diag = [arm for arm in ORDER if arm in ("P0", "P")
        and "muted_g0" in cache[f"{arm}|{SEEDS[0]}|{a.minutes}"]]
if diag:
    print("\n--- DIAGNOSTIC (not pre-registered) -----------------------------------")
    print("`run/audience.py:assay` runs at the PlasticConfig default pred_gain=1.0 in")
    print("every arm, so a learned W_pred writes onto perception during the assay even")
    print("in P0. Re-assayed with the test-time gain forced to 0 for comparison:")
    print(f"\n{'arm':<4}{'DiD intact g=1':>16}{'DiD intact g=0':>16}"
          f"{'DiD muted g=1':>15}{'DiD muted g=0':>15}")
    for arm in diag:
        print(f"{arm:<4}"
              f"{float(np.mean(per_seed(arm, 'intact', did))):>+16.4f}"
              f"{float(np.mean(per_seed(arm, 'intact_g0', did))):>+16.4f}"
              f"{float(np.mean(per_seed(arm, 'muted', did))):>+15.4f}"
              f"{float(np.mean(per_seed(arm, 'muted_g0', did))):>+15.4f}")

# --------------------------------------------------------------------------- falsifiers
print("\n=== PRE-REGISTERED FALSIFIERS (section 4) =============================")

have = lambda *n: all(x in summary for x in n)

# 1. Primary
print("\n1. PRIMARY -- 'Muted audience-specific DiD < +0.10.'")
if have("P"):
    dm = summary["P"][2]
    fires = dm < 0.10
    print(f"   arm P muted DiD = {dm:+.4f}, threshold +0.10")
    print(f"   -> {'FIRES' if fires else 'CLEAR'}: "
          + ("the Pavlovian route does not produce an audience effect either; "
             "H2f should be read as NOT SUPPORTED on its own terms."
             if fires else
             "W_pred produces a muted audience-specific effect at or above the bar."))
else:
    print("   NOT EVALUABLE: arm P not run.")

# 2. Specificity
print("\n2. SPECIFICITY -- 'Food-channel muted DiD >= +0.05.'")
if have("P"):
    raw = float(np.mean(per_seed("P", "muted", lambda r: r.food_effect)))
    line = f"   arm P raw muted food effect (food_aud - food_alone) = {raw:+.4f}"
    if have("S"):
        base = float(np.mean(per_seed("S", "muted", lambda r: r.food_effect)))
        vs_s = raw - base
        line += f"\n   same, differenced against the S baseline = {vs_s:+.4f}"
    else:
        vs_s = raw
    print(line)
    fires = vs_s >= 0.05
    fires_raw = raw >= 0.05
    print(f"   -> {'FIRES' if fires else 'CLEAR'} (on the S-baselined figure), "
          f"{'FIRES' if fires_raw else 'CLEAR'} (on the raw figure)")
    print("      " + ("indiscriminate elevation -- the effect is not about calling in "
                      "company." if fires or fires_raw else
                      "the food channel does not show an audience effect; the alarm "
                      "effect, if any, is specific."))
else:
    print("   NOT EVALUABLE: arm P not run.")

# 3. Gain
print("\n3. GAIN -- 'The muted DiD at pred_gain=0 is within 0.03 of the value at "
      "pred_gain>0.'")
if have("P", "P0"):
    d = summary["P"][2] - summary["P0"][2]
    per = per_seed("P", "muted", did) - per_seed("P0", "muted", did)
    m, se = mean_se(per)
    fires = abs(d) <= 0.03
    print(f"   muted DiD: P {summary['P'][2]:+.4f}, P0 {summary['P0'][2]:+.4f}, "
          f"difference {d:+.4f} (paired {m:+.4f} +/- {se:.4f} SE)")
    print(f"   -> {'FIRES' if fires else 'CLEAR'}: "
          + ("the effect is not W_pred's at all." if fires else
             "the gain manipulation moves the effect, so it is W_pred's."))
else:
    print("   NOT EVALUABLE: arms P and P0 both needed.")

# 4. Reproduction
print("\n4. REPRODUCTION -- 'The hebbian_readout arm does not reproduce E096's muted "
      f"+{E096_MUTED_DID:.4f} within 0.03.'")
if have("H"):
    dm = summary["H"][2]
    d = dm - E096_MUTED_DID
    fires = abs(d) > 0.03
    print(f"   arm H muted DiD = {dm:+.4f}, E096 = {E096_MUTED_DID:+.4f}, "
          f"difference {d:+.4f}")
    print(f"   -> {'FIRES' if fires else 'CLEAR'}: "
          + ("the harness does not reproduce its own positive control; nothing else in "
             "this run is interpretable." if fires else
             "the harness reproduces E096, so the rest of the run is interpretable."))
else:
    print("   NOT EVALUABLE: arm H not run.")

print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
