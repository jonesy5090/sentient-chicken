"""E098 Part B -- `W_pred` on the repaired audience assay (E098 section 5).

"E097's four arms unchanged, on the repaired assay, at the `hawk_period_s` Part A
selects. 8 seeds, 30 min rearing, each brain assayed twice."

TWO ASSAY VARIANTS ARE RUN, because the repair as landed does not reach the pathway.

  `landed`  `audience.assay(p, cfg, n, ps=ps_end)` -- exactly what section 5 repair 2
            specifies. `assay()` calls `simulate.rollout` at its default
            `pc=simulate.NO_PLASTICITY`, whose `pred_enabled` is False, so `_one_step`
            leaves `pred_from` at None and `brain.step` sources the prediction from
            *instantaneous* `rate(x)`; the repaired trace update, gated on
            `pred_enabled`, is skipped too. The reared `ps` therefore rides in the scan
            carry unread. Measured, not argued:
            `scratchpad/e098_reach_probe.py` finds this variant BIT-IDENTICAL to the
            no-`ps` call, which is E097's own call.

  `live`    the same four staged rollouts with the assay-time `PlasticConfig` made
            explicit: `NO_PLASTICITY._replace(pred_enabled=True, pred_centred=True,
            pred_bar_freeze_s=60.0)`. Learning is still off -- `W_pred` is written only
            in `consolidate`, which stays gated on `enabled` -- but the traces now
            advance and `W_pred` is read through the centred lagged trace it was
            trained on. `pred_gain` is left at 1.0 for every arm, exactly as E097's
            assay ran it, so the ONLY thing that changes is the source of the
            prediction. This is what section 2(a) describes as the defect.

Arm P is additionally reared at E097's own `hawk_period_s=900` so the reachability
falsifier compares like with like; that cell is checked per-seed against
`scratchpad/e097_cache.json` rather than against the rounded published table.

    PYTHONPATH=. .venv/bin/python scratchpad/e098b_wpred_fair.py --hawk-period 60
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
ap.add_argument("--hawk-period", type=float, required=True,
                help="the world Part A selects")
ap.add_argument("--reach-period", type=float, default=900.0,
                help="E097's own world, for the reachability falsifier")
ap.add_argument("--cache", default="scratchpad/e098b_cache.json")
ap.add_argument("--arms", default="S,H,P0,P")
a = ap.parse_args()

BASE = spec.DEFAULT_COOP._replace(n_hens=16, food_deplete_rate=0.0)
SEEDS = list(range(a.seeds))
STEPS = int(a.minutes * 60.0 / BASE.dt)

# Section 5's table, copied verbatim from scratchpad/e097_wpred_audience.py.
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

# Assay-time config for the `live` variant. Learning off, exploration off, traces on.
PC_LIVE = simulate.NO_PLASTICITY._replace(
    pred_enabled=True, pred_centred=True, pred_bar_freeze_s=60.0)

CELLS = [(arm, a.hawk_period) for arm in ORDER]
if abs(a.reach_period - a.hawk_period) > 1e-9 and "P" in ORDER:
    CELLS.append(("P", a.reach_period))


def assay_live(p, cfg, n_hens, ps, steps=300):
    """`audience.assay` with the assay-time PlasticConfig made explicit."""
    def call_rate(aud, hawk, food, channel):
        w = audience._staged(cfg, n_hens, audience=aud, hawk=hawk, food=food)
        x = brain.initial_state(p, n_hens)
        *_, tr = simulate.rollout(w, x, p, jax.random.key(11),
                                  cfg._replace(n_hens=n_hens), steps,
                                  ps=ps, pc=PC_LIVE)
        return float(jnp.mean(tr.motor[:, 0, channel]))

    return audience.AudienceResult(
        alarm_alone=call_rate(False, True, False, spec.M_CALL_AERIAL),
        alarm_audience=call_rate(True, True, False, spec.M_CALL_AERIAL),
        food_alone=call_rate(False, False, True, spec.M_CALL_FOOD),
        food_audience=call_rate(True, False, True, spec.M_CALL_FOOD),
    )


def run_cell(seed: int, arm: str, period: float):
    """Rear once; assay that one brain four ways."""
    pc = ARMS[arm]
    cfg = BASE._replace(hawk_period_s=period)
    muted = cfg._replace(channel_mode="none")
    k = jax.random.key(seed)
    p = connectome.build(jax.random.fold_in(k, 1), regions.DEFAULT_REGIONS,
                         n_hens=cfg.n_hens, auditory_scaffold=True)
    w = world.reset(k, cfg)
    x = brain.initial_state(p, cfg.n_hens)
    ps = plasticity.initial_state(p, cfg.n_hens, pc)
    _w, _x, p_end, ps_end, *_ = simulate.rollout_quiet(
        w, x, p, jax.random.fold_in(k, 2), cfg, STEPS, pc=pc, ps=ps)
    return {
        "intact": list(audience.assay(p_end, cfg, cfg.n_hens, ps=ps_end)),
        "muted": list(audience.assay(p_end, muted, cfg.n_hens, ps=ps_end)),
        "intact_live": list(assay_live(p_end, cfg, cfg.n_hens, ps_end)),
        "muted_live": list(assay_live(p_end, muted, cfg.n_hens, ps_end)),
        "w_pred_abs": float(jnp.mean(jnp.abs(p_end.W_pred))),
    }


# --------------------------------------------------------------------------- run
cache = json.load(open(a.cache)) if os.path.exists(a.cache) else {}
t0 = time.perf_counter()
for arm, period in CELLS:
    for s in SEEDS:
        ck = f"{arm}|{s}|{a.minutes}|{period}"
        if ck not in cache:
            cache[ck] = run_cell(s, arm, period)
            json.dump(cache, open(a.cache, "w"))
            print(f"  reared+assayed {arm} @{period:.0f}s seed {s} "
                  f"({time.perf_counter() - t0:.0f} s elapsed)", flush=True)


# --------------------------------------------------------------------------- report
def cell(arm, seed, which, period=None):
    period = a.hawk_period if period is None else period
    return audience.AudienceResult(*cache[f"{arm}|{seed}|{a.minutes}|{period}"][which])


def per_seed(arm, which, f, period=None):
    return np.array([f(cell(arm, s, which, period)) for s in SEEDS])


def did(r):
    return (r.alarm_audience - r.alarm_alone) - (r.food_audience - r.food_alone)


def mean_se(v):
    m = float(np.mean(v))
    if len(v) < 2:
        return m, 0.0
    return m, float(np.std(v, ddof=1)) / (len(v) ** 0.5)


crit = _t_critical(len(SEEDS) - 1)

print(f"\nE098 Part B -- `W_pred` on the repaired audience assay. {len(SEEDS)} matched "
      f"seeds (0-{SEEDS[-1]}), {a.minutes:.0f} min rearing, {BASE.n_hens} hens, "
      f"food_deplete_rate={BASE.food_deplete_rate}, "
      f"hawk_period_s={a.hawk_period:.0f}")
print("Every arm reared ONCE; the same reared brain assayed twice per variant -- audio")
print('intact, and audio muted at test (channel_mode="none"). The muted contrast is')
print("primary. DiD = (alarm_aud - alarm_alone) - (food_aud - food_alone).")


def table(which_i, which_m, label, note):
    print(f"\n=== {label} ==============================================")
    print(note)
    hdr = (f"{'arm':<4}{'audio':<8}{'alarm_alone':>12}{'alarm_aud':>11}"
           f"{'food_alone':>12}{'food_aud':>10}{'alarm_eff':>11}{'food_eff':>10}"
           f"{'DiD':>9}")
    print()
    print(hdr)
    print("-" * len(hdr))
    for arm in ORDER:
        for which, tag in ((which_i, "intact"), (which_m, "MUTED")):
            m = lambda f: float(np.mean(per_seed(arm, which, f)))
            print(f"{arm:<4}{tag:<8}"
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
    out = {}
    for arm in ORDER:
        di, sei = mean_se(per_seed(arm, which_i, did))
        dm, sem = mean_se(per_seed(arm, which_m, did))
        out[arm] = (di, sei, dm, sem)
        frac = (100.0 * dm / di) if abs(di) > 1e-9 else float("nan")
        print(f"{arm:<4}{di:>+12.4f}{sei:>9.4f}{dm:>+12.4f}{sem:>9.4f}{frac:>9.0f}%")

    print(f"\n(paired t critical at df={len(SEEDS) - 1}: {crit:.3f})")
    for arm in ORDER:
        m, se = mean_se(per_seed(arm, which_m, did))
        t = abs(m) / (se + 1e-12)
        print(f"  {arm:<4} muted DiD {m:>+8.4f} +/- {se:.4f} SE   t={t:>7.2f}   "
              f"{'SIGNIFICANT' if t > crit else 'not significant'}")
    return out


landed = table("intact", "muted", "VARIANT `landed`: assay(ps=reared), default pc",
               "This is repair 2 exactly as specified. It is a no-op -- see the\n"
               "bit-identity check below and scratchpad/e098_reach_probe.py.")
live = table("intact_live", "muted_live",
             "VARIANT `live`: traces advanced, W_pred read through the centred lag",
             "This is the read section 2(a) says the projection was trained for.\n"
             "Learning still off; only the source of the prediction changes.")

# --------------------------------------------------------------------- no-op check
print("\n--- is the landed repair a no-op? --------------------------------------")
print("Per-seed comparison of arm P against E097's own cache (same world, same seeds).")
try:
    e097 = json.load(open("scratchpad/e097_cache.json"))
except FileNotFoundError:
    e097 = {}
rp = a.reach_period if ("P", a.reach_period) in CELLS else a.hawk_period
for which, e_key in (("intact", "intact"), ("muted", "muted"),
                     ("intact_live", "intact"), ("muted_live", "muted")):
    n_exact, n_cmp, biggest = 0, 0, 0.0
    for s in SEEDS:
        ek = f"P|{s}|{a.minutes}"
        if ek not in e097:
            continue
        n_cmp += 1
        got = cache[f"P|{s}|{a.minutes}|{rp}"][which]
        want = e097[ek][e_key]
        n_exact += (got == want)
        biggest = max(biggest, max(abs(g - w_) for g, w_ in zip(got, want)))
    if n_cmp:
        print(f"  P@{rp:.0f}s {which:<12} vs E097 {e_key:<7}: "
              f"{n_exact}/{n_cmp} bit-identical, largest |delta| {biggest:.6f}")

# -------------------------------------------------------------------- falsifiers
print("\n=== PRE-REGISTERED FALSIFIERS (E098 section 4) ========================")
print("Verdicts are read off the `live` variant. The `landed` variant reproduces E097")
print("bit-for-bit, so evaluating a falsifier on it would only re-report E097.")

print("\n1. INERTNESS (hard gate) -- evaluated separately by scratchpad/e098_gate.py")
print("   and the test suite, before this run. Reported as already passed.")

print("\n2. REACHABILITY -- 'Arm P's numbers are unchanged after the fix.'")
E097_P_DID = {"intact": -0.0101, "muted": -0.0236}
if "P" in ORDER:
    print(f"   arm P re-reared at E097's own world (hawk_period_s={rp:.0f}), "
          f"matched seeds 0-{SEEDS[-1]}:")
    names = ["alarm_alone", "alarm_aud", "food_alone", "food_aud"]
    biggest = 0.0
    # Guarded: without E097's per-seed cache there is nothing to compare against, and a
    # NaN delta must not fall through `max()` and be reported as a match.
    comparable = all(f"P|{s}|{a.minutes}" in e097 for s in SEEDS)
    for which, e_key, tag in (("intact_live", "intact", "intact"),
                              ("muted_live", "muted", "MUTED")):
        got = [float(np.mean([cache[f"P|{s}|{a.minutes}|{rp}"][which][i]
                              for s in SEEDS])) for i in range(4)]
        want = ([float(np.mean([e097[f"P|{s}|{a.minutes}"][e_key][i]
                                for s in SEEDS])) for i in range(4)]
                if comparable else [float("nan")] * 4)
        for n, g, e in zip(names, got, want):
            print(f"     {tag:<7}{n:<12}{g:>9.4f}   E097 {e:>8.4f}   delta {g - e:+.4f}")
            biggest = max(biggest, abs(g - e))
    d_i = float(np.mean(per_seed("P", "intact_live", did, rp)))
    d_m = float(np.mean(per_seed("P", "muted_live", did, rp)))
    print(f"     DiD intact {d_i:+.4f} (E097 {E097_P_DID['intact']:+.4f}), "
          f"DiD muted {d_m:+.4f} (E097 {E097_P_DID['muted']:+.4f})")
    if not comparable:
        print("   NOT EVALUABLE: scratchpad/e097_cache.json has no matching arm-P "
              "entries for these seeds/minutes.")
    else:
        print(f"   largest |delta| on the four raw numbers = {biggest:.4f}")
        fires = biggest < 0.005
        print(f"   -> {'FIRES' if fires else 'CLEAR'}: "
              + ("arm P is unchanged, so the projection still is not reading the "
                 "trained signal and section 2(a) is not the defect it appears to be."
                 if fires else
                 "arm P moved, so the repaired read reaches the pathway -- section "
                 "2(a) was a real defect."))
else:
    print("   NOT EVALUABLE: arm P not run.")

print("\n3. PRIMARY -- 'Muted audience-specific DiD < +0.10 for `W_pred` on the repaired")
print("   assay at a rearing world where the pairing occurs >20 times.'")
if "P" in live:
    dm, sem = live["P"][2], live["P"][3]
    fires = dm < 0.10
    print(f"   arm P muted DiD = {dm:+.4f} +/- {sem:.4f} SE, threshold +0.10")
    print(f"   -> {'FIRES' if fires else 'CLEAR'}: "
          + ("H2f's falsifier has been fairly attempted and not met -- and section 4 "
             "licenses that reading this time, because 2(a) and 2(c) are both removed."
             if fires else
             "`W_pred` produces a muted audience-specific effect at or above the bar."))
else:
    print("   NOT EVALUABLE: arm P not run.")

print("\n4. SPECIFICITY -- 'Food-channel muted DiD >= +0.05.'")
if "P" in live:
    raw = float(np.mean(per_seed("P", "muted_live", lambda r: r.food_effect)))
    line = f"   arm P raw muted food effect (food_aud - food_alone) = {raw:+.4f}"
    if "S" in live:
        base = float(np.mean(per_seed("S", "muted_live", lambda r: r.food_effect)))
        vs_s = raw - base
        line += f"\n   same, differenced against the S baseline = {vs_s:+.4f}"
    else:
        vs_s = raw
    print(line)
    fires, fires_raw = vs_s >= 0.05, raw >= 0.05
    print(f"   -> {'FIRES' if fires else 'CLEAR'} (on the S-baselined figure), "
          f"{'FIRES' if fires_raw else 'CLEAR'} (on the raw figure)")
    print("      " + ("indiscriminate elevation -- the effect is not about calling in "
                      "company." if fires or fires_raw else
                      "the food channel shows no audience effect; the alarm effect, "
                      "if any, is specific."))
else:
    print("   NOT EVALUABLE: arm P not run.")

print("\n--- |W_pred| after rearing (did the rule learn anything at all?) -------")
for arm, period in CELLS:
    v = float(np.mean([cache[f"{arm}|{s}|{a.minutes}|{period}"]["w_pred_abs"]
                       for s in SEEDS]))
    print(f"  {arm:<3} @{period:>4.0f}s   mean |W_pred| = {v:.6f}")

print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
