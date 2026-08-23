"""E098 instrument check -- does carrying the reared `ps` into `assay()` reach anything?

`run/audience.py:assay` calls `simulate.rollout` at its default
`pc=simulate.NO_PLASTICITY`, which has `pred_enabled=False`. In `_one_step` that means
(a) `pred_from` stays None, so `brain.step` sources the prediction from *instantaneous*
`rate(x)` rather than from `ps.z_lag`, and (b) the repaired trace update is skipped,
because it is gated on `pc.pred_enabled`. If both hold, the reared `ps` is dead weight in
the scan carry and repair 2 cannot move a number.

Measured here rather than argued: rear arm P briefly, then assay three ways.

    PYTHONPATH=. .venv/bin/python scratchpad/e098_reach_probe.py
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from coop import spec, world
from hen import brain, connectome, plasticity, regions
from hen.plasticity import PlasticConfig
from run import audience, simulate

ap = argparse.ArgumentParser()
ap.add_argument("--minutes", type=float, default=3.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--hawk-period", type=float, default=60.0)
a = ap.parse_args()

CFG = spec.DEFAULT_COOP._replace(n_hens=16, food_deplete_rate=0.0,
                                 hawk_period_s=a.hawk_period)
STEPS = int(a.minutes * 60.0 / CFG.dt)
PC_P = PlasticConfig(enabled=True, pred_enabled=True, pred_gain=1.0,
                     pred_centred=True, pred_bar_freeze_s=60.0)
# The same rule with learning off: traces stay live and centred, so `W_pred` is read
# through the representation it was trained on. This is what repair 1 was written for.
PC_ASSAY = PC_P._replace(enabled=False, explore_sigma=0.0)


def assay_with_pc(p, cfg, n_hens, ps, pc, steps=300):
    """`audience.assay`, with the assay-time PlasticConfig made explicit."""
    def call_rate(aud, hawk, food, channel):
        w = audience._staged(cfg, n_hens, audience=aud, hawk=hawk, food=food)
        x = brain.initial_state(p, n_hens)
        *_, tr = simulate.rollout(w, x, p, jax.random.key(11),
                                  cfg._replace(n_hens=n_hens), steps, ps=ps, pc=pc)
        return float(jnp.mean(tr.motor[:, 0, channel]))

    return audience.AudienceResult(
        alarm_alone=call_rate(False, True, False, spec.M_CALL_AERIAL),
        alarm_audience=call_rate(True, True, False, spec.M_CALL_AERIAL),
        food_alone=call_rate(False, False, True, spec.M_CALL_FOOD),
        food_audience=call_rate(True, False, True, spec.M_CALL_FOOD),
    )


t0 = time.perf_counter()
k = jax.random.key(a.seed)
p = connectome.build(jax.random.fold_in(k, 1), regions.DEFAULT_REGIONS,
                     n_hens=CFG.n_hens, auditory_scaffold=True)
w = world.reset(k, CFG)
x = brain.initial_state(p, CFG.n_hens)
ps0 = plasticity.initial_state(p, CFG.n_hens, PC_P)
_w, _x, p_end, ps_end, *_ = simulate.rollout_quiet(
    w, x, p, jax.random.fold_in(k, 2), CFG, STEPS, pc=PC_P, ps=ps0)

print(f"reared arm P, {a.minutes:.0f} min, seed {a.seed}, "
      f"hawk_period_s={a.hawk_period:.0f}  ({time.perf_counter() - t0:.0f} s)")
print(f"|W_pred| mean {float(jnp.mean(jnp.abs(p_end.W_pred))):.6f}  "
      f"max {float(jnp.max(jnp.abs(p_end.W_pred))):.6f}   "
      f"(at init {float(jnp.mean(jnp.abs(p.W_pred))):.6f})")
print(f"|z_lag| mean {float(jnp.mean(jnp.abs(ps_end.z_lag))):.6f}   "
      f"|z_lag - z_lag_bar| mean "
      f"{float(jnp.mean(jnp.abs(ps_end.z_lag - ps_end.z_lag_bar))):.6f}")
print(f"ps.age_s {float(jnp.mean(ps_end.age_s)):.1f} s "
      f"(pred_bar_freeze_s = {PC_P.pred_bar_freeze_s})\n")

variants = {
    "A  assay(), no ps            (E097's call)":
        lambda: audience.assay(p_end, CFG, CFG.n_hens),
    "B  assay(), ps=reared        (E098 repair 2 as landed)":
        lambda: audience.assay(p_end, CFG, CFG.n_hens, ps=ps_end),
    "C  ps=reared, pc=pred_enabled (traces live + centred)":
        lambda: assay_with_pc(p_end, CFG, CFG.n_hens, ps_end, PC_ASSAY),
    "D  ps=None,   pc=pred_enabled (traces live, from zero)":
        lambda: assay_with_pc(p_end, CFG, CFG.n_hens, None, PC_ASSAY),
}

hdr = f"{'variant':<56}{'alarm_alone':>12}{'alarm_aud':>11}{'food_alone':>12}{'food_aud':>10}"
print(hdr)
print("-" * len(hdr))
got = {}
for name, fn in variants.items():
    r = fn()
    got[name[0]] = np.array(list(r))
    print(f"{name:<56}" + "".join(f"{v:>12.6f}" if i == 0 else
                                  f"{v:>11.6f}" if i == 1 else
                                  f"{v:>12.6f}" if i == 2 else f"{v:>10.6f}"
                                  for i, v in enumerate(r)))

print("\nbit-identity:")
for x_, y_ in (("A", "B"), ("A", "C"), ("B", "C"), ("C", "D")):
    same = bool(np.array_equal(got[x_], got[y_]))
    print(f"  {x_} == {y_}: {same}   max |delta| = "
          f"{float(np.max(np.abs(got[x_] - got[y_]))):.6f}")

print("\nIf A == B, repair 2 does not reach the pathway on its own and Part B as")
print("specified would reproduce E097 exactly -- a spurious reachability FIRES.")
print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
