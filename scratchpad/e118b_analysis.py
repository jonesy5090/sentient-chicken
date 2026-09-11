"""E118 -- the contrasts the summary table cannot show.

Two things need computing separately.

(1) `sel-ctl` is not comparable across arms with different mutation rates. The control
    degrades under mutation load, so a higher sigma gives selection more damage to repair
    and inflates the difference without selection being any more effective. Comparing
    arms means comparing the *selected* trajectories.

(2) `baseline` and `drives_only` differ only in the selection criterion, and with
    `select=False` the criterion is never consulted -- so the two arms share their control
    exactly (verified below). That makes their selected lineages paired on founders,
    worlds, mutation draws and everything else, which is the cleanest contrast in the
    experiment.

Run:  PYTHONPATH=. python scratchpad/e118b_analysis.py
"""

import json
import sys

import numpy as np

CACHE = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/e118_cache.json"
c = json.load(open(CACHE))


def imp(arm, cond):
    return np.array([r["improve"] for r in c[arm][cond]])


def paired(a, b, label):
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / se if se > 1e-12 else float("nan")
    print(f"  {label:<44} {d.mean():+.4f}+-{se:.4f}  t={t:+6.2f}  (df={len(d)-1}, "
          f"bar 3.182)")
    return d.mean(), se, t


print("Sanity check: arms that differ only in the selection criterion must share their")
print("control exactly, because a control never consults fitness.")
if "baseline" in c and "drives_only" in c:
    same = np.allclose(imp("baseline", "control"), imp("drives_only", "control"))
    print(f"  baseline control == drives_only control: {same}")
    if not same:
        print("  !! they differ -- the two arms are not the matched pair assumed below")

print("\nSelected-lineage improvement per arm (hunger at gen 0 minus at gen 15;")
print("positive = the flock got better). Controls shown for reference.")
for arm in c:
    s, k = imp(arm, "selected"), imp(arm, "control")
    print(f"  {arm:>12}: selected {s.mean():+.4f} +- {s.std(ddof=1)/2:.4f}   "
          f"control {k.mean():+.4f}")

print("\nThe criterion contrast -- paired per lineage, identical in every other respect:")
if "baseline" in c and "drives_only" in c:
    paired(imp("drives_only", "selected"), imp("baseline", "selected"),
           "drives_only - baseline (selected lineages)")

print("\nOther arms against baseline, selected lineages, paired per lineage.")
print("(Not identical in other respects -- mutation load differs -- so read as")
print(" descriptive rather than as a clean causal contrast.)")
for arm in c:
    if arm in ("baseline",):
        continue
    paired(imp(arm, "selected"), imp("baseline", "selected"), f"{arm} - baseline")

print("\nWhen the search stops: generation by which 90% of total improvement is reached.")
for arm in c:
    g = []
    for r in c[arm]["selected"]:
        h = np.array(r["hunger"])
        tot = h[0] - h[-1]
        if tot <= 1e-9:            # never improved; a horizon is meaningless
            g.append(np.nan)
            continue
        reached = np.where((h[0] - h) >= 0.9 * tot)[0]
        g.append(int(reached[0]) if len(reached) else len(h))
    g = np.array(g, dtype=float)
    n_ok = int(np.sum(~np.isnan(g)))
    print(f"  {arm:>12}: gen {np.nanmean(g) if n_ok else float('nan'):5.1f} "
          f"({n_ok}/{len(g)} lineages improved at all)")

print("\nDiversity retained at generation 15, as a fraction of founding.")
for arm in c:
    d_out = np.mean([r["div_frac"] for r in c[arm]["selected"]]) * 100
    d_w = np.mean([r["diversity_w"][-1] / (r["diversity_w"][0] + 1e-12)
                   for r in c[arm]["selected"]]) * 100
    print(f"  {arm:>12}: W_out {d_out:5.1f}%   W {d_w:5.1f}%")

print("\nMotor profile at the final generation -- the saturation check from section 4.")
print(f"  {'arm':>12} {'forward':>8} {'peck':>7} {'crouch':>7} {'flee':>7}")
for arm in c:
    m = np.mean([r["motor"] for r in c[arm]["selected"]], axis=0)
    flag = "  <- saturated" if max(m[0], m[3], m[5], m[6]) > 0.9 else ""
    print(f"  {arm:>12} {m[0]:8.2f} {m[3]:7.2f} {m[5]:7.2f} {m[6]:7.2f}{flag}")
