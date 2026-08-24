"""E106b: the behavioural half of E106's degeneracy falsifier.

Section 4 fires on ANY ethogram assay changing state, or on the flock's welfare getting
worse on hunger and predation together. The representation result means nothing if the
interneuron leaves a hen who cannot feed or cannot hide.
"""
import time
import jax, numpy as np
from coop import spec, world
from hen import brain, connectome, plasticity, regions
from run import probes, simulate

BASE = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=60.0)
REAR, SEEDS = int(30 * 60 / BASE.dt), 8
PC = plasticity.PlasticConfig(enabled=True, hebbian_readout=True,
                              readout_scaling_strength=0.3)
reg = regions.DEFAULT_REGIONS

t0 = time.perf_counter()
print("E106b -- the behavioural falsifier\n")

# --- the ethogram, at the arm the representation result comes from ----------
for label, rec in (("baseline", 0.0), ("interneuron 1.0", 1.0)):
    results = probes.run_all(spec.DEFAULT_COOP._replace(recurrent_lateral=rec))
    names = [fn.__name__ for fn in probes.ALL]
    passed = {n: r.passed for n, r in zip(names, results)}
    n_pass = sum(passed.values())
    xf = [n for n in names if not passed[n] and n in probes.EXPECTED_FAILURES]
    bad = [n for n in names if not passed[n] and n not in probes.EXPECTED_FAILURES]
    print(f"  {label:>16}: {n_pass}/{len(names)} pass, "
          f"{len(xf)} registered xfail, {len(bad)} unexpected failures")
    if bad:
        print(f"{'':>20}unexpected: {', '.join(bad)}")
    if label == "baseline":
        baseline_state = dict(passed)
    else:
        moved = [n for n in names if passed[n] != baseline_state[n]]
        print(f"{'':>20}assays that CHANGED STATE: "
              f"{', '.join(moved) if moved else 'none'}")

# --- welfare, matched seeds -------------------------------------------------
print(f"\n  welfare over 30 min of rearing, {SEEDS} matched seeds")
print(f"{'arm':>18}{'hunger':>10}{'caught/dive':>13}")
hung, caught = {}, {}
for label, rec in (("baseline", 0.0), ("interneuron 1.0", 1.0)):
    cfg = BASE._replace(recurrent_lateral=rec)
    h, c = [], []
    for s in range(SEEDS):
        k = jax.random.key(s)
        p0 = connectome.build(jax.random.fold_in(k, 1), reg, n_hens=16)
        w = world.reset(k, cfg)
        x = brain.initial_state(p0, 16)
        st = plasticity.initial_state(p0, 16, PC)
        w2, _x, _p, _ps, _k = simulate.rollout_quiet(
            w, x, p0, jax.random.fold_in(k, 2), cfg, REAR, st, PC)
        h.append(float(np.mean(np.asarray(w2.hunger))))
        # `n_caught_any` over `n_dives`, the metric E101/E102 used. NOT `n_struck`,
        # which counts contact *steps* and which E027 recorded as uninterpretable
        # because it moves with how long a hen lingers rather than with how often she
        # is caught. The first version of this script used it and produced a
        # "caught/dive" of 54.3, which is not a rate at all.
        c.append(float(np.sum(np.asarray(w2.n_caught_any))
                       / max(float(np.sum(np.asarray(w2.n_dives))), 1.0)))
    hung[label], caught[label] = np.array(h), np.array(c)
    print(f"{label:>18}{np.mean(h):>10.4f}{np.mean(c):>13.4f}")

for name, d in (("hunger", hung), ("caught/dive", caught)):
    diff = d["interneuron 1.0"] - d["baseline"]
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    t = diff.mean() / (se + 1e-12)
    print(f"    {name:>12} paired difference {diff.mean():+.4f} +/- {se:.4f}  "
          f"t={t:+.2f}  (df={len(diff)-1}, crit 2.365)")

print("\nfalsifier: fires if any assay changes state, or if BOTH hunger and")
print("caught/dive are significantly worse under the interneuron.")
print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")
