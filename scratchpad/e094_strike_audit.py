"""E094 Part A: does fixing the m-sampling defect move the weights at all?

E067 confirmed the mechanism -- m was sampled at the consolidation boundary rather than
traced, so a discrete reward event reached consolidate() on ~2% of occurrences -- and
explicitly did not adopt any reinterpretation of a prior result. This screens whether it
could have.

Run at the H4 configuration (hawk_period_s=20), where CLAUDE.md records strike_penalty as
87.3% of reward variance. If |W| does not move HERE it moves nowhere, and the audit closes
with "real, confirmed, inconsequential".

Reports the strike-event count and how many landed on a consolidation boundary, to confirm
E067's ~2% in this harness rather than inheriting the figure.
"""
import sys, time
sys.path.insert(0, 'scratchpad')
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from coop import spec, world
from hen import brain, connectome, plasticity, regions
from run import simulate

HENS, SEEDS, MINUTES = 16, 8, 30.0
# The configuration the defect is most exposed at -- E027 measured the strike term at
# 87.3% of reward variance here, against 0.0% at the 900 s default.
CFG = spec.DEFAULT_COOP._replace(n_hens=HENS, hawk_period_s=20.0)
STEPS = int(MINUTES * 60 / CFG.dt)


@partial(jax.jit, static_argnames=("cfg", "pc", "n"))
def run(w, x, p, ps, key, cfg, pc, n):
    def step(carry, i):
        prev_events = carry[0].n_strike_events
        carry, _o = simulate._one_step(carry, None, cfg=cfg, pc=pc)
        wl, q = carry[0], carry[3]
        ev = jnp.sum(carry[0].n_strike_events - prev_events)
        # did this step land on a consolidation boundary?
        on_boundary = ((i + 1) % pc.interval) == 0
        return carry, (ev, ev * on_boundary)
    _c, (ev, ev_b) = jax.lax.scan(step, (w, x, p, ps, key), jnp.arange(n))
    return _c[2], jnp.sum(ev), jnp.sum(ev_b)


def wnorm(p):
    return float(jnp.sum(jnp.abs(p.W) * p.mask[None, :, :])
                 / (p.W.shape[0] * jnp.sum(p.mask) + 1e-9))


print(f"E094 Part A -- does the m-sampling fix move |W|? {SEEDS} seeds, {MINUTES:.0f} min")
print(f"H4 configuration: hawk every {CFG.hawk_period_s:.0f}s "
      f"(strike term = 87.3% of reward variance here, 0.0% at the 900s default)\n")
t0 = time.perf_counter()
print(f"{'seed':>5}{'strikes':>9}{'on boundary':>13}{'|W| legacy':>12}{'|W| fixed':>11}"
      f"{'diff':>10}")
rows = []
for s in range(SEEDS):
    k = jax.random.key(s)
    out = {}
    for legacy in (True, False):
        pc = plasticity.PlasticConfig(enabled=True, legacy_m_sampling=legacy)
        p0 = connectome.build(jax.random.fold_in(k, 1), regions.DEFAULT_REGIONS,
                              n_hens=HENS)
        w = world.reset(k, CFG)
        x = brain.initial_state(p0, HENS)
        ps = plasticity.initial_state(p0, HENS, pc)
        w0 = wnorm(p0)
        p_end, ev, ev_b = run(w, x, p0, ps, jax.random.fold_in(k, 9), CFG, pc, STEPS)
        out[legacy] = (wnorm(p_end) - w0, float(ev), float(ev_b))
    dl, ev, ev_b = out[True]
    df, _, _ = out[False]
    rel = (df - dl) / max(abs(dl), 1e-12)
    rows.append((ev, ev_b, dl, df, rel))
    print(f"{s:>5}{ev:>9.0f}{ev_b:>13.0f}{dl:>12.6f}{df:>11.6f}{100*rel:>9.1f}%")

ev, ev_b, dl, df, rel = (np.array([r[i] for r in rows]) for i in range(5))
frac = ev_b.sum() / max(ev.sum(), 1)
print(f"\nstrike events: {ev.sum():.0f} total, {ev_b.sum():.0f} on a consolidation "
      f"boundary = {100*frac:.1f}%")
print(f"  (E067 measured ~2% by exhaustive timing sweep; this is the same quantity "
      f"measured in a running flock)")
print(f"\n|W| drift: legacy {dl.mean():+.6f}, fixed {df.mean():+.6f}, "
      f"relative difference {100*np.mean(rel):+.1f}%")
d = df - dl
t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
print(f"paired: {d.mean():+.6f} +/- {d.std(ddof=1)/np.sqrt(len(d)):.6f}, t={t:+.2f} "
      f"vs t(7)=2.365 -> {'SIGNIFICANT' if abs(t) > 2.365 else 'not significant'}")
print(f"\nwall clock: {time.perf_counter()-t0:.0f} s")
print("--- pre-registered falsifier (E094 section 4) ---")
print(f"screening  |W| relative difference {100*abs(np.mean(rel)):.1f}% (closes the audit "
      f"if <5%) -> {'CLOSES -- real, confirmed, inconsequential' if abs(np.mean(rel)) < 0.05 else 'Part B required'}")


# --- Direction check (added after Part A cleared on magnitude) --------------
# |W| is a scalar summary: two conditions can show identical drift while moving different
# synapses in different directions. The pre-registered screen used magnitude, so this is
# the stricter test it should have been. Per-synapse cosine similarity between the two
# conditions' weight changes -- 1.0 means the fix changed nothing at all.
if __name__ == "__main__":
    print("\n--- direction check: is the weight CHANGE itself the same? ---")
    print(f"{'seed':>5}{'cosine(dW_legacy, dW_fixed)':>30}{'max |elementwise diff|':>25}")
    cos = []
    for s in range(4):
        k = jax.random.key(s)
        dW = {}
        for legacy in (True, False):
            pc = plasticity.PlasticConfig(enabled=True, legacy_m_sampling=legacy)
            p0 = connectome.build(jax.random.fold_in(k, 1), regions.DEFAULT_REGIONS,
                                  n_hens=HENS)
            w = world.reset(k, CFG)
            x = brain.initial_state(p0, HENS)
            ps = plasticity.initial_state(p0, HENS, pc)
            p_end, _e, _b = run(w, x, p0, ps, jax.random.fold_in(k, 9), CFG, pc, STEPS)
            dW[legacy] = np.asarray(p_end.W - p0.W).ravel()
        a, b = dW[True], dW[False]
        c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        cos.append(c)
        print(f"{s:>5}{c:>30.6f}{np.max(np.abs(a - b)):>25.3e}")
    print(f"\nmean cosine {np.mean(cos):.6f} -- 1.0 would mean the fix changes nothing")
