"""E067 -- diagnostic, not pre-registered. Independent verification of a red-team
finding: is `m` (the reward-modulation factor `consolidate()` uses to gate the
recurrent weight update `W`) a genuine trace with memory, or a single-step snapshot
that only reflects a discrete reward event if that event happens to land exactly on
a consolidation boundary?

Written from scratch against the real `hen/plasticity.py` functions and formulas --
not a re-run of the red-team reviewer's own script (which was not saved to disk;
this is an independent re-derivation, per this project's own red-team discipline).

Checks both of this project's discrete, per-event reward terms: `sickness_penalty`
(T2, added in E066) and `strike_penalty` (used since ~E014, throughout H2/H4/T1's
history).
"""
import jax, jax.numpy as jnp
from hen import connectome, plasticity, regions
from hen.plasticity import PlasticConfig

H = 1
p = connectome.build(jax.random.key(0), regions.DEFAULT_REGIONS, n_hens=H)


def capture_rate(penalty: float, interval: int, baseline_tau_s: float,
                 dt: float = 0.01, threshold: float = 0.1) -> float:
    """Sweep every possible offset of a single discrete `-penalty` reward spike
    within one consolidation interval, and count what fraction produce |m| above
    `threshold` exactly at the boundary step consolidate() actually reads.
    """
    a_b = dt / baseline_tau_s
    visible = 0
    for onset in range(1, interval + 1):   # onset==interval lands ON the boundary
        b = 0.0
        m_at_boundary = None
        for t in range(1, interval + 1):
            reward_now = -penalty if t == onset else 0.0
            b = b + a_b * (reward_now - b)
            m = reward_now - b
            if t == interval:
                m_at_boundary = m
        if abs(m_at_boundary) > threshold:
            visible += 1
    return visible / interval


pc = PlasticConfig(enabled=True, sickness_penalty=1.0)
print(f"pc.interval={pc.interval}, pc.baseline_tau_s={pc.baseline_tau_s}, "
      f"pc.strike_penalty={pc.strike_penalty}, pc.sickness_penalty={pc.sickness_penalty}")

# Worked example: a sickness onset at step 23 (chosen arbitrarily, not aligned to any
# boundary), tracing m step by step through the first consolidation boundary at t=50.
ps = plasticity.initial_state(p, H, pc)
a_b = 0.01 / pc.baseline_tau_s
baseline = ps.baseline
m_trace = []
for t in range(100):
    reward_now = jnp.array([-1.0]) if t == 23 else jnp.array([0.0])
    baseline = baseline + a_b * (reward_now - baseline)
    m_trace.append(float((reward_now - baseline)[0]))
print(f"\nWorked example: -1.0 spike at t=23, pc.interval=50")
print(f"  m at t=23 (the spike itself): {m_trace[23]:.4f}")
print(f"  m at t=49 (one step before the boundary): {m_trace[49]:.6f}")
print(f"  m at t=50 (the boundary consolidate() actually reads): {m_trace[50]:.6f}")
print(f"  => {'VISIBLE' if abs(m_trace[50]) > 0.1 else 'INVISIBLE'} to consolidate()")

print(f"\nsickness_penalty capture rate: "
      f"{capture_rate(pc.sickness_penalty, pc.interval, pc.baseline_tau_s):.1%}")
print(f"strike_penalty capture rate:   "
      f"{capture_rate(pc.strike_penalty, pc.interval, pc.baseline_tau_s):.1%}")
print("\n(both discrete events go through the identical code path: reward() -> "
     "m = reward - ps.baseline -> consolidate(p, ps, m, pc), called only when "
     "w_next.t % pc.interval == 0, using whatever m happens to be AT that step.)")
