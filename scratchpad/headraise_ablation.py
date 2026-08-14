"""Is the channel a message or an interrupt? (E026)

E018 section 5 pre-registered this exact ablation, conditional on a strike reduction
appearing: "separates 'she hides because she was told' from 'she looks up and sees it
herself'". E024 produced a large strike reduction. It was never run.

The arithmetic says the answer in advance and nobody checked it: the scaffold drives
crouch to sigmoid(1.5 - 2.5) = 0.269, and hiding from a hawk requires
motor[M_CROUCH] > 0.5 (coop/world.py). So a heard alarm CANNOT hide a hen. The only
live route is call -> stop pecking -> head up -> see the hawk yourself -> visual
reflex at weight 8.0 -> crouch 0.996.

Splitting the scaffold in half tests that directly.
"""
from functools import partial

import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, regions
from run import simulate

HENS, SECONDS = 16, 300.0
SEEDS = range(4)


def variant(p, keep):
    """keep: 'both' | 'respond' (crouch/flee only) | 'interrupt' (head-raise only)."""
    r = np.asarray(p.reflex).copy()
    aer = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)
    gnd = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_GROUND)
    if keep in ("interrupt", "none"):          # delete the response half
        r[spec.M_CROUCH, aer] = 0.0
        r[spec.M_FLEE, gnd] = 0.0
    if keep in ("respond", "none"):            # delete the head-raise half
        for c in (aer, gnd):
            r[spec.M_PECK, c] = 0.0
            r[spec.M_SCRATCH, c] = 0.0
    return p._replace(reflex=jnp.asarray(r))


print(f"{'condition':<34}{'caught/event':>14}{'at risk':>9}{'caught rate*':>14}"
      f"{'exposed':>9}{'fed %':>8}")
print("  caught/event = P(struck | inside the radius when the dive began) -- the")
print("  denominator is fixed before any response, so the treatment cannot move it.")
print("  caught rate* = struck/exposed, CONFOUNDED: crouching immobilises, so a")
print("  partially-crouching hen lingers in the radius and inflates her own denominator.\n")
for label, mode, keep, scaf in (
        ("no channel", "none", "both", True),
        ("intact, full scaffold", "intact", "both", True),
        ("intact, response only (no head-raise)", "intact", "respond", True),
        ("intact, head-raise only (no response)", "intact", "interrupt", True),
        ("yoked, full scaffold", "yoked", "both", True)):
    cr, fd, ex, cpe, ar = [], [], [], [], []
    for seed in SEEDS:
        cfg = spec.DEFAULT_COOP._replace(n_hens=HENS, hawk_period_s=60.0,
                                         channel_mode=mode)
        w = world.reset(jax.random.key(seed), cfg)
        p = connectome.build(jax.random.fold_in(jax.random.key(seed), 1),
                             regions.DEFAULT_REGIONS.with_pallium(1.5),
                             n_hens=HENS, auditory_scaffold=scaf)
        p = variant(p, keep)
        x = brain.initial_state(p, HENS)
        w_end, *_ = simulate.simulate(w, x, p, jax.random.fold_in(jax.random.key(seed), 2),
                                      cfg, SECONDS, 60.0, simulate.NO_PLASTICITY)
        s_, e_ = float(jnp.sum(w_end.n_struck)), float(jnp.sum(w_end.n_exposed))
        a_, c_ = float(jnp.sum(w_end.n_at_risk)), float(jnp.sum(w_end.n_caught))
        cr.append(s_ / max(e_, 1.0))
        cpe.append(c_ / max(a_, 1.0)); ar.append(a_)
        fd.append(float(jnp.sum(w_end.n_fed) / (HENS * SECONDS / cfg.dt)) * 100)
        ex.append(e_ / HENS)
    print(f"{label:<34}{np.mean(cpe):>14.3f}{np.mean(ar):>9.1f}{np.mean(cr):>14.3f}"
          f"{np.mean(ex):>9.0f}{np.mean(fd):>8.2f}")

print("\nIf 'response only' returns to the no-channel baseline, the channel is an")
print("interrupt and its content is doing no work. Read caught/event, not caught rate*:")
print("on the confounded metric the two disagreed about the sign of the effect.")
