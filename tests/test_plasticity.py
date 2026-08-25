"""Phase 1 tests: learning must change the right things and nothing else.

Most of these guard invariants rather than behaviour. Whether the rule actually
produces adaptive learning is a *hypothesis* (H2), tested by `run/experiment.py` as a
matched-seed contrast -- not something to assert in a unit test. Connectome churn is
easy to produce and proves nothing on its own.
"""

import jax
import numpy as np
import jax.numpy as jnp
import pytest

from coop import spec, world
from hen import brain, connectome, plasticity, regions
from hen.plasticity import PlasticConfig
from run import simulate

CFG = spec.DEFAULT_COOP._replace(n_hens=4)
LEARN = PlasticConfig(enabled=True)
LEARN_NO_GROWTH = PlasticConfig(enabled=True, growth_enabled=False)


@pytest.fixture(scope="module")
def flock():
    key = jax.random.key(0)
    w = world.reset(key, CFG)
    p = connectome.build(jax.random.key(1), regions.DEFAULT_REGIONS,
                         n_hens=CFG.n_hens)
    x = brain.initial_state(p, CFG.n_hens)
    return w, x, p


def _run(flock, pc, steps=3_000, seed=9):
    w, x, p = flock
    ps = plasticity.initial_state(p, CFG.n_hens, pc)
    return simulate.rollout_quiet(w, x, p, jax.random.key(seed), CFG, steps, ps, pc)


# --- Learning happens, and only where it should ---------------------------

def test_fixed_hen_does_not_change(flock):
    _, _, p0 = flock
    _w, _x, p1, *_ = _run(flock, PlasticConfig(enabled=False))
    assert jnp.array_equal(p0.W, p1.W)
    assert jnp.array_equal(p0.W_out, p1.W_out)


def test_learning_changes_weights(flock):
    _, _, p0 = flock
    _w, _x, p1, *_ = _run(flock, LEARN_NO_GROWTH)
    assert not jnp.allclose(p0.W, p1.W)


def test_reflex_arc_is_never_plastic(flock):
    """A chick's innate responses stay fixed for life, as they do in a real bird."""
    _, _, p0 = flock
    _w, _x, p1, *_ = _run(flock, LEARN)
    assert jnp.array_equal(p0.reflex, p1.reflex)
    assert jnp.array_equal(p0.b_motor, p1.b_motor)
    assert jnp.array_equal(p0.W_in, p1.W_in)
    assert jnp.array_equal(p0.dale, p1.dale)


# --- Invariants that must survive learning --------------------------------

def test_dale_law_survives_learning(flock):
    """Learning must not be able to turn an inhibitory neuron excitatory."""
    _w, _x, p1, *_ = _run(flock, LEARN)
    for j in (0, 7, 100, p1.W.shape[-1] - 1):
        col = p1.W[:, :, j]
        nz = col[col != 0.0]
        if nz.size:
            assert bool(jnp.all(jnp.sign(nz) == p1.dale[j])), f"neuron {j}"


def test_dale_law_holds_on_the_readout(flock):
    """...including on `W_out`, which is the pathway that reaches a muscle.

    This test checked `p.W` only, and `W_out` was drawn from a zero-mean normal and
    clipped symmetrically -- so every motor-stub neuron sent excitation to some muscles
    and inhibition to others. Measured before the fix: **0 of 48 columns compliant, all
    48 mixed**, with 10 of the source neurons inhibitory.

    E022 found this, filed it under "verified -- adopt", and it fell off the action list
    for four experiments until E027 re-measured it. Checked at hatch *and* after
    learning, because it was broken in both places and fixing one would have looked
    like fixing both.
    """
    _, _, p0 = flock
    _w, _x, p1, *_ = _run(flock, LEARN)
    n_motor = p0.W_out.shape[-1]
    src = p0.dale[-n_motor:]
    assert float(jnp.sum(src < 0)) > 0, "no inhibitory motor-stub neurons; test is vacuous"

    for name, W in (("at hatch", p0.W_out), ("after learning", p1.W_out)):
        for j in range(n_motor):
            nz = W[:, :, j][W[:, :, j] != 0.0]
            if nz.size:
                assert bool(jnp.all(jnp.sign(nz) == src[j])), (
                    f"{name}: motor-stub neuron {j} has outgoing weights of both signs; "
                    "an inhibitory cell is excitatory on the pathway to the muscles")


def test_hebbian_readout_ignores_reward_sign():
    """H2f/E055: `hebbian_readout` must genuinely remove reward-gating from the
    readout update, not merely rename it. With the flag on, identical traces under
    opposite-signed `m` (reward vs. punishment) must produce an identical `W_out`
    update; with it off (default), they must differ -- otherwise the flag does
    nothing and the H2f test built on it would not be testing what it claims to.
    """
    p = connectome.build(jax.random.key(1), regions.DEFAULT_REGIONS, n_hens=CFG.n_hens)

    def w_out_after(hebbian, m_sign):
        pc = PlasticConfig(enabled=True, hebbian_readout=hebbian)
        ps = plasticity.initial_state(p, CFG.n_hens, pc)
        k1, k2 = jax.random.split(jax.random.key(0))
        # bar stays at its initial 0, so dz_slow/dz_motor equal these traces directly.
        ps = ps._replace(z_slow=jax.random.normal(k1, ps.z_slow.shape) * 0.1,
                         z_motor=jax.random.normal(k2, ps.z_motor.shape) * 0.1)
        m = jnp.full((CFG.n_hens,), m_sign * 5.0)
        return plasticity.consolidate(p, ps, m, pc).W_out

    assert jnp.array_equal(w_out_after(True, +1.0), w_out_after(True, -1.0)), (
        "hebbian_readout=True still depends on the sign of m -- reward-gating was "
        "not actually removed")
    assert not jnp.array_equal(w_out_after(False, +1.0), w_out_after(False, -1.0)), (
        "sanity check failed: the default (instrumental) rule should still depend "
        "on m's sign")


def test_readout_scaling_bounds_hebbian_growth():
    """E055 follow-up: `readout_scaling_strength` must actually keep `W_out`'s row
    sums bounded near the innate baseline under sustained one-directional growth --
    otherwise it does not fix the runaway E055 measured (cortical drive 2-2.7x
    reflex magnitude under `hebbian_readout` with no scaling).

    Repeats `consolidate()` with a fixed, persistently-correlated trace pattern
    (the worst case for an unbounded rule) and compares final row sums with scaling
    on vs. off.
    """
    p = connectome.build(jax.random.key(1), regions.DEFAULT_REGIONS, n_hens=CFG.n_hens)
    innate_row_sum_out = jnp.sum(jnp.abs(p.W_out), axis=2)

    def final_row_sum(scaling_strength):
        pc = PlasticConfig(enabled=True, hebbian_readout=True,
                           readout_scaling_strength=scaling_strength)
        ps = plasticity.initial_state(p, CFG.n_hens, pc)
        k1, k2 = jax.random.split(jax.random.key(2))
        z_slow = jnp.abs(jax.random.normal(k1, ps.z_slow.shape)) * 0.2
        z_motor = jnp.abs(jax.random.normal(k2, ps.z_motor.shape)) * 0.2
        ps = ps._replace(z_slow=z_slow, z_motor=z_motor)   # bar=0 -> persistent, same-sign
        p_cur = p
        m = jnp.ones((CFG.n_hens,))
        for _ in range(200):
            p_cur = plasticity.consolidate(p_cur, ps, m, pc)
        return jnp.sum(jnp.abs(p_cur.W_out), axis=2)

    unscaled = final_row_sum(0.0)
    scaled = final_row_sum(0.3)

    assert float(jnp.mean(unscaled)) > float(jnp.mean(innate_row_sum_out)) * 1.5, (
        "test setup is not actually stressing growth -- unscaled row sums did not "
        "grow well past the innate baseline")
    assert float(jnp.mean(scaled)) < float(jnp.mean(unscaled)), (
        "readout_scaling_strength did not reduce growth relative to the unscaled case"
    )


def test_weights_stay_bounded(flock):
    _w, _x, p1, *_ = _run(flock, LEARN)
    assert bool(jnp.all(jnp.isfinite(p1.W)))
    assert float(jnp.max(jnp.abs(p1.W))) <= LEARN.w_max + 1e-6


def test_no_autapses_after_growth(flock):
    """Growth must never wire a neuron to itself."""
    _w, _x, p1, *_ = _run(flock, LEARN, steps=25_000)
    diag = jnp.diagonal(p1.W, axis1=1, axis2=2)
    assert float(jnp.max(jnp.abs(diag))) == 0.0


def test_growth_respects_anatomy(flock):
    """A synapse can only appear where an axon could reach.

    Without this, a hypothalamic neuron could sprout onto the sensory stub purely
    because the two happened to correlate.
    """
    _w, _x, p1, *_ = _run(flock, LEARN, steps=25_000)
    live = p1.W != 0.0
    illegal = live & ~p1.growable[None, :, :]
    assert int(jnp.sum(illegal)) == 0


def test_growth_respects_budget(flock):
    _, _, p0 = flock
    ps = plasticity.initial_state(p0, CFG.n_hens, LEARN)
    _w, _x, p1, *_ = _run(flock, LEARN, steps=25_000)
    live = jnp.sum(p1.W != 0.0, axis=(1, 2))
    assert float(jnp.max(live)) <= float(ps.budget) * 1.1


def test_structural_change_actually_occurs(flock):
    """Growth and pruning both fire -- otherwise the budget test passes trivially."""
    _w, _x, p1, *_ = _run(flock, LEARN, steps=25_000)
    d = plasticity.divergence_from_genome(p1)
    assert int(jnp.sum(d["grown"])) > 0
    assert int(jnp.sum(d["pruned"])) > 0


# --- Stability ------------------------------------------------------------

def test_plastic_run_stays_finite(flock):
    """Ten minutes of chicken time with learning on, no divergence."""
    _w, x1, p1, *_ = _run(flock, LEARN, steps=60_000)
    assert bool(jnp.all(jnp.isfinite(x1)))
    assert float(jnp.max(jnp.abs(x1))) < 50.0
    assert bool(jnp.all(jnp.isfinite(p1.W)))


def test_plastic_determinism(flock):
    """Learning must not introduce nondeterminism."""
    a = _run(flock, LEARN, steps=2_000, seed=3)
    b = _run(flock, LEARN, steps=2_000, seed=3)
    assert jnp.array_equal(a[2].W, b[2].W)
    assert jnp.array_equal(a[0].pos, b[0].pos)


# --- The neuromodulator ---------------------------------------------------

def test_reward_is_drive_reduction(flock):
    """Eating when hungry is rewarding; going hungry is not."""
    w, _, _ = flock
    fed = w._replace(hunger=w.hunger - 0.01)
    starved = w._replace(hunger=w.hunger + 0.01)
    r_fed = plasticity.reward(w, fed, CFG, LEARN)
    r_starved = plasticity.reward(w, starved, CFG, LEARN)
    assert float(jnp.mean(r_fed)) > 0.0
    assert float(jnp.mean(r_starved)) < 0.0


def test_being_caught_is_aversive(flock):
    """One strike *event* must cost her something.

    Reads `n_strike_events`, not `n_struck`: E028 moved the reward onto the event
    counter, and this test failed on the old field -- which is the rule working. A
    contact-step counter is still incremented for diagnostics, and bumping it now
    changes the reward by exactly zero, so asserting on it would pass forever without
    testing anything.
    """
    w, _, _ = flock
    struck = w._replace(n_strike_events=w.n_strike_events + 1.0)
    assert float(jnp.mean(plasticity.reward(w, struck, CFG, LEARN))) < 0.0

    contact_only = w._replace(n_struck=w.n_struck + 1.0)
    assert float(jnp.mean(plasticity.reward(w, contact_only, CFG, LEARN))) == 0.0, (
        "the per-step contact counter still reaches the reward; it is diagnostics only")


def test_discrete_reward_event_reaches_consolidation_regardless_of_timing():
    """E067 guard: a single-step reward event (a strike, a sickness onset) must
    reach `consolidate()` no matter which step within the `interval`-step window it
    lands on.

    Before this fix, `m` was `reward - baseline` read fresh every step, but only the
    value at the exact consolidation-boundary step (`t % interval == 0`) was ever
    passed to `consolidate()` -- an event landing anywhere else was invisible. An
    exhaustive sweep over every possible timing offset found this happened on only
    2% of occurrences (`scratchpad/e067_reward_eligibility_check.py`, following an
    adversarial review). `m_acc` (accumulated every step in `update_traces`, reset on
    consolidation) fixes this: because it is a *sum* over the window, a single
    spike's contribution to the mean is the same regardless of exactly when within
    the window it occurred.
    """
    p = connectome.build(jax.random.key(1), regions.DEFAULT_REGIONS, n_hens=1)
    pc = PlasticConfig(enabled=True)
    r = jnp.zeros((1, p.b.shape[0]))
    motor = jnp.zeros((1, p.W_out.shape[1]))

    def mean_m_after_spike(onset_step):
        ps = plasticity.initial_state(p, 1, pc)
        for t in range(1, pc.interval + 1):
            reward_now = jnp.array([-1.0]) if t == onset_step else jnp.array([0.0])
            ps = plasticity.update_traces(ps, r, motor, reward_now, CFG, pc)
        return float(ps.m_acc[0] / pc.interval)

    m_early = mean_m_after_spike(1)               # far from the boundary
    m_mid = mean_m_after_spike(pc.interval // 2)
    m_late = mean_m_after_spike(pc.interval)       # exactly on the boundary

    for label, m in (("early", m_early), ("mid", m_mid), ("late", m_late)):
        assert m < -0.015, (
            f"a discrete reward event at the {label} offset must still reach "
            f"consolidation (expected ~-1/{pc.interval}), got m={m}")
    assert max(m_early, m_mid, m_late) - min(m_early, m_mid, m_late) < 0.005, (
        f"timing within the window should barely matter: early={m_early}, "
        f"mid={m_mid}, late={m_late}")


def test_sickness_onset_is_aversive_only_when_opted_in():
    """T2 (E066): `sickness_penalty` defaults to 0.0, matching
    `readout_scaling_strength`'s own precedent -- adding this term must change
    nothing for any experiment that doesn't explicitly opt in, since `sick_on`
    (E060) exists unconditionally on every `World`, in every hypothesis's runs, not
    just T2's own.
    """
    w = world.reset(jax.random.key(0), CFG)
    onset = w._replace(sick_on=jnp.ones((CFG.n_hens,), dtype=bool))

    assert float(jnp.mean(plasticity.reward(w, onset, CFG, LEARN))) == 0.0, (
        "sickness_penalty=0.0 by default; a sickness onset must not move reward")

    pc = LEARN._replace(sickness_penalty=1.0)
    assert float(jnp.mean(plasticity.reward(w, onset, CFG, pc))) < 0.0

    # Still sick, not newly sick -- must not charge every step of an already-sick
    # window, the same rising-edge discipline `struck` already uses (E014's fix).
    still_sick = onset._replace(sick_on=jnp.ones((CFG.n_hens,), dtype=bool))
    assert float(jnp.mean(plasticity.reward(onset, still_sick, CFG, pc))) == 0.0


# --- Guards against the E010 confound ---------------------------------------

def test_fixed_control_is_actually_fixed():
    """A condition named 'fixed' must not silently carry exploration noise.

    E010 compared a control running at explore_sigma=0.6 -- inherited from a default
    added two experiments earlier -- against a historical noiseless one, and read the
    difference as a result. The contrast varied two things at once.
    """
    from run.experiment import PHASE1
    fixed = PHASE1[0]
    assert not fixed.pc.enabled
    assert fixed.pc.explore_sigma == 0.0, (
        f"'{fixed.name}' carries explore_sigma={fixed.pc.explore_sigma}; "
        "a fixed control must be deterministic")


def test_every_condition_states_exploration_explicitly():
    """No condition may inherit the exploration default -- it must be written down."""
    import inspect
    from run import experiment
    src = inspect.getsource(experiment)
    block = src[src.index("PHASE1 = ("):src.index(")\n", src.index("PHASE1 = ("))]
    n_conditions = block.count("Condition(")
    n_explicit = block.count("explore_sigma=")
    assert n_explicit == n_conditions, (
        f"{n_conditions} conditions but only {n_explicit} state explore_sigma; "
        "an inherited default is how E010 went wrong")


def test_disabled_plasticity_leaves_weights_untouched_even_with_noise(flock):
    """Noise must perturb behaviour without ever writing to the connectome."""
    _, _, p0 = flock
    pc = PlasticConfig(enabled=False, explore_sigma=0.6)
    _w, _x, p1, *_ = _run(flock, pc)
    assert jnp.array_equal(p0.W, p1.W)
    assert jnp.array_equal(p0.W_out, p1.W_out)


# --- Guards on the three E019 defects ---------------------------------------
#
# All three survived eighteen experiments because nothing tested for them, and two of
# them were invisible at the flock size the rest of this file uses. These run at
# n_hens=16, the default, deliberately.

E019_CFG = spec.DEFAULT_COOP           # 16 hens: the size the defects appeared at


def test_a_flock_with_nothing_to_report_is_quiet_on_the_alarm_channels():
    """A hen who is not alarm-calling must emit nothing on the alarm channels.

    Motor channels are sigmoids, so a resting bird sits at sigmoid(-2.5) = 0.076 on
    every channel including the four call ones. Until E019 that floor was emitted as a
    real call, and summed across 15 flockmates it pinned every audio channel at 1.0.

    Scoped to the two alarm channels on purpose. The *food* channel does saturate at
    the default flock size, and that is real behaviour rather than a floor: the innate
    arc food-calls on the sight of food out to 10 m, so a clumped flock near a feeder
    has twelve of sixteen birds genuinely calling. It carries no information for the
    same reason, but the cause is the reflex arc and the fix is a design decision, not
    a bug fix. Tracked in docs/backlog.md rather than silently patched here.
    """
    from coop import sensing
    w = world.reset(jax.random.key(0), E019_CFG)
    p = connectome.build(jax.random.key(1), regions.DEFAULT_REGIONS,
                         n_hens=E019_CFG.n_hens)
    x = brain.initial_state(p, E019_CFG.n_hens)
    w, x, *_ = simulate.rollout_quiet(w, x, p, jax.random.key(5), E019_CFG, 6_000)
    audio = sensing.observe(w, E019_CFG)[:, spec.AUDIO_LO:spec.AUDIO_HI]
    for name, motor_ch in (("aerial", spec.M_CALL_AERIAL),
                           ("ground", spec.M_CALL_GROUND)):
        i = spec.CALL_MOTOR_IDX.index(motor_ch)
        level = float(jnp.max(audio[:, i]))
        assert level < 0.3, (
            f"resting flock hears {level:.3f} on the {name} alarm channel with no "
            "predator anywhere; it is saturated before anyone has said anything")


def test_a_call_is_audible_to_a_flockmate():
    """The whole project depends on this one number being non-zero.

    E019 measured it at exactly 0.0000 with 16 hens: a full-amplitude alarm from an
    adjacent bird moved the receiver's channel not at all. Every experiment about
    communication was running on a constant.
    """
    from coop import sensing
    w = world.reset(jax.random.key(0), E019_CFG)
    p = connectome.build(jax.random.key(1), regions.DEFAULT_REGIONS,
                         n_hens=E019_CFG.n_hens)
    x = brain.initial_state(p, E019_CFG.n_hens)
    w, x, *_ = simulate.rollout_quiet(w, x, p, jax.random.key(5), E019_CFG, 6_000)

    i = spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)
    ch = spec.AUDIO_LO + i
    before = float(sensing.observe(w, E019_CFG)[0, ch])
    loud = w._replace(calls=w.calls.at[1, i].set(1.0))
    after = float(sensing.observe(loud, E019_CFG)[0, ch])
    assert after - before > 0.1, (
        f"a full-amplitude alarm call from a flockmate moves the channel by "
        f"{after - before:+.4f}; nothing downstream can learn from that")


def test_call_floor_matches_the_resting_motor_output():
    """`CALL_FLOOR` and `REST_BIAS` live in different modules and must not drift."""
    from hen import innate
    resting = float(jax.nn.sigmoid(jnp.array(innate.REST_BIAS)))
    assert abs(spec.CALL_FLOOR - resting) < 1e-3, (
        f"CALL_FLOOR={spec.CALL_FLOOR} but a resting hen emits {resting:.4f}")


def test_what_the_pallium_sends_to_the_muscles_depends_on_the_situation(flock):
    """The cortical drive must vary as behaviour unfolds, not sit at a constant.

    This is the property E019 found missing: what the pallium sent to the motor system
    varied by 0.7% of its own magnitude over three seconds. It was an offset, and it
    slid peck downward -- the hen learned to stop eating, which is the whole of the harm
    E013-E016 spent four experiments characterising.

    **The metric here is deliberately not the rank of `dW_out`.** The review that found
    this defect measured rank, and rank is the wrong test: a rank-one change
    `dW_out = u v^T` contributes `u (v . motor_stub)` to the drive, which varies
    perfectly well as `motor_stub` varies. Rank stayed at 0.999 through the fix while
    state-dependence improved elevenfold, so rank would have failed a working rule.
    What matters is whether the drive tracks the situation, so that is what is asserted.
    """
    from coop import sensing
    w, x, p0 = flock
    _w, _x, p1, *_ = _run(flock, LEARN_NO_GROWTH, steps=20_000)
    assert not jnp.allclose(p0.W_out, p1.W_out), "readout did not move at all"

    cort = []
    for t in range(200):
        obs = sensing.observe(w, CFG)
        x, motor, drives = brain.step(x, obs, p1, CFG.dt)
        w = world.step(w, motor, jax.random.fold_in(jax.random.key(3), t), CFG)
        cort.append(drives.cortical)
    cort = jnp.stack(cort)
    variability = float(jnp.mean(jnp.std(cort, axis=0))
                        / (jnp.mean(jnp.abs(cort)) + 1e-9))
    assert variability > 0.02, (
        f"cortical drive varies by {variability:.4f} of its magnitude; the pallium is "
        "applying a constant offset to the muscles rather than a state-dependent one")


def test_reward_is_not_dominated_by_one_component(hawk_period_s=900.0):
    """No component may carry the reward signal on its own.

    E019: the vigour (call-cost) term was 98.1% of reward *variance* -- a hen was taught
    almost nothing except 'did you just call'. That term had been moved out of H2's
    metric by E012 and into the teaching signal, where nobody checked it.

    **This test measures variance in a rollout, and the reason is embarrassing.** The
    first version, written the same day E019 found the defect, perturbed each drive by
    an identical -0.01 and compared the reward *response*. Under the broken code all
    four components entered `d_drive` with the same coefficient, so each scored exactly
    25% and the guard passed on the bug it was written for (E022 3d). Sensitivity was
    never the problem: vigour dominated because vigour *varied*, sd 0.23, while hunger
    barely moved. Guard the quantity that broke, not the one that is easy to poke.

    Each candidate's contribution is measured by *freezing* it -- recomputing the reward
    with that field unchanged between the two steps -- and taking the variance of the
    difference. That works whether or not the field is in the formula, so the test does
    not have to be told which terms count. A second version of this test enumerated
    components by name and failed for the opposite reason: it scored vigour's variance
    after E019 had already removed vigour from the reward.

    Run at 16 hens over 30 s, not on the module's 4-hen fixture. Feeding is sparse and
    bursty -- hunger only moves when a hen actually eats -- so over a short window at a
    small flock no hen eats at all and `cold` trivially scores 100%. That is the window
    being too small, not the reward being broken, and a third version of this test
    failed that way before the size was fixed.
    """
    from coop import sensing
    pc = PlasticConfig()
    CFG = E019_CFG._replace(hawk_period_s=hawk_period_s)
    w = world.reset(jax.random.key(0), CFG)
    p = connectome.build(jax.random.key(1), regions.DEFAULT_REGIONS,
                         n_hens=CFG.n_hens)
    x = brain.initial_state(p, CFG.n_hens)
    # `n_strike_events` rather than `n_struck`: the reward reads the event counter, and
    # freezing the field the reward does not read would score 0% and prove nothing.
    fields = ("hunger", "thirst", "cold", "vigour", "n_strike_events")
    rewards, contrib = [], {k: [] for k in fields}
    for t in range(3_000):
        obs = sensing.observe(w, CFG)
        x, motor, _ = brain.step(x, obs, p, CFG.dt)
        wn = world.step(w, motor, jax.random.fold_in(jax.random.key(4), t), CFG)
        r = plasticity.reward(w, wn, CFG, pc)
        rewards.append(jnp.mean(r))
        for k in fields:
            frozen = wn._replace(**{k: getattr(w, k)})
            contrib[k].append(jnp.mean(r - plasticity.reward(w, frozen, CFG, pc)))
        w = wn

    assert float(jnp.var(jnp.array(rewards))) > 0.0, "reward never varied; assay is dead"
    var = {k: float(jnp.var(jnp.array(v))) for k, v in contrib.items()}
    total = sum(var.values())
    shares = {k: round(v / max(total, 1e-12), 3) for k, v in var.items()}
    for name, v in var.items():
        assert v / max(total, 1e-12) < 0.8, (
            f"{name} carries {100 * v / max(total, 1e-12):.0f}% of the variance the "
            f"reward actually responds to; a modulator dominated by one component "
            f"teaches only that component ({shares})")


def test_being_caught_does_not_dominate_the_reward_where_hawks_are_common():
    """The strike penalty is a *cost*, not the lesson. Guarded where hawks exist.

    E027. `n_struck` incremented every step of contact, so one catch during a 12 s dive
    was worth up to ~1000 against per-step drive terms of 0.1-0.5. At `hawk_period_s=20`
    -- the rate every H4 experiment runs at -- it carried **87.3%** of the reward
    variance. E014 had removed the `/dt` and left the per-step accumulation; E022 filed
    the remainder as owed and nobody verified it.

    The test above cannot catch this and never could: at the 900 s default no hawk
    arrives in its window, so the strike share is 0.0% by absence.

    **Two things have to hold for this test to mean anything**, and the second is the
    one the project keeps getting wrong: the strike term must be small, *and* strikes
    must actually have happened. A 30 s window at this predator rate still contains no
    strike, so it would pass for exactly the vacuous reason the 900 s config does. 100 s
    is the shortest window measured to contain one -- which makes a single hardcoded
    seed fragile to any change that shifts the RNG stream (e.g. `OBS_DIM` growing, which
    changes what a fixed connectome-build key draws even though nothing about predation
    changed): seed 0 went from >=1 strike to a closest approach of 1.549 m against a
    1.5 m strike radius, a miss by seconds of simulated proximity, after E048's
    personal-space channel was added. A sweep of 8 seeds found strikes in 6 of them
    (0 and 3 miss), confirming this is seed sensitivity, not a behavioural regression --
    so this test tries a short, fixed list of seeds rather than trusting one.
    """
    from coop import sensing
    pc = PlasticConfig()
    CFG = E019_CFG._replace(hawk_period_s=20.0)

    for seed in (0, 1, 2):
        w = world.reset(jax.random.key(seed), CFG)
        p = connectome.build(jax.random.key(seed + 1), regions.DEFAULT_REGIONS,
                             n_hens=CFG.n_hens)
        x = brain.initial_state(p, CFG.n_hens)

        strike_contrib, other, events = [], [], 0.0
        for t in range(10_000):
            obs = sensing.observe(w, CFG)
            x, motor, _ = brain.step(x, obs, p, CFG.dt)
            wn = world.step(w, motor, jax.random.fold_in(jax.random.key(4), t), CFG)
            r = plasticity.reward(w, wn, CFG, pc)
            frozen = wn._replace(n_strike_events=w.n_strike_events)
            strike_contrib.append(jnp.mean(r - plasticity.reward(w, frozen, CFG, pc)))
            other.append(jnp.mean(r))
            events += float(jnp.sum(wn.n_strike_events - w.n_strike_events))
            w = wn

        if events > 0:
            break

    assert events > 0, (
        "no hen was struck in the whole window at any of 3 tried seeds, so this guard "
        "proves nothing -- the same vacuous pass that let the defect survive at "
        "hawk_period_s=900")

    v_strike = float(jnp.var(jnp.array(strike_contrib)))
    v_total = float(jnp.var(jnp.array(other)))
    share = v_strike / max(v_total, 1e-12)
    assert share < 0.2, (
        f"being caught carries {100 * share:.0f}% of the reward variance over {events:.0f} "
        "strike events; a hen learning in this world is taught strike avoidance and "
        "almost nothing else")


def test_vigour_is_a_cost_in_the_world_but_not_in_the_reward(flock):
    """Calling still costs something -- just not in the teaching signal.

    Removing vigour from `reward()` must not remove the cost entirely, or
    audience-sensitivity has nothing to emerge from. It still drains with calling and
    still attenuates what flockmates hear.
    """
    w, _, _ = flock
    pc = PlasticConfig()
    spent = w._replace(vigour=w.vigour - 0.5)
    r = float(jnp.mean(plasticity.reward(w, spent, CFG, pc)))
    assert abs(r) < 1e-9, f"vigour still enters the reward ({r:+.6f})"


# --- Guards on the E018 auditory scaffold -----------------------------------

def test_bare_arc_has_no_auditory_response():
    """Without the scaffold, hearing a call must drive nothing at all.

    This is the baseline every E018 number is measured against. If the scaffold ever
    leaks into the default, the 2x2's control condition silently becomes the treatment
    and the experiment measures nothing -- the same shape of failure as E010.
    """
    from hen import innate
    r = innate.reflex_matrix()
    audio = r[:, spec.AUDIO_LO:spec.AUDIO_HI]
    assert float(jnp.max(jnp.abs(jnp.asarray(audio)))) == 0.0


def test_scaffold_is_off_by_default():
    """The default connectome is the one E001-E017 were run on."""
    from hen import innate
    assert jnp.array_equal(jnp.asarray(innate.reflex_matrix()),
                           jnp.asarray(innate.reflex_matrix(auditory_scaffold=False)))
    p = connectome.build(jax.random.key(0), regions.DEFAULT_REGIONS, n_hens=2)
    assert float(jnp.max(jnp.abs(p.reflex[:, spec.AUDIO_LO:spec.AUDIO_HI]))) == 0.0


def test_scaffold_wires_what_it_says_and_nothing_else():
    """The scaffold must not quietly acquire channels it was pre-registered without.

    Specifically: no relay (hearing a call must never drive producing one), and
    nothing on the food or contact channels, which stay predator-neutral so that a
    second-order conditioning test remains available.
    """
    from hen import innate
    r = jnp.asarray(innate.reflex_matrix(auditory_scaffold=True))
    aerial = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)
    ground = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_GROUND)
    contact = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_CONTACT)
    food = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_FOOD)

    assert float(r[spec.M_CROUCH, aerial]) == innate.SCAFFOLD_WEIGHT
    assert float(r[spec.M_FLEE, ground]) == innate.SCAFFOLD_WEIGHT
    # head comes up, or the call restores no information she did not have
    for call in (aerial, ground):
        assert float(r[spec.M_PECK, call]) == -innate.SCAFFOLD_WEIGHT
        assert float(r[spec.M_SCRATCH, call]) == -innate.SCAFFOLD_WEIGHT
    # no relay: no heard call may drive any produced call
    for heard in (aerial, ground, contact, food):
        for produced in spec.CALL_MOTOR_IDX:
            assert float(r[produced, heard]) == 0.0, "scaffold must not relay calls"
    # the neutral channels stay neutral
    assert float(jnp.max(jnp.abs(r[:, [contact, food]]))) == 0.0


def test_scaffold_never_outweighs_seeing_it_yourself():
    """First-hand information must always beat second-hand.

    A hen who can see the hawk should not be talked out of it, so the auditory weight
    has to stay well under the visual one. E018 fixes it at 1.5 against 8.0; this
    guards the ordering rather than the value.
    """
    from hen import innate
    r = jnp.asarray(innate.reflex_matrix(auditory_scaffold=True))
    aerial_call = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)
    assert float(r[spec.M_CROUCH, aerial_call]) < float(r[spec.M_CROUCH,
                                                          spec.IDX_AERIAL]) / 3.0


def test_scaffold_leaves_the_arc_fixed_under_learning(flock):
    """The scaffold is part of the reflex arc, so learning must never touch it."""
    _, _, _p0 = flock
    p_scaf = connectome.build(jax.random.key(1), regions.DEFAULT_REGIONS,
                              n_hens=CFG.n_hens, auditory_scaffold=True)
    x = brain.initial_state(p_scaf, CFG.n_hens)
    w = world.reset(jax.random.key(0), CFG)
    ps = plasticity.initial_state(p_scaf, CFG.n_hens, LEARN)
    _w, _x, p1, *_ = simulate.rollout_quiet(
        w, x, p_scaf, jax.random.key(9), CFG, 3_000, ps, LEARN)
    assert jnp.array_equal(p_scaf.reflex, p1.reflex)


def _aud_bounds():
    s_lo, s_hi = regions.DEFAULT_REGIONS.bounds(regions.SENSORY)
    p_lo, p_hi = regions.DEFAULT_REGIONS.bounds(regions.PALLIUM)
    n_aud_s = max(1, round((s_hi - s_lo) * regions.AUD_FRACTION))
    n_aud_p = max(1, round((p_hi - p_lo) * regions.AUD_FRACTION))
    return s_lo, s_lo + n_aud_s, s_hi, p_lo, p_lo + n_aud_p, p_hi


def test_modality_segregation_is_off_by_default():
    """The default connectome is the one E001-E034 were run on: fully mixed."""
    p0 = connectome.build(jax.random.key(0), regions.DEFAULT_REGIONS, n_hens=2)
    p1 = connectome.build(jax.random.key(0), regions.DEFAULT_REGIONS, n_hens=2,
                          modality_segregated=False)
    assert jnp.array_equal(p0.mask, p1.mask)
    assert jnp.array_equal(p0.W_in, p1.W_in)


def test_modality_segregation_cuts_exactly_the_cross_terms():
    """Field L must not hear from the visual stub, and vice versa -- nothing else
    about the sensory<->pallium block, or any other region pair, should move."""
    s_lo, s_split, s_hi, p_lo, p_split, p_hi = _aud_bounds()
    key = jax.random.key(0)
    p0 = connectome.build(key, regions.DEFAULT_REGIONS, n_hens=2)
    p1 = connectome.build(key, regions.DEFAULT_REGIONS, n_hens=2,
                          modality_segregated=True)

    assert not bool(p1.mask[p_split:p_hi, s_lo:s_split].any()), \
        "rest-of-pallium must not hear from the auditory stub slice"
    assert not bool(p1.mask[p_lo:p_split, s_split:s_hi].any()), \
        "Field L must not hear from the visual stub slice"
    assert bool(p0.mask[p_split:p_hi, s_lo:s_split].any()), \
        "sanity: the mixed connectome must have had those connections to cut"

    # growth cannot regrow what was cut
    assert not bool(p1.growable[p_split:p_hi, s_lo:s_split].any())
    assert not bool(p1.growable[p_lo:p_split, s_split:s_hi].any())

    # nothing outside the sensory<->pallium block moved
    for r_id in range(regions.N_REGIONS):
        if r_id in (regions.SENSORY, regions.PALLIUM):
            continue
        lo, hi = regions.DEFAULT_REGIONS.bounds(r_id)
        assert jnp.array_equal(p0.mask[lo:hi], p1.mask[lo:hi]), regions.REGION_NAMES[r_id]


def test_modality_segregation_afferents_do_not_cross():
    """Audio channels must not reach the visual stub slice, and no other exteroceptive
    channel may reach the auditory stub slice."""
    s_lo, s_split, s_hi, _, _, _ = _aud_bounds()
    p = connectome.build(jax.random.key(0), regions.DEFAULT_REGIONS, n_hens=2,
                         modality_segregated=True)
    audio = slice(spec.AUDIO_LO, spec.AUDIO_HI)
    assert float(jnp.max(jnp.abs(p.W_in[s_split:s_hi, audio]))) == 0.0
    non_audio_vis = spec.VIS_LO
    assert float(jnp.max(jnp.abs(p.W_in[s_lo:s_split, non_audio_vis]))) == 0.0
    # and each slice still gets *something*
    assert float(jnp.max(jnp.abs(p.W_in[s_lo:s_split, audio]))) > 0.0
    assert float(jnp.max(jnp.abs(p.W_in[s_split:s_hi, non_audio_vis]))) > 0.0


def test_sensory_pallium_density_is_off_by_default():
    """The default connectome is the one E001-E040 were run on: density 0.30."""
    p0 = connectome.build(jax.random.key(0), regions.DEFAULT_REGIONS, n_hens=2)
    p1 = connectome.build(jax.random.key(0), regions.DEFAULT_REGIONS, n_hens=2,
                          sensory_pallium_density=None)
    assert jnp.array_equal(p0.mask, p1.mask)
    p2 = connectome.build(jax.random.key(0), regions.DEFAULT_REGIONS, n_hens=2,
                          sensory_pallium_density=0.30)
    assert jnp.array_equal(p0.mask, p2.mask), "0.30 must reproduce the hardcoded default"


def test_sensory_pallium_density_moves_only_that_block():
    """Lowering density must touch the sensory->pallium block and nothing else, and
    the resulting fan-in must genuinely be lower (not just fewer synapses drawn and
    silently compensated somewhere else)."""
    s_lo, s_hi = regions.DEFAULT_REGIONS.bounds(regions.SENSORY)
    p_lo, p_hi = regions.DEFAULT_REGIONS.bounds(regions.PALLIUM)
    key = jax.random.key(0)
    p_default = connectome.build(key, regions.DEFAULT_REGIONS, n_hens=2)
    p_sparse = connectome.build(key, regions.DEFAULT_REGIONS, n_hens=2,
                                sensory_pallium_density=0.05)

    block_default = p_default.mask[p_lo:p_hi, s_lo:s_hi]
    block_sparse = p_sparse.mask[p_lo:p_hi, s_lo:s_hi]
    assert float(block_sparse.mean()) < float(block_default.mean())

    for r_id in range(regions.N_REGIONS):
        if r_id in (regions.SENSORY, regions.PALLIUM):
            continue
        lo, hi = regions.DEFAULT_REGIONS.bounds(r_id)
        assert jnp.array_equal(p_default.mask[lo:hi], p_sparse.mask[lo:hi]), \
            regions.REGION_NAMES[r_id]

    fan_in_default = jnp.sum(p_default.mask, axis=1)[p_lo:p_hi].mean()
    fan_in_sparse = jnp.sum(p_sparse.mask, axis=1)[p_lo:p_hi].mean()
    assert float(fan_in_sparse) < float(fan_in_default)


def test_reward_components_are_commensurate(flock):
    """No single reward component may dwarf the others.

    E014: `struck * strike_penalty / dt` treated a discrete event as a rate, so one
    predator strike contributed -100 against a feeding step worth ~0.5 -- about 150x
    -- and the modulator slammed every eligible synapse at once. It took eight
    experiments to find, and it should have been caught by construction.

    The comparison is against a real *feeding* step, not against baseline hunger
    drift. Drift is tiny by design, and measuring against it would flag a healthy
    reward as broken -- which it did on the first version of this test.
    """
    w, _, _ = flock
    pc = PlasticConfig()
    # One timestep of successful feeding, as coop/world.py computes it.
    eaten = CFG.dt * CFG.peck_food_rate * w.hunger
    fed = w._replace(hunger=w.hunger - eaten)
    struck = w._replace(n_struck=w.n_struck + 1.0)

    r_fed = abs(float(jnp.mean(plasticity.reward(w, fed, CFG, pc))))
    r_struck = abs(float(jnp.mean(plasticity.reward(w, struck, CFG, pc))))
    ratio = r_struck / max(r_fed, 1e-9)
    assert ratio < 20.0, (
        f"a strike is worth {ratio:.0f}x a step of feeding; reward components must "
        "stay within an order of magnitude, or the modulator is dominated by one "
        "event and learning becomes a shock response")


# --- Guards on the H4 channel ladder (E026) ---------------------------------
#
# Before E026 nothing in this suite touched `channel_mode` at all -- the single most
# load-bearing manipulation in the headline experiment, in a repo whose CLAUDE.md has
# a standing rule about exactly this. E024 ran a control that kept 98% of the
# information it was supposed to destroy, and no test could have caught it.

def _heard_vs_hawk(mode, seed=0, steps=9_000):
    """corr(aerial audio, a hawk is inside MY strike radius) under a channel mode.

    Jitted, and staged so hawks actually arrive. The first version ran a Python-level
    loop over a 40 s window with a hawk every 60 s, so it usually observed no hawk at
    all and returned "the assay is dead" -- a guard that cannot see the thing it guards.
    `hawk_period_s` is cut to 15 s here purely to make the test observable; it is a
    fixture setting, not the experiment's.

    The warm-up matters: the yoked buffer is `cfg.call_log_steps` deep and its lags reach
    most of the way back, so a run that starts measuring immediately reads unwritten
    zeros and would pass for the wrong reason.
    """
    from functools import partial
    from coop import sensing
    aer = spec.CALL_MOTOR_IDX.index(spec.M_CALL_AERIAL)
    # `call_log_steps` is off by default -- the buffer costs throughput and only the
    # yoked control needs it (E026). A test that forgets it gets a loud ValueError
    # rather than a silently zeroed channel, which would pass for the wrong reason.
    cfg = spec.DEFAULT_COOP._replace(
        n_hens=16, hawk_period_s=15.0, channel_mode=mode,
        call_log_steps=(spec.YOKE_LOG_STEPS if mode == "yoked" else 1))

    @partial(jax.jit, static_argnames=("cfg",))
    def trace(w, x, p, key, cfg):
        def step(carry, _):
            w, x, key = carry
            key, k = jax.random.split(key)
            obs = sensing.observe(w, cfg)
            x, motor, _ = brain.step(x, obs, p, cfg.dt)
            d = jnp.linalg.norm(w.pos - w.hawk_pos[None, :], axis=-1)
            near = (d < cfg.hawk_strike_radius) & (w.hawk_on > 0.5)
            w = world.step(w, motor, k, cfg)
            return (w, x, key), (obs[:, spec.AUDIO_LO + aer], near)
        return jax.lax.scan(step, (w, x, key), None, length=steps)[1]

    w = world.reset(jax.random.key(seed), cfg)
    p = connectome.build(jax.random.fold_in(jax.random.key(seed), 1),
                         regions.DEFAULT_REGIONS, n_hens=16, auditory_scaffold=True)
    x = brain.initial_state(p, 16)
    w, x, *_ = simulate.rollout_quiet(w, x, p, jax.random.key(9), cfg,
                                      cfg.call_log_steps + 200)
    heard, near = trace(w, x, p, jax.random.key(7), cfg)
    h, n = np.asarray(heard).ravel(), np.asarray(near).ravel()
    assert n.sum() > 0, "no hawk ever reached a hen; the assay is dead, not the channel"
    assert h.std() > 0, "the audio channel never varied; nothing to correlate"
    return float(np.corrcoef(h, n.astype(float))[0, 1])


def test_the_intact_channel_carries_information():
    """If this fails the experiment has no signal to detect, never mind a control."""
    c = _heard_vs_hawk("intact")
    assert c > 0.2, f"intact channel correlates {c:.3f} with a hawk being on her"


def test_the_yoked_control_destroys_the_information():
    """The control must actually be uninformative. E024's did not, and shipped.

    Measured at E026: intact +0.56, permuted +0.55 (98% kept), yoked -0.13.
    """
    c = _heard_vs_hawk("yoked")
    assert abs(c) < 0.2, (
        f"yoked control still correlates {c:.3f} with a hawk being on her; "
        "a control that carries the signal is not a control")


def test_shuffled_is_not_a_control_and_is_labelled_so():
    """Regression: the permutation must stay available and stay disclaimed.

    Kept so E024 reproduces. The guard is that nobody quietly promotes it back to
    being the headline control -- the source has to say it is not one.
    """
    import inspect
    from coop import sensing
    src = inspect.getsource(sensing._channel)
    assert "NOT A CONTROL" in src.upper(), (
        "the shuffled mode must be documented as not a valid control")


def test_every_channel_mode_is_reachable():
    """A typo in a mode name must fail loudly, not silently fall through to intact."""
    from coop import sensing
    for mode in ("intact", "none", "severed", "self", "yoked", "shuffled"):
        cfg = spec.DEFAULT_COOP._replace(
            n_hens=4, channel_mode=mode,
            call_log_steps=(spec.YOKE_LOG_STEPS if mode == "yoked" else 1))
        w = world.reset(jax.random.key(0), cfg)
        assert sensing.observe(w, cfg).shape == (4, spec.OBS_DIM)
    with pytest.raises(ValueError):
        cfg = spec.DEFAULT_COOP._replace(n_hens=4, channel_mode="nonsense")
        sensing.observe(world.reset(jax.random.key(0), cfg), cfg)


# --- E105: decorrelating readout, temporal adaptation ---------------------
#
# Both mechanisms are off at the shipped defaults, and both guards run at the
# configuration where the defect they address appears: `hebbian_readout`, the rule
# whose collapse E100 measured, at 16 hens rather than the suite's usual 4.


def test_e105_mechanisms_are_off_by_default():
    assert PlasticConfig().readout_decorrelate == 0.0
    assert spec.DEFAULT_COOP.sensory_adapt_tau_s is None


def test_decorrelation_makes_the_output_channels_learn_different_directions():
    """The update must stop being rank one.

    Under the current rule the presynaptic factor is one vector shared by every output
    row, so all twelve channels move along the same pallial direction every
    consolidation -- cosine 1.0 between any two rows of `dw_out`. That is the collapse
    E100 measured, at its source. Deflation must break it without silencing the update,
    which is the degeneracy the experiment's own falsifier watches for.
    """
    n_hens = 16
    p = connectome.build(jax.random.key(3), regions.DEFAULT_REGIONS, n_hens=n_hens)
    pc = PlasticConfig(enabled=True, hebbian_readout=True)
    ps = plasticity.initial_state(p, n_hens, pc)
    # Traces at plausible non-zero values, so the update is not identically zero, and
    # a grown readout so the deflation term is not negligible against the presynaptic
    # vector -- at hatch `readout_scale` is 0.05 and it would barely bite.
    key = jax.random.key(4)
    ps = ps._replace(
        z_slow=jax.random.uniform(jax.random.fold_in(key, 1), ps.z_slow.shape),
        z_motor=jax.random.uniform(jax.random.fold_in(key, 2), ps.z_motor.shape))
    p = p._replace(W_out=p.W_out * 20.0)
    m = jnp.ones((n_hens,))

    def measure(decorrelate):
        p2 = plasticity.consolidate(p, ps, m, pc._replace(
            readout_decorrelate=decorrelate, w_max=1e9))
        dw = p2.W_out - p.W_out
        u = dw / (jnp.linalg.norm(dw, axis=2, keepdims=True) + 1e-12)
        c = jnp.abs(jnp.einsum("hmk,hnk->hmn", u, u))
        off = jnp.broadcast_to(1.0 - jnp.eye(c.shape[-1])[None, :, :], c.shape)
        return (float(jnp.sum(c * off) / jnp.sum(off)),
                float(jnp.mean(jnp.linalg.norm(dw, axis=2))))

    align_off, mag_off = measure(0.0)
    align_on, mag_on = measure(1.0)
    assert align_off > 0.99, (
        f"the current rule should give a rank-one update -- rows aligned at "
        f"{align_off:.4f}. If this is already low the test proves nothing.")
    # 0.9997 -> ~0.91 measured: the residual left to each channel differs by two
    # orders of magnitude more than the rule's own numerical noise. A single
    # consolidation cannot make twelve rows orthogonal; what matters is that they stop
    # being the same vector, so the accumulation over a rearing can span more than one
    # direction.
    assert align_on < 0.96, (
        f"deflation left the output rows aligned at {align_on:.4f}; the update is "
        "still effectively rank one and the mechanism is inert")
    assert mag_on > 0.2 * mag_off, (
        f"deflation cut the update magnitude to {mag_on:.3e} from {mag_off:.3e}; "
        "forcing the channels apart by silencing them is not the mechanism")


def test_temporal_adaptation_passes_change_not_level():
    """A relay unit held at a constant input must fall silent; a step must show.

    This is the property the mechanism exists for. Run directly on `brain.step` with
    a held-constant observation, so nothing about the world's own dynamics can supply
    the decay.
    """
    n_hens, tau, dt = 4, 2.0, spec.DEFAULT_COOP.dt
    p = connectome.build(jax.random.key(5), regions.DEFAULT_REGIONS, n_hens=n_hens)
    obs = jnp.full((n_hens, spec.OBS_DIM), 0.6)
    pool = p.lateral_pool > 0
    adapt = jnp.zeros((n_hens, p.b.shape[0]))
    x = brain.initial_state(p, n_hens)
    first = None
    for i in range(int(20 * tau / dt)):
        x, _motor, d = brain.step(x, obs, p, dt, adapt_bar=adapt)
        adapt = adapt + (dt / tau) * (d.current - adapt)
        driven = float(jnp.mean(jnp.abs((d.current - adapt)[:, pool])))
        if i == 0:
            first = driven
    assert first > 1e-3, "the constant input must actually drive the relay at first"
    assert driven < 0.05 * first, (
        f"after 20 time constants of an unchanging input the relay still passes "
        f"{driven:.5f} against an initial {first:.5f}; it is passing level, not change")
    # And a change must still get through.
    x, _motor, d = brain.step(x, obs * 0.0, p, dt, adapt_bar=adapt)
    stepped = float(jnp.mean(jnp.abs((d.current - adapt)[:, pool])))
    assert stepped > 0.5 * first, (
        f"a step change passed only {stepped:.5f} against {first:.5f} at rest; "
        "adaptation has suppressed the signal along with the baseline")


# --- E106: pooled interneurons in the recurrent regions -------------------


def test_recurrent_lateral_is_off_by_default():
    assert spec.DEFAULT_COOP.recurrent_lateral == 0.0


def test_the_interneuron_pools_its_own_region_and_no_other():
    """Pallium and motor stub each lose their own mean; nothing else is touched.

    The guard exists because `_lateral_pool` got exactly this wrong in E104 and was
    caught before running: it swept in the hypothalamus's interoceptive units, so
    hunger would have been subtracted from visual contrast. Here the hypothalamus is
    excluded for a second reason -- an absolute drive level *is* the message, and a hen
    who could only perceive being hungrier than usual would be worse off.
    """
    from hen import neurons
    n_hens = 4
    p = connectome.build(jax.random.key(7), regions.DEFAULT_REGIONS, n_hens=n_hens)
    n = p.b.shape[0]
    r = jax.random.uniform(jax.random.key(8), (n_hens, n)) + 0.5
    q = neurons.pooled(r, p.region_pools, 1.0)

    for r_id in (regions.PALLIUM, regions.MOTOR):
        lo, hi = regions.DEFAULT_REGIONS.bounds(r_id)
        assert abs(float(jnp.mean(q[:, lo:hi]))) < 1e-5, (
            f"region {r_id} still carries a common component after pooling")
        assert float(jnp.std(q[:, lo:hi])) > 0.1, (
            f"region {r_id} lost its variation along with its mean")
    for r_id in (regions.SENSORY, regions.HIPPOCAMPUS,
                 regions.ARCOPALLIUM, regions.HYPOTHALAMUS):
        lo, hi = regions.DEFAULT_REGIONS.bounds(r_id)
        assert bool(jnp.allclose(q[:, lo:hi], r[:, lo:hi])), (
            f"region {r_id} was modified; only the pallium and motor stub have an "
            "interneuron, and the hypothalamus must keep its absolute drive levels")


def test_the_interneuron_does_not_change_any_neurons_own_rate():
    """It changes what targets receive, not what the neuron does.

    `ctrnn_step` must still return the true rate. If this ever stops holding, every
    diagnostic in the project that reads `neurons.rate(x)` is silently measuring
    something else.
    """
    from hen import neurons
    n_hens = 4
    p = connectome.build(jax.random.key(9), regions.DEFAULT_REGIONS, n_hens=n_hens)
    x = brain.initial_state(p, n_hens)
    current = jax.random.uniform(jax.random.key(10), x.shape)
    r_proj = neurons.pooled(neurons.rate(x), p.region_pools, 1.0)
    _x1, r1 = neurons.ctrnn_step(x, p.W, current, p.b, p.tau, CFG.dt)
    _x2, r2 = neurons.ctrnn_step(x, p.W, current, p.b, p.tau, CFG.dt, r_proj)
    assert bool(jnp.array_equal(r1, r2)), (
        "the interneuron changed the reported rate; it must only change the projection")
    assert not bool(jnp.allclose(_x1, _x2)), (
        "the interneuron changed nothing downstream -- it is inert, not subtractive")


# --- E107: the direction-stability metric --------------------------------
#
# These guard the defect that invalidated E100-E106's headline: a statistic that
# pooled sixteen hens before taking the mean direction, so it reported between-hen
# spread as if it were within-hen state-dependence. The guard runs on a synthetic
# case built to separate the two, because on real trajectories the two numbers can
# happen to agree -- they do for the pallium and the motor stub, and that near-agreement
# is exactly why nobody noticed the readout's disagreed.


def test_pooling_hens_is_not_within_hen_state_dependence():
    """Hens that each vary but point different ways: per-hen low, pooled high.

    Each hen's output sweeps a wide arc around her own private axis. Within a hen the
    direction genuinely varies, so `direction_stability` must be well below 1. Pooled,
    the private axes dominate and the statistic is near the between-hen number instead.
    """
    from run import metrics
    rng = np.random.default_rng(0)
    n_t, n_h, n_d = 400, 16, 12
    axes = rng.normal(size=(n_h, n_d))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    a = np.empty((n_t, n_h, n_d))
    for h in range(n_h):
        wobble = rng.normal(size=(n_t, n_d))
        wobble -= (wobble @ axes[h])[:, None] * axes[h][None, :]
        wobble /= np.linalg.norm(wobble, axis=1, keepdims=True)
        a[:, h] = axes[h][None, :] + 1.2 * wobble

    per_hen = metrics.direction_stability(a)
    pooled = metrics.pooled_direction_stability(a)
    between = metrics.between_hen_alignment(a)

    assert per_hen < 0.75, (
        f"per-hen stability {per_hen:.4f}: each hen's direction genuinely varies here, "
        "so this must be well below 1.0 or the metric cannot see state-dependence")
    assert abs(pooled - between) < abs(pooled - per_hen), (
        f"pooled {pooled:.4f} should track the between-hen number {between:.4f} rather "
        f"than the per-hen one {per_hen:.4f}; if it does not, this synthetic case no "
        "longer separates the two and the guard proves nothing")


def test_a_fixed_direction_reads_as_one_whatever_its_magnitude():
    """The metric's own definition: magnitude alone varying must give 1.0."""
    from run import metrics
    rng = np.random.default_rng(1)
    axis = rng.normal(size=8)
    scale = rng.uniform(0.01, 5.0, size=(300, 4, 1))
    a = scale * axis[None, None, :]
    assert abs(metrics.direction_stability(a) - 1.0) < 1e-6
    assert abs(metrics.pooled_direction_stability(a) - 1.0) < 1e-6


def test_dc_share_uses_axis_not_ord():
    """A vector that never moves is 100% DC; numpy's second positional arg is `ord`."""
    from run import metrics
    a = np.tile(np.array([3.0, 4.0, 0.0]), (50, 2, 1))
    assert abs(metrics.dc_share(a) - 1.0) < 1e-9
    rng = np.random.default_rng(2)
    noisy = rng.normal(size=(500, 2, 3))
    assert metrics.dc_share(noisy) < 0.2, (
        "zero-mean noise must have a near-zero DC share")


# --- E110: the postsynaptic factor ---------------------------------------


def test_postsynaptic_factor_defaults_to_the_motor_output():
    assert PlasticConfig().postsynaptic_factor == "motor"


def test_the_perturbation_is_carried_out_and_is_zero_without_a_key():
    """`Drives.noise` must be the perturbation actually added, not a re-draw.

    The whole point of E110 is crediting the exploration noise, so if this field ever
    stops being the same sample that reached the drive, node perturbation silently
    becomes noise-correlated-with-nothing.
    """
    n_hens = 4
    p = connectome.build(jax.random.key(11), regions.DEFAULT_REGIONS, n_hens=n_hens)
    x = brain.initial_state(p, n_hens)
    obs = jnp.full((n_hens, spec.OBS_DIM), 0.3)

    _x, _m, quiet = brain.step(x, obs, p, CFG.dt, key=None)
    assert float(jnp.max(jnp.abs(quiet.noise))) == 0.0, (
        "an assay runs with key=None and must have no perturbation at all")

    key = jax.random.key(12)
    _x, motor, d = brain.step(x, obs, p, CFG.dt, key=key, sigma=0.5)
    # Reconstruct: the drive without noise is reflex + cortical + b_motor.
    drive = d.reflex + d.cortical + p.b_motor[None, :]
    assert bool(jnp.allclose(motor, jax.nn.sigmoid(drive + d.noise), atol=1e-6)), (
        "Drives.noise is not the perturbation that produced this motor output")
    assert float(jnp.std(d.noise)) > 0.1, "sigma=0.5 should give a visible perturbation"


def test_swapping_the_postsynaptic_factor_changes_direction_not_magnitude():
    """E110's magnitude control: arms must differ in direction alone.

    E089's lesson. If an arm also changes the update's size, a behavioural difference is
    confounded with learning rate and the experiment cannot attribute it to direction.
    """
    n_hens = 16
    p = connectome.build(jax.random.key(13), regions.DEFAULT_REGIONS, n_hens=n_hens)
    key = jax.random.key(14)
    m = jnp.ones((n_hens,))
    sizes, directions = {}, {}
    for factor in ("motor", "noise", "cortical"):
        pc = PlasticConfig(enabled=True, postsynaptic_factor=factor)
        ps = plasticity.initial_state(p, n_hens, pc)
        ps = ps._replace(
            z_slow=jax.random.uniform(jax.random.fold_in(key, 1), ps.z_slow.shape),
            z_motor=jax.random.uniform(jax.random.fold_in(key, 2), ps.z_motor.shape),
            z_post=jax.random.uniform(jax.random.fold_in(key, 3), ps.z_post.shape))
        dw = plasticity.consolidate(p, ps, m, pc._replace(w_max=1e9)).W_out - p.W_out
        sizes[factor] = float(jnp.mean(jnp.linalg.norm(dw, axis=(1, 2))))
        directions[factor] = dw / (jnp.linalg.norm(dw) + 1e-12)

    for factor in ("noise", "cortical"):
        ratio = sizes[factor] / sizes["motor"]
        assert 0.5 < ratio < 2.0, (
            f"{factor} changed the update's magnitude by {ratio:.2f}x; the rescale to "
            "||dz_motor|| is meant to leave direction as the only difference")
        cos = float(jnp.sum(directions[factor] * directions["motor"]))
        assert abs(cos) < 0.9, (
            f"{factor} produced an update {cos:.3f} aligned with the default -- it is "
            "not a different direction, so the intervention is inert")
