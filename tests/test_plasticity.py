"""Phase 1 tests: learning must change the right things and nothing else.

Most of these guard invariants rather than behaviour. Whether the rule actually
produces adaptive learning is a *hypothesis* (H2), tested by `run/experiment.py` as a
matched-seed contrast -- not something to assert in a unit test. Connectome churn is
easy to produce and proves nothing on its own.
"""

import jax
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
    w, _, _ = flock
    struck = w._replace(n_struck=w.n_struck + 1.0)
    assert float(jnp.mean(plasticity.reward(w, struck, CFG, LEARN))) < 0.0


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


def test_reward_is_not_dominated_by_one_component(flock):
    """No component may carry the reward signal on its own.

    E019: the vigour (call-cost) term was 98.1% of reward variance -- a hen was taught
    almost nothing except 'did you just call'. That term had been moved out of H2's
    *metric* by E012 and into the teaching signal, where nobody checked it. The general
    failure this guards is a term being verified in the home it just left.
    """
    w, _, _ = flock
    pc = PlasticConfig()
    base = plasticity.reward(w, w, CFG, pc)
    contribs = {}
    for name, field, delta in (("hunger", "hunger", -0.01),
                               ("thirst", "thirst", -0.01),
                               ("cold", "cold", -0.01),
                               ("vigour", "vigour", -0.01)):
        w2 = w._replace(**{field: getattr(w, field) + delta})
        contribs[name] = abs(float(jnp.mean(plasticity.reward(w, w2, CFG, pc) - base)))
    total = sum(contribs.values())
    for name, v in contribs.items():
        assert v / max(total, 1e-12) < 0.6, (
            f"{name} carries {100 * v / total:.0f}% of the reward response; "
            f"components must stay comparable ({contribs})")


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
