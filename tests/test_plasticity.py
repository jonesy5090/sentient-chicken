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
