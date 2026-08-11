"""Phase 0 acceptance tests.

The throughput test is the one to watch. Everything downstream of this phase depends
on the simulation staying far faster than real time; if a change makes the loop slow,
that is a correctness problem for the project even though nothing computes a wrong
answer.
"""

import jax
import jax.numpy as jnp
import pytest

from coop import sensing, spec, world
from hen import brain, connectome, neurons, regions
from run import probes, simulate

CFG = spec.DEFAULT_COOP


@pytest.fixture(scope="module")
def flock():
    key = jax.random.key(0)
    w = world.reset(key, CFG)
    p = connectome.build(jax.random.key(1), regions.DEFAULT_REGIONS,
                         n_hens=CFG.n_hens)
    x = brain.initial_state(p, CFG.n_hens)
    return w, x, p


# --- Structure ------------------------------------------------------------

def test_observation_layout_is_consistent():
    assert spec.OBS_DIM == spec.AUDIO_HI
    assert spec.vis_index(0, 0) == 0
    assert spec.vis_index(spec.N_BINS - 1, spec.N_VIS_CLASSES - 1) == spec.VIS_HI - 1
    assert len(spec.CALL_MOTOR_IDX) == spec.N_CALLS


def test_dale_law_holds(flock):
    """A neuron's outgoing weights must all share its sign."""
    _, _, p = flock
    w = p.W[0]
    for j in (0, 5, p.W.shape[-1] - 1):
        col = w[:, j]
        nz = col[col != 0.0]
        if nz.size:
            assert bool(jnp.all(jnp.sign(nz) == p.dale[j]))


def test_no_autapses(flock):
    _, _, p = flock
    assert not bool(jnp.any(jnp.diag(p.mask)))


def test_integration_is_stable_by_construction():
    ratio = neurons.stability_ratio(jnp.asarray(regions.REGION_TAU), CFG.dt)
    assert ratio < 0.5, f"dt/tau = {ratio}, forward Euler will misbehave"


# --- Dynamics -------------------------------------------------------------

def test_determinism(flock):
    """Same seed, same trajectory. Nothing downstream is debuggable without this."""
    w, x, p = flock
    k = jax.random.key(42)
    a = simulate.rollout_quiet(w, x, p, k, CFG, 200)
    b = simulate.rollout_quiet(w, x, p, k, CFG, 200)
    assert jnp.array_equal(a[0].pos, b[0].pos)
    assert jnp.array_equal(a[1], b[1])


def test_different_seeds_diverge(flock):
    """The RNG stream must actually reach the world.

    Predator arrivals are the only stochastic element once the flock is placed, and
    at the default 15-minute period none occurs in a short rollout -- two runs would
    then be identical for the right reason. Shorten the period so hawks arrive.
    """
    w, x, p = flock
    cfg = CFG._replace(hawk_period_s=2.0)
    a = simulate.rollout_quiet(w, x, p, jax.random.key(1), cfg, 2_000)
    b = simulate.rollout_quiet(w, x, p, jax.random.key(2), cfg, 2_000)
    assert not jnp.allclose(a[0].pos, b[0].pos)


def test_reset_seeds_differ():
    a = world.reset(jax.random.key(1), CFG)
    b = world.reset(jax.random.key(2), CFG)
    assert not jnp.allclose(a.pos, b.pos)


def test_long_run_stays_finite(flock):
    """Ten minutes of chicken time with no drift, saturation or NaN."""
    w, x, p = flock
    w2, x2, *_ = simulate.rollout_quiet(w, x, p, jax.random.key(3), CFG, 60_000)
    assert bool(jnp.all(jnp.isfinite(x2)))
    assert float(jnp.max(jnp.abs(x2))) < 50.0
    assert bool(jnp.all(jnp.isfinite(w2.pos)))
    assert bool(jnp.all((w2.pos >= 0.0) & (w2.pos <= CFG.size)))


def test_hens_stay_inside_the_run(flock):
    w, x, p = flock
    w2, *_ = simulate.rollout_quiet(w, x, p, jax.random.key(4), CFG, 5_000)
    assert float(jnp.min(w2.pos)) >= 0.0
    assert float(jnp.max(w2.pos)) <= CFG.size


# --- The head-down gate ---------------------------------------------------

def test_head_down_gates_the_aerial_channel(flock):
    """Unit-level check of the mechanism the whole language phase rests on."""
    w, _, _ = flock
    w = w._replace(hawk_pos=w.pos[0], hawk_on=jnp.array(1.0),
                   head_down=jnp.zeros((CFG.n_hens,)))
    up = sensing.observe(w, CFG)[0, spec.IDX_AERIAL]
    down = sensing.observe(
        w._replace(head_down=jnp.ones((CFG.n_hens,))), CFG)[0, spec.IDX_AERIAL]
    assert float(up) > 0.5
    assert float(down) == pytest.approx(0.0, abs=1e-6)


# --- Null control ---------------------------------------------------------

def test_blind_hen_does_nothing_purposeful(flock):
    """With a zeroed observation, no action channel fires except tonic walking.

    This is what makes the ethogram assays meaningful: their positives come from
    sensory drive, not from resting bias leaking through the motor sigmoid.
    """
    _, x, p = flock
    obs = jnp.zeros((CFG.n_hens, spec.OBS_DIM))
    _, motor = brain.step(x, obs, p, CFG.dt)
    for ch in (spec.M_PECK, spec.M_CROUCH, spec.M_FLEE, spec.M_SCRATCH,
               spec.M_CALL_CONTACT, spec.M_CALL_FOOD,
               spec.M_CALL_AERIAL, spec.M_CALL_GROUND):
        assert float(jnp.max(motor[:, ch])) < 0.5


# --- Behaviour ------------------------------------------------------------

@pytest.mark.parametrize("probe", probes.ALL, ids=lambda f: f.__name__)
def test_neonatal_ethogram(probe):
    result = probe(CFG)
    assert result.passed, result.detail


# --- Throughput -----------------------------------------------------------

def test_sustains_real_time_factor(flock):
    """Guard the project's binding constraint: developmental wall-clock time.

    The threshold is deliberately loose so the test does not flake on a loaded or
    shared machine. `python -m bench.envelope` is the real measurement.
    """
    import time
    w, x, p = flock
    simulate.rollout_quiet(w, x, p, jax.random.key(5), CFG, 200)[0].pos.block_until_ready()

    t0 = time.perf_counter()
    out = simulate.rollout_quiet(w, x, p, jax.random.key(5), CFG, 2_000)
    jax.block_until_ready((out[0].pos, out[1]))
    factor = 2_000 * CFG.dt / (time.perf_counter() - t0)

    assert factor > 5.0, f"only {factor:.1f}x real time; a rearing run would crawl"
