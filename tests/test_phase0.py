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
from hen import brain, connectome, innate, neurons, regions
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
    assert spec.OBS_DIM == spec.GAKEL_PLACE_HI
    assert spec.PLACE_LO == spec.AUDIO_HI
    assert spec.GAKEL_PLACE_LO == spec.PLACE_HI
    assert spec.GAKEL_PLACE_HI - spec.GAKEL_PLACE_LO == spec.N_PLACE
    assert spec.N_PLACE == spec.PLACE_GRID ** 2
    assert spec.CALL_MOTOR_IDX[spec.GAKEL_CALL_IDX] == spec.M_CALL_GAKEL
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


# --- Personal space (E025) -------------------------------------------------

def test_crowding_channel_activates_only_inside_personal_space(flock):
    """CLS_CROWDING must stay exactly zero at ordinary flocking distance and turn on
    once a flockmate is well inside PERSONAL_SPACE_THRESHOLD.

    Isolates hens 0 and 1 and parks the rest of the flock far away, since with 16
    hens at their reset positions some other bird could otherwise be close enough
    to hen 0 to contaminate the reading.
    """
    w, _, _ = flock
    far_corner = jnp.full((CFG.n_hens - 2, 2), CFG.size - 0.5)
    pos = w.pos.at[0].set(jnp.array([5.0, 5.0])).at[2:].set(far_corner)
    heading = jnp.zeros((CFG.n_hens,))

    def max_crowding(sep):
        p = pos.at[1].set(jnp.array([5.0 + sep, 5.0]))
        obs = sensing.observe(w._replace(pos=p, heading=heading), CFG)
        return max(float(obs[0, spec.vis_index(b, spec.CLS_CROWDING)])
                  for b in range(spec.N_BINS))

    assert max_crowding(2.0) == pytest.approx(0.0, abs=1e-6)   # ordinary flocking range
    assert max_crowding(0.2) > 0.5                            # well inside contact range


def test_personal_space_reflex_dominates_attraction_at_contact():
    """The repulsion weight must exceed the attraction weight, or CLS_CROWDING only
    damps CLS_FLOCKMATE's pull instead of reversing it -- a linear reflex arc cannot
    produce attract-then-repel any other way. See the derivation in hen/innate.py.
    """
    r = innate.reflex_matrix()
    for b in innate._LEFT:
        attract = r[spec.M_TURN_L, spec.vis_index(b, spec.CLS_FLOCKMATE)]
        repel = r[spec.M_TURN_R, spec.vis_index(b, spec.CLS_CROWDING)]
        assert repel > attract
    for b in innate._RIGHT:
        attract = r[spec.M_TURN_R, spec.vis_index(b, spec.CLS_FLOCKMATE)]
        repel = r[spec.M_TURN_L, spec.vis_index(b, spec.CLS_CROWDING)]
        assert repel > attract


# --- Food-discovery pulse (E053) --------------------------------------------

def test_food_call_drive_spikes_on_arrival_and_decays():
    """IDX_FOOD_ARRIVAL must fire only on the rising edge of reaching a food patch,
    not stay high while a hen remains there -- the whole point of the fix.
    """
    cfg = CFG._replace(n_hens=1)
    w = world.reset(jax.random.key(0), cfg)
    w = w._replace(pos=jnp.array([[10.0, 10.0]]), heading=jnp.array([0.0]),
                   food_pos=jnp.array([[10.05, 10.0]]))

    # Step 1: she is already at the patch and was not there before (at_food_prev=0
    # from reset) -- this must be the rising edge.
    motor = jnp.zeros((1, spec.MOTOR_DIM))
    w1 = world.step(w, motor, jax.random.key(1), cfg)
    assert float(w1.food_call_drive[0]) == pytest.approx(1.0, abs=1e-6)

    # Hold her there for well past the decay window; drive must fall close to zero
    # despite continuous presence.
    wn = w1
    for t in range(500):
        wn = world.step(wn, motor, jax.random.fold_in(jax.random.key(1), t), cfg)
    assert float(wn.food_call_drive[0]) < 0.05

    obs = sensing.observe(w1, cfg)
    assert float(obs[0, spec.IDX_FOOD_ARRIVAL]) == pytest.approx(1.0, abs=1e-6)


# --- Wall avoidance ---------------------------------------------------------

def test_wall_escape_channels_point_away_from_the_nearest_wall(flock):
    """IDX_WALL_ESCAPE_L/R must fire on the side that turns the hen away from
    whichever wall she is nearest, and stay zero away from all walls.
    """
    w, _, _ = flock

    def escape(pos, heading):
        p = w.pos.at[0].set(jnp.array(pos))
        h = w.heading.at[0].set(heading)
        obs = sensing.observe(w._replace(pos=p, heading=h), CFG)
        return float(obs[0, spec.IDX_WALL_ESCAPE_L]), float(obs[0, spec.IDX_WALL_ESCAPE_R])

    # Near the left wall (x=0); escaping means heading 0 (+x). Facing +y (heading
    # pi/2), escape is a right turn (clockwise, south of due-east); facing -y
    # (heading -pi/2), it's a left turn. Facing exactly 0 or pi is the boundary case
    # (already escaping, or a coin flip) and deliberately not tested here.
    l, r = escape([0.1, 10.0], jnp.pi / 2)
    assert r > 0.5 and l == pytest.approx(0.0, abs=1e-6)

    l, r = escape([0.1, 10.0], -jnp.pi / 2)
    assert l > 0.5 and r == pytest.approx(0.0, abs=1e-6)

    # Middle of the arena, far from every wall: both channels off.
    l, r = escape([10.0, 10.0], 0.0)
    assert l == pytest.approx(0.0, abs=1e-6) and r == pytest.approx(0.0, abs=1e-6)


def test_wall_escape_reflex_turns_a_cornered_hen_away(flock):
    """End-to-end: a hen 0.1 m from a wall, facing straight into it, must be
    measurably further from it a few seconds later -- the reflex, not just the
    channel, has to work. No food/water/flockmates/predators to confound it.
    """
    w, x, p = flock
    cfg = CFG._replace(hawk_period_s=1e9, ground_pred_period_s=1e9)
    far = jnp.array([cfg.size - 1, cfg.size - 1])
    w0 = w._replace(
        pos=w.pos.at[0].set(jnp.array([0.1, cfg.size / 2])),
        heading=w.heading.at[0].set(jnp.pi),
        food_pos=jnp.tile(far, (cfg.n_food, 1)),
        water_pos=jnp.tile(far, (cfg.n_water, 1)))
    w1, *_ = simulate.rollout_quiet(w0, x, p, jax.random.key(6), cfg, 4_000)
    assert float(w1.pos[0, 0]) > 0.3   # started at 0.1 m


# --- T2 contamination/sickness scaffold (E060) ------------------------------

def test_contamination_only_changes_on_epoch_transition():
    """A real bug caught while building this: an earlier version recomputed
    `food_contaminated` unconditionally every step, silently overriding any staged
    value back to whatever epoch 0 resolves to. It must persist within an epoch
    (the hawk_on/hawk_t pattern -- state, not a recomputation) so it can be staged.
    """
    cfg = CFG._replace(contamination_enabled=True, n_hens=1, n_food=1, contamination_period_s=300.0)
    w = world.reset(jax.random.key(0), cfg)
    w = w._replace(food_contaminated=jnp.array([False]))   # stage: clean, epoch 0
    motor = jnp.zeros((1, spec.MOTOR_DIM))
    for t in range(50):
        w = world.step(w, motor, jax.random.fold_in(jax.random.key(1), t), cfg)
    assert bool(w.food_contaminated[0]) is False, (
        "staged contamination was overwritten within the same epoch")


def test_sick_channel_only_fires_for_sick_flockmates(flock):
    """CLS_SICK must be exactly zero for a healthy flockmate and nonzero for a
    sick one, mirroring CLS_CROWDING's own gating test.
    """
    w, _, _ = flock
    far_corner = jnp.full((CFG.n_hens - 2, 2), CFG.size - 0.5)
    pos = w.pos.at[0].set(jnp.array([5.0, 5.0])).at[1].set(jnp.array([6.0, 5.0])) \
                .at[2:].set(far_corner)
    heading = jnp.zeros((CFG.n_hens,))

    def max_sick(sick):
        sick_on = jnp.zeros((CFG.n_hens,), dtype=bool).at[1].set(sick)
        obs = sensing.observe(w._replace(pos=pos, heading=heading, sick_on=sick_on),
                              CFG)
        return max(float(obs[0, spec.vis_index(b, spec.CLS_SICK)])
                  for b in range(spec.N_BINS))

    assert max_sick(False) == pytest.approx(0.0, abs=1e-6)
    assert max_sick(True) > 0.5


def test_avoid_sick_reflex_dominates_attraction():
    """The repulsion weight from CLS_SICK must exceed CLS_FLOCKMATE's attraction
    weight, or a sick flockmate only damps the pull toward her instead of reversing
    it -- the same algebraic requirement CLS_CROWDING has, for the same reason (a
    linear reflex arc cannot produce attract-then-repel from one channel alone).
    """
    r = innate.reflex_matrix()
    for b in innate._LEFT:
        attract = r[spec.M_TURN_L, spec.vis_index(b, spec.CLS_FLOCKMATE)]
        repel = r[spec.M_TURN_R, spec.vis_index(b, spec.CLS_SICK)]
        assert repel > attract
    for b in innate._RIGHT:
        attract = r[spec.M_TURN_R, spec.vis_index(b, spec.CLS_FLOCKMATE)]
        repel = r[spec.M_TURN_L, spec.vis_index(b, spec.CLS_SICK)]
        assert repel > attract


def test_sickness_onset_sets_timer_and_decays_but_outlasts_the_call():
    """Eating contaminated food must set a bounded sickness timer and spike the
    gakel-call drive together, on the same step -- and the sickness timer must
    clearly outlast the call's own short decay (E060's design: a discovery pulse for
    the call, a much longer physiological state for the sickness itself).
    """
    cfg = CFG._replace(contamination_enabled=True, n_hens=1, n_food=1)
    w = world.reset(jax.random.key(0), cfg)
    w = w._replace(pos=jnp.array([[10.0, 10.0]]), heading=jnp.array([0.0]),
                   food_pos=jnp.array([[10.05, 10.0]]),
                   food_contaminated=jnp.array([True]))
    motor = jnp.zeros((1, spec.MOTOR_DIM)).at[0, spec.M_PECK].set(1.0)

    w1 = world.step(w, motor, jax.random.key(1), cfg)
    assert float(w1.sick_t[0]) == pytest.approx(cfg.sickness_duration_s, abs=1e-6)
    assert bool(w1.sick_on[0])
    assert float(w1.sick_call_drive[0]) == pytest.approx(1.0, abs=1e-6)

    wn = w1
    for t in range(int(cfg.gakel_call_decay_s / cfg.dt) + 50):
        wn = world.step(wn, motor, jax.random.fold_in(jax.random.key(1), t), cfg)
    assert float(wn.sick_call_drive[0]) < 0.05   # the call has decayed
    assert bool(wn.sick_on[0])                   # she is still sick


# --- Allocentric place cells (T2 Stage 2 prerequisite, E063) ----------------

def test_place_cells_peak_near_the_nearest_grid_center():
    """A hen standing exactly on a grid centre must read (near) peak activation on
    that cell and materially less on every other -- the basic geometry claim, checked
    directly rather than assumed.
    """
    cfg = CFG._replace(place_cells_enabled=True, n_hens=1)
    w = world.reset(jax.random.key(0), cfg)
    edges = jnp.linspace(0.0, cfg.size, spec.PLACE_GRID + 2)[1:-1]
    center = jnp.array([edges[0], edges[0]])
    w = w._replace(pos=center[None, :], heading=jnp.zeros((1,)))
    place = sensing.observe(w, cfg)[0, spec.PLACE_LO:spec.PLACE_HI]
    assert float(place[0]) == pytest.approx(1.0, abs=1e-5)
    assert float(jnp.sort(place)[-2]) < 0.5   # every other cell clearly lower


def test_place_cells_are_independent_of_heading():
    """The whole point of this channel is that it survives a change of heading, unlike
    every egocentric bin in `vis`. Rotating a hen in place must not move her place-cell
    pattern at all.
    """
    cfg = CFG._replace(place_cells_enabled=True, n_hens=1)
    w = world.reset(jax.random.key(0), cfg)
    w = w._replace(pos=jnp.array([[8.3, 12.7]]))
    p0 = sensing.observe(w._replace(heading=jnp.array([0.0])), cfg)[
        0, spec.PLACE_LO:spec.PLACE_HI]
    p1 = sensing.observe(w._replace(heading=jnp.array([2.1])), cfg)[
        0, spec.PLACE_LO:spec.PLACE_HI]
    assert jnp.allclose(p0, p1, atol=1e-6)


def test_place_cells_discriminate_distinct_locations():
    """A positive control (CLAUDE.md's instrument discipline): two hens on opposite
    sides of the arena must produce measurably different place-cell patterns, or the
    channel carries no usable location information regardless of what plasticity does
    with it.
    """
    cfg = CFG._replace(place_cells_enabled=True, n_hens=2)
    w = world.reset(jax.random.key(0), cfg)
    w = w._replace(pos=jnp.array([[2.0, 2.0], [18.0, 18.0]]),
                   heading=jnp.zeros((2,)))
    place = sensing.observe(w, cfg)[:, spec.PLACE_LO:spec.PLACE_HI]
    corr = jnp.corrcoef(place[0], place[1])[0, 1]
    assert corr < 0.1, "opposite corners should not share a place-cell pattern"




# --- Shared allocentric map (T2-revised mechanism 2) ------------------------

def test_shared_place_map_transfers_by_place_and_only_by_place():
    """Testimony about place P must evoke the same representation as *being* at P,
    and must not evoke the representation of any other place.

    The first half is what makes the association transferable at all: `W_pred` sources
    from the pallium, so a binding learned while hearing about P is written onto units
    that must still be active when she later walks to P. Without shared afferents there
    is no overlap and no transfer.

    The second half is what makes it a *map* rather than a blanket merge of the two
    channels. If testimony about P also lit up place Q, the hen would learn that
    everywhere is dangerous -- which would look like success on any aggregate metric
    while being the opposite of the referential claim T2 exists to test.

    Magnitude is checked too: the patterns must be congruent but not identical, or a
    hen could not distinguish "I am here" from "someone called from there".
    """
    import numpy as np

    def stub_response(p, block_lo, k):
        obs = jnp.zeros((1, spec.OBS_DIM)).at[0, block_lo + k].set(1.0)
        return np.asarray((obs @ p.W_in.T)[0])

    off = connectome.build(jax.random.key(0), regions.DEFAULT_REGIONS, n_hens=1)
    on = connectome.build(jax.random.key(0), regions.DEFAULT_REGIONS, n_hens=1,
                          shared_place_map=True, testimony_gain=0.5)

    # Off by default: the flag must change nothing unless asked for.
    assert jnp.array_equal(off.W_in, connectome.build(
        jax.random.key(0), regions.DEFAULT_REGIONS, n_hens=1,
        shared_place_map=False).W_in)

    same, cross, ratios = [], [], []
    for k in range(spec.N_PLACE):
        self_k = stub_response(on, spec.PLACE_LO, k)
        test_k = stub_response(on, spec.GAKEL_PLACE_LO, k)
        if self_k.std() > 0 and test_k.std() > 0:
            same.append(np.corrcoef(self_k, test_k)[0, 1])
            ratios.append(test_k.sum() / max(self_k.sum(), 1e-9))
        # a different place, chosen far away on the grid
        other = (k + spec.PLACE_GRID // 2 + 1) % spec.N_PLACE
        self_other = stub_response(on, spec.PLACE_LO, other)
        if self_other.std() > 0 and test_k.std() > 0:
            cross.append(np.corrcoef(self_other, test_k)[0, 1])

    assert np.mean(same) > 0.99, (
        f"testimony about P must evoke P's own representation; got {np.mean(same):.3f}")
    assert np.mean(same) - np.mean(cross) > 0.5, (
        f"testimony about P is nearly as similar to other places ({np.mean(cross):.3f}) "
        f"as to P itself ({np.mean(same):.3f}) -- this is a blanket merge, not a map")
    assert np.mean(ratios) == pytest.approx(0.5, abs=0.02), (
        "testimony must be weaker than first-hand experience, or the two are "
        "indistinguishable once summed")


# --- Gakel withdrawal scaffold (T2-revised mechanism 1) ---------------------

def test_gakel_scaffold_is_off_by_default_and_narrowly_scoped():
    """It must change nothing unless asked for, and when asked for must touch only
    the gakel audio channel.

    The second half is the load-bearing part. This scaffold is the single behavioural
    anchor T2-revised adds, and its whole claim to not smuggling in the answer is that
    it is wired to the *call* and to nothing else -- in particular not to the place
    channels, which E063 deliberately left unwired so that the location-specific part
    stays genuinely learned.
    """
    off = innate.reflex_matrix()
    assert jnp.array_equal(jnp.asarray(off),
                           jnp.asarray(innate.reflex_matrix(gakel_scaffold=False)))

    on = innate.reflex_matrix(gakel_scaffold=True)
    gakel = spec.AUDIO_LO + spec.CALL_MOTOR_IDX.index(spec.M_CALL_GAKEL)

    assert on[spec.M_PECK, gakel] < 0.0, "must suppress ingestion"

    # And must NOT touch locomotion (E083, correcting E082's finding). Damping
    # M_FORWARD makes a hen who is already at the aversive place stay at it, because
    # actuation.py derives speed from it -- lingering, where avoidance needs leaving.
    # It is also a functional freeze, which the anti-predator clause below forbids by
    # another route. Departure is meant to come via hunger: she keeps walking, declines
    # to eat, stays hungry, and hunger drives M_FORWARD harder.
    assert on[spec.M_FORWARD, gakel] == off[spec.M_FORWARD, gakel], (
        "the scaffold suppressed forward drive; that produces lingering, not avoidance")

    # No other call channel moves -- the response is to this call, not to hearing.
    for i in range(spec.N_CALLS):
        ch = spec.AUDIO_LO + i
        if ch != gakel:
            assert jnp.array_equal(jnp.asarray(off[:, ch]), jnp.asarray(on[:, ch])), (
                f"audio channel {i} changed; the scaffold must be gakel-specific")

    # Nothing location-specific anywhere, which is the anti-question-begging claim.
    assert jnp.array_equal(jnp.asarray(off[:, spec.PLACE_LO:]),
                           jnp.asarray(on[:, spec.PLACE_LO:])), (
        "the scaffold touched a place channel; which places are aversive must be "
        "learned, never wired")

    # And no anti-predator borrowing -- bad food is not a hawk.
    for m in (spec.M_CROUCH, spec.M_FLEE):
        assert on[m, gakel] == off[m, gakel]


# --- Gakel-call location cue (T2 Stage 2 prerequisite, E064) ----------------

def test_gakel_cue_is_zero_when_nobody_is_calling():
    """No gakel call anywhere must mean no location cue anywhere -- the channel should
    never manufacture a signal from silence.
    """
    cfg = CFG._replace(place_cells_enabled=True, n_hens=3)
    w = world.reset(jax.random.key(0), cfg)
    obs = sensing.observe(w, cfg)
    assert float(obs[:, spec.GAKEL_PLACE_LO:spec.GAKEL_PLACE_HI].max()) == pytest.approx(
        0.0, abs=1e-6)


def test_gakel_cue_points_at_the_caller_not_the_listener():
    """A listener's cue must peak at the *caller's* place-cell pattern, not her own --
    the entire point of this channel over the plain audio amplitude it's built from.
    """
    cfg = CFG._replace(place_cells_enabled=True, n_hens=2)
    w = world.reset(jax.random.key(0), cfg)
    # (2,2) to (12,12) is ~14.1 m apart -- within hear_range=15.0 so the call is
    # actually audible, but far enough that the two hens map to clearly distinct cells.
    w = w._replace(pos=jnp.array([[2.0, 2.0], [12.0, 12.0]]), heading=jnp.zeros((2,)),
                   calls=jnp.zeros((2, spec.N_CALLS)).at[1, spec.GAKEL_CALL_IDX].set(0.9))
    cue = sensing.observe(w, cfg)[:, spec.GAKEL_PLACE_LO:spec.GAKEL_PLACE_HI]
    truth_caller = sensing._place_cells(w.pos[1:2], cfg)[0]
    truth_listener = sensing._place_cells(w.pos[0:1], cfg)[0]
    assert int(cue[0].argmax()) == int(truth_caller.argmax())
    assert int(cue[0].argmax()) != int(truth_listener.argmax())


def test_gakel_cue_fades_with_distance():
    """A more distant caller must produce a weaker (not just differently-located) cue
    -- the "coarse" part of coarse directional hearing, and the reason this channel
    needs no artificial noise injected to be honestly imprecise.
    """
    cfg = CFG._replace(place_cells_enabled=True, n_hens=3)
    w = world.reset(jax.random.key(0), cfg)
    calls = jnp.zeros((3, spec.N_CALLS)).at[1, spec.GAKEL_CALL_IDX].set(0.9) \
                                        .at[2, spec.GAKEL_CALL_IDX].set(0.9)
    w = w._replace(pos=jnp.array([[10.0, 10.0], [11.0, 10.0], [14.0, 10.0]]),
                   heading=jnp.zeros((3,)), calls=calls)
    cue = sensing.observe(w, cfg)[:, spec.GAKEL_PLACE_LO:spec.GAKEL_PLACE_HI]
    near_strength = float(cue[0].max())  # listener 0 re: caller 1, 1 m away
    # isolate caller 2's own contribution by re-running with only she calling
    w2 = w._replace(calls=jnp.zeros((3, spec.N_CALLS)).at[2, spec.GAKEL_CALL_IDX].set(0.9))
    far_strength = float(
        sensing.observe(w2, cfg)[:, spec.GAKEL_PLACE_LO:spec.GAKEL_PLACE_HI][0].max())
    assert far_strength < near_strength


def test_yoked_gakel_cue_uses_the_callers_position_when_she_called_not_now():
    """The correctness property this whole channel exists to get right: under
    `channel_mode='yoked'`, a listener must be cued to where the caller *was* when she
    called, never her current position -- handing over the current one would leak
    exactly the real-time contingency the yoked control exists to destroy (the same
    class of leak E024's shuffled control had for plain audibility).
    """
    cfg = CFG._replace(place_cells_enabled=True, n_hens=2, channel_mode="yoked", call_log_steps=spec.YOKE_LOG_STEPS,
                       yoke_min_lag_s=1.0)
    key = jax.random.key(0)
    w = world.reset(key, cfg)
    w = w._replace(pos=jnp.array([[10.0, 10.0], [2.0, 2.0]]), heading=jnp.zeros((2,)))
    silent = jnp.zeros((2, spec.MOTOR_DIM))
    calling = jnp.zeros((2, spec.MOTOR_DIM)).at[1, spec.M_CALL_GAKEL].set(1.0)

    w = world.step(w, calling, jax.random.fold_in(key, 1), cfg)   # logs (~2, 2)
    w = w._replace(pos=w.pos.at[1].set(jnp.array([18.0, 18.0])))   # she then moves far

    lag0 = int(cfg.yoke_min_lag_s / cfg.dt)
    for t in range(lag0 - int(w.t)):
        w = world.step(w, silent, jax.random.fold_in(key, t + 2), cfg)

    cue = sensing.observe(w, cfg)[:, spec.GAKEL_PLACE_LO:spec.GAKEL_PLACE_HI]
    truth_old = sensing._place_cells(jnp.array([[2.0, 2.0]]), cfg)[0]
    truth_new = sensing._place_cells(jnp.array([[18.0, 18.0]]), cfg)[0]
    assert int(cue[0].argmax()) == int(truth_old.argmax())
    assert float(cue[0, int(truth_new.argmax())]) < 1e-6


# --- Null control ---------------------------------------------------------

def test_blind_hen_does_nothing_purposeful(flock):
    """With a zeroed observation, no action channel fires except tonic walking.

    This is what makes the ethogram assays meaningful: their positives come from
    sensory drive, not from resting bias leaking through the motor sigmoid.
    """
    _, x, p = flock
    obs = jnp.zeros((CFG.n_hens, spec.OBS_DIM))
    _, motor, _ = brain.step(x, obs, p, CFG.dt)
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
