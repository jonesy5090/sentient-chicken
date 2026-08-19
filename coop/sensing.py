"""World -> observation vector. The optic tectum analogue.

The single most consequential line in this file is the head-down gate on the aerial
channel. A hen with her beak in the litter cannot see a hawk. That asymmetry is not
decoration: emergent-communication work is consistent that signalling only evolves
when a receiver lacks information the sender has. If every hen could always see the
sky, no alarm call would ever be worth making, however many neurons the bird was
given. The pressure for language lives in the environment, not the brain.
"""

import jax
import jax.numpy as jnp

from coop import spec
from coop.spec import CoopConfig

_HALF_FOV = jnp.deg2rad(spec.FIELD_OF_VIEW_DEG) / 2.0
_BIN_RAD = jnp.deg2rad(spec.BIN_WIDTH_DEG)


def _bin_proximity(pos, heading, ent_pos, valid, cfg: CoopConfig):
    """Nearest-object proximity per angular bin -> (H, N_BINS).

    `valid` broadcasts to (H, E): it masks entities that are absent (a hawk that is
    not there, a hen's view of herself).
    """
    rel = ent_pos[None, :, :] - pos[:, None, :]                 # (H, E, 2)
    d = jnp.linalg.norm(rel, axis=-1)                           # (H, E)

    ang = jnp.arctan2(rel[..., 1], rel[..., 0]) - heading[:, None]
    ang = (ang + jnp.pi) % (2 * jnp.pi) - jnp.pi

    prox = jnp.clip(1.0 - d / cfg.vision_range, 0.0, 1.0)
    prox = prox * (jnp.abs(ang) <= _HALF_FOV) * valid

    bin_idx = jnp.clip(((ang + _HALF_FOV) / _BIN_RAD).astype(jnp.int32),
                       0, spec.N_BINS - 1)
    onehot = jax.nn.one_hot(bin_idx, spec.N_BINS)               # (H, E, N_BINS)
    return jnp.max(onehot * prox[..., None], axis=1)            # (H, N_BINS)


def _place_cells(pos: jax.Array, cfg: CoopConfig) -> jax.Array:
    """(..., N_PLACE) allocentric place-cell activation -- a Gaussian bump per cell on
    a fixed grid tiling the arena, independent of heading. The one channel in this file
    that is *not* egocentric; see `coop/spec.py`'s module comment for why.

    `pos` may carry any number of leading batch dims (`(..., 2)`); used both for a
    hen's own location (`(H, 2)`, E063) and, batched over listener and caller together,
    for the gakel location cue below (`(H, H, 2)`, E064).
    """
    edges = jnp.linspace(0.0, cfg.size, spec.PLACE_GRID + 2)[1:-1]
    cx, cy = jnp.meshgrid(edges, edges, indexing="ij")
    centers = jnp.stack([cx.ravel(), cy.ravel()], axis=-1)      # (N_PLACE, 2)
    d2 = jnp.sum((pos[..., None, :] - centers) ** 2, axis=-1)
    return jnp.exp(-d2 / (2.0 * cfg.place_sigma ** 2))


def _gakel_location_cue(weight: jax.Array, caller_pos: jax.Array,
                        cfg: CoopConfig) -> jax.Array:
    """(H, N_PLACE) loudness-weighted mixture of gakel callers' own place-cell
    patterns, as heard by each listener -- see `coop/spec.py`'s module comment for the
    design reasoning. Deliberately *un*normalised: a faint, distant call contributes a
    faint, low-confidence pattern rather than a full-strength one, which is the
    "coarse" part of coarse directional hearing.

    `weight[i, j]` is how loudly listener i currently hears j's gakel call --
    `atten[i, j] * gakel_amplitude[j]`, already routed through `_channel` for the H4
    ladder and, under `channel_mode='yoked'`, already the per-listener lagged amplitude.
    `caller_pos[i, j]` is where j was when that call was made -- `w.pos` broadcast for
    the ordinary case, or the matching lagged slice of `w.pos_log` under 'yoked', so a
    yoked listener never receives a caller's *current* position alongside a
    time-shifted call (that would leak exactly the real-time contingency the control
    exists to destroy).
    """
    place = _place_cells(caller_pos, cfg)                        # (H, H, N_PLACE)
    return jnp.clip(jnp.einsum("ij,ijp->ip", weight, place), 0.0, 1.0)


def _channel(atten: jax.Array, w, cfg: CoopConfig) -> jax.Array:
    """Rewire who hears whom, for the H4 condition ladder. (H, H) -> (H, H).

    `atten[i, j]` is how loudly hen i hears hen j. Every mode below leaves the *cost*
    of calling untouched and changes only the routing, which is the point: the ladder
    has to vary information and nothing else.

    **`shuffled` is NOT A CONTROL, and the paragraph that used to sit here claiming
    otherwise is why E024 was invalid.** It said a reassigned sender carries "no
    information about *her* surroundings". Measured (E026): the permutation keeps 98%
    of the intact channel's correlation with "a hawk is on me". Every hen already
    hears every other -- `hear_range` 15 m in a 20 m arena, mean 15.0 of 15 audible --
    so any within-timestep permutation preserves "someone is calling right now", which
    in this coop is almost the whole signal. Retained only so E024 reproduces.

    `yoked` is the control: the flock's real calls, shifted in time per hen by more
    than a hawk dive. Same rate, amplitudes and cost; correlation -0.13 against
    intact's +0.56.
    """
    h = atten.shape[0]
    mode = cfg.channel_mode

    if mode == "intact":
        return atten
    if mode == "none" or mode == "severed":
        # Both deliver silence. They are meant to differ in whether she still *pays*
        # to call, which lives in `call_vigour_drain`, not here -- so as conditions
        # they are only distinct if the caller sets that too. E024 ran them as separate
        # rungs and got bit-identical results on all 8 seeds (E026); its prediction
        # "C? ~ C0" was scored against a duplicate row.
        return jnp.zeros_like(atten)
    if mode == "self":
        # Cs: only her own voice, at the amplitude a flockmate beside her would hear.
        # `atten` has the diagonal suppressed (world.py adds 1e6 to self-distance),
        # so it is restored explicitly rather than read back off the matrix.
        return jnp.eye(h)
    if mode == "yoked":
        # THE CONTROL. She hears the flock's real calls, shifted in time by a per-hen
        # lag longer than a hawk dive. Identical rate, amplitude distribution, burst
        # structure and cost; no contingency with her current world.
        #
        # A within-timestep permutation cannot do this. Every hen already hears every
        # other (hear_range 15 m in a 20 m arena), so permuting the sender preserves
        # "someone is calling right now" -- which in this coop is almost the whole
        # signal. Measured (E026), correlation with "a hawk is on me":
        #     intact 0.5595 | permuted 0.5496 (98% kept) | yoked -0.1288 (destroyed)
        # The surviving component is temporal, which is why no amount of dispersal
        # rescued it.
        #
        # Implemented at the caller: `world.step` keeps a ring buffer of past calls and
        # hands `sensing.observe` the lagged slice, because the lag has to reach back
        # further than one step. Here the routing is simply intact -- the shift has
        # already been applied to `w.calls`.
        return atten
    if mode == "shuffled":
        epoch = jnp.floor(w.t * cfg.dt / cfg.shuffle_period_s).astype(jnp.int32)
        perm = jax.random.permutation(jax.random.fold_in(jax.random.key(0xC0FFEE),
                                                         epoch), h)
        # Row i keeps its own attenuation profile and receives hen perm[i]'s voices,
        # so loudness statistics are preserved and only the source identity changes.
        return atten[perm]
    raise ValueError(f"unknown channel_mode {mode!r}")


def observe(w, cfg: CoopConfig = spec.DEFAULT_COOP) -> jax.Array:
    """Build the (H, OBS_DIM) observation for the whole flock."""
    h = cfg.n_hens

    # --- Lateral vision: five classes across twelve bins ---
    food = _bin_proximity(w.pos, w.heading, w.food_pos,
                          (w.food_amount > 0.01)[None, :], cfg)
    water = _bin_proximity(w.pos, w.heading, w.water_pos,
                           jnp.ones((1, cfg.n_water)), cfg)
    flock = _bin_proximity(w.pos, w.heading, w.pos,
                           1.0 - jnp.eye(h), cfg)
    threat = _bin_proximity(w.pos, w.heading, w.fox_pos[None, :],
                            w.fox_on[None, None], cfg)
    # Personal space (E025). Derived from `flock`, not a new distance computation --
    # zero until a flockmate is well inside PERSONAL_SPACE_THRESHOLD of the vision
    # range, then ramps to 1 at contact. The reflex arc wires this to turn *away*,
    # opposing CLS_FLOCKMATE's turn-toward with a larger weight.
    crowding = jnp.clip(
        (flock - spec.PERSONAL_SPACE_THRESHOLD) / (1.0 - spec.PERSONAL_SPACE_THRESHOLD),
        0.0, 1.0)
    # Sick-flockmate visibility (T2, E025 location fix, E060). Same _bin_proximity
    # computation as `flock`, gated to only currently-sick hens -- the visual referent
    # that lets a flockmate locate *where* something bad happened, which the gakel
    # call alone cannot carry (the same acoustic-location gap already flagged for the
    # ordinary food call).
    sick = _bin_proximity(w.pos, w.heading, w.pos,
                          (1.0 - jnp.eye(h)) * w.sick_on[None, :], cfg)

    vis = jnp.stack([food, water, flock, threat, crowding, sick], axis=-1)  # (H,N_BINS,6)
    vis = vis.reshape(h, spec.N_BINS * spec.N_VIS_CLASSES)      # index = bin*6 + cls

    # --- Overhead channel, gated by posture ---
    d_hawk = jnp.linalg.norm(w.pos - w.hawk_pos[None, :], axis=-1)
    aerial = jnp.clip(1.0 - d_hawk / cfg.vision_range, 0.0, 1.0) * w.hawk_on
    aerial = aerial * (1.0 - w.head_down)        # <-- the vigilance/foraging trade-off

    # --- Interoception ---
    d_hens = jnp.linalg.norm(w.pos[:, None, :] - w.pos[None, :, :], axis=-1)
    d_hens = d_hens + jnp.eye(h) * 1e6
    isolation = jnp.clip(jnp.min(d_hens, axis=-1) / 3.0, 0.0, 1.0)
    intero = jnp.stack([w.hunger, w.thirst, w.cold, isolation, w.food_call_drive,
                        w.sick_call_drive], axis=-1)

    # --- Somatic ---
    d_wall = jnp.min(jnp.minimum(w.pos, cfg.size - w.pos), axis=-1)
    wall = jnp.clip(1.0 - d_wall / 1.0, 0.0, 1.0)

    # Directional wall-escape (wall-avoidance fix). `wall` has no direction, so pick the
    # nearest wall's outward normal as the escape heading, express it relative to the
    # hen's own heading (the same frame `_bin_proximity` uses for everything else), and
    # split into two non-negative "turn this way" magnitudes.
    margins = jnp.stack([w.pos[:, 0], cfg.size - w.pos[:, 0],
                         w.pos[:, 1], cfg.size - w.pos[:, 1]], axis=-1)   # (H,4): L,R,B,T
    nearest = jnp.argmin(margins, axis=-1)
    normals = jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    escape_dir = normals[nearest]                                        # (H, 2)
    escape_ang = jnp.arctan2(escape_dir[:, 1], escape_dir[:, 0]) - w.heading
    escape_ang = (escape_ang + jnp.pi) % (2 * jnp.pi) - jnp.pi
    wall_escape_l = wall * jnp.clip(jnp.sign(escape_ang), 0.0, 1.0)
    wall_escape_r = wall * jnp.clip(-jnp.sign(escape_ang), 0.0, 1.0)

    somatic = jnp.stack([wall, w.speed / cfg.flee_speed,
                         wall_escape_l, wall_escape_r], axis=-1)

    # --- Place cells: allocentric, independent of channel_mode and everything
    # auditory below -- location is not a call, and does not get rewired by the H4
    # ladder. ---
    place = (_place_cells(w.pos, cfg) if cfg.place_cells_enabled
             else jnp.zeros((h, spec.N_PLACE)))

    # --- Audition: flockmates' calls, attenuated by distance ---
    #
    # Voices combine in *intensity*, not amplitude: two birds calling at 0.5 are not
    # as loud as one at 1.0. Summing amplitudes linearly and clipping -- which is what
    # this did until E019 -- means fifteen flockmates saturate the channel between
    # them, and a real call from the bird beside you adds nothing measurable. Summing
    # power and taking the root keeps a genuine call dominant over any amount of
    # background, while still letting a chorus be louder than a soloist.
    atten = jnp.clip(1.0 - d_hens / cfg.hear_range, 0.0, 1.0)   # self excluded by 1e6
    atten = _channel(atten, w, cfg)

    if cfg.channel_mode == "yoked":
        # Each receiver hears the whole flock as it was `lag_i` ago. Lags are fixed per
        # hen and spread across the buffer so no two receivers share a timeline, and
        # every lag exceeds a hawk dive -- otherwise the shifted stream still overlaps
        # the event it is meant to be decorrelated from.
        if cfg.call_log_steps < spec.YOKE_LOG_STEPS:
            raise ValueError(
                "channel_mode='yoked' needs cfg.call_log_steps=spec.YOKE_LOG_STEPS; "
                f"got {cfg.call_log_steps}. The buffer is off by default because it "
                "costs throughput the other conditions should not pay for.")
        lag0 = int(cfg.yoke_min_lag_s / cfg.dt)
        span = cfg.call_log_steps - lag0
        lags = lag0 + (jnp.arange(h) * (span // max(h, 1))) % max(span, 1)
        idx = (w.t - lags) % cfg.call_log_steps                 # (H,)
        heard = w.call_log[idx]                                 # (H, H, N_CALLS)
        heard_pos = w.pos_log[idx]                               # (H, H, 2)
        power = jnp.einsum("ij,ijc->ic", atten ** 2, heard ** 2)
        gakel_weight = atten * heard[:, :, spec.GAKEL_CALL_IDX]
        gakel_cue = (_gakel_location_cue(gakel_weight, heard_pos, cfg)
                     if cfg.place_cells_enabled else jnp.zeros((h, spec.N_PLACE)))
        return jnp.concatenate(
            [vis, aerial[:, None], intero, somatic,
             jnp.clip(jnp.sqrt(power), 0.0, 1.0), place, gakel_cue], axis=-1)

    if cfg.legacy_audio:
        audio = jnp.clip(atten @ w.calls, 0.0, 1.0)             # pre-E019, for E021
    else:
        audio = jnp.clip(jnp.sqrt((atten ** 2) @ (w.calls ** 2)), 0.0, 1.0)

    gakel_weight = atten * w.calls[:, spec.GAKEL_CALL_IDX][None, :]
    caller_pos = jnp.broadcast_to(w.pos[None, :, :], (h, h, 2))
    gakel_cue = (_gakel_location_cue(gakel_weight, caller_pos, cfg)
                 if cfg.place_cells_enabled else jnp.zeros((h, spec.N_PLACE)))

    return jnp.concatenate(
        [vis, aerial[:, None], intero, somatic, audio, place, gakel_cue], axis=-1)


def observability(w, cfg: CoopConfig = spec.DEFAULT_COOP) -> jax.Array:
    """(H, OBS_DIM) mask: 1 where a channel carries information this step, 0 where not.

    The head-down gate does not make the aerial channel *zero*, it makes it
    *unobserved* — she is not seeing an empty sky, she is not looking. That
    distinction matters the moment anything tries to learn to predict the channel:
    training a predictor on censored samples teaches it that calls mean no hawk,
    which is the opposite of the association we want it to form.
    """
    h = cfg.n_hens
    mask = jnp.ones((h, spec.OBS_DIM))
    return mask.at[:, spec.IDX_AERIAL].set(1.0 - w.head_down)
