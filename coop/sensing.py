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


def _channel(atten: jax.Array, w, cfg: CoopConfig) -> jax.Array:
    """Rewire who hears whom, for the H4 condition ladder. (H, H) -> (H, H).

    `atten[i, j]` is how loudly hen i hears hen j. Every mode below leaves the *cost*
    of calling untouched and changes only the routing, which is the point: the ladder
    has to vary information and nothing else.

    The shuffle is the load-bearing one. It gives each hen a randomly reassigned
    flockmate's voice at her own attenuation profile, so she receives the same number
    of calls, at the same amplitudes, with the same statistics -- carrying no
    information about *her* surroundings. If an intact flock beats a shuffled one, it
    is information transfer doing the work and not the mere presence of a channel.

    The permutation is re-drawn every `shuffle_period_s`. A fixed permutation is a
    stable mapping, and a stable mapping is something a flock could in principle learn
    to invert; re-drawing keeps the channel genuinely uninformative rather than merely
    scrambled once.
    """
    h = atten.shape[0]
    mode = cfg.channel_mode

    if mode == "intact":
        return atten
    if mode == "none" or mode == "severed":
        # C0 severs at the receiver, not the sender: she still pays to call.
        return jnp.zeros_like(atten)
    if mode == "self":
        # Cs: only her own voice, at the amplitude a flockmate beside her would hear.
        # `atten` has the diagonal suppressed (world.py adds 1e6 to self-distance),
        # so it is restored explicitly rather than read back off the matrix.
        return jnp.eye(h)
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

    # --- Lateral vision: four classes across twelve bins ---
    food = _bin_proximity(w.pos, w.heading, w.food_pos,
                          (w.food_amount > 0.01)[None, :], cfg)
    water = _bin_proximity(w.pos, w.heading, w.water_pos,
                           jnp.ones((1, cfg.n_water)), cfg)
    flock = _bin_proximity(w.pos, w.heading, w.pos,
                           1.0 - jnp.eye(h), cfg)
    threat = _bin_proximity(w.pos, w.heading, w.fox_pos[None, :],
                            w.fox_on[None, None], cfg)

    vis = jnp.stack([food, water, flock, threat], axis=-1)      # (H, N_BINS, 4)
    vis = vis.reshape(h, spec.N_BINS * spec.N_VIS_CLASSES)      # index = bin*4 + cls

    # --- Overhead channel, gated by posture ---
    d_hawk = jnp.linalg.norm(w.pos - w.hawk_pos[None, :], axis=-1)
    aerial = jnp.clip(1.0 - d_hawk / cfg.vision_range, 0.0, 1.0) * w.hawk_on
    aerial = aerial * (1.0 - w.head_down)        # <-- the vigilance/foraging trade-off

    # --- Interoception ---
    d_hens = jnp.linalg.norm(w.pos[:, None, :] - w.pos[None, :, :], axis=-1)
    d_hens = d_hens + jnp.eye(h) * 1e6
    isolation = jnp.clip(jnp.min(d_hens, axis=-1) / 3.0, 0.0, 1.0)
    intero = jnp.stack([w.hunger, w.thirst, w.cold, isolation], axis=-1)

    # --- Somatic ---
    d_wall = jnp.min(jnp.minimum(w.pos, cfg.size - w.pos), axis=-1)
    wall = jnp.clip(1.0 - d_wall / 1.0, 0.0, 1.0)
    somatic = jnp.stack([wall, w.speed / cfg.flee_speed], axis=-1)

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
    if cfg.legacy_audio:
        audio = jnp.clip(atten @ w.calls, 0.0, 1.0)             # pre-E019, for E021
    else:
        audio = jnp.clip(jnp.sqrt((atten ** 2) @ (w.calls ** 2)), 0.0, 1.0)

    return jnp.concatenate(
        [vis, aerial[:, None], intero, somatic, audio], axis=-1)


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
