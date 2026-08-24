"""Building the innate connectome.

Two things are genetic and shared by every hen in the flock: the connection *mask*
(which regions wire to which, sampled once from `REGION_CONNECTIVITY`) and each
neuron's excitatory/inhibitory identity. Individual hens differ only in their initial
weight values -- developmental noise, not different blueprints.

The mask is stored explicitly and separately from the weights even though phase 0
never changes it. That is deliberate: phase 1's structural growth is implemented as
flipping a mask bit and initialising a weight, with no reallocation and no
recompilation. Preallocating the maximum connectivity budget now is the one decision
that would be expensive to revisit later.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from coop import spec
from hen import innate, regions
from hen.regions import Regions


class BrainParams(NamedTuple):
    W: jax.Array          # (H, N, N) recurrent weights, Dale-signed and masked
    mask: jax.Array       # (N, N) bool -- the innate connectome, kept as a reference
    growable: jax.Array   # (N, N) bool -- where an axon could ever reach
    W_in: jax.Array       # (N, OBS_DIM) sensory afferents (shared, fixed)
    W_out: jax.Array      # (H, MOTOR_DIM, n_motor) cortical motor readout
    W_pred: jax.Array     # (H, OBS_DIM, N) top-down associative projection
    # (H, MOTOR_DIM, n_motor) descending gate on the reflex arc (E101-B). Starts at zero
    # so `sigmoid(0 + GATE_OPEN_BIAS)` ~ 1: a hatchling suppresses no reflex at all, and
    # only learning can close the gate. Unused unless `PlasticConfig.reflex_gate`.
    W_gate: jax.Array
    # (H, MOTOR_DIM, n_motor) striatal drive for E102's competitive gate. Zero at hatch,
    # so `sigmoid(BIAS + 0 - beta*0)` ~ 1 and a hatchling's reflexes arrive intact.
    W_str: jax.Array
    # (N,) 1.0 for units that receive sensory afferents, 0 elsewhere (E104). The pooled
    # inhibitory interneuron averages over *these* units -- a real interneuron pools the
    # relay it sits in, not the whole brain, most of which gets no afferents at all.
    lateral_pool: jax.Array
    pred_src: jax.Array   # (N,) bool -- which neurons may source a prediction
    b: jax.Array          # (N,) resting bias
    tau: jax.Array        # (N,) membrane time constants, seconds
    reflex: jax.Array     # (MOTOR_DIM, OBS_DIM) innate arc, fixed
    b_motor: jax.Array    # (MOTOR_DIM,)
    dale: jax.Array       # (N,) +1 excitatory / -1 inhibitory


def _lateral_pool(reg: Regions, n: int, place_to_hippocampus: bool) -> jax.Array:
    """Which units the pooled inhibitory interneuron averages over (E104).

    The *exteroceptive* relays only. A first attempt used "every unit with a nonzero
    `W_in` row", which silently swept in the hypothalamus's 16 interoceptive units --
    so hunger and thirst would have subtracted from visual contrast. An interneuron
    pools the relay it sits in; hunger is not part of the visual relay.

    The hippocampus joins the pool only when `place_to_hippocampus` routes place cells
    there (E086), because that is when it becomes an afferent-receiving sensory stage.
    """
    pool = np.zeros(n, dtype=np.float32)
    s_lo, s_hi = reg.bounds(regions.SENSORY)
    pool[s_lo:s_hi] = 1.0
    if place_to_hippocampus:
        h_lo, h_hi = reg.bounds(regions.HIPPOCAMPUS)
        pool[h_lo:h_hi] = 1.0
    return jnp.asarray(pool)


def _region_of(reg: Regions) -> np.ndarray:
    """(N,) array giving each neuron's region id."""
    return np.concatenate([np.full(n, r) for r, n in enumerate(reg.sizes)])


def _modality_bounds(reg: Regions, aud_fraction: float):
    """Split points carving an auditory-only slice out of the sensory stub and a
    matching "Field L" slice out of the pallium. First `aud_fraction` of each region,
    by neuron index -- see `regions.AUD_FRACTION`.
    """
    s_lo, s_hi = reg.bounds(regions.SENSORY)
    p_lo, p_hi = reg.bounds(regions.PALLIUM)
    n_aud_s = max(1, int(round((s_hi - s_lo) * aud_fraction)))
    n_aud_p = max(1, int(round((p_hi - p_lo) * aud_fraction)))
    return s_lo, s_lo + n_aud_s, s_hi, p_lo, p_lo + n_aud_p, p_hi


def _segregate_modality(mat: jax.Array, reg: Regions, aud_fraction: float) -> jax.Array:
    """Cut the two cross-modal corners of the sensory->pallium block.

    The auditory stub slice keeps its connections into the Field L pallial slice; the
    visual (rest-of-stub) slice keeps its connections into the rest of the pallium.
    Neither talks to the other's target. Everything else -- afferents into the stub,
    recurrence within the pallium (including between the two slices), and every other
    region pair -- is untouched; this mirrors the ablation E017/E034 measured
    (`scratchpad/why_pallium_collapses.py`'s "targeted (Field L)" condition) exactly,
    now built into the connectome instead of a post-hoc surgery on one already built.

    Applied to both `mask` and `growable` so structural growth cannot regrow the
    cross-modal connections that start absent -- the segregation is meant to be an
    anatomical fact, not just an initial condition.
    """
    s_lo, s_split, s_hi, p_lo, p_split, p_hi = _modality_bounds(reg, aud_fraction)
    mat = mat.at[p_split:p_hi, s_lo:s_split].set(False)   # rest-of-pallium <- aud stub
    mat = mat.at[p_lo:p_split, s_split:s_hi].set(False)   # Field L <- visual stub
    return mat


def build(key: jax.Array, reg: Regions = regions.DEFAULT_REGIONS,
          n_hens: int = spec.DEFAULT_COOP.n_hens,
          gain: float = 0.95, readout_scale: float = 0.05,
          auditory_scaffold: bool = False,
          scaffold_gain: float = 1.0,
          modality_segregated: bool = False,
          aud_fraction: float = regions.AUD_FRACTION,
          sensory_pallium_density: float | None = None,
          legacy_food_call: bool = False,
          gakel_scaffold: bool = False,
          gakel_scaffold_gain: float = 1.0,
          gakel_peck_weight: float | None = None,
          hunger_peck_weight: float = 0.0,
          shared_place_map: bool = False,
          testimony_gain: float = 0.5,
          balanced_ei: bool = False,
          place_to_hippocampus: bool = False) -> BrainParams:
    """Sample a newly hatched flock.

    `readout_scale` is small on purpose: at hatch the cortical pathway is near-silent
    and behaviour is dominated by the innate reflex arc. The pallium is present and
    connected, but it has nothing to say yet.

    `auditory_scaffold` gives her an innate response to hearing an alarm call (E018).
    Off by default, because switching it on changes the innate repertoire and so the
    comparison basis for E001-E017.

    `modality_segregated` gives audition its own slice of the sensory stub and its own
    pallial target ("Field L"), instead of the default fully-mixed sensory->pallium
    projection -- the anatomical separation a real bird has (nucleus ovoidalis -> Field
    L kept apart from nucleus rotundus -> entopallium, two separate thalamic relays).
    `aud_fraction` sets how much of each region that slice gets (default 1/6, matching
    the hand-cut probe this replaces). Off by default, same reason as
    `auditory_scaffold`. **Does not help separability** -- E017/E034's 2.06x/1.45x
    figures were unpaired ratio-of-means on a quantity with ~6x genome-to-genome spread;
    a paired, properly fan-in-normalised re-measurement (E035) found no effect
    (t=0.04). Kept for anatomical realism and because the implementation is what caught
    the error, not because it is a working fix for H2d.

    `gain` has been re-baselined twice. It was 0.9 through E009, dropped to 0.70 in
    E010 when the pallium was found saturated, and set to **0.95** in E023 after E022
    found the E/I bug: until then the pallium contained no inhibitory neurons at all,
    so every earlier gain figure describes a different network.

    Measured across 6 genomes, relative separability of "saw a hawk" against "heard an
    aerial alarm", on the corrected connectome:

        gain   mean pallial rate   genome sd   separability
        0.60         0.172           0.001         3.8%
        0.70         0.189           0.002         4.5%
        0.85         0.228           0.004         5.8%
        0.95         0.276           0.009         7.4%   <- default
        1.00         0.320           0.016         9.4%
        1.05         0.415           0.041         9.4%
        1.10         0.603           0.061         4.5%   <- transition
        1.40         0.916           0.008         1.0%

    **The knife edge is gone, and that is what the fix bought.** The old network
    bifurcated between 0.75 and 0.78 (rate 0.35 -> 0.50) and was unusable by 0.90. This
    one climbs smoothly from 0.60 to 1.00 with genome-to-genome spread under 0.02, and
    does not break until past 1.05. The usable band is roughly four times wider.

    0.95 rather than the peak at 1.00-1.05, on E010's reasoning, which still holds:
    weights move during learning, so leave margin. 0.95 sits ~0.12 below the
    transition where the old 0.70 sat ~0.08 below its own, and its genome spread is
    3.3% of the mean rate against 10% at 1.05. Take 0.90 if a run is ever seen to
    drift its operating point upward.

    **What the fix did not buy is separability.** At 0.95 it is 7.4%, against 7.5% for
    the old usable default. Comparable, not better -- and the review that found the E/I
    bug predicted a 1.4x improvement that did not materialise (E023). H2d's problem is
    untouched: a hen still barely distinguishes a heard alarm from a seen hawk. What
    changed is that the number no longer depends on holding a parameter to two decimal
    places.

    Separability still varies a lot between genomes, which is individual variation in
    how well a given hen's wiring separates her world. It means per-seed results are
    noisy and contrasts need replicates.

    `sensory_pallium_density`, if given, overrides `REGION_CONNECTIVITY`'s sensory
    (row 0) -> pallium (column 1) entry (default 0.30) for this connectome only.
    E017/E034 localised H2d's representational loss to fan-in dilution at exactly this
    projection: each pallial unit sums ~19 stub inputs, of which one or two carry any
    given distinction, so a real difference lands as a small perturbation on a large
    common-mode drive. Lower density means fewer, less diluted inputs per pallial unit;
    `fan_in` below is computed from the actual sampled mask, so weights are
    automatically re-normalised for whatever density results -- same mechanism
    `modality_segregated` relies on, applied to density instead of a partition. None of
    the numbers in the table above hold once this is changed; re-measure, don't assume.

    `legacy_food_call` restores the pre-E053 innate food call (continuous on raw sight,
    not the discovery pulse). Off by default; exists only as E054's ablation condition.
    """
    k_mask, k_w, k_in = jax.random.split(key, 3)
    n = reg.total
    rid = _region_of(reg)

    # --- Genetic mask: P[source region, target region], W[i, j] is j -> i ---
    p = np.asarray(regions.REGION_CONNECTIVITY, dtype=np.float32)
    if sensory_pallium_density is not None:
        p = p.copy()
        p[regions.SENSORY, regions.PALLIUM] = sensory_pallium_density
    p_ij = p[rid[None, :], rid[:, None]]                     # (N, N)
    mask = jax.random.uniform(k_mask, (n, n)) < jnp.asarray(p_ij)
    mask = mask & ~jnp.eye(n, dtype=bool)                    # no autapses

    # Where a synapse could ever appear. Phase 1 growth is confined to this: axons
    # only reach regions their tract projects to, so a hypothalamic neuron cannot
    # sprout onto the sensory stub however well correlated the two happen to be.
    growable = jnp.asarray(p_ij > 0.0) & ~jnp.eye(n, dtype=bool)

    if modality_segregated:
        mask = _segregate_modality(mask, reg, aud_fraction)
        growable = _segregate_modality(growable, reg, aud_fraction)

    # --- Dale's law: a neuron's outgoing weights all share its sign ---
    #
    # Stratified *within each region*, not by flat index across the whole array. The
    # old version took the first 80% of neurons by index; since regions are laid out
    # contiguously that cut fell mid-arcopallium and left the pallium 100% excitatory,
    # the hypothalamus and motor stub 100% inhibitory (E022). Interneurons are
    # distributed throughout a real pallium, and a recurrent pool without them sits on
    # a saddle-node, which is what made the gain a two-decimal-place quantity.
    #
    # Which neurons within a region are inhibitory is a genetic fact shared by the
    # flock, so it is drawn from the mask key rather than the per-hen weight key.
    rng_dale = np.random.default_rng(int(jax.random.randint(k_mask, (), 0, 2**30)))
    dale_np = np.empty(n, dtype=np.float32)
    for r_id, size in enumerate(reg.sizes):
        lo, hi = reg.bounds(r_id)
        n_exc = int(round(regions.EXCITATORY_FRACTION * size))
        signs = np.full(size, -1.0, dtype=np.float32)
        signs[:n_exc] = 1.0
        rng_dale.shuffle(signs)
        dale_np[lo:hi] = signs
    dale = jnp.asarray(dale_np)

    fan_in = jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1.0)
    w_raw = jnp.abs(jax.random.normal(k_w, (n_hens, n, n))) * (gain / jnp.sqrt(fan_in))
    w = w_raw * mask[None] * dale[None, None, :]

    if balanced_ei:
        # E072. Every recurrent projection here carries a large positive DC by
        # construction: 80% of neurons are excitatory (EXCITATORY_FRACTION) and both
        # signs draw magnitudes from the same |normal|, so excitation outweighs
        # inhibition ~4:1 in total. Measured on the default connectome, the
        # sensory->pallium block nets **+0.93 per pallial unit**.
        #
        # That is exactly the quantity E017/E034 localised H2d's 14.5-17x loss to --
        # "a clean difference lands as a small perturbation on a large common-mode
        # drive" -- and exactly the defect E027 already fixed for `W_out`, whose
        # comment reads: "The stub is 80% excitatory, so signing a rectified draw
        # leaves every motor channel with a large positive DC bias where the old
        # zero-mean draw was balanced." `W_out` now nets -0.000000. `W` never received
        # the same treatment.
        #
        # Same construction as `w_out`'s: scale each postsynaptic unit's inhibitory
        # inputs so their total magnitude matches its excitatory total, making the row
        # sum zero. Balanced per row rather than by neuron count, which is only correct
        # in expectation -- the same reasoning E027 gives.
        exc = w_raw * mask[None] * (dale > 0)[None, None, :]
        inh = w_raw * mask[None] * (dale < 0)[None, None, :]
        scale = exc.sum(-1, keepdims=True) / (inh.sum(-1, keepdims=True) + 1e-9)
        balanced = exc - inh * scale
        # Hold total synaptic magnitude constant. Scaling inhibition up to meet
        # excitation raises sum|W| by ~60%, which would confound "balanced E/I" with
        # "more total weight" -- the identical confound that invalidated both of
        # E017/E034's modality-segregation figures (E035: segregation was never
        # separated from less total drive to the segregated slice). Renormalising here
        # makes the flag change the E/I *ratio* and nothing else.
        w = balanced * (jnp.sum(jnp.abs(w)) / (jnp.sum(jnp.abs(balanced)) + 1e-9))

    # --- Afferents. Exteroceptive channels reach the sensory stub; interoceptive
    # drives reach the hypothalamus. Nothing else gets direct input.
    w_in = np.zeros((n, spec.OBS_DIM), dtype=np.float32)
    s_lo, s_hi = reg.bounds(regions.SENSORY)
    h_lo, h_hi = reg.bounds(regions.HYPOTHALAMUS)

    rng = np.random.default_rng(int(jax.random.randint(k_in, (), 0, 2**30)))
    extero = [i for i in range(spec.OBS_DIM)
              if not (spec.INTERO_LO <= i < spec.INTERO_HI)]

    def _random_afferents(lo, hi, channels):
        return rng.gamma(2.0, 0.5, (hi - lo, len(channels))) \
            * (rng.random((hi - lo, len(channels))) < 0.3)

    if modality_segregated:
        # Same afferent statistics (gamma magnitude, 30% density) as the mixed case,
        # just partitioned: audio channels reach only the auditory stub slice, every
        # other exteroceptive channel reaches only the rest of the stub.
        s_split = _modality_bounds(reg, aud_fraction)[1]
        audio_ch = list(range(spec.AUDIO_LO, spec.AUDIO_HI))
        vis_ch = [i for i in extero if i not in audio_ch]
        w_in[s_lo:s_split, audio_ch] = _random_afferents(s_lo, s_split, audio_ch)
        w_in[s_split:s_hi, vis_ch] = _random_afferents(s_split, s_hi, vis_ch)
    else:
        w_in[s_lo:s_hi, extero] = _random_afferents(s_lo, s_hi, extero)

    if place_to_hippocampus:
        # T2-revised mechanism 2, second attempt (E086).
        #
        # The problem. `regions.py` names HIPPOCAMPUS "place and spatial memory" and
        # E063 was written up as giving it its first real function -- but E063 added
        # place *channels*, and `W_in` writes only into the sensory stub above, so the
        # region never received them. Measured before this change: of the 64 units
        # taking place afferents, 64 were in the sensory stub and 0 in the hippocampus.
        # Position therefore reached `pred_src` only after competing with 113 other
        # channels through a 64-unit bottleneck, and E085 measured the result -- about
        # four points above chance while the hen moves, against 84.6% parked.
        #
        # The fix is a routing change, deliberately not a magnitude change: the same
        # afferent statistics used everywhere else (gamma(2.0, 0.5), 30% density), into
        # a region that already projects to the pallium at 0.20. Nothing is scaled up;
        # place simply stops sharing a stub with hunger, vision and audio.
        #
        # Both spatial blocks are routed, and the `shared_place_map` relationship
        # between them is preserved below, so E064's "a place is the same place however
        # you learned of it" continues to hold in the new region.
        hp_lo, hp_hi = reg.bounds(regions.HIPPOCAMPUS)
        place_ch = list(range(spec.PLACE_LO, spec.PLACE_HI))
        w_in[hp_lo:hp_hi, place_ch] = _random_afferents(hp_lo, hp_hi, place_ch)

    if shared_place_map:
        # T2-revised mechanism 2: a shared allocentric map.
        #
        # The problem it solves. A hen learns "something bad happened at P" from a call
        # heard across the coop, so her own place cells (PLACE_LO..PLACE_HI) encode
        # where *she* is, while the caller's location arrives separately on
        # GAKEL_PLACE_LO..GAKEL_PLACE_HI. E064 deliberately built the second from the
        # identical grid geometry as the first -- but they are different observation
        # indices, and the afferent draw above is independent per column, so the
        # pallial pattern evoked by "testimony about P" shares nothing with the one
        # evoked by "standing at P". `W_pred` sources from the pallium
        # (`pred_src`), so an association learned from testimony would be written onto
        # units that are silent when she later walks to P. Zero transfer. Without this,
        # testimony can only ever teach her about the spot she was standing on when she
        # heard the news.
        #
        # The fix, and why it is a connectome prior rather than a behavioural one:
        # give the testimony block the *same* afferent pattern as the self-location
        # block, scaled. Stub unit i's response to hearing-about-P then becomes a
        # scaled copy of its response to being-at-P, so any `W_pred` weight learned on
        # one is directly readable from the other. Nothing about which places are good
        # or bad is implied -- only that a place is the same place however you learned
        # of it, which is what a cognitive map *is*.
        #
        # Scaled rather than identical, deliberately. At `testimony_gain=1.0` the two
        # channels would be indistinguishable once summed, and a hen could not tell "I
        # am here" from "someone called from there". A gain below 1 keeps the
        # representations congruent in *pattern* (so the association transfers) while
        # separable in *magnitude*, and encodes the same ordering `innate.py` already
        # argues for the alarm scaffold: first-hand information beats second-hand.
        w_in[:, spec.GAKEL_PLACE_LO:spec.GAKEL_PLACE_HI] = (
            testimony_gain * w_in[:, spec.PLACE_LO:spec.PLACE_HI])

    w_in[h_lo:h_hi, spec.INTERO_LO:spec.INTERO_HI] = rng.gamma(
        2.0, 1.0, (h_hi - h_lo, spec.INTERO_HI - spec.INTERO_LO))

    m_lo, m_hi = reg.bounds(regions.MOTOR)
    # Dale's law applies here too, and until E027 it did not. This drew from a zero-mean
    # normal, so every motor-stub neuron sent excitation to some muscles and inhibition
    # to others -- measured at 0 of 48 columns compliant, all 48 mixed, with 10 of the
    # source neurons inhibitory. E022 found it, marked it "verified -- adopt", and it
    # fell off the action list for four experiments.
    #
    # Same construction as the recurrent weights: draw a magnitude, then take the sign
    # from the presynaptic neuron's identity.
    #
    # **The inhibitory weights are scaled up, and without it this fix breaks the bird.**
    # The stub is 80% excitatory, so signing a rectified draw leaves every motor channel
    # with a large positive DC bias where the old zero-mean draw was balanced. Measured
    # on the first attempt: hens crouched permanently and were struck 0 times in 3000
    # steps at a predator rate that had produced 1000, and hunger went from 43% to 92%
    # of the reward variance. The recurrent path never showed this because synaptic
    # scaling renormalises its row sums; the readout has no such correction.
    #
    # Balancing E against I in total magnitude is also the biologically right reading --
    # inhibitory interneurons are outnumbered and individually stronger.
    # Balanced exactly per (hen, motor channel) rather than by neuron count, which is
    # only correct in expectation and left a residual net drive of 0.44 against a total
    # magnitude of 2.66. Scaling the inhibitory group so its total matches the excitatory
    # one makes the untrained readout contribute exactly zero DC to every muscle -- which
    # is the stated design intent anyway ("starts nearly silent", and has to *earn*
    # influence), now true by construction instead of by luck.
    d_out = dale[m_lo:m_hi]
    mag = jnp.abs(jax.random.normal(
        jax.random.fold_in(k_w, 1), (n_hens, spec.MOTOR_DIM, m_hi - m_lo)))
    exc = mag * (d_out > 0)[None, None, :]
    inh = mag * (d_out < 0)[None, None, :]
    scale = exc.sum(-1, keepdims=True) / (inh.sum(-1, keepdims=True) + 1e-9)
    w_out = (exc - inh * scale) * readout_scale

    # Top-down projection onto the sensory representation the reflex arc reads.
    # Starts at exactly zero: a newly hatched hen predicts nothing and perceives only
    # what is in front of her. Every association she ever has is one she formed.
    w_pred = jnp.zeros((n_hens, spec.OBS_DIM, n))

    # Predictions come from the pallium only, never from the sensory stub. E008 found
    # the first version was circular: sourced from the whole brain, it learned "when
    # in hawk-state, predict hawk", because the sensory stub carries the hawk percept
    # directly and dominates the association. Association cortex, one step removed
    # from the relay, is also where top-down predictions come from in a real brain.
    p_lo, p_hi = reg.bounds(regions.PALLIUM)
    pred_src = jnp.zeros((n,), dtype=bool).at[p_lo:p_hi].set(True)
    if place_to_hippocampus:
        # The half that makes the routing above useful: `W_pred` may now source from
        # units that actually carry position. Either change alone is close to pointless
        # -- afferents into a region nothing reads, or a readable region with no place
        # in it.
        #
        # This knowingly re-creates the shape E008 fixed -- a `pred_src` region with
        # direct sensory afferents -- and E086 section 5 records why that is acceptable
        # *here* and what it obliges next. In short: the association T2 needs is
        # cross-modal (source place, write the gakel call, which is not a hippocampal
        # afferent), the source is the lagged centred trace rather than instantaneous
        # rate, and E086 plants rather than learns. **A learning run on this must carry
        # an autoencoder control**, because `shared_place_map` also routes the testimony
        # channels here, and predicting testimony-about-P from being-at-P would be
        # circular in exactly E008's sense while looking like successful association.
        hp_lo, hp_hi = reg.bounds(regions.HIPPOCAMPUS)
        pred_src = pred_src.at[hp_lo:hp_hi].set(True)

    tau = jnp.asarray(np.asarray(regions.REGION_TAU, dtype=np.float32)[rid])

    return BrainParams(
        W=w,
        mask=mask,
        growable=growable,
        W_in=jnp.asarray(w_in),
        W_out=w_out,
        W_pred=w_pred,
        W_gate=jnp.zeros_like(w_out),
        W_str=jnp.zeros_like(w_out),
        lateral_pool=_lateral_pool(reg, n, place_to_hippocampus),
        pred_src=pred_src,
        b=jnp.full((n,), -2.0),
        tau=tau,
        reflex=jnp.asarray(
            innate.reflex_matrix(auditory_scaffold, scaffold_gain, legacy_food_call,
                                 gakel_scaffold, gakel_scaffold_gain,
        gakel_peck_weight=gakel_peck_weight,
        hunger_peck_weight=hunger_peck_weight)),
        b_motor=jnp.asarray(innate.reflex_bias()),
        dale=dale,
    )


def stats(p: BrainParams, reg: Regions = regions.DEFAULT_REGIONS) -> dict:
    """Descriptive numbers for the README and the benchmark banner."""
    n = reg.total
    n_syn = int(jnp.sum(p.mask))
    return {
        "neurons": n,
        "synapses": n_syn,
        "mean_fan_out": n_syn / n,
        "density": n_syn / (n * n),
        "regions": dict(zip(regions.REGION_NAMES, reg.sizes)),
    }
