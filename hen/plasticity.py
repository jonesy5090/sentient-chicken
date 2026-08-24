"""Phase 1: learning.

A three-factor rule. Two of the factors are local -- traces of pre- and postsynaptic
activity -- and the third is a global neuromodulator standing in for dopamine:

    dW[i,j]  =  eta(age) * m * (z_slow[i] - z_slow_bar[i]) * (z_fast[j] - z_fast_bar[j])

The subtracted terms are each trace's own slow mean, and they are not decoration. Rates
are sigmoids -- strictly positive, and both traces are low-pass filters that barely
move -- so without centring the outer product is dominated by the product of the two
means and is the *same matrix* every consolidation. E019 measured that version: rank
one, with the cortical drive varying by 0.7% of its magnitude. It could only slide a
constant offset, and it slid peck downward. Centring makes the rule a covariance rule,
signed and state-dependent.

The neuromodulator `m` is a reward *prediction error*: homeostatic improvement minus
a running expectation of it. Reward is drive reduction -- eating when hungry, warming
when cold, escaping when threatened. Nothing here is a task objective handed down
from outside; a hen is rewarded for staying alive and comfortable, which is the only
thing a real animal optimises.

Three design points worth stating, because each is a place this could have gone wrong:

**Eligibility is factored, not stored per synapse.** A true pre-post correlation trace
would need an (H, N, N) tensor -- doubling the memory traffic of a simulation that is
already bandwidth bound. Filtering pre and post separately and taking the outer
product at update time is the standard approximation and costs two vectors per hen.

**Weights are written every `interval` steps, not every step.** The forward pass
already reads W each step; writing it too would double bandwidth and halve the
real-time factor. The traces run continuously and consolidation samples them
periodically, which is also closer to how synaptic consolidation actually works.
Note that nothing *accumulates* between consolidations -- the traces are EMAs, and
`consolidate` reads them at that instant. With `tau_fast = 0.02 s` the presynaptic
factor has a short memory against a 50-step interval, so the effective credit window
is `tau_slow`, 0.2 s. Anything that has to bridge a longer gap than that is not
learnable by this rule as written.

**The reflex arc is untouched.** Plasticity applies to the cortical pathway only.
A chick's innate responses stay fixed for life, as they do in a real bird -- what
develops is the pallium's ability to modulate them.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from coop.spec import CoopConfig
from hen.connectome import BrainParams


class PlasticConfig(NamedTuple):
    enabled: bool = True

    # Eligibility traces. The asymmetry between the two time constants is what gives
    # the rule a direction: post integrates over a longer window than pre, a
    # rate-coded stand-in for the timing asymmetry of STDP.
    tau_fast: float = 0.02        # presynaptic, seconds
    tau_slow: float = 0.20        # postsynaptic
    tau_motor: float = 0.10

    # Learning rate and the critical period. Plasticity is highest at hatch and
    # decays with age, so early experience shapes the connectome disproportionately.
    eta: float = 4e-3
    # The readout rate is 10x the recurrent one, and that asymmetry is load-bearing.
    # The pallium starts near-silent by design (readout_scale=0.05), so it has to
    # *earn* influence over behaviour before anything it learns can reach a muscle.
    # Measured over 10 min of chicken time (E002): at 2e-3 the readout does not grow
    # at all (1.00x) and cortical drive stays at 0.16 of the reflex arc; at 2e-2 it
    # grows 1.17x and reaches 0.51. At 2e-1 cortical drive overwhelms the innate arc
    # (1.72x) and behaviour gets worse -- a hen who overrides her reflexes with an
    # untrained pallium is worse off than one who does not.
    eta_out: float = 2e-2
    critical_period_s: float = 3 * 86400.0

    # Neuromodulation
    reward_scale: float = 60.0
    strike_penalty: float = 1.0
    # T2 (E066, following E065's null): `reward()` had no term for sickness at all --
    # checked directly, there was none -- so the only teaching signal available to
    # `W`'s reward-gated update (the pathway that actually routes place-cell
    # information toward motor output; `hebbian_readout` only touches `W_out`, which
    # reads from the motor region's own rates, not from sensory input) had nothing to
    # learn T2's outcome from. Same discrete-event shape as `strike_penalty`, same
    # default magnitude (1.0) as a first-pass calibration -- getting poisoned is a
    # real cost (60s of impaired mobility) but not obviously as severe as a predator
    # strike, and no attempt is made here to derive a "correct" relative weighting.
    #
    # Off by default (0.0), matching `readout_scaling_strength`'s own precedent: this
    # changes reward dynamics for *any* plastic run in a world with T2's contamination
    # scaffold active, not just T2's own experiments, so every other hypothesis that
    # happens to run with plasticity enabled against `DEFAULT_COOP` must opt in
    # explicitly rather than have its established results silently move underneath it.
    sickness_penalty: float = 0.0
    baseline_tau_s: float = 20.0

    # Weight on flockmates' welfare in a hen's own reward signal. This is kin
    # selection, which is the standard explanation for why alarm calling evolves in
    # group-living birds at all: warning a relative propagates your own genes. A
    # flock of hens is typically related.
    #
    # It is needed here for a specific reason. Under purely individual reward a call
    # can never pay -- the caller bears the energetic cost and the *listener* gets
    # the benefit -- so a hen would learn to fall silent and H3 could not be tested.
    #
    # Note what this does NOT do: nothing here rewards calling *when an audience is
    # present*. It rewards flockmates doing well, and it charges for calling. That
    # audience-conditional calling is the optimal policy is the hypothesis, not the
    # instruction.
    kin_weight: float = 0.5

    # Whether the kin term is weighted by audibility. See `reward()`: the flat flock
    # mean cannot assign credit to a caller, which is E005's diagnosis of its own
    # null. False reproduces the E005 condition.
    kin_audible: bool = True

    # Consolidation. The forward pass already reads W every step; writing it too is
    # what costs, so eligibility accumulates continuously and consolidates every
    # `interval` steps. Measured: interval=10 costs 3x throughput, 50 costs ~25%,
    # and beyond 50 the returns flatten. 50 steps is 500 ms of chicken time, far
    # finer than any behaviour we care about.
    interval: int = 50            # steps between weight writes
    w_max: float = 0.5
    scaling_strength: float = 0.02  # synaptic scaling back toward innate total

    # Exploration. Young animals are behaviourally variable, and that variability is
    # the substrate reinforcement acts on -- a deterministic policy can only ever be
    # rewarded for actions it already takes. E006 established that without this the
    # hen cannot acquire any stimulus-response pairing her innate reflex arc does not
    # already visit, which blocks every hypothesis about learned communication.
    #
    # Decays with age on the same schedule as the critical period: a chick explores,
    # an adult exploits.
    explore_sigma: float = 0.6      # s.d. on the motor drive, at hatch
    explore_tau_s: float = 3600.0   # halves roughly every hour of chicken time

    # Top-down association (H2c). Learned by a masked delta rule that is NOT
    # reward-modulated: association between co-occurring stimuli needs no reward and
    # no attribution of benefit to anyone, which is exactly the credit-assignment
    # problem that sank E005 and E006.
    pred_enabled: bool = False      # opt-in, so it is a contrast rather than a change
    pred_gain: float = 1.0          # how strongly prediction augments perception
    eta_pred: float = 5e-3
    tau_lag: float = 1.5            # seconds; bridges cue to outcome
    pred_max: float = 0.05          # per-synapse cap on the top-down projection
    # Centre the prediction pathway's source before projecting it (E071). Off by
    # default so it is a contrast rather than a change, matching `pred_enabled`'s own
    # precedent -- E042/E043 ran H2c against the uncentred rule and their recorded
    # numbers describe that rule. Applies to both halves: the readout in `brain.py`
    # and `W_pred`'s own update below, which had the identical defect.
    pred_centred: bool = False
    # Time constant for `z_lag_bar`, the prediction pathway's centring baseline.
    # `None` inherits `baseline_tau_s`, which is what this shared with until E087 --
    # two unrelated quantities on one constant, and not stated anywhere as a choice.
    # They want opposite things: the reward baseline should track on the timescale of
    # reinforcement, while this one must be *slow compared to whatever the prediction
    # is about* or it subtracts the signal along with the DC. E086 measured the cost of
    # getting that wrong: at 20 s against dwell times of 17-75 s, the baseline tracks
    # the hen's position and removes it, costing ~20 points of decodability.
    pred_bar_tau_s: float | None = None
    # Freeze `z_lag_bar` once the hen reaches this age, holding it constant thereafter.
    # `None` never freezes, which is the behaviour every result before E088 was measured
    # under.
    #
    # Why a frozen baseline rather than a slower one. E087 swept `pred_bar_tau_s` and
    # found the cost of centring is the baseline *tracking*, not its timescale: longer
    # taus made decodability worse, not better, and destroyed selectivity (at 300 s the
    # prediction at a control place exceeded the prediction at the target). What did
    # work was a *constant* baseline -- selectivity 5.00 at 89.8% place decodability,
    # against the moving average's 23.28 at 73.7%. But E087's constant was the mean
    # across settled place states, which requires knowing the places in advance and is
    # therefore a diagnostic, not a mechanism. Freezing is the causal version of it: a
    # developmental calibration of what "average activity" means, estimated once and
    # then fixed, rather than re-estimated forever.
    pred_bar_freeze_s: float | None = None
    # E101-A. Drop the relu on the top-down prediction, so `W_pred` may SUBTRACT from the
    # observation the reflex arc reads instead of only adding to it.
    #
    # `brain.py` has always relu'd it, with the stated reason that "association can only
    # add percepts, never suppress real ones". That is a design choice and E101 measured
    # what it costs: with it, the learned pathway is 98.4% excitatory and its strongest
    # possible opposition is ~46x too small to counter the 8.0 crouch reflex. A hen can
    # learn to imagine a hawk and cannot learn to ignore one.
    #
    # Vertebrate development is substantially the forebrain acquiring the ability to
    # suppress brainstem reflexes. This is the sensory-gating half of that: she learns to
    # stop *perceiving* what she should ignore. The clip in `brain.py` still prevents her
    # perceiving more vividly than reality; this lets her perceive less.
    pred_signed: bool = False
    # E101-B. A learned multiplicative gate on the reflex arc's output:
    # `drive = reflex * sigmoid(W_gate @ motor_stub) + cortical`. Descending gain control
    # -- the arc still fires, the forebrain decides how much of it reaches the muscles.
    # This is the mechanism the additive meeting point at `brain.py:87` cannot express.
    reflex_gate: bool = False
    # Bound on `W_gate`, so a runaway gate cannot silence the arc outright. At -8.0 the
    # gate reaches sigmoid(4.0 - 8.0) = 0.018: near-total suppression is reachable but
    # requires the rule to drive there deliberately.
    gate_max: float = 8.0
    # E102. Replace E101-B's independent per-channel gates with a basal-ganglia-style
    # competitive one: `gate = sigmoid(BIAS + s - bg_lateral * mean(s))`, where `s` is a
    # striatal drive read from the motor stub.
    #
    # The point is the subtraction. Uniform changes cancel -- if learning drives every
    # channel's `s` down together, `s - mean(s)` stays zero and the gate does not move.
    # **Global suppression is architecturally unavailable; only redistribution is.**
    #
    # E101-B, a free gate, learned to close 11 of 12 channels (turning to 0.099, pecking
    # to 0.116) and got its predation benefit from a mundane route: `mobility = 1 - crouch`,
    # so a crouching hen cannot move and stays in the strike radius. A largely inert hen
    # drifts out. That is what a gate with no competition converges on, and it is the
    # third appearance of the degeneracy E100 found in `W_out`.
    #
    # Real basal ganglia hold motor programs under tonic inhibition and *selectively
    # release* them, with lateral competition in striatum making release focused. Birds
    # have a well-developed medial striatum and pallidum; this model had neither.
    bg_gate: bool = False
    bg_lateral: float = 1.0

    # Restores E067's pre-fix behaviour: `m` read as a single-step snapshot at the
    # consolidation boundary rather than averaged over the window since the last one.
    # Exists purely so E075 can ask what E067's fix changed, exactly as `legacy_audio`
    # exists so E021 could ask what E019's audio fix changed. Never a default -- the
    # snapshot behaviour is the defect E067 measured at a 2% capture rate for discrete
    # reward events.
    legacy_m_sampling: bool = False

    # Readout rule kind (H2f, E055). `W_pred` above is already non-reward-gated, but it
    # only ever augments *perception* (it writes onto the observation the reflex arc
    # reads) -- it has no route to a motor decision like "call more when someone is
    # listening", which only `W_out` can produce. H2f's falsifier calls for a rule
    # "closer to Pavlovian association" tested on the same task the reward-modulated
    # one failed; this is that rule applied to the one pathway that could actually
    # move the needle on it. When True, `consolidate()` drops the reward-modulation
    # factor `m` from the readout update entirely (replaced with a constant), turning
    # it into unsupervised Hebbian/covariance learning: whatever motor pattern
    # reliably co-occurs with a cortical state gets wired in, regardless of whether it
    # helped. Off by default -- the instrumental rule is the one every other
    # experiment in this tree assumes.
    hebbian_readout: bool = False

    # Readout stability (E055 follow-up). `W` gets a synaptic-scaling correction
    # pulling its row sums back toward the innate connectome (below); `W_out` never
    # has. Under the reward-modulated rule this went unexercised in practice -- a
    # reward-prediction error averages toward zero over time, which happened to keep
    # readout growth bounded without anyone relying on it. `hebbian_readout` removes
    # that incidental check and nothing replaces it: E055 measured cortical drive
    # reaching 2-2.7x reflex magnitude, the documented "behaviour gets worse" regime,
    # with every calling channel elevated regardless of condition -- a broken readout,
    # not a targeted policy. This applies the same correction `W` already has to
    # `W_out`. Off by default (0.0), so it changes nothing for any existing result;
    # opt in alongside `hebbian_readout` to actually test H2f's falsifier cleanly.
    readout_scaling_strength: float = 0.0

    # Structural growth
    growth_enabled: bool = True
    growth_interval: int = 10_000   # steps (100 s of chicken time)
    growth_rate: float = 0.02       # P(new synapse) at mean coactivity
    prune_threshold: float = 5e-3   # |w| below this is a candidate for pruning
    prune_rate: float = 0.05
    synapse_budget: float = 1.5     # multiple of the innate synapse count


class PlasticState(NamedTuple):
    z_fast: jax.Array      # (H, N) presynaptic trace
    z_slow: jax.Array      # (H, N) postsynaptic trace
    z_motor: jax.Array     # (H, MOTOR_DIM) motor output trace
    # Slow means of the three traces above. The rule learns on *deviations* from
    # these, not on raw activity -- see `consolidate`.
    z_fast_bar: jax.Array   # (H, N)
    z_slow_bar: jax.Array   # (H, N)
    z_motor_bar: jax.Array  # (H, MOTOR_DIM)
    z_err: jax.Array       # (H, OBS_DIM) masked prediction error trace
    z_lag: jax.Array       # (H, N) slow trace: what the brain was doing before
    # Slow mean of `z_lag`, for centring the prediction pathway (E071). `z_lag` is a
    # rate trace: strictly positive, mean ~0.23, with the across-stimulus signal
    # measured at 3.7% of that DC baseline (E070). Projecting it raw through `W_pred`
    # lets the DC term dominate -- E070 measured a planted place association predicting
    # 1.0000 at its own place and 0.9637 at a different one. This is the same defect
    # E019 found for `W`/`W_out`, which is why the fix is the same: subtract the
    # trace's own slow mean, exactly as `z_slow_bar` does.
    z_lag_bar: jax.Array   # (H, N)
    baseline: jax.Array    # (H,) running expectation of reward
    # Accumulated reward-prediction-error since the last consolidation (E067). `m`
    # itself is not a trace anywhere else in this file -- unlike z_fast/z_slow/
    # z_motor, which are continuously-updated EMAs, `reward_now - baseline` used to be
    # read fresh every step and only the value at the exact consolidation boundary
    # (every `interval` steps) ever reached `consolidate()`. A discrete, single-step
    # reward event (`strike_penalty`, `sickness_penalty`) has no reason to land on
    # that boundary, and an exhaustive sweep over every possible timing offset found
    # it does so 2% of the time -- confirmed independently
    # (`scratchpad/e067_reward_eligibility_check.py`), following an adversarial
    # review. `m_acc` is a running sum, reset to zero every consolidation; the caller
    # divides by `pc.interval` to get the mean reward-prediction-error over the
    # window, which any discrete event within it now always contributes to, rather
    # than only when its timing happens to coincide with the write.
    m_acc: jax.Array       # (H,)
    age_s: jax.Array       # scalar, seconds since hatch
    innate_row_sum: jax.Array   # (H, N) reference total input weight per neuron
    innate_row_sum_out: jax.Array  # (H, MOTOR_DIM) same, for W_out (E055 follow-up)
    budget: jax.Array      # scalar, maximum live synapses per hen


def initial_state(p: BrainParams, n_hens: int, pc: PlasticConfig) -> PlasticState:
    n = p.b.shape[0]
    row_sum = jnp.sum(jnp.abs(p.W), axis=2)
    row_sum_out = jnp.sum(jnp.abs(p.W_out), axis=2)
    return PlasticState(
        z_fast=jnp.zeros((n_hens, n)),
        z_slow=jnp.zeros((n_hens, n)),
        z_motor=jnp.zeros((n_hens, p.W_out.shape[1])),
        z_fast_bar=jnp.zeros((n_hens, n)),
        z_slow_bar=jnp.zeros((n_hens, n)),
        z_motor_bar=jnp.zeros((n_hens, p.W_out.shape[1])),
        z_err=jnp.zeros((n_hens, p.W_pred.shape[1])),
        z_lag=jnp.zeros((n_hens, n)),
        z_lag_bar=jnp.zeros((n_hens, n)),
        baseline=jnp.zeros((n_hens,)),
        m_acc=jnp.zeros((n_hens,)),
        age_s=jnp.array(0.0),
        innate_row_sum=row_sum,
        innate_row_sum_out=row_sum_out,
        budget=jnp.sum(p.mask) * pc.synapse_budget,
    )


def reward(w_prev, w_next, cfg: CoopConfig, pc: PlasticConfig) -> jax.Array:
    """Homeostatic improvement per second, minus the cost of being caught.

    Drive *reduction* is positive. There is no task term: the hen is rewarded for
    being fed, watered, warm and uneaten, and any foraging strategy she develops has
    to be discovered from that alone.

    A fraction `kin_weight` of the flock's mean welfare is added to each hen's own.
    See `PlasticConfig.kin_weight` -- without it a call can never repay its energetic
    cost, because the caller pays and the listener benefits.
    """
    # Drives are costs, so reduction is positive.
    #
    # Vigour is deliberately NOT here, and the history is worth keeping. E012 found
    # `call_energy_cost` charged to hunger was destroying H2's *metric*, and moved it
    # into its own `vigour` budget -- which was correct, and which quietly relocated it
    # into the *teaching signal*, where nobody checked it. E019 measured the result:
    #
    #     component   share of reward variance
    #     hunger                 0.0%
    #     thirst                 0.0%
    #     cold                   1.9%
    #     vigour                98.1%
    #
    # A hen was being taught almost nothing except "did you just call". The cost is
    # still real -- calling still drains vigour, and vigour still attenuates what
    # flockmates hear (`coop/world.py`), so vocal effort remains self-limiting and
    # audience-sensitivity still has a gradient. It is simply not a reward term.
    #
    # The general lesson, which cost three defects to learn: when a term is moved,
    # measure it in its new home, not the one it left.
    d_drive = ((w_prev.hunger - w_next.hunger)
               + (w_prev.thirst - w_next.thirst)
               + (w_prev.cold - w_next.cold)) / cfg.dt
    # Being caught is a discrete *event*, not a rate, and it must not be divided by
    # dt. It was, until E014. At dt=0.01 that turned a single strike into -100 in a
    # hen's reward -- roughly 150x anything the drive terms contribute (~0.6) -- and
    # the baseline, which tracks over 20 s, cannot absorb a spike like that. The
    # modulator then slams every eligible synapse at once.
    #
    # Measured: seeds where hens were struck lost 50-75% of their connectome; a seed
    # that happened to take zero strikes lost 17% and was fine. Ablating this term
    # restored every seed to ~83%. The units error was the whole of the erosion that
    # E013 attributed to a random walk.
    #
    # **E014 fixed half of it.** Removing the `/dt` left `n_struck` still incrementing
    # every *step* of contact, so one catch during a 12 s dive was still worth up to
    # ~1000. E027 measured the result at the H4 configuration (hawk every 20 s): this
    # term carried **87.3%** of the reward variance, against 0.0% at the 900 s default
    # the guard test runs at -- where no hawk arrives at all. Reading the event counter
    # is the other half of E014's fix, six experiments late.
    struck = w_next.n_strike_events - w_prev.n_strike_events
    # Sickness onset (T2, E066): a discrete event, not a rate, the same reasoning
    # `struck` above already applies -- dividing by dt would let one poisoning event
    # dominate the same way one strike used to before E014's fix. `sick_on` is a level
    # (True for the whole `sickness_duration_s` window), so the onset is its own
    # rising edge, not `w_next.sick_on` directly, which would double-charge every step
    # she stays sick.
    sick_onset = (w_next.sick_on & ~w_prev.sick_on).astype(jnp.float32)
    own = (d_drive * pc.reward_scale - struck * pc.strike_penalty
          - sick_onset * pc.sickness_penalty)

    if pc.kin_weight == 0.0:
        return own

    n = own.shape[0]
    if not pc.kin_audible:
        # Flat flock mean over the *other* hens: (sum - self) / (n - 1).
        # E005 showed this cannot work: the quantity is nearly identical for every
        # hen, so it moves the same way for the silent bird as for the caller, and
        # no hen can discover she is responsible for a call. Kept as a condition to
        # contrast against, not as a default.
        others = (jnp.sum(own) - own) / jnp.maximum(n - 1, 1)
        return own + pc.kin_weight * others

    # Audibility-weighted: a hen's reward reflects the welfare of the flockmates who
    # could actually hear her. Now the kin term differs per hen and covaries with who
    # was in earshot, which is the correlation the learning rule needs to credit her
    # own call. It is also better biology -- kin selection operates on the relatives
    # you actually help, not on a population average.
    d = jnp.linalg.norm(w_next.pos[:, None, :] - w_next.pos[None, :, :], axis=-1)
    d = d + jnp.eye(n) * 1e6                       # exclude self
    audible = jnp.clip(1.0 - d / cfg.hear_range, 0.0, 1.0)
    heard = (audible @ own) / (jnp.sum(audible, axis=-1) + 1e-6)
    return own + pc.kin_weight * heard


def update_traces(ps: PlasticState, r: jax.Array, motor: jax.Array,
                  reward_now: jax.Array, cfg: CoopConfig, pc: PlasticConfig,
                  pred_err: jax.Array = None) -> PlasticState:
    """Decay the eligibility traces toward current activity. Called every step."""
    a_f = cfg.dt / pc.tau_fast
    a_s = cfg.dt / pc.tau_slow
    a_m = cfg.dt / pc.tau_motor
    a_b = cfg.dt / pc.baseline_tau_s
    a_pb = cfg.dt / (pc.pred_bar_tau_s if pc.pred_bar_tau_s is not None
                     else pc.baseline_tau_s)
    # 1.0 while still calibrating, 0.0 once frozen. A float rather than a bool so it
    # multiplies the update cleanly under jit.
    frozen = (1.0 if pc.pred_bar_freeze_s is None
              else (ps.age_s < pc.pred_bar_freeze_s).astype(jnp.float32))
    err = ps.z_err if pred_err is None else ps.z_err + a_s * (pred_err - ps.z_err)
    a_l = cfg.dt / pc.tau_lag
    # The slow means track each trace on the same time constant as the reward
    # baseline, so "unusually active" and "better than expected" are judged over
    # comparable windows.
    baseline = ps.baseline + a_b * (reward_now - ps.baseline)
    # Accumulate this step's reward-prediction-error (E067) -- see PlasticState.m_acc.
    # Uses the just-updated `baseline`, matching what the old instantaneous `m =
    # reward - ps.baseline` computation in `_one_step` used.
    m_acc = ps.m_acc + (reward_now - baseline)
    return ps._replace(
        z_err=err,
        z_lag=ps.z_lag + a_l * (r - ps.z_lag),
        # Frozen once past `pred_bar_freeze_s` (E088). Traced rather than branched:
        # `age_s` is a JAX scalar, so a Python conditional here would bake in whichever
        # value it held at trace time.
        z_lag_bar=ps.z_lag_bar + a_pb * frozen * (ps.z_lag - ps.z_lag_bar),
        z_fast=ps.z_fast + a_f * (r - ps.z_fast),
        z_slow=ps.z_slow + a_s * (r - ps.z_slow),
        z_motor=ps.z_motor + a_m * (motor - ps.z_motor),
        z_fast_bar=ps.z_fast_bar + a_b * (ps.z_fast - ps.z_fast_bar),
        z_slow_bar=ps.z_slow_bar + a_b * (ps.z_slow - ps.z_slow_bar),
        z_motor_bar=ps.z_motor_bar + a_b * (ps.z_motor - ps.z_motor_bar),
        baseline=baseline,
        m_acc=m_acc,
        age_s=ps.age_s + cfg.dt,
    )


def _enforce_dale(w: jax.Array, dale: jax.Array, w_max: float) -> jax.Array:
    """Keep every weight on the sign of its presynaptic neuron, and bounded.

    Learning must not be able to turn an inhibitory neuron excitatory. Clamping the
    magnitude at zero also means a weight driven to zero simply leaves the
    connectome, which is what makes pruning fall out of the same operation.
    """
    magnitude = jnp.clip(w * dale[None, None, :], 0.0, w_max)
    return magnitude * dale[None, None, :]


def consolidate(p: BrainParams, ps: PlasticState, m: jax.Array,
                pc: PlasticConfig) -> BrainParams:
    """Write accumulated eligibility into the weights. Called every `interval` steps."""
    eta = pc.eta / (1.0 + ps.age_s / pc.critical_period_s)
    eta_out = pc.eta_out / (1.0 + ps.age_s / pc.critical_period_s)

    # Both trace factors enter as *deviations from their own slow mean*, not as raw
    # activity. This is the difference between a Hebbian rule and a covariance rule,
    # and E019 showed it is the difference between learning a policy and learning a
    # constant.
    #
    # Rates are sigmoids: strictly positive, mean ~0.27, and both traces are low-pass
    # filters that barely move. Their outer product is therefore dominated by the
    # product of the two means -- the same matrix every consolidation, scaled. Measured
    # on the old rule: `dW_out` was rank one (top singular value share 0.9981) and the
    # cortical drive varied by 0.7% of its own magnitude across three seconds of
    # behaviour. It could slide a constant offset up or down and do nothing else. What
    # it slid was peck, downward; the hen learned to stop eating.
    #
    # Centring makes the update signed and state-dependent: strengthen when this
    # neuron was *more active than usual* while the target was *more active than
    # usual* and reward beat expectation. That is the standard covariance form, and it
    # is also the biologically motivated one -- a synapse has no access to an absolute
    # activity scale, only to changes against its own recent history.
    dz_fast = ps.z_fast - ps.z_fast_bar
    dz_slow = ps.z_slow - ps.z_slow_bar
    dz_motor = ps.z_motor - ps.z_motor_bar

    # Recurrent weights. Only synapses that already exist are updated; growing new
    # ones is a separate, much rarer operation.
    dw = eta * m[:, None, None] * dz_slow[:, :, None] * dz_fast[:, None, :]
    w = p.W + dw * (p.W != 0.0)

    # Synaptic scaling: pull each neuron's total input weight back toward what it
    # was born with. Without it, a sustained positive modulator ratchets every
    # weight upward until the network saturates.
    row_sum = jnp.sum(jnp.abs(w), axis=2)
    correction = 1.0 + pc.scaling_strength * (
        ps.innate_row_sum / (row_sum + 1e-6) - 1.0)
    w = w * correction[:, :, None]
    w = _enforce_dale(w, p.dale, pc.w_max)

    # Motor readout: this is how the pallium earns influence over behaviour. Without
    # it the cortical pathway stays near-silent forever and nothing it learns can
    # reach a muscle.
    #
    # `m_out` is where H2f's alternative rule lives (E055): with `hebbian_readout`,
    # the reward-modulation factor is replaced by a constant, so the update no longer
    # asks "did this help" -- only "did this pre/post pair deviate from its own mean
    # together". The covariance form (dz_motor, dz_slow are both already deviations
    # from their slow means) still lets the update go negative for anti-correlated
    # pairs; what changes is that helpfulness no longer gates it either way.
    n_motor = p.W_out.shape[-1]
    m_out = jnp.ones_like(m) if pc.hebbian_readout else m
    dw_out = (eta_out * m_out[:, None, None]
              * dz_motor[:, :, None] * dz_slow[:, None, -n_motor:])
    # Readout scaling (E055 follow-up), the same correction `W` gets above, applied
    # to `W_out`. Off by default: only relevant once something removes the reward
    # gate's incidental stabilising effect, and changing this pathway's dynamics for
    # every existing reward-modulated result is exactly the silent-comparison-basis
    # change this project's conventions exist to avoid.
    w_out_raw = p.W_out + dw_out
    if pc.readout_scaling_strength > 0.0:
        row_sum_out = jnp.sum(jnp.abs(w_out_raw), axis=2)
        correction_out = 1.0 + pc.readout_scaling_strength * (
            ps.innate_row_sum_out / (row_sum_out + 1e-6) - 1.0)
        w_out_raw = w_out_raw * correction_out[:, :, None]

    # Dale's law holds here too. This was a symmetric clip until E027, so learning could
    # walk a motor-stub neuron's outgoing weight across zero and turn an inhibitory cell
    # excitatory -- the precise thing the invariant forbids, on the one pathway that
    # reaches a muscle. `_enforce_dale` expects the presynaptic sign per source column,
    # which for the readout is the motor stub's slice of `dale`.
    w_out = _enforce_dale(w_out_raw, p.dale[-n_motor:], pc.w_max)

    w_pred = p.W_pred
    if pc.pred_enabled:
        # Masked delta rule: move the prediction toward the observation, using only
        # channels that were actually observable. Factored into traces for the same
        # reason as the recurrent rule -- storing a per-synapse error would double the
        # memory traffic of a simulation that is already bandwidth bound.
        # Error now, against what the brain was doing a moment ago, and only from
        # neurons allowed to source a prediction.
        # Centred for the same reason the recurrent rule centres its own factors: an
        # uncentred, strictly-positive source makes the update dominated by the DC
        # term, so `W_pred` learns to predict from "the pallium is on" rather than
        # from what distinguishes this state from a typical one.
        lag = ps.z_lag - ps.z_lag_bar if pc.pred_centred else ps.z_lag
        d_pred = (eta_pred_of(ps, pc) * ps.z_err[:, :, None]
                  * (lag * p.pred_src[None, :])[:, None, :])
        w_pred = jnp.clip(p.W_pred + d_pred, -pc.pred_max, pc.pred_max)

    # E101-B: the descending gate learns on the same rule and schedule as the readout it
    # sits beside. Same postsynaptic factor (`dz_motor`), same presynaptic source (the
    # motor stub's slow trace) -- so if the gate stays open it is because nothing taught
    # it to close, not because it was denied the signal `W_out` gets.
    w_gate = p.W_gate
    if pc.reflex_gate and not pc.bg_gate:
        d_gate = (eta_out * m_out[:, None, None]
                  * dz_motor[:, :, None] * dz_slow[:, None, -n_motor:])
        w_gate = jnp.clip(p.W_gate - d_gate, -pc.gate_max, pc.gate_max)

    # E102's striatum learns on exactly the same rule and schedule the free gate used, so
    # a null cannot be blamed on a withheld signal -- only on the competition.
    w_str = p.W_str
    if pc.bg_gate:
        d_str = (eta_out * m_out[:, None, None]
                 * dz_motor[:, :, None] * dz_slow[:, None, -n_motor:])
        w_str = jnp.clip(p.W_str - d_str, -pc.gate_max, pc.gate_max)

    return p._replace(W=w, W_out=w_out, W_pred=w_pred, W_gate=w_gate, W_str=w_str)


def eta_pred_of(ps: PlasticState, pc: PlasticConfig):
    """Association rate, on the same critical-period decay as everything else."""
    return pc.eta_pred / (1.0 + ps.age_s / pc.critical_period_s)


def restructure(p: BrainParams, ps: PlasticState, key: jax.Array,
                pc: PlasticConfig) -> BrainParams:
    """Grow and prune synapses. Called every `growth_interval` steps.

    Growth is stochastic and gated on coactivity: a pair of neurons that fire
    together is a candidate for wiring together. It is also anatomically
    constrained -- a synapse can only appear where the innate connectivity priors
    allow an axon to be, so a hypothalamic cell cannot sprout onto the sensory stub
    just because the two happen to correlate.

    Pruning is not a separate mechanism. A weight driven to zero by learning is
    already gone; this only removes the ones loitering near zero.
    """
    live = p.W != 0.0
    n_live = jnp.sum(live, axis=(1, 2), keepdims=True)

    coactive = ps.z_slow[:, :, None] * ps.z_fast[:, None, :]
    mean_co = jnp.mean(coactive, axis=(1, 2), keepdims=True) + 1e-9
    p_grow = jnp.clip(pc.growth_rate * coactive / mean_co, 0.0, 0.5)
    p_grow = jnp.where(n_live >= ps.budget, 0.0, p_grow)   # respect the budget

    # One random tensor serves both decisions. Growth only considers dead synapses
    # and pruning only live ones, so the two draws are over disjoint supports and
    # cannot correlate. At (H, N, N) these allocations are the expensive part of
    # restructuring, so three of them cost real throughput and one does not.
    u = jax.random.uniform(key, p.W.shape)

    born = (~live) & p.growable[None, :, :] & (u < p_grow)
    doomed = live & (jnp.abs(p.W) < pc.prune_threshold) & (u < pc.prune_rate)

    # New synapses start weak, on the sign their presynaptic neuron is allowed.
    seed_w = pc.prune_threshold * 2.0 * p.dale[None, None, :]

    w = jnp.where(born, seed_w, p.W)
    w = jnp.where(doomed, 0.0, w)
    return p._replace(W=w)


def divergence_from_genome(p: BrainParams) -> dict:
    """How far a hen's connectome has drifted from the one she hatched with."""
    live = p.W != 0.0
    innate = p.mask[None, :, :]
    return {
        "live_synapses": jnp.sum(live, axis=(1, 2)),
        "grown": jnp.sum(live & ~innate, axis=(1, 2)),
        "pruned": jnp.sum(~live & innate, axis=(1, 2)),
    }
