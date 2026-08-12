# The hypothesis tree

The purpose of this file is to stop the project drifting into testing whatever is
convenient. Every experiment must name the hypothesis it feeds, and every result must
come back here and change something — a status, a prediction, or the tree itself.

**Rule: no experiment is run without a parent hypothesis recorded here.** If a
question does not ladder up to H0, either it belongs in the backlog as a new branch,
or it does not belong in the project.

Status values: `SUPPORTED` · `UNDER TEST` · `NOT STARTED` · `REFUTED` · `ABANDONED`

---

## H0 — root

> A neural model of a chicken, in an environment simple enough that its vision and
> motor budget can be spent elsewhere, can be given a communication channel that
> **measurably changes what the flock is able to do**.

Two halves, and they fail independently: the bird has to be a credible bird, and the
channel has to do real work. H1–H3 are the first half. H4–H6 are the second.

**Falsified if:** a flock with an intact channel performs no better than a
capacity-matched flock with a shuffled one, at any capacity, on any task requiring
private information.

---

## H1 — a small rate-coded network with innate reflexes behaves like a hatchling

**Status: SUPPORTED** (phase 0)

**Prediction:** documented neonatal chick behaviours appear without learning —
indiscriminate pecking at contrast, crouching at overhead looming, fleeing ground
threats, distress calling when isolated, approaching flockmates when cold, and
*distinct* aerial vs terrestrial alarm calls.

**Evidence:** 7/7 assays in `run/probes.py`. Referential alarm separation measured at
hawk→(aerial 0.99, ground 0.12), fox→(aerial 0.08, ground 0.94).

**Falsifier:** any assay failing, or passing only under hand-tuned staging.

---

## H1a — the environment creates an information asymmetry the flock cannot resolve individually

**Status: SUPPORTED** (phase 0)

This is the load-bearing precondition for the whole language half. If every hen can
always see every threat, no signal is ever worth making.

**Prediction:** a foraging hen is measurably blind to aerial threats.

**Evidence:** aerial channel 0.01 while pecking vs 0.87 head-up; crouch 0.06 vs 0.98,
in the same bird moments apart. Across an hour of flock time, hens are head-down ~64%
of the time.

**Falsifier:** head-down fraction near zero (nobody is ever blind) or near one
(nobody ever sees anything, so nobody has information to share).

---

## H2 — three-factor plasticity produces measurable behavioural improvement

**Status: SUPPORTED** (phase 1) — *for learning without structural growth, at 20 min
of chicken time.* Growth is explicitly not supported; see H2a.

**Prediction:** a flock learning under a reward-prediction-error rule regulates its
drives better over a rearing run than a genome-matched, coop-matched flock that
cannot learn. Operationalised as hunger declining across a run relative to control.

**Experiment:** `run/experiment.py`, matched-seed contrast across three conditions
(fixed / learning without growth / learning with growth).

**Falsifier:** no reliable difference from the fixed control across seeds. That would
mean the rule is producing connectome churn without behavioural consequence — which
is a real possibility and the reason this is stated as a hypothesis rather than a
feature.

**Evidence — positive but underpowered.**
[E001](experiments/E001-does-plasticity-help.md) returned a null.
[E002](experiments/E002-can-the-pallium-reach-a-muscle.md) found why: the cortical
readout was frozen, so nothing the pallium learned could reach a muscle.
[E003](experiments/E003-does-the-fixed-readout-rescue-learning.md) reran E001 with
only that fixed, and the effect tripled — mean hunger now *falls* across a run
(0.321 → 0.295) where the fixed control *rises* (0.306 → 0.370), with feeding rate up
from 5.2% to 6.5% of timesteps.

**[E004](experiments/E004-replication-at-twelve-seeds.md) settles it at twelve seeds:
−0.063 ± 0.016 SE, t=3.93 against a 2.23 threshold, p≈0.002.** Both pre-registered
predictions held. The learning hen feeds on 7.4% of timesteps against the control's
6.3%.

The effect size shrank 30% from E003's four-seed estimate (−0.090 → −0.063), which
the pre-registration had flagged as a possibility — E003's estimate *was* inflated by
small-sample luck, and it cleared anyway because precision improved faster than the
estimate fell. E003 also caught a bug in the *analysis*: the harness had used a 2-SE
threshold, which at four seeds would have called p≈0.09 a result.

**What the null turned out to be.** The cortical readout was learning at 2e-3, at
which rate `|W_out|` grew 1.00x over ten minutes — frozen. E001 was measuring a hen
incapable of acting on anything she had learned. E002 also found a ceiling that was
not anticipated: at `eta_out=2e-1` cortical drive overwhelms the innate arc and
behaviour gets *worse*, because a hen who overrides her reflexes with an untrained
pallium is worse off than one who does not. The default is now 2e-2, near the
optimum.

**Remaining scope limits.** Supported at 20 min of chicken time and for a single
learning-rate setting. Untested: whether the effect grows over a realistic rearing
(days), and whether it survives the growth rule (see H2a).

---

## H2a — does structural growth *hurt* learning?

**Status: NOT STARTED** — opened by E004, and it inverts the design's expectation.

Growth was the weaker condition in all three runs of the phase 1 contrast (E001
t=1.00 vs 0.78; E003 0.77 vs 2.50; E004 1.75 vs 3.93), and E004 attached a cost to
it: the no-growth hen reaches significance with **21,148 synapses**, having pruned
42% of her innate connectome, while the growth hen ends with **40,753** and does not
clear.

**Prediction:** the growth rule adds synapses on coactivity, which is orthogonal to
whether those synapses help, so continuous rewiring destabilises what has been
learned. Behavioural improvement should fall monotonically with growth rate above
some small optimum.

**Falsifier:** a growth-rate sweep showing a flat or rising relationship, which would
make the three-run ordering a coincidence of the on/off contrast.

**Note:** the default stays `growth_enabled=True` until this is settled. Three runs
of a binary contrast is not enough to overturn a default with a biological
justification behind it.

---

## H2c — a learned cue can recruit an innate response via top-down association

**Status: NOT STARTED** — opened after E007, and it supersedes the resolution E007
proposed.

E007 left the project at a fork: the innate and learned pathways sum into one motor
drive, so there is no setting where the pallium can both initiate new behaviour and
avoid overriding good reflexes. E007 proposed multiplicative gating of the reflex arc.

**That proposal does not work, and the reason is worth recording.** Gating scales an
existing reflex. In the case that matters — hearing an alarm call with *no hawk
visible* — the reflex input is zero, and any gain applied to zero is zero. Gating
cannot create a response to a cue the innate arc does not already respond to.

**The correction: wire the learned pathway to the sensory representation, not to the
motor output.** Let the pallium project back onto the observation the reflex arc
reads. Then hearing an alarm call does not have to *recreate crouching* — it only has
to recreate the *percept of a hawk*, and the innate reflex, which is already strong
enough, does the rest.

This is Pavlovian conditioning stated architecturally: the conditioned stimulus comes
to activate the representation the unconditioned stimulus would. It is also why the
magnitudes suddenly work. Supplying crouch drive directly needs +2.50 against a
measured cortical capacity of 0.002. Supplying it *through* the aerial channel needs
only ~0.3 in sensory units, because the innate reflex multiplies it by 8.0.

**Where the association comes from, and why it needs no reward.** A hen who is
head-up both sees the hawk and hears the flockmate's call — the two co-occur, and a
plain Hebbian rule associates them. Later, head-down and blind, the call alone
reconstructs the percept. She learns what the call means during the moments she can
check for herself, and can then use it when she cannot.

That resolves the credit-assignment problem that sank E005 and E006 as well, because
association needs no reward signal and no attribution of benefit to a caller.

**Prediction:** comprehension — crouching to a played-back alarm with no predator
visible — rises above zero with a top-down associative projection and stays at zero
without it. Measured on the existing assay, which E006 showed is sound.

**Falsifiers:** comprehension flat with the projection added; or comprehension
appearing but the hen now hallucinating percepts with no cue, which would mean the
projection is unconstrained rather than associative.

**Risk to watch:** a pathway that writes into a hen's own senses can make her
perceive things that are not there. That is the intended mechanism and also the
obvious failure mode, so the assay must check the no-cue baseline, not just the
cued response.

---

## H3 — learned usage reproduces the audience effect without being programmed

**Status: UNDER TEST** (phase 2/3) — two nulls recorded, and **blocked by H2b**.

Real cockerels alarm- and food-call far more readily with a hen present, graded by
audience type (conspecific > other species > empty cage). Call *production* is innate
in chickens and is wired in; *usage* is learned and deliberately is not.

**Prediction:** after rearing in a social coop, calling rate becomes conditional on a
flockmate being within earshot, without any term rewarding that.

**Why it matters:** this is the model predicting a documented behaviour it was never
told about. It is the strongest available validation that the learning rule is
capturing something real, and it is cheap — the behaviour is already measurable.

**Falsifier:** calling rate independent of audience after rearing.

**Evidence — [E005](experiments/E005-does-the-audience-effect-emerge.md): null.** The
food-call effect rose in the mean (+0.032) but the between-seed spread swamps it
(t=0.64). The alarm-call effect went *against* prediction (−0.031, t=1.43) and did so
more consistently than the food effect went with it. The assay itself is sound: the
fixed control is flat to three decimals in every cell.

**Leading suspect — structural, same class as E002.** The kin term adds
`kin_weight × mean(other hens' reward)`, which is nearly identical for every hen in
the flock. For a hen to learn that *her call helped*, the modulator must move
*because she called*; a flock average moves the same way for the silent bird beside
her. So the reward gives the flock a reason for calls to exist and gives no
individual a way to discover she is responsible for one — leaving the energy **cost**
as the only component correlated with her own calling, which predicts suppression,
which is the sign observed.

**[E006](experiments/E006-audibility-weighted-kin-reward.md): second null, and it
found the real blocker.** Audibility-weighting the kin term did not rescue the effect
(food t=0.90 vs flat-kin's t=0.64 — indistinguishable). But the comprehension assay
added in the same run returned the decisive number: **hearing an alarm call changes a
hen's behaviour by ~0.0005, before and after rearing, in every condition.** Nothing
responds to calls, so a call cannot help anyone, so it cannot repay its cost — leaving
the private energy cost as the only thing correlated with her own calling, which
predicts suppression, which is the sign seen on the alarm channel in both experiments.

**H3 is blocked by H2b, not by its own design.** See below.

---

## H2b — the learning rule cannot acquire behaviours outside the innate repertoire

**Status: SUPPORTED as a limitation** — established by
[E006](experiments/E006-audibility-weighted-kin-reward.md), refined by
[E007](experiments/E007-exploration-does-not-rescue-comprehension.md). **This blocks
H3, H4 and H5.**

The three-factor rule strengthens synapses that were active when the modulator moved.
It has no way to reinforce an action that never happened — and the model is
deterministic. No motor noise, no stochastic action selection, no exploration of any
kind.

So every behaviour a hen will ever perform is one the innate reflex arc already
produces. Learning can re-weight *when* existing behaviours fire; it cannot acquire a
stimulus-response pairing the reflex arc never visits. To learn "crouch when you hear
an alarm", she must at some point crouch on hearing one — and she never does, because
crouching is driven by *seeing* a hawk and `hen/innate.py` deliberately wires nothing
from the auditory channels.

**Why this was invisible until now.** H2 only required tuning behaviours the hen
already performs — approach food, peck, huddle. The limit binds the moment the
question is about acquiring something genuinely new, which is every hypothesis from
here on.

**[E007](experiments/E007-exploration-does-not-rescue-comprehension.md) refines this
and the refinement matters.** Exploration was added and comprehension did not move —
nor did it with 30x the predator density, nor with both. The measurement that
explains it:

```
drive needed for crouch > 0.5     : +2.50
innate reflex, hawk overhead      : +8.00
cortical drive to crouch, playback: +0.002   -- roughly 170x short
```

**The real limit is that the learned pathway can modulate behaviour but cannot
initiate it.** Exploration cannot help, because the problem is not that the action is
never sampled — it is that once sampled, the learned pathway has no way to reproduce
it. There is nothing for the credit to attach to.

**This is the same tension E002 found, seen from the other side.** E002: cortical
influence too high and behaviour degrades, because an untrained pallium overrides good
reflexes. E007: cortical influence low enough to be safe, and nothing new can be
learned. The two pathways sum into one motor drive and compete on a single axis, so
there is no setting that satisfies both.

**Open architectural decision** — additive competition versus multiplicative gating.
If the cortical pathway set a per-channel *gain on the reflex arc* instead of adding
its own drive, a learned signal could recruit an innate behaviour without generating
it from nothing, and could not run away because it only scales existing responses.
That is closer to how learned modulation of tectal circuits is thought to work. It is
a change to the core architecture and has not been made.

---

## H4 — an intact channel beats a shuffled one on a task requiring private information

**Status: NOT STARTED** (phase 4) — **this is the headline**

**Prediction:** on a task where information is private, costly to acquire alone, and
changes at an intermediate rate, a flock with an intact channel outperforms a
capacity-matched flock whose channel is shuffled between hens.

**Design:** the six-way condition ladder in `docs/backlog.md`. The headline contrast
is **L vs C?** (shuffled), not L vs a smaller brain — a reduced-capacity control
confounds capacity with language and cannot answer this.

**Falsifiers:** L ≈ C? at every capacity; or muting the channel in a *trained* L
flock costs nothing; or C− (extra capacity, no channel) ≈ L, meaning the neurons did
the work.

---

## H5 — compositional structure requires a transmission bottleneck

**Status: NOT STARTED** (phase 4+)

**Prediction:** with generational turnover — naive chicks acquiring the code from a
limited sample of incumbents — emergent signals become compositional (rising
topographic similarity). Without turnover, they stay holistic: one arbitrary symbol
per situation, no reusable parts.

**Counter-prediction worth stating up front:** raw capacity should *increase* channel
use while *decreasing* compositionality, because a large enough network can memorise
an arbitrary mapping instead of building a structured one. If both rise together,
something we do not understand is happening and it is worth stopping to find out.

---

## H6 — a hen's internal state is recoverable from her signals

**Status: NOT STARTED** (phase 4+)

**Prediction:** a post-hoc translator trained on (signal, ground-truth state) pairs
recovers internal state above chance, and its accuracy rises over generations.

**Caveat, which is not optional:** translator output is a learned mapping from a
scalar, not a report. See `docs/ethics.md` §6.

---

## Changelog

| date | change |
|---|---|
| phase 0 | H1, H1a recorded as `SUPPORTED`; tree established |
| phase 1 | H2 opened, experiment harness built |
| E001 | H2 null recorded. Churn without behavioural consequence. |
| E002 | Diagnosed the null: readout frozen. `eta_out` 2e-3 → 2e-2. |
| E003 | Effect appeared (t=2.50), short of threshold. Analysis bug found and fixed. |
| E004 | H2 → `SUPPORTED` (t=3.93, p≈0.002). H2a opened. Predator exposure retired. |
| E005 | H3 null. Kin reward cannot assign credit to the caller; E006 opened. |
| E006 | H3 null again. Comprehension is zero: no exploration. H2b opened, blocking H3-H5. |
| E007 | Exploration added; comprehension still zero. H2b refined: the learned pathway cannot *initiate*, only modulate. Architectural fork opened. |
