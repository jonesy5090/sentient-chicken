# Backlog: the language experiment

The hypothesis: **a flock with a productive communication channel can reach a goal
that an equivalent flock without one cannot.**

This document specifies how to test that without fooling ourselves. It is design
only -- nothing here is built yet.

---

## 1. The control has to be capacity-matched

The obvious control is a flock with reduced neural capacity that behaves like a
normal hen. That is a useful *reference point*, but it is not the control that tests
the hypothesis, because it varies two things at once: capacity and language. If the
small-brained flock fails, we cannot say whether it failed for want of language or
for want of neurons -- and since the extra capacity was added specifically to support
language, that confound is not a technicality. It is the whole question.

The fix is a ladder in which only one thing changes at a time.

| condition | capacity | channel | isolates |
|---|---|---|---|
| **N** natural | default | innate calls only | the "normal hen" reference |
| **C−** capacity control | expanded | none | did the extra neurons alone do it? |
| **C0** severed | expanded | emits, nobody hears | the motor cost of calling |
| **C?** shuffled | expanded | hears a *random* flockmate | **the real control** |
| **Cs** self-only | expanded | hears only herself | channel as private memory |
| **L** language | expanded | intact | the hypothesis |

**The headline comparison is L vs C?, not L vs N.** A shuffled channel delivers the
same bandwidth, the same signal statistics and the same energetic cost, carrying zero
information about the world. If L beats C?, information transfer is doing the work.
If L ≈ C?, the channel is decorative however impressive the transcripts look.

**Cs** matters more than it appears. A channel can help a *single* agent by acting as
external memory -- writing a symbol now and reading it back later -- which is not
communication at all. Without Cs in the ladder, that failure mode is invisible.

Keep **N** in the design anyway. It answers a different and genuinely interesting
question: does the language flock beat a naturalistic bird, or merely beat a
lobotomised one?

---

## 2. Don't design for a wall

The instinct is a task that is impossible without language. A hard binary is worth
avoiding: if the control scores zero, we learn only that the task was impossible, not
how much language helped or at what capacity it started paying off.

A graded task with a large, clean effect is strictly more informative, because it
yields a **dose-response curve** across the capacity ladder. Run every condition at
several pallium sizes (say 0.5x, 1x, 2x, 4x the default 256 units). The interesting
result is not "language wins" but *where the curve bends* -- the capacity at which
the channel starts carrying real information.

Predict two things that may come apart, per the emergent-communication literature:
more capacity should increase channel **use**, while *reducing* structural
**compositionality**, because a large enough network can memorise an arbitrary
holistic code instead of building a compositional one. If both rise together,
something interesting is happening.

---

## 3. Task candidates

The requirement is information that is **private, useful, costly to acquire
individually, and cannot be resolved by looking.**

The critical parameter is how fast the answer changes. Social learning pays only at
*intermediate* rates of environmental change: too stable and every hen just learns it
herself, too fast and nothing helps anyone. **The answer must change faster than an
individual can learn it, but slower than the flock can propagate it.** Finding that
band is a sweep, and finding it is itself a result.

### T1 — Shared vigilance — **done, see the T1 node in `docs/hypothesis.md`**

~~Nearly free... Metric: food intake at matched predation risk (compare Pareto
frontiers, not raw intake)... Prediction: L forages more than C? at equal risk...~~

**Built and run (E045), finally with the Pareto treatment this section always called
for** instead of the four single-point mean comparisons that preceded it (E026, E028,
E028b, plus a zero-compute re-analysis). **The intake prediction is not supported** —
null at every capacity across a 0.5×–4.0× sweep, nine checks total now agreeing. **What
is supported**: a real, capacity-robust safety benefit (L safer than C? at all five
capacities), which is genuine Pareto improvement even without the intake side.

~~Per-hen vigilance falling with flock size was never tested — still open, lower
priority~~ **Done (E046): falsified, and the reason corrects the mechanism.** `head_down`
falls with flock size for *both* L and C? at similar rates — no channel-specific
vigilance relaxation. Most likely a chorus effect (power-summed audio gets louder with
more hens, triggering call-suppression on loudness regardless of content), not a
strategy the reflex arc is structurally capable of having. The safety advantage itself
turned out to be flock-size-dependent (null at 4–8 hens, significant at 32) — a
population-statistics story (more independent chances of a true warning), not a
behavioural one. **Both of T1's original predictions are now settled.** Full results in
the T1 node.

### T2 — The rotating poisoned feeder (the headline experiment)

> **Extended and fully specified** following a design review that found two real gaps
> in the sketch below: no call in this project (real or built) is specific to bad
> food, and no contamination mechanic exists anywhere in the world model. Both are
> designed here rather than assumed. First formal pre-registration:
> [E060](experiments/E060-t2-contamination-scaffold.md) (Stage 1: build and validate
> the innate scaffold — the learning test is a later, separate experiment).

Several feeders; one is contaminated; which one rotates on a period tuned to the band
above. Eating from it is costly but survivable. Contamination is invisible until
tasted, so the information is genuinely private and genuinely transferable.

- **Metric**: total flock sickness per rotation period.
- **Prediction**: L converges toward one hen's mistake per rotation (the discoverer
  pays, everyone else is warned); C? pays roughly N times that.
- **Why this one**: it tests *reference* -- the signal must carry **which** feeder,
  not merely that something is wrong. And it extends a behaviour chickens genuinely
  have, since their food calls are already functionally referential.

#### What was missing, and the fix for each

**1. No signal for "this is bad."** Every existing call is either positive (food) or
about a predator (aerial/ground). Rather than invent a bespoke "danger: bad food" call
with no real-world counterpart, this uses a real, documented chicken vocalisation: the
**gakel-call** ("gackering"), a frustration / negative-expectation call given when an
anticipated good outcome turns out bad — described in the welfare literature as a
candidate indicator of negative affect, and (like the calls already built) it comes
with its own audience-sensitivity findings to eventually check the model against.
Production is innate, wired to fire once, on the rising edge of a sickness event —
architecturally identical to `IDX_FOOD_ARRIVAL` → `M_CALL_FOOD` (E053): a discovery
pulse, not continuous nagging for the whole sick period.

**2. No contamination mechanic.** New `World` state: which feeders are currently bad
(invisible — `food_amount`'s appearance is unaffected), rotating on a period. Eating
from a contaminated feeder (the existing `fed` rising edge, gated on contamination)
triggers a sickness event.

**3. Sickness needs a real, embodied consequence, not just a number.** Per direct
instruction: the hen who eats bad food should be **visibly slow and still for a
duration, then recover** — not an instant, invisible reward penalty. Modelled as a
timed state (`sick_t`, same idiom as `hawk_t`/`fox_t`'s dive/dwell timers), set on the
sickness rising edge and decaying to zero. While sick, a mobility multiplier
(mechanical, in `actuation.py`, the same way `crouch` mechanically zeroes locomotion —
this is a physiological constraint happening *to* her, not a decision the reflex arc
or learning ever gets to make) cuts her speed sharply, not to zero: **slowed**, not
frozen solid, matching "visibly slow / still" rather than a hard freeze. The gakel call
fires at the *onset* of this state (the discovery-pulse pattern above); recovery is
silent — she is simply moving normally again once `sick_t` reaches zero.

**4. The location problem** (flagged when T2 first came up: the food call — and by the
same logic, the gakel call — cannot carry *which* feeder acoustically; a receiver would
need to see the caller). **Fixed with a new vision class, not a new acoustic
dimension**: `CLS_SICK`, a flockmate-proximity signal (same `_bin_proximity` mechanism
as `CLS_FLOCKMATE`) gated to only the currently-sick hen(s). A flockmate now has a
directly perceivable, *located* cue — a hen slowed and visible at a specific bearing
and distance — for as long as the sickness lasts, which is exactly the duration a
receiver needs to notice her, orient toward her, and register where she is. The gakel
call and the visual cue play complementary, not redundant, roles: audio broadcasts
"something bad just happened" past visual range and gives the pathway a learnable
*stimulus* to condition on; vision supplies *where*. Neither alone would solve the
reference problem; together they do, the same two-channel structure (a call plus a
visual referent) real referential alarm calling already uses in this model (a hawk
call plus the sight of the hawk once you look up).

**5. Innate anchor, or learning has nothing to amplify.** H2f (E055–E057) established
directly, this session, that this project's only working non-instrumental learning
mechanism *amplifies an existing innate anchor into a targeted policy* — it does not
build an association from nothing (E058, E059 both confirmed this cleanly, the second
with a mechanistic explanation). T2's learned claim ("avoid *this* feeder, even once
the sick hen has recovered and left") is exactly the "build a new, durable,
location-bound contingency" kind of claim that formula cannot do unaided. So T2 needs
its own anchor, the same way the audience-effect task had one: an innate reflex that
turns a hen **away** from `CLS_SICK` — a hardwired, momentary avoidance of a visibly
sick flockmate's immediate vicinity, mirroring `CLS_CROWDING`'s personal-space
mechanism exactly (same wiring pattern, opposite turn channel, weight chosen the same
way). This is not invented from nothing either: avoidance of sick-looking conspecifics
is a broadly documented cross-species phenomenon (a "behavioural immune system"), not
chicken-specific literature this project can cite with the same precision as the
alarm-call work, but a real, plausible basis rather than an assumption. **The anchor
gives immediate, momentary spatial avoidance while the cue is visible; what learning
would have to add is durability — continuing to avoid the location after the hen has
recovered and the visual cue is gone.** That gap between the innate anchor and the
durable, referential claim is precisely what T2 tests.

#### Innate vs. learned, stated plainly (the question asked directly)

**Innate, all of it, matching this project's consistent split between wired
production/physiology and learned usage/durable association:**
- Contamination existing and rotating — a fact about the world, not a behaviour.
- The sickness event and its physiological consequence (slowed movement) — mechanical,
  like `crouch`, not mediated by any learned weight.
- Gakel-call production, firing on the sickness-onset pulse — call production is
  innate throughout this entire project (Konishi's finding), and this is no exception.
- The `CLS_SICK` visual channel — a sensory fact, like any other vision class.
- The proposed turn-away-from-`CLS_SICK` reflex — the anchor, hardwired like every
  other reflex in `hen/innate.py`.

**Learned — this is the actual hypothesis, the reason to build any of the above:**
whether a flock, using the same non-reward-gated readout rule validated for H2f
(`hebbian_readout` + `readout_scaling_strength` — the only mechanism this project has
ever gotten to build a targeted contingency), can turn the innate anchor's momentary
reaction into a durable, location-specific avoidance that persists after the sick hen
and her visual cue are gone. **Both, exactly as before** — an innate scaffold precise
enough to give the flock a fighting chance, wrapped around a learned claim the scaffold
does not itself satisfy. Whether the plain instrumental rule (H2's own, separately
null on general foraging) does anything here either is a natural secondary condition
to run alongside it, not the primary hypothesis.

#### Staging, following this project's own rule about testing the instrument first

Four stages, not two — the original plan (build, then test) skips exactly the
population-level checks that caught real defects for H4 (E024's control) and the
personal-space reflex (E025→E048), and T2's scaffold is bigger than either of those
(a world mechanic, a call, a vision class and a reflex, all new together).

~~**Stage 1**: build the world mechanic, the gakel call, the visual channel and the
innate anchor; validate all four against the ethogram, no learning involved.~~ **Done
([E060](experiments/E060-t2-contamination-scaffold.md)).** All four falsifier checks
pass (12/12 ethogram, 70/70 suite). Two real bugs found and fixed during validation,
not before it: contamination was being silently overwritten every step regardless of
staging (caught by a probe's own negative control), and `viz/web/app.js` hardcoded a
call-channel stride that would have misaligned rendering once `N_CALLS` grew.

~~**Stage 1b**: the same population-level check H4's audio channel and the
personal-space reflex both needed before anyone could trust a headline number — full
flock, no learning, and specifically: does contamination get discovered at a workable
rate, is the gakel call actually audible to nearby flockmates (measured, not assumed —
the E019/E024/E026 lesson), and does the anchor produce real dispersal away from a sick
hen.~~ **Done ([E061](experiments/E061-t2-population-scaffold-check.md)).** All three
checks pass at 16 hens: 22.3 discovery events / 20 min, gakel audibility real (11×
heard\|sick vs. heard\|not), anchor dispersal real (5.44m vs. 2.71m stripped). One
caveat recorded, not a failure: the sender-shuffle control retains 82% of intact's
correlation, the same architectural shape E024 found for the alarm channel — the flock
clumps, so it isn't specific to gakel and doesn't affect T2's claim.

~~**Stage 1c**: calibrate `contamination_period_s` against the band named in §2 above —
fast enough to matter, slow enough to propagate — a sweep, not a guessed constant.~~
**Done ([E062](experiments/E062-t2-contamination-period-calibration.md)).** Swept
{100,200,300,450,600}s, 16 hens, no learning. Audience saturates flat at every period
(gregariousness already clumps the flock tightly, E025) — propagation is nearly free
here regardless of period. Overlap (a rotation firing while a hen is still sick) held
43–61% at *every* period including 600s — ten times `sickness_duration_s` — because at
~5.6 discovery events per rotation (E061), cumulative sick-time (~335s) exceeds the
300s rotation itself; a measured picture of the "C? pays ~N times" no-learning
baseline, not a period defect. `contamination_period_s` stays at 300.0 — no evidence
favoured any other candidate. Per-feeder sickness attribution doesn't exist yet, so the
narrower stale-cue-misattribution risk this stage set out to check remains unmeasured;
worth building only if Stage 2's results look consistent with it.

**Prerequisite found and filled before Stage 2 could be designed
([E063](experiments/E063-allocentric-place-cells.md)).** T2's literal claim — durable
avoidance of *this specific feeder*, outlasting the visible cue, recognised from a
later approach in any direction — is not representable by anything built through
Stage 1c: every existing channel is egocentric by design (`coop/sensing.py`'s own
docstring), so a hen loses any trace of a location the instant she turns. Added a
fixed 5×5 innate place-cell grid, giving `hen/regions.py`'s previously-generic
`hippocampus` region its first real function. `OBS_DIM` 88 → 113. No reflex reads it
(raw location carries no innate meaning alone, by design); 77/77 full suite including
an unchanged ethogram, confirming nothing already-validated moved. Scaffolding, not a
result — Stage 2 can now actually be designed.

**Second prerequisite found and filled ([E064](experiments/E064-gakel-location-cue.md)).**
E063's place cells solve durable location memory only for a hen who directly *witnesses*
a sickness event (`CLS_SICK` and `CLS_FOOD` co-occur in her own egocentric view, so her
own place cells tag the moment). A hen who only *hears* the gakel call from beyond
visual range gets nothing spatial at all — audio in this model has never carried
direction, and self-location alone doesn't tell you where someone *else* is. Added a
loudness-weighted mixture of gakel callers' own place-cell patterns, reusing E063's
grid rather than asking the pallium to learn trigonometry from a bearing it isn't even
given. Required extending `World` with a `pos_log` ring buffer (matching `call_log`'s
existing pattern) so the yoked control hands a listener the caller's position *when she
called*, not her current one — the same class of leak E024's shuffled control had for
plain audibility, caught here before it could contaminate a result rather than after.
`OBS_DIM` 113 → 138.

**Effort assessment, not scoped: making place-cell tuning learned rather than fixed
at birth.** `README.md` already flags that E063's place cells depart from real biology
— in every species place cells have been studied in (most thoroughly rodents; the
avian literature is real but thinner, and mostly behavioural in domestic chicks
specifically, e.g. Vallortigara and colleagues' geometric-reorientation work following
Cheng — direct single-unit evidence of chick place-field *development* specifically
isn't something this project can point to with confidence) a cell's tuning forms
through exploring a particular environment, not before. Raised while reviewing
whether that departure is worth closing. Three tiers, increasing cost:

- **Tier 1 — experience-gated reliability.** Keep the current ready-made geometry, but
  scale the channel's strength by a per-hen, per-cell visitation trace (an EMA
  matching the decaying-pulse idiom `food_call_drive`/`sick_call_drive` already use,
  indexed by cell instead of scalar) — a totally novel area reads near-zero regardless
  of how close the hen actually is. Captures "you can't use spatial memory of
  somewhere you've never been" without touching the underlying map. One new `(H,
  N_PLACE)` `World` field, a small `world.step` update, a multiply in `sensing.py` —
  comparable in scope to E063 itself, a single experiment.
- **Tier 2 — learned tuning via competitive Hebbian plasticity.** Seed hippocampal
  units with broad or random initial receptive fields and add a new plasticity rule
  (`hen/plasticity.py`) that nudges each unit's preferred location toward wherever the
  hen currently is, weighted by its own response — a standard competitive-learning
  update, still using ground-truth position as the (imperfect) teaching signal but
  letting the *mapping* adapt rather than fixing it at `reset()`. A genuinely new kind
  of plasticity for this project — every existing rule adjusts connection weights, not
  the parameters of a sensory computation — needing its own validation ladder (geometry
  checks, a positive control that distinct hens develop distinct maps, full
  regression). Realistically a multi-experiment arc, comparable to how long H2f's
  single validated rule took to nail down (E055–E057) — on the order of 1–2 weeks at
  this project's pace, not a single PR.
- **Tier 3 — full path-integration + landmark-binding emergence.** No ground-truth
  position anywhere; place-like coding would have to emerge purely from integrating
  self-motion (which isn't currently fed back as a signal at all) with visual landmark
  recognition, via continuous-attractor or self-organising-map dynamics — the actual
  mechanisms the computational-neuroscience literature proposes for real place/grid
  cell emergence. A dedicated sub-architecture, not a channel addition, and in real
  tension with `dense JIT-compiled scan, throughput is a correctness constraint`
  (attractor dynamics typically need many settling steps per update). Weeks of
  research-engineering with a real chance of not converging to anything
  biologically-recognisable — a research project, not a backlog item with an effort
  estimate that means much.

**Recommendation:** none of this blocks Stage 2 — E063's fixed tuning is a stated,
deliberate simplification, not a functional gap. Tier 1 is cheap enough to be worth
doing opportunistically if a future result looks like it's exploiting unrealistic
birth-complete spatial knowledge (e.g. implausibly fast apparent "learning" in Stage 2
that's really the ready-made map doing the work) — a diagnostic-driven trigger, not a
scheduled task. Tier 2 is a real future direction, worth it only once a specific
hypothesis needs map quality itself to be a variable. Tier 3 is out of scope for this
project's cadence.

~~**Stage 2**: the actual L vs. C? contrast — does the flock, with the validated and
calibrated scaffold and the H2f-style learning rule, converge toward one hen's mistake
per rotation, as the original prediction states.~~ **Done, null
([E065](experiments/E065-t2-stage2-learning-contrast.md)).** 16 hens, H2f's rule
(E057), L vs. C? vs. a fixed baseline S, 8 seeds, 18 rotations. The falsifier fired:
L's early-to-late sickness-per-rotation change (+0.875) was not smaller than C?'s
(−0.250) — primary contrast +1.125 ± 0.715, t=1.57, not significant, wrong sign. S's
own change (+1.719) was nominally largest of the three, consistent with a within-run
trend unrelated to learning or channel content. `|W_out|` drift identical between C?
and L — a real limit on what this coarse diagnostic can distinguish from a small,
localised effect, not proof the rule was inactive. **T2 is `NOT SUPPORTED`** at this
configuration — the first time it has actually been tested against data, not just
designed. Not closed: the local-vs-aggregate weight-change question, a longer run, or
a different rule all remain genuinely open and unpursued.

**E065 withdrawn and corrected ([E066](experiments/E066-t2-stage2-corrected-rerun.md)).**
`hen/plasticity.py`'s `reward()` had no term for sickness at all, checked directly —
E065's null was very likely not a fair test, since the reward-gated pathway that
routes place-cell information toward motor output (`W`, unaffected by
`hebbian_readout`, which only touches `W_out`) had nothing to learn T2's outcome
from. Fixed with `sickness_penalty` (off by default). Also added a pre-registered
split of sickness onsets into *witnessed* (another already-sick hen within
`vision_range` — explainable by the innate anchor alone, identical across all three
conditions) vs. *testimony-only* (not witnessed — the only case the auditory channel
could plausibly help with), addressing the concern that the innate reflex could
dilute a small real effect in the aggregate metric. **Result: still null, more
solidly.** Primary contrast flipped to the predicted sign but stayed tiny
(−0.19 ± 1.25, t=0.15). Testimony-only onsets were ~1/3 the volume of witnessed ones
(the dilution concern was real) but showed no effect either (+0.06 ± 0.19, t=0.32,
wrong sign) — ruling out "the aggregate washed out a real effect." T2 stays
`NOT SUPPORTED`, now on solid methodological ground: both live objections to E065
checked directly, conclusion held.

**Candidate future task, scoped enough to know it is blocked: spatial memory for
feeder locations.** Deplete a feeder, and test whether a hen remembers where it was and
returns faster once it regrows — memory of location should speed up search. Attractive
because it needs *no new mechanisms*: `W_pred` could bind place-pattern → `CLS_FOOD`,
`pred_gain` injects that into `reflex_in`, and the existing innate food-approach reflex
fires. A complete loop from parts already built and off by default.

**Blocked by the environment, measured before building** (the check T2 never got):
**a hen can see food 98.3% of hen-steps**, and with depletion enabled at its default
only 1.7% of steps have no food visible anywhere, with feeders below `CLS_FOOD`'s
visibility gate just 0.33% of the time. In a 20 m arena with `vision_range=10` and four
feeders she never needs to remember where food is — she can look. A perfect spatial
memory would buy nothing measurable, so the metric cannot show a positive result
regardless of whether the capability exists. `CLAUDE.md`'s sixth instrument check,
answered no.

Before this task is worth building, the environment needs to make food *findable but
not visible*: fewer feeders, shorter `vision_range`, a larger arena, or much harsher
deplete/regrow dynamics. None is free — `vision_range` also governs predator detection,
arena size touches flock density and every dispersal result (E025, E048, E062), and
depletion has already been relitigated once (E025 → E037). Needs its own scoped design
and a sweep to locate settings where memory has room to pay, before any build.

A second, narrower blocker to resolve in that design: `CLS_FOOD` is *egocentric*
(bin-indexed, rotating with heading), and this model exposes no allocentric heading
channel — so a place→food association can support *recognition* ("I am at a food
place") but not obviously *navigation* ("food is that way"). The same gap that blocked
the gakel bearing problem in E064. Recognition alone may still pay via area-restricted
search, but that is the weaker claim and should be the one stated.

~~**Earlier framing of the same idea: food-call homing via place cells.**~~ Raised while reviewing whether E063 should also change the ordinary food
call's design — it shouldn't (see the reasoning below), but it surfaces a genuinely
new, separate, testable idea. The food call itself still cannot carry direction, for
the same reason gakel's plain amplitude couldn't (no bearing in this model's audio, and
place cells encode a listener's *own* location, not a caller's or a resource's) — so
wiring the food call to *innately* pull hens toward it would both fail mechanically and
break this project's central design principle (production innate, comprehension/usage
learned; hardwiring "call → approach" collapses exactly the question H2/H3/T2 exist to
test). What E063 genuinely opens up: a hen who has personally visited a food patch
before could *learn* to associate her own place-cell pattern there with "food," and use
that memory to navigate back to a remembered location out of sight, on hearing or
seeing some later food-related cue — a real capability that didn't exist pre-E063, and
learned rather than reflexive. Distinct from T2 (this is about *finding* food faster,
not avoiding a danger) and not scoped or pre-registered — a candidate for its own future
task if a hypothesis needs it.

### T2-revised — the same task, built associatively instead of instrumentally

> **Design only, nothing built.** Supersedes T2's Stage 2 architecture, not T2's
> question. The scaffold (E060–E062) and both location channels (E063, E064) are
> unchanged and reused as-is.

**Why the first architecture failed, precisely.**
[E069](experiments/E069-t2-positive-control.md) settled it: no `sickness_penalty`
magnitude across a thousandfold sweep produces learned avoidance, while the connectome
survives intact and the metric is demonstrably adequate. The reason is structural and
was predictable from two facts the project had already established separately —
E058/E059 (the reward-gated rule *amplifies an existing innate anchor* and never builds
an association from nothing) and E063 (place cells were deliberately given no reflex:
*"raw location carries no innate meaning alone, by design"*). Jointly those guarantee
no reward magnitude can produce place-specific avoidance. Tuning cannot fix it.

**The reframe.** T2's Stage 2 asked an *instrumental* question — "was that good for
me?" — routed through `m`, the reward-gated pathway. The behaviour it actually needs is
*associative* — "these two things co-occurred" — which needs no reward and no credit
assignment at all. `hen/plasticity.py`'s own docstring says exactly this about
`W_pred`: association between co-occurring stimuli "needs no reward and no attribution
of benefit to anyone, which is exactly the credit-assignment problem that sank E005 and
E006."

Building it associatively **sidesteps E067's eligibility defect and E069's finding
entirely** — neither applies to a rule that isn't reward-gated. This routes around the
broken machinery rather than through it.

#### The chain, and what each link needs

| link | mechanism | status |
|---|---|---|
| a hen falls sick, calls | gakel call, innate production on sickness onset | **built** (E060) |
| flockmates hear it | audio channel, audibility measured at 11× | **built** (E061) |
| the call carries *where* | gakel location cue, caller's place-pattern, loudness-weighted | **built** (E064) |
| the call means *something bad* | innate withdrawal response to hearing gakel | **new (1)** |
| testimony and experience share a map | shared allocentric population for both place channels | **new (2)** |
| "that place is bad" is remembered | `W_pred` binds place-pattern → gakel, non-reward-gated | **built, off by default** |
| the memory changes behaviour | `pred_gain` injects prediction into `reflex_in` | **built** |
| food there is avoided too | withdrawal competes with food approach | **free, if (1) acts on approach** |

Only two pieces are genuinely new. The associative machinery already exists and already
feeds perception: `hen/brain.py`'s `predicted = einsum(W_pred, src * pred_src)` followed
by `reflex_in = clip(obs + pred_gain * relu(predicted), 0, 1)` — a learned prediction is
added directly to what the reflex arc reads. A hen who has bound place P to the gakel
channel will, on returning to P, *perceive a call that is not there*.

#### New mechanism 1 — an innate withdrawal response to hearing the gakel call

A fixed reflex: heard gakel amplitude → reduced forward drive (and/or turning away).
Wired in `hen/innate.py` alongside the existing call-production weights.

**Why this is not smuggling in the answer**, which the earlier "aversion signal →
avoid" sketch genuinely was. The anchor is on the *call*, not on any place. Nothing
location-specific is wired anywhere. Which places are aversive is entirely learned, and
learned only from co-occurrence. This is the same production-innate / usage-learned
split the project applies everywhere, and structurally identical to `CLS_SICK`'s
turn-away being innate while *which hen is sick* is not.

**Biological grounding, and its limit.** `auditory_scaffold`'s own justification applies:
naive chicks do show innate responses to a conspecific's fear call before any learning.
That literature is about *alarm* calls; whether naive birds respond innately to the
gakel call specifically — a frustration/negative-affect vocalisation — is not something
this project can cite with confidence. State it as a modelling assumption, not a
finding. It should follow `auditory_scaffold`'s precedent exactly: opt-in, off by
default, documented as scaffolding, and never on in a headline condition without saying
so.

#### New mechanism 2 — a shared allocentric representation

The gap that makes this necessary: she hears the call *from a distance*, so her own
place cells (`PLACE_LO..PLACE_HI`) encode where **she** is, not where the bad feeder is.
E064's cue (`GAKEL_PLACE_LO..GAKEL_PLACE_HI`) carries the caller's location and
deliberately reuses E063's identical grid geometry — but they are **different
observation indices**. A pattern learned on the testimony channel does not transfer to
the self-location channel when she later walks there. Without this, testimony can only
ever teach her about the spot she was standing on when she heard the news.

The fix is a **connectome prior, not a behavioural one**: both channels project onto a
common allocentric population (naturally, the `hippocampus` region E063 gave its first
real function), so cell *k* of that population responds to place *k* whether the
evidence arrived by testimony or by being there. Nothing about which place is good or
bad is implied — only that a place is the same place however you learned of it, which is
what a cognitive map *is*.

Implemented in `hen/connectome.py` as a structured projection replacing the default
random afferent draw for these two blocks specifically — the same kind of targeted
wiring `modality_segregated` already does for the auditory pathway.

#### Why L vs C? remains a real test

This is the objection that killed the earlier sketch and it does not apply here. Under
`channel_mode='yoked'`, a hen hears the flock's real gakel calls time-shifted, so
`W_pred` binds them to **the wrong places** — her associations are as strong as L's and
point somewhere harmless. Under `intact` they land on locations that were genuinely
contaminated. Both conditions learn equally hard; only L learns something *true*. The
contrast measures information transfer and nothing else, which is what
`docs/backlog.md` §1 requires of the headline comparison.

E064's `pos_log` already guarantees a yoked listener receives the caller's position
*as of when she called*, not her current one — so the wrong-place binding is genuinely
decorrelated rather than accidentally informative.

#### Staging

1. **Wire and validate mechanism 1 in isolation.** Ethogram probe: a hen hearing a
   staged gakel call withdraws; one hearing other calls does not. No learning.
2. **Wire and validate mechanism 2 in isolation.** Positive control, and the one that
   matters most: stage a hen at place P, confirm the shared population responds
   similarly whether P is signalled by self-location or by testimony. If this fails
   nothing downstream can work, and it fails silently.
3. **Positive control for the whole chain, before any contrast.** Hand-plant a
   `W_pred` association (place P → gakel) and confirm a hen avoids P with no learning
   involved. **This is the step E065 skipped and three experiments paid for.** If a
   hand-wired success is undetectable, stop.
4. **Only then** the L vs C? contrast, with the same metric and MDE E069 characterised
   (19–35% of baseline at n=8 — adequate for T2's predicted effect).

#### Known unknowns, stated before building

- **`tau_lag = 1.5 s` may be too short.** It bridges cue→outcome for `W_pred`. Binding
  is roughly simultaneous here (she hears the call while the location cue is active),
  so 1.5 s is plausibly enough — but this is an assumption, and step 3's positive
  control is where it gets tested rather than assumed.
- **`pred_gain` has never run in a headline condition.** E058/E059 used
  `pred_gain=0.0`. Its interaction with the reflex arc at nonzero gain is largely
  uncharacterised, and `hen/brain.py`'s `relu` + `clip` means a strong prediction can
  saturate a channel.
- **Hallucination is the failure mode to watch.** A hen who over-predicts gakel calls
  perceives danger everywhere and stops foraging. `W_pred`'s existing `pred_max` clip
  bounds this; whether it bounds it *enough* at a behaviourally useful `pred_gain` is
  unknown.
- **This does not rescue the reward-gated rule**, and is not meant to. E067's
  `strike_penalty` audit remains open and untouched by any of this.

### T3 — The safe corridor (stretch)

Food beyond a region where some ground is dangerous; the safe route changes
periodically. A hen who survives the crossing knows something no one else can see.

- **Tests**: compositionality (direction *and* distance) and **displacement** --
  reference to a place not currently visible, which is the real Rubicon and the thing
  most likely *not* to emerge without explicit pressure.
- Hardest to build and to interpret. Do it last, and treat a negative result as a
  finding rather than a failure.

### Rejected: time-of-day gating

A feeder that opens on a schedule is individually learnable -- every hen can discover
it alone, at low cost, and then it is hers forever. No information asymmetry survives
past the first day, so there is nothing to communicate. Worth stating explicitly so
it does not get proposed again.

---

## 4. Generational turnover is not optional

Compositional structure needs a transmission bottleneck: each generation acquiring
the code from a limited sample of the last. Without it the expected outcome is a
holistic code -- one arbitrary symbol per situation, no reusable parts.

So the protocol needs periodic replacement of a fraction of the flock with naive
chicks who must learn the code from incumbents. This is both the mechanism that
drives structure and, conveniently, just chickens breeding.

---

## 5. Measurement

**Primary**: task performance per condition per capacity.

**Causal efficacy — the one that actually proves it.** Take a *trained* L flock and
mute the channel at test time. The drop in performance is the amount of work language
was doing. Correlational evidence that hens call a lot is not evidence that calling
matters.

**Channel content**: mutual information between emitted symbols and world state
(which feeder, which predator class, where).

**Comprehension via playback**: present a recorded call to a hen with no
corresponding stimulus and measure the behavioural response. This is precisely the
protocol Evans & Marler used on real fowl, which makes the results directly
comparable to the animal literature — a rare opportunity and worth designing for.

**Compositionality**: topographic similarity (Spearman correlation between distances
in world-state space and Levenshtein distances between messages), plus positional and
bag-of-symbols disentanglement.

**Legibility**: a post-hoc translator mapping symbols to English by correlating
against ground-truth state. Do not constrain the language to be readable — let it
evolve, then translate. Translator accuracy over generations is itself a metric: it
measures how much of a hen's internal state is recoverable from her calls.

---

## 6. What would falsify the hypothesis

Worth writing down now, while it is still cheap to be honest:

- **L ≈ C?** at every capacity — the channel carries no usable information.
- **Muting a trained L flock costs nothing** — the channel was correlated with
  behaviour but not driving it.
- **C− ≈ L** — the extra neurons did the work and language was incidental.
- **No rotation period exists where social learning beats individual learning** — the
  task does not actually require communication, and needs redesigning.

---

## 7. Order of work

> **Rewritten (2026-08-20).** The list below had gone badly stale: it still ranked T1
> and the condition harness as future work when both were long done, and its blocking
> item cited E013–E015, which E020 superseded around sixty experiments ago. What it
> described was the project as of roughly E015. Kept as a record beneath the current
> ordering, because the route is part of the record.

**Current order (revised 2026-08-20, after E082–E085):**

1. ~~**T2-revised's whole-chain positive control with a discriminative plant.**~~ **Done,
   and it opened three further instrument defects rather than closing one.**
   [E082](experiments/E082-t2-chain-control-redone.md) established that the chain
   conducts end to end. [E083](experiments/E083-gakel-anchor-produces-leaving.md) found
   the plant **anti-selective** in the live run (0.656 at the target, 1.244 elsewhere) —
   fitted on a *parked* hen, read back on a *moving* one — and removed a functional freeze
   from the gakel anchor. [E084](experiments/E084-live-place-decoding.md) found the
   occupancy metric could resolve only 18.3% at n=4 against a pre-registered 15%, so both
   falsifiers were guaranteed to fire. [E085](experiments/E085-repaired-instrument.md)
   repaired both and measured the result.
2. ~~**Rewrite the H2d node around E081.**~~ **Done** (PR #55).
3. ~~**Fix the `|W_out|` "is the rule active?" diagnostic.**~~ **Done.** `run/simulate.py`
   now also reports `w_norm` — mean `|W|` over live synapses — and `run/diagnose.py`
   prints both with the distinction stated. Measured on a 3-minute default run: `|W_out|`
   is **flat at 0.0608 → 0.0608 (1.00×)** while `|W|` moves 0.08643 → 0.08557. Under
   `hebbian_readout` the neuromodulator is replaced by a constant for `W_out` only, so it
   drifts identically whether or not a reward arrived; `W` stays reward-gated. Masked,
   because ~86% of the matrix is structurally zero.

**Blocking everything in T2 — the place representation.** E085 measured position as
linearly decodable from pallial state at about **four points above chance while the hen is
moving** (54.3% on balanced-split seeds, t=+2.95), against 84.6% when she is parked. There
is not enough place signal for `W_pred` to bind a place to a call. The architecture review
in the same session traced why, and it is concrete:

- **The hippocampus is not in the loop.** `regions.py:17` names it "place and spatial
  memory" and E063 was written up as giving it its first real function. Measured: of the
  64 units receiving place afferents, **64 are in the sensory stub and 0 are in the
  hippocampus** — and the hippocampus (320–400) is **excluded from `pred_src`**, so even
  if place reached it, `W_pred` could not read it.
- **A 64-unit bottleneck carries all 138 observation channels**, so 50 spatial channels
  compete with hunger, vision and audio at 30% afferent density.
- **Nothing amplifies place.** The asymmetry that made H2f work is that the rule amplifies
  what innate wiring emphasises (E058/E059, E069); E063 deliberately gave the place
  channels no innate anchors, correctly, and the cost is that position has no route to
  dominance in pallial variance.

Fixes, in order: **(a)** route place cells to the hippocampus and add it to `pred_src`;
**(b)** a weak innate place→hippocampus projection, wiring *that there are places* and not
*which place is aversive*; **(c)** widen the place population; **(d)** failing all three,
re-scope T2 as not reachable in this architecture, which is a finding rather than a
failure. **Do not run the L vs C? contrast against the current representation.**

4. **Audit `strike_penalty`'s reward-eligibility defect.** HIGH PRIORITY and unresolved:
   [E067](experiments/E067-reward-eligibility-sampling-defect.md) measured discrete
   reward events reaching `consolidate()` ~2% of the time, and **no prior conclusion has
   been checked against it.** `CLAUDE.md` records `strike_penalty` as 87% of reward
   variance at the H4 configuration, so this plausibly touches several recorded results.
5. **T3 — the safe corridor.** The only original task candidate never attempted, and the
   only remaining item from the old list.
6. **Generational turnover** (§4 above, "not optional"), and the behaviours the tree
   needs but the code lacks — spatial memory (currently environment-blocked, see T2's
   section), social hierarchy.

**~~Superseded ordering, kept for the record:~~**

1. ~~**Phase 1 plasticity — REOPENED, and it blocks everything below.** E013 is the
   first clean test and learning comes out significantly worse than no learning
   (t=3.85)... Nothing below is worth running until a rule exists that does not make the
   bird worse.~~ **Superseded by [E020](experiments/E020-h2-after-the-fixes.md)**: after
   E019's three fixes the harm is gone (+0.062 → +0.001, t=0.08). H2 is a clean null,
   not a harm.
2. ~~Condition harness: the six-way ladder as a config.~~ **Done** — `run/h4.py`'s
   `LADDER` carries all six plus a lesioned arm.
3. ~~T1 shared vigilance.~~ **Done** — [E045](experiments/E045-t1-pareto.md),
   [E046](experiments/E046-t1-vigilance-flocksize.md). Both original predictions settled.
4. ~~Rotation-period sweep to locate the social-learning band.~~ **Done** —
   [E062](experiments/E062-t2-contamination-period-calibration.md); no reason found to
   move off 300 s.
5. ~~T2 poisoned feeder — the headline experiment.~~ **Run, and answered instrumentally**
   — `NOT SUPPORTED` (E065/E066/E068/E069). The associative route is item 1 above.
6. ~~Playback and lesion assays.~~ **Done** — `run/audience.py` for playback, `h4.py`'s
   `Lx lesioned` condition for lesions.
7. ~~T3 safe corridor.~~ Still open — item 5 above.

## 7-. What to do next, after the second review ([E022](experiments/E022-second-review-verified.md))

This supersedes the ordering in §7 and §7a. Verified independently; the review's own
top-ranked item is **not** here, because its headline number did not replicate.

1. **Fix the `dale` sampling.** The pallium has **zero inhibitory neurons** — E/I is
   assigned by flat index over a region-ordered array, so it segregates by region
   instead of mixing. Sensory, pallium and hippocampus are 100% excitatory; hypothalamus
   and motor stub are 100% inhibitory. Two lines. Invalidates every genome, so it needs a
   deliberate re-baselining like E010's. Almost certainly the source of the knife-edge
   gain that `connectome.py:78-81` complains about without diagnosing.

2. **Promote `fed %` to the primary metric.** Hunger change measures the *sign of*
   `f − 6.17%`, and hens start at hunger 0.30 which is exactly that equilibrium. It
   correlates −0.94 with `fed %`, which is already printed beside it, and `fed %` has the
   better pairing correlation (ρ +0.914 against +0.791) as well as no knife edge.

   **Do not block on food layout.** The review ranked that first as a free 5× win; it is
   the opposite. Measured on the quantity that actually powers a matched-seed contrast —
   the paired-difference sd, not the marginal spread both earlier checks used — pinning
   the layout *raises* sd_d (hunger 0.0387 → 0.0443; fed % 1.69 → 2.89) and destroys the
   pairing (ρ +0.791 → +0.335; +0.914 → **−0.525**). Layout is a *shared* nuisance, so
   pairing already cancels it for free; removing it removes the correlated component that
   was making the pairing work. E022 addendum.

3. ~~**Run H4 with no plasticity at all.**~~ **Done and the control failed
   ([E024](experiments/E024-h4-without-plasticity.md)).** The ladder works; T1 does not.
   A shuffled sender still reports your hawk because 38.8% of the flock shares it. See
   item 0 below — **flock dispersal now blocks the headline experiment outright.**

0. ~~**Make the flock spread out.** Promoted above everything.~~ **Demoted — the
   diagnosis was wrong.** I promoted this on the reasoning that clumping defeated the
   sender-scrambling control. E025 tested it directly: food depletion did not disperse
   the flock (23.0% → 21.9% strike-radius overlap), and even at gregariousness weight
   **zero** the permutation retained 91% of the information. E026 found the surviving
   component was **temporal, not spatial** — every hen hears every other, so scrambling
   *who* you hear preserves "someone is calling right now", which is nearly the whole
   signal. The control that works is yoked. Dispersal was never the problem, and this
   item sat at the top of the list for two experiments on my mistake.

   Still worth doing eventually for T2, where a signal naming a feeder is useless if
   everyone stands at one. Not a blocker for anything now.

~~3. **Run H4 with no plasticity at all.**~~ *(superseded, see above)* §7 says phase 1 blocks everything below; it does
   not block H4, whose prediction mentions no learning. Production is innate and passes
   7/7; calls are audible since E019; comprehension can be innate via the E018 scaffold.
   Needs the capacity ladder wired into `run_condition` (it hardcodes `DEFAULT_REGIONS`)
   and a channel-shuffle path in `sensing.py`. **Neither exists, and this is the
   project's actual thesis.**

**Owed checks** (from the same review). Two are now settled:

- ~~`strike_penalty` still charging per-step~~ — **verified and fixed
  ([E028](experiments/E028-instrument-repair.md)).** It was real, and worse than filed:
  87.3% of the reward variance at the H4 configuration. Now 0.2%.
- ~~Dale's law on `W_out`~~ — **verified and fixed (E028).** 0 of 48 columns complied.
  E022 marked this "verified — adopt" and it then fell off the action list entirely.

Still unverified: **the 0.2 s credit window against a 10–30 s approach task** — now the
oldest and most consequential owed item, because `hen/plasticity.py:34-37` states in the
source that anything bridging a longer gap "is not learnable by this rule as written",
which would make every H2 null uninformative by `CLAUDE.md` check 6. Also outstanding:
the audience assay's saturated aerial channel in its audience cell; E006's kin term
moving the teaching signal by 0.5–3%; `W_pred` costing ~11% of throughput
unconditionally.

**And a positive control has still never been run**, though `CLAUDE.md` says one is not
optional. No experiment has shown the harness detecting a deliberately planted
improvement. Until one does, "the rule did not learn X" is not a statement about the
rule.

**Abandon:** H2a (six comparisons, never significant, sign flipped in E020) and the E016
"last word" follow-up (a workaround for a harm that no longer exists).

## 7a. The three E019 defects — ~~blocking everything~~ **all fixed**

All three verified and fixed in [E019](experiments/E019-three-verified-defects.md) §7,
with six guard tests at `n_hens=16`.

1. ~~**Make calls audible.**~~ **Done.** Floor-subtraction at emission plus power-domain
   combination at sensing. A full-amplitude alarm from an adjacent bird now moves the
   receiver's channel by **+0.908**, from 0.0000. Holds 4–32 hens.

2. ~~**Give `W_out` more than one degree of freedom.**~~ **Done**, and the framing was
   wrong: rank is not the right measure. A rank-one `ΔW_out = u vᵀ` still contributes a
   state-*dependent* `u (v · motor_stub)`. Centring both traces made the rule a
   covariance rule and raised drive variability 11× (0.007 → 0.080) while rank stayed
   at 0.999. Whether 8% is *enough* is what re-running E013 answers.

3. ~~**Take the vigour term out of `reward()`.**~~ **Done.** Reward is now hunger 54%,
   cold 46%. The cost stays real in the world.

**The meta-item stands and is the durable part.** All three were quantities checked in
the place they had just been moved *from*. When a term is relocated, measure it in its
new home.

### Immediate follow-ups created by the fixes

- ~~**Re-run E013.**~~ **Done — [E020](experiments/E020-h2-after-the-e019-fixes.md).**
  The harm is gone (+0.062 → +0.001, t=0.08) and the erosion with it (48% → 2.5%). H2
  moves from `REFUTED at this timescale` to a clean null. E013, E015 and E016 superseded.

- ~~**Does learning repay the cost of its own exploration?**~~ **Struck — tested and
  failed ([E021](experiments/E021-the-cost-of-exploration.md)).** On fresh seeds
  (12–23) learning is **+0.021 ± 0.027 against noise-only, the wrong sign and not
  significant**. H2's one nearly-positive result is gone. Recorded rather than quietly
  dropped: it was flagged post-hoc, tested properly, and falsified, which is the process
  working.

- ~~**Why did exploration became costly?**~~ **Moot ([E021](experiments/E021-the-cost-of-exploration.md)).**
  There is no cost to explain. It did not replicate in either audio regime; the
  current-audio pair went from +0.032 (t=3.84) to −0.000 (t=0.01).

- ~~**Re-check every tree status that rests on a single seed block.**~~ **Closed for
  E004/E016.** E016 was marked superseded at the time. E004's t=3.93 is the most
  re-checked number in the tree — E010 first (t=3.93 → t=0.08 on the same design,
  corrected gain alone), then independently by the full E013→E020→E021→E023→E037 chain,
  landing on today's +0.0003 ± 0.0156, t=0.02. See `docs/hypothesis.md`'s H2 section.

- **Report per-seed spread in `run/experiment.py`**, not just the mean and SE. A
  homogeneous block should be visible while the run is happening, not two experiments
  later.

- **Attribution ladder.** Four things changed between E013 and E020. Now that the status
  has actually moved, which one did it is worth the runs.

- ~~**Re-run E018**, unchanged in design — only its instrument was broken.~~ **Done
  (E036).** Falsifier fired: `S+L − S = -0.005, t=2.25`, wrong-signed. New node H2f.
- ~~**Re-measure H2d** against a call channel that now varies.~~ **Done (E034).**
- ~~**The innate food call fires on sight, out to 10 m.** Twelve of sixteen hens
  food-call continuously, so that channel is saturated by *genuine* calling and carries
  no information. Real cockerels food-call on *finding* food, and are audience-sensitive
  about it. This is a reflex-arc change and needs its own hypothesis node — it is not a
  bug fix and should not be done as one.~~ **Done ([E053](experiments/E053-food-call-discovery-pulse.md)).**
  Replaced continuous `CLS_FOOD` sight-gating with a discovery pulse (`IDX_FOOD_ARRIVAL`):
  fires on the rising edge of arriving at a patch, decays over 4s regardless of how long
  she stays. Flock-wide calling fraction dropped 42.8% → 4.2%, no hen left above 50%
  (was 4/16). Production stays innate and audience-blind, unchanged — only the temporal
  trigger changed. `OBS_DIM` 71 → 72.
- ~~**The flock clumps** — nearest-neighbour 0.23 m in a 20 × 20 m run. Flagged by the
  same review. T1 (divided vigilance) and T2 (which feeder is poisoned) both assume
  hens are somewhere different from each other. Nothing disperses them: food never
  depletes, so there is no foraging competition.~~ **Stale — food depletion was added
  and tested (E025, file written retrospectively).** It does not disperse the flock
  either; gregariousness's attraction-only wiring is the actual cause, still unfixed,
  still needs a crowding/individual-distance channel. Depletion stayed on regardless and
  turned out to have a large, previously unmeasured side effect on foraging baselines at
  20+ minutes (E037) — see the new item below. **The crowding channel itself is now
  built (E048)**: a new `CLS_CROWDING` vision class, zero until a flockmate is well
  inside personal-space range then ramping to 1 at contact, wired to turn the hen
  *away* at a weight that beats attraction, without the huddling/feeding collapse full
  removal of gregariousness causes. **Whether it disperses the flock enough to matter
  is a separate, now-answered question: [E050](experiments/E050-shuffle-info-recheck.md)
  found the hawk-targeted clustering H4's control depends on unchanged from before the
  fix** (38.4%, replicated on two 8-seed blocks, vs. E024's original 38.8%) —
  E048's own 3-seed reading of an improvement there does not replicate. `OBS_DIM` moves
  59 → 71 regardless of that finding —
  everything using the vision layout re-baselines from here, the same way E023 did.
- ~~**Audit `food_deplete_rate`'s effect on other tree results.**~~ **Closed — all four
  highest-stakes results checked directly.** H2 (E037): confound real, number corrected.
  H2e/E032/E033 (E038): confound real, sign reversed, status reverted `REFUTED` →
  `UNDER TEST`. H4 (E039): confound present at this duration too (one feeder to 4.3% by
  minute 10) but not consequential — result holds, significant even at reduced n. H2f/E036
  (E040): also not consequential — result reproduces to three decimal places. Two
  affected, two not; checking each was the only way to know which, and reasoning by
  analogy from one result to another was tried (for H4 and H2f both) and worked for
  neither — both got checked anyway rather than left on the argument.

  **Genuinely still open**, narrower than before: whether anything beyond
  `n_hens=16`/20–30 minute durations (other flock sizes, other durations, any future H3
  harness) needs the same treatment. Not exhaustively swept — the four results actually
  in the tree were the ones checked.

## 8. Open items from experiments

- **THE STRATEGIC QUESTION, now live rather than rhetorical.**
  [E115](experiments/E115-a-real-basal-ganglia.md) tested the best-motivated structural
  hypothesis this project has had — that the missing subpallium was why the model cannot
  select actions — and it failed like the five before it. **Six explanations for H2's
  null, six failures**, and the sixth was different in kind. Fixing the brain has now been
  tried. The two routes that remain are:
  1. **Generational selection** (§4 above, never started). H0 does not require
     within-lifetime plasticity — H4 is `SUPPORTED` and runs with it *off*. Evolving
     connectomes across generations sidesteps the whole H2 arc, and the transmission
     bottleneck it needs is what H5 requires anyway. **It is the only route that reaches
     H5 at all.**
  2. **Write up the null.** [E111](experiments/E111-is-there-headroom.md) made it
     informative by proving the headroom is real (0.21 hunger units, t≈7, replicated), so
     "this rule in this architecture cannot acquire a foraging policy, and here is what
     six mechanisms ruled out" is a result rather than an absence of one.
  Adding a seventh mechanism is not on this list. The base rate is six from six.

- **Why turning specifically?** [E114](experiments/E114-does-the-gate-work-through-vigilance.md)
  showed the gate's benefit is carried by turn suppression, not vigilance, and the obvious
  reading — straighter travel out of the strike radius — is untested. Path straightness and
  time-from-hawk-onset-to-leaving-the-radius would settle it, and both are cheap.
- **The rule cannot stop at the useful channels.** E114 measured the learned gate
  underperforming a hand-lesioned version of itself by −0.064/−0.072, because the
  incidental 1–8% suppression it adds to the other nine channels costs back about half the
  gain. Whether a sparser update or a threshold on the striatal drive recovers that is the
  first mechanism this project would be proposing against a **measured** deficit rather
  than a suspected one.
- ~~**The head-down test**~~ **Done (E114): refuted.** — the one part of E102's story
  [E113](experiments/E113-permuted-gate-control.md) could not establish. E113 showed the
  learned gate's channel assignment beats a scrambled one (pooled z=−4.40) and a flat one
  (z=−7.00), so selectivity is real. It did **not** show the benefit runs through
  `head_down` specifically, which is what E102 claims. Break the `peck/scratch → head_down`
  link with the learned gate held fixed: if the benefit survives, the story is wrong even
  though the selectivity is not.
- **A fourth block for E113 if it matters enough.** One of three blocks met the
  pre-registered per-block bar; all three shared the sign and the pooled estimate is
  decisive, but pooling was post-hoc and 8 seeds is where this project has been burned
  repeatedly.
- **The residual gap has a name: staying put.**
  [E112](experiments/E112-repair-the-peck-reflex.md) repaired the innate peck reflex —
  P(peck | on food) 39.7% → 97.9% with `arrival_peck_weight=4.0` and
  `peck_stops_walking=1.0` together — and closed only **26%** of E111's gap, with learning
  adding **+0.0014, t=+0.12** on top. So the arc was not the limitation. What the camped
  oracle does that the repaired hen still cannot is **stay on its patch**: at a patch 4.8%
  against 3.3%, despite the repaired hen pecking almost perfectly while she is there.
  Persistence needs "I am foraging here" held across seconds against a hunger drive that
  keeps her walking — a policy, not a stimulus→response mapping. Any mechanism proposed
  for it must be pre-registered against
  [E109](experiments/E109-what-the-rule-writes.md)'s constraint: a rule that cannot
  redirect behaviour cannot produce persistence either.
- ~~**E112 — wire `IDX_FOOD_ARRIVAL` to `M_PECK`, and re-ask H2 against the repaired
  baseline.**~~ **Done.** The arc really is mis-aimed and the fix is real and small. [E111](experiments/E111-is-there-headroom.md) measured the reflex hen
  pecking **39.65% of the time she is on food against 59.59% when she is not** — the peck
  reflex is keyed to a vision channel reading 0.9100 on food and 0.9430 away. The channel
  that would tell her, `IDX_FOOD_ARRIVAL`, exists and carries the fact at AUC 0.87–0.99;
  it is wired to the food *call* and not to the peck. This is a **fix to the innate arc,
  not to learning**, and it must be pre-registered as one: it moves every baseline in the
  project. The interesting question is what happens afterwards — if repairing her closes
  most of the 0.21 gap, then what ten experiments were asking learning to discover was a
  defect in the hand-written wiring, which is a different and more uncomfortable finding
  than any of the five mechanisms.
- ~~**E111 — is a positive result reachable in this environment at all?**~~ **Done.
  There is: 0.21–0.23 hunger units, replicated at t≈7.** The environment is not the
  excuse, and the nulls are about the rule. Target recorded: camped-oracle hunger
  **0.4223**, reflex **0.6332**. [E110](experiments/E110-postsynaptic-factor.md) closed the fifth and
  last mechanistic explanation for H2's null, and left a number worth staring at: hunger
  equilibrates at **~0.63 in every arm, including a frozen readout that cannot learn**.
  If a reflex-only hen is already near the achievable optimum, then no learning rule can
  demonstrate a benefit and H2 is *unanswerable in this coop* rather than false. E019
  contains this failure already — "hens start at hunger 0.30, which **is** the
  equilibrium; the metric was a coin flip". Measure the ceiling: a hand-written
  near-optimal forager, or a hen with the food channel wired straight to approach, against
  the reflex baseline. If the gap is small, the environment needs changing before any rule
  is tested again. Cheap, and this project's own rule says a null is only informative if
  the instrument could have shown a positive.
- ~~**E110 — a postsynaptic factor with a direction of its own.**~~ **Done, and null.**
  [E109](experiments/E109-what-the-rule-writes.md) measured `dz_motor` at cosine 0.98 to
  the reflex arc's own deviation (0.99 where the update lands), so the readout can only
  ever add more of what the arc already does. The direct consequence is a rule whose
  postsynaptic factor is *not* the motor output — credit the cortical contribution alone,
  or the exploration noise alone, so the update has a direction the arc does not dictate.
  **But this is the fifth mechanism proposed against this null and the previous four were
  each proposed with the same confidence.** Pre-register it with a falsifier that can end
  the line rather than extend it, and note that E109 established a sufficient obstacle,
  not a proven cause.
- ~~**E109 — decode `dz_motor`, the one factor in the rule never measured.**~~ **Done.**
  [E108](experiments/E108-what-the-rule-reads.md) closed the third upstream explanation
  for H2's null: the rule can see its own teaching event (`m` at AUC 0.955) and the state
  that caused it (`dz_slow` at 0.731). Both factors are present at consolidation time, so
  **the failure is in what the rule does with them, not in what reaches it.** The leading
  candidate is the postsynaptic factor: `dz_motor` traces the *motor output*, which is
  dominated by the reflex arc, so the rule may only ever reinforce what the arc already
  does at a food patch — which is already the right behaviour, and reinforcing it changes
  nothing. Directly measurable: decode, from `dz_motor` at the boundaries, how much of it
  is reflex-driven versus cortical. Cheap, and it is the only factor in the rule that has
  never been looked at.
  **No fourth upstream mechanism.** Three have been proposed and none survived.
- **The direction-stability line is closed
  ([E107](experiments/E107-red-team-review-2026-08-24.md)).** The metric E100–E106 chased
  pooled sixteen hens with sixteen different readouts; per hen the cortical drive runs
  0.9932 untrained → 1.0000 reared and never collapsed. Six mechanisms were built to
  explain it. **No seventh.** Any future experiment on this line must state in §5 which
  statistic it tests and why that statistic answers the question, and use
  `run/metrics.direction_stability` rather than a local copy.
- **Three verified defects, each needing a fix before the affected assay is used again.**
  (a) The audience assay's arms differ in 14 observation channels, not one — its "absent"
  flockmates are fenced back to 13.3 m inside a 15 m hearing range, and a relayed alarm
  channel saturates at 1.0000. Needs three ablation arms: flockmates deleted rather than
  parked, audience gagged, audience invisible. (b) E101/E102's untrained control is inert
  by construction (`W_gate`/`W_str` start at zero, gate sits at 0.982), so "the
  interaction is the evidence" carries nothing; needs the permuted-gate control — take a
  reared `W_gate`, permute its rows, re-run the 2×2, and if predation still falls the
  benefit is "less crouching". (c) No multiplicity control anywhere: declare the contrast
  count in §5 and divide α by it, and require both seed blocks to clear independently
  rather than pooling after seeing the first.
- ~~**E107 — re-calibrate the readout under the interneuron. The one thing that should
  happen next.**~~ **Superseded before it ran**, by the review that took E107's number.
  The premise was E106's magnitude collapse mattering because its representation result
  was large; per hen that result is a tenth the size and misses its own bar. [E106](experiments/E106-recurrent-inhibition.md) fixed the
  representation and left the learned pathway two orders of magnitude quieter
  (|cortical| 1.606 → 0.020): the common mode *was* the magnitude. `eta_out`,
  `readout_scale` and `readout_scaling_strength` were every one of them calibrated
  against the old regime, where the presynaptic signal was ~100× larger. This is a
  single-variable experiment with a clear falsifier, and it is the first time in this
  project that the learned pathway has had anything worth amplifying. Arm E of E106
  (`recurrent_lateral` + `sensory_lateral`, cortical stability 0.5735) is where to re-run
  it if it works.
- **THE critical path, restated and now quantified end-to-end
  ([E105](experiments/E105-decorrelating-readout.md)), and now partly resolved by
  [E106](experiments/E106-recurrent-inhibition.md).** The item below names pallial
  separability as the blocker for three hypotheses. E105 measured the same defect at the
  *output* stage and it is terminal: the motor stub, the only thing the learned readout
  reads, is **99.98% a constant vector at hatch** (deviation 7.18% of its own mean), and
  a gain sweep shows a reared `W_out` tracking its input's direction stability to within
  0.01 — so **the readout was never the limitation and six interventions aimed at it
  could not have worked**. The chain is observation 0.6375 → sensory stub 0.9707 →
  pallium 0.9934 → motor stub 0.9930. E103 fixed the zeroth stage's cause and
  [E104](experiments/E104-lateral-inhibition.md) built a relay interneuron that works
  there and **does not survive the two recurrent stages after it**. **No further
  readout-side mechanism should be attempted.** The next intervention has to act on the
  recurrent stages, and the question it raises — whether a strictly-positive rate code
  with excitatory-dominant recurrence can hold a varying representation at all — is
  architectural, not experimental. It belongs in a design decision, not an E106.
  **Answered by E106**: it can. A pooled interneuron in the two unaddressed stages takes
  the pallium to 0.7105 and the motor stub to 0.7400, replicating to within 0.005 on
  fresh seeds. The remaining half of the problem is magnitude, not representation.
- ~~E071 interaction check: is downstream `z_lag` centring redundant once the source is
  fixed?~~ **Closed unrun (E105 §8).** The source is fixed at one of three stages;
  asking whether a downstream patch for the same defect is redundant cannot be
  interpreted until the other two are addressed.

- ~~**BLOCKING T2: the sickness reward is ~4 orders of magnitude too weak, needs
  calibrating, and E014's erosion history makes "just raise it" unsafe.**~~
  **Answered and closed by [E069](experiments/E069-t2-positive-control.md).**
  Calibration is not the fix: a `sickness_penalty` sweep across a thousandfold range
  produced no learned avoidance at any magnitude. The signal does reach the weights
  (mean `|W−W₀|` +26% at penalty 1000), but as undirected perturbation rather than
  behaviour. The E014 safety concern is also retired — the connectome retains 97.5%
  of its innate synapses at penalty 1000 vs 98.0% at zero, so raising the term is
  safe and simply useless. The metric was never the limiting factor either (E069
  Part A: resolves 19–35% of baseline at n=8, far finer than T2's own predicted
  effect). **T2 now needs a place-linked innate anchor for the rule to amplify, a
  different rule capable of building associations from nothing, or acceptance as
  answered in the negative for this architecture** — see E069 §8, which sets out the
  hazard in option 1 (an anchor specific enough to make T2 learnable risks hardwiring
  the answer T2 exists to discover).
- **THE critical path, now on evidence rather than assertion: pallial separability
  (H2d/E017).** [E071](experiments/E071-pred-centring.md) established that it is the
  proximate blocker for **three** hypotheses at once — T2-revised, H2c and H3 all need
  the pallium to distinguish *which* stimulus, and none can while place-to-place
  pallial correlation sits at **0.94–0.96** (E070's measurement). E071 also rules out
  the cheap substitute: centring the prediction readout (below) helps materially but
  leaves the prediction *variable* across places rather than *selective* for one, with
  a distractor place driving stronger withdrawal than the planted one. E017 relocated
  this to fan-in dilution at sensory→pallium; E041 reframed it around density; neither
  resolved it. **Whatever is tried next should be measured against a concrete target:
  place-to-place pallial correlation must come down from 0.94–0.96 before any linear
  readout off the pallium can be referential.**
- ~~**HIGH PRIORITY: `W_pred` never received E019's centring fix.**~~ **Fixed behind a
  flag ([E071](experiments/E071-pred-centring.md)), and it is necessary but not
  sufficient.** [E070](experiments/E070-t2-revised-chain-positive-control.md)
  measured a hand-planted place→gakel association predicting **1.0000 at the planted
  place and 0.9637 at a different place entirely** — 3.6% selectivity. Traced: the place
  block is perfectly orthogonal in the observation (0.0000), and the information
  survives into the pallium (across-place variation has real structure, top singular
  values 0.294/0.117/0.097/0.066) — but it is **3.7% of the DC baseline**, and
  `brain.py`'s readout is `W_pred @ (src * pred_src)` on raw, uncentred rates, so the DC
  term dominates. `plasticity.py` already documents this exact failure for `W` and
  `W_out` ("without centring the outer product is dominated by the product of the two
  means") and fixes it there via `z_fast_bar`/`z_slow_bar`/`z_motor_bar`. **`W_pred`
  has no equivalent in either its learning or its readout.** Ruled out by measurement,
  not argument: not a settling artefact (identical at 1–300 settle steps), and not
  dilution by competing channels (a dedicated place-only stub slice at 10/21/32 units
  leaves pallium similarity at 0.9997). Proposed fix: subtract a running mean of pallial
  rate from `src` before projection, the direct analogue of `z_slow_bar`. Needs its own
  pre-registration and a guard test — it is a core change to `brain.py` on the pathway
  H2c and H3 also depend on. Note this is a *more specific and more tractable* claim
  than H2d/E017's standing "projection problem" framing: for place information at least,
  the projection is fine and the readout is not.
- **Fix whenever T2 or any reward-gated experiment next runs**: the `|W_out|` "is the
  rule active?" diagnostic reads a pathway that structurally cannot respond under
  `hebbian_readout` (its update is not reward-gated), and returned a falsely
  reassuring identical value across E065, E066 and E068. It should read `|W|`, as
  E068's and E069's own follow-up diagnostics do.
- **HIGH PRIORITY, unresolved: does `strike_penalty`'s reward-eligibility defect
  affect any prior hypothesis's actual conclusion?**
  [E067](experiments/E067-reward-eligibility-sampling-defect.md) confirmed, via an
  adversarial review and independent re-verification, that `m` (the factor gating
  `consolidate()`'s update to the recurrent weights `W`) is sampled at the exact
  consolidation-boundary step rather than traced, so a discrete single-step reward
  event — `strike_penalty`, used since ~E014, throughout H2/H4/T1's history — reaches
  `consolidate()` on only ~2% of occurrences. Confirmed as a real mechanism, not
  adopted as a reinterpretation of any specific prior result: whether T1's
  Pareto-safety finding, any of H4's states, or H2's own clean nulls actually
  *depended* on the discrete strike-event term surviving to a boundary — versus being
  adequately explained by the continuous `d_drive` pathway, which is a genuine trace
  and unaffected — has not been checked for any of them. This needs its own scoped
  investigation per hypothesis before any status changes; the mechanism alone does
  not tell you which conclusions, if any, actually move.
- ~~**Does the cortical pathway ever influence behaviour?**~~ **Answered by
  [E002](experiments/E002-can-the-pallium-reach-a-muscle.md):** it does, but only
  once the readout can learn. At the E001 setting `|W_out|` grew 1.00x — frozen.
  `eta_out` raised 2e-3 → 2e-2.
- ~~**Is synaptic scaling cancelling the learning signal?**~~ Moot — E003/E004 found
  the effect once the readout could learn, so scaling was not the blocker.
- **H2a: does structural growth hurt learning?** Weaker in *all five* runs that have
  compared them, now including E013 (+0.077 vs +0.062, both significantly worse than
  no learning). Growth has never helped in any regime. Largely subsumed by the
  erosion question above.
- **The innate/learned control balance is a parameter worth studying, not tuning.**
  E002 found that too much cortical influence makes behaviour *worse* — an untrained
  pallium overriding good reflexes. There is an optimum, and it plausibly maps onto
  the trade-off real precocial birds face between hatching competent and staying
  plastic. Worth a proper sweep rather than a single tuned value.
- ~~**Why does the rule erode?**~~ **Answered (E014):** a units error made one
  predator strike worth −100 in reward. Fixed; the connectome recovers. It was **not**
  the cause of the behavioural harm, and pruning turns out to be nearly free (E015:
  22% of the connectome lost for +0.010).
- ~~**Is the superadditivity a moving-target problem?**~~ **No (E016).** Staging was
  tested and the prediction was falsified: letting the pallium settle first does
  nothing (t=0.25), while doing the readout first cuts harm 69% (+0.052 → +0.016,
  significant). The moving-target story is withdrawn.
- **Does the pathway that learns *last* dominate the harm?** (E016) That is the new
  reading, and it makes a sharp cheap test: append a short pallium-only stage to an
  otherwise simultaneous schedule and see whether it absorbs the harm regardless of
  what came before. Worth one run — but note this is a **workaround**, not a fix.
  Even at its best, staging leaves the hen worse than not learning at all.
- **Fix the representation (H2d).** Both pathways learn from or into a pallium whose
  states for "heard an alarm" and "saw a hawk" differ by ~6% of mean rate. The gain
  correction helped 8x and was not enough. ~~Likely needs something that decorrelates
  the sensory projections rather than the random overlapping ones we have.~~
  **[E017](experiments/E017-where-separability-is-lost.md) relocated the problem:** the
  sensory projections are already near-orthogonal (separability 1.055 at the stub,
  9% shared units). The loss is fan-in dilution at sensory → pallium, and it is not
  recurrence — zeroing recurrence makes it slightly worse. **Still the critical path**
  — H2, H2b, H2c and H3 all trace back to it — but the fix is a projection problem,
  not a decorrelation problem.

- ~~**Wire an innate auditory reflex arc.** (E017) `hen/innate.py` has *no* response to
  hearing any call... Needs its own hypothesis node and a falsifier that distinguishes
  "the scaffold works" from "we wired in the answer".~~ **Done — built (E018,
  `auditory_scaffold=True`) and tested to a falsifiable conclusion (E036).** The scaffold
  works as specified (comprehension manipulation check 0.19, matching prediction) and
  the falsifier fired: supplying it did not let learning add a contingent audience
  effect. See H2f — the open question moved from "does she have a foothold" to "is the
  rule the right kind."

- ~~**Modality-segregated afferents.** (E017) Audition currently shares the sensory stub
  and its pallial targets with vision. Real birds keep them apart — Field L via nucleus
  ovoidalis, entopallium via rotundus, two separate thalamic relays. A hand-cut
  segregation measured **2.06x** separability. Cheap, biologically motivated, and not
  sufficient on its own (2x against a 17x loss). Should be done via the connectivity
  prior in `regions.py`, not a slice.~~ **Done (E035): built into
  `connectome.build(modality_segregated=True)`, and the 2.06x (and E034's 1.45x
  re-measurement) does not replicate on a paired genome sample — t=0.04, no effect.
  Both were unpaired ratio-of-means on a quantity with ~6x genome-to-genome spread.
  Not a live candidate fix for H2d until/unless a properly powered test finds
  otherwise.**

- **Sensory→pallium connectivity density.** (E041) Naive prediction was that *lowering*
  density would reduce dilution of the informative signal and improve H2d's separability.
  Falsified — separability rises monotonically with density instead, all the way to full
  connectivity (density 1.0, ~2× the default). Regression-checked same session:
  throughput unaffected (dense-with-mask architecture, density doesn't change compute),
  H2's contrast not broken at 8 seeds (not itself evidence of anything — see E041's own
  caveat about this project's block-variance lesson). **The strongest H2d lever found so
  far, still not adopted as a default.** ~~Next: a properly powered (24-seed, pooled)
  version of the H2 contrast at full density if this is going to be relied on, and/or
  testing whether it actually unblocks H2c's associative comprehension mechanism — which
  is the thing H2d was blocking in the first place and hasn't been re-tested since.~~

  **The comprehension check is done (E042): it doesn't, or not by enough to tell.**
  Comprehension at full density vs default, t=1.17, not significant; absolute
  comprehension in both conditions (0.005–0.007) is ~1/30th the auditory scaffold's
  hand-wired 0.19. `\|W_pred\|` grew to under 1% of its cap in every condition — at 20
  minutes' rearing and H4-standard predator density, the rule barely updated at all,
  which is at least as plausible a bottleneck as remaining separability. ~~**New owed
  item**: isolate exposure/duration from separability... e.g. a longer rearing window or
  higher predator density... to see whether `\|W_pred\|` ever moves meaningfully given
  enough exposure.~~

  **Done (E043): mixed answer, and the mixture is the finding.** Doubling predator
  density (hawk every 10s) left *mean* `\|W_pred\|` completely flat versus E042 — the
  registered falsifier for "exposure is the bottleneck" fires exactly as written. But
  *max* `\|W_pred\|` jumped to 30–40% of cap, from near-zero — not registered as the
  primary metric, reported as exploratory rather than promoted past that. Comprehension
  itself barely moved either way (still ~1/25th the scaffold).

  ~~**New, better-specified owed item**: a structural read of `W_pred`...~~ **Done
  (E044): genuinely mixed, and the mixture is the answer.** `IDX_AERIAL` ranks 30th of
  59 target channels by weight (below average — not the dominant thing `W_pred` is
  learning), but *within* the aerial-specific weights, correlation with
  call-responsiveness is significantly positive across 6 seeds (r=+0.304, t=2.61,
  barely clears threshold), inconsistent in 2 of 6. **Reading: a narrow, real trace of
  the correct association exists, buried inside a matrix whose largest-scale behaviour
  is about something else** — what that something else is remains unidentified (would
  need the same rank/correlation analysis repeated for whichever channels *do* dominate
  `W_pred`'s weight, not assumed).

  **This closes out the H2c/H2d/W_pred thread for now, by design.** Three experiments
  (E042, E043, E044) converged on "something real but small and partial" without
  surfacing a lever that moves it further — pushing density further, exposure further,
  or reading the weights more closely all found the same story. Continuing to escalate
  this one mechanism is not obviously the best use of the next round of work; stepping
  back to the wider queue (T1/T2 task design, H3's audience effect, or the E025-adjacent
  world-model gaps) is more likely to be productive than a fourth pass at `W_pred`.
  **The 24-seed H2-contrast replication (from E041) is still separately owed**,
  unaffected by any of this.

  ~~A structurally different candidate: food-call saturation crowding out pallium
  capacity for the alarm channel.~~ **Tested and closed (E054): not the answer either.**
  Fixing the saturation (E053) and re-running the exact E042 comprehension check with
  only `legacy_food_call` varied gave a clean, non-significant null (t=0.70) — the first
  test of a competing-channel-capacity account rather than the alarm channel's own
  representation, and it failed the same way as the other three. Strengthens H2f over
  any remaining named precondition.

  ~~H3's audience effect~~ **Done (E047): H3's original design re-run on the corrected
  system, and it closes rather than opens a direction.** Comprehension confirmed exactly
  zero without a scaffold (H2b's diagnosis is architectural, not an artefact of the
  since-fixed inaudible channel), both `alarm_effect` and `food_effect` stay null, and
  E005's one promising lead (`food_effect` +0.032, t=0.64) does not replicate. Combined
  with E036/E040, H3 has failed both ways this project could imagine it working. Not a
  live next-direction candidate anymore — it converges on the same open question as
  H2c/H2f, not a separate one.

  ~~The E025-adjacent world-model gap: gregariousness's attraction-only wiring~~ **Done
  (E048): the crowding channel is built, without breaking huddling or feeding — but
  E050 found it does not disperse the hens a hawk actually targets.** See above.
  **T1/T2 task design remains the live candidate from this list**,
  and now sits on a flock that actually spreads out, which both tasks assume.

- ~~**Is the learning rule the wrong *kind*?** (E017, open — no node yet.)~~ **Has a
  node now: H2f**, opened by E036's falsifier firing. One correction to this item's own
  framing, found while investigating: `W_pred`'s existing rule (`hen/plasticity.py`) is
  **already** a Pavlovian-style delta rule — no reward term touches it — so "wrong
  routing, wrong rule" was half right for the wrong half. E008/E009 tested this exact
  rule via `W_pred` and found it null, but for a *representational* reason (H2d), not a
  rule-kind reason. H2f's own evidence (E036/E040) is about the *other* pathway (`W_out`,
  genuinely reward-modulated, tested via learned audience-sensitive calling). The two
  questions are related but distinct and should not be conflated: H2d blocks `W_pred`
  from having anything to associate; H2f is about whether `W_out`'s instrumental rule can
  build contingent behaviour at all, even with a foothold supplied.
- ~~**Does the learning effect grow over a realistic rearing?**~~ Moot until a
  non-destructive rule exists. Running longer with the current rule strips more of
  the connectome, not less.
- ~~**H2f's falsifier, attempted twice, not yet decisively resolved either way**~~
  **Resolved (E057): the falsifier clears, replicated.** A non-reward-gated readout rule
  (`hebbian_readout`, stabilised with `readout_scaling_strength`) produces a real,
  twice-replicated, predominantly audience-conditional effect (+0.232, t=40.90 pooled)
  significantly larger than a smaller, context-specific (not indiscriminate — a
  food-channel control ruled that out) general-elevation component (+0.122, t=14.38
  pooled). H2f -> `SUPPORTED`, narrower than a clean audience-only ideal. See
  [E055](experiments/E055-hebbian-readout.md),
  [E056](experiments/E056-hebbian-readout-scaled.md),
  [E057](experiments/E057-separating-audience-from-elevation.md) for the full arc.
- **New item, opened by E057's own consequence section**: a structural read of what the
  trained `W_out` potentiates when a real hawk is present (audience or not) — is the
  general-elevation component a sensible "threat salience" association, or something
  less interpretable? Not required for E057's result to stand; a follow-up explanation,
  the same kind of analysis E044 did for `W_pred`.
- ~~**A fresh, direct pass at H2c** using the same non-reward-gated-and-bounded rule
  family and the same mandatory-diagnostic-before-trusting discipline that got H2f's
  result right on the third attempt.~~ **Done (E058): fails cleanly.** Crouch nominally
  significant (t=2.58) but matched by three unrelated control channels at the same tiny
  magnitude (~0.004) — general excitability, not comprehension. The mandatory
  diagnostic (built in this time, not added after a surprise) caught it immediately.
  H2c stays `NOT STARTED`. Narrows H2f's mechanism: it amplifies an existing anchor
  (the scaffold), it does not build one from nothing.
- ~~**Does comprehension emerge via the readout rule with a longer rearing duration or
  higher hawk rate**~~ **Closed (E059): no, mechanistically.** Doubled exposure
  reproduced E058's numbers almost exactly; `\|W_out\|` drift measured directly at
  0.054 regardless of exposure level — `readout_scaling_strength` reaches a dynamic
  equilibrium independent of how much co-occurrence data rearing supplies, unlike
  `W_pred`'s hard clip (which does respond to exposure, per E043). Not an exposure
  problem; any future attempt on this specific pathway needs a different stabiliser,
  not more rearing time.
- ~~**Predator exposure as a metric.**~~ Retired. Uninformative in both E003 and E004
  (SEs of 1460-2469 on means of 13-44). E001's apparent 43% difference was noise, as
  it was flagged at the time.

---

## Sources

- [Many-eyes effect and the vigilance/foraging trade-off](https://royalsocietypublishing.org/doi/10.1098/rsos.150135)
- [Evolutionarily stable vigilance as a function of group size](https://www.sciencedirect.com/science/article/abs/pii/S0003347205810231)
- [Conditions that favour cumulative cultural evolution](https://royalsocietypublishing.org/rstb/article/378/1872/20210400/109147/Conditions-that-favour-cumulative-cultural)
- [Measuring non-trivial compositionality in emergent communication](https://www.researchgate.net/publication/344945098_Measuring_non-trivial_compositionality_in_emergent_communication)
- [Structural inductive biases in emergent communication](https://arxiv.org/pdf/2002.01335)
- [On the meaning of alarm calls: functional reference in an avian vocal system](https://www.sciencedirect.com/science/article/abs/pii/S0003347283711589)
- [Audience effects on alarm calling in chickens](https://pubmed.ncbi.nlm.nih.gov/3396311/)
