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

Several feeders; one is contaminated; which one rotates on a period tuned to the band
above. Eating from it is costly but survivable. Contamination is invisible until
tasted, so the information is genuinely private and genuinely transferable.

- **Metric**: total flock sickness per rotation period.
- **Prediction**: L converges toward one hen's mistake per rotation (the discoverer
  pays, everyone else is warned); C? pays roughly N times that.
- **Why this one**: it tests *reference* -- the signal must carry **which** feeder,
  not merely that something is wrong. And it extends a behaviour chickens genuinely
  have, since their food calls are already functionally referential.

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

1. **Phase 1 plasticity — REOPENED, and it blocks everything below.**
   [E013](experiments/E013-clean-test-of-h2.md) is the first clean test and learning
   comes out *significantly worse* than no learning (t=3.85). E004's positive result is
   best explained as an artefact of the saturated regime.

   The reason is **not** connectome damage, though E013 said so at the time: that was a
   units bug, fixed in [E014](experiments/E014-strike-units-bug.md), and
   [E015](experiments/E015-decomposing-the-harm.md) measured pruning as nearly free
   (22% of the connectome for +0.010). The harm is the learned pathways themselves,
   superadditively. Nothing below is worth running until a rule exists that does not
   make the bird worse.
2. Condition harness: the six-way ladder as a config, sharing one simulation path.
   `run/experiment.py` has the matched-seed skeleton; it currently knows three
   conditions, not six.
3. T1 shared vigilance — validates the harness against a known biological effect.
4. Rotation-period sweep to locate the social-learning band.
5. T2 poisoned feeder — the headline experiment.
6. Playback and lesion assays.
7. T3 safe corridor.

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
