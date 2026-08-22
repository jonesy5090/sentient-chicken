# The hypothesis tree

The purpose of this file is to stop the project drifting into testing whatever is
convenient. Every experiment must name the hypothesis it feeds, and every result must
come back here and change something — a status, a prediction, or the tree itself.

**Rule: no experiment is run without a parent hypothesis recorded here.** If a
question does not ladder up to H0, either it belongs in the backlog as a new branch,
or it does not belong in the project.

Status values: `SUPPORTED` · `UNDER TEST` · `NOT STARTED` · `REFUTED` · `ABANDONED`

> ## ⚠ Re-baselined at E023 — every number below predates it
>
> [E022](experiments/E022-second-review-verified.md) found that the pallium contained
> **no inhibitory neurons at all**: E/I identity was assigned by flat index over a
> region-ordered array, so it segregated by region. Fixed in
> [E023](experiments/E023-ei-fix-and-rebaseline.md), which also moved the gain default
> from 0.70 to 0.95.
>
> **Every measurement recorded in this file was taken on the old network.** None is
> known to be *wrong*; all are unrepeated on the corrected connectome, which is a
> different and weaker claim. Statuses are retained rather than reset, and each should
> be read as "established on the pre-E023 brain" until re-run. The queue is in E023 §6.

> ## ⚠ OBS_DIM moved 59 → 71 → 73 → 74 → 88 at E048/E051/E053/E060 — narrower than E023, still worth flagging
>
> [E048](experiments/E048-personal-space-fix.md) added `CLS_CROWDING`, a fifth vision
> class giving the reflex arc a personal-space signal (E025's diagnosed fix for flock
> clumping): 59 → 71. [E051](experiments/E051-wall-avoidance.md) added
> `IDX_WALL_ESCAPE_L/R`, wiring a wall-avoidance reflex that hadn't existed at all: 71 →
> 73. [E053](experiments/E053-food-call-discovery-pulse.md) added `IDX_FOOD_ARRIVAL`,
> replacing the food-call reflex's continuous sight-gating with a discovery pulse: 73 →
> 74. [E060](experiments/E060-t2-contamination-scaffold.md) (T2 Stage 1) added
> `CLS_SICK` and `IDX_SICKNESS_ONSET` together: 74 → 88, and `MOTOR_DIM` also moved for
> the first time (11 → 12, the gakel call), which none of the earlier additions
> touched. [E063](experiments/E063-allocentric-place-cells.md) (T2 Stage 2
> prerequisite) added a 25-cell allocentric place-cell grid, the first channel in this
> file that is *not* egocentric: 88 → 113. [E064](experiments/E064-gakel-location-cue.md)
> (T2 Stage 2 prerequisite) added a second 25-cell block, the same grid reused for the
> *caller's* location as heard by a listener beyond visual range: 113 → 138, now the
> biggest single jump. Unlike E023, none of these touch neuron
> identity or any existing channel's values —
> the observation layout is symbolic and offset-based (`coop/spec.py`), so every prior
> channel keeps its old index and meaning, just at a new absolute position. What they do
> change: the connectome's sensory fan-in size, and therefore anything sensitive to
> `OBS_DIM` as a raw number (e.g. `sensory_pallium_density`'s fan-in-dilution math,
> E041) **and any fixed-seed test whose connectome-build RNG stream shifts as a result**
> (E051 found and fixed one: `test_being_caught_does_not_dominate_the_reward_where_hawks_are_common`
> was already running at a marginal single-seed threshold and flipped from a hit to a
> 1.549 m miss against a 1.5 m strike radius). Behavioural endpoints (fed %, caught %,
> comprehension) are not expected to shift from `OBS_DIM` alone. Statuses are retained,
> not reset; treat this as a smaller, scoped version of the E023 caveat rather than a
> second full re-baseline.

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

~~**Status: UNDER TEST — and the null is now UNINFORMATIVE** —
[E031](experiments/E031-the-credit-window-is-not-the-blocker.md).~~

**Status: UNDER TEST — the null's informativeness is open again** —
[E033](experiments/E033-e032-second-block-pooled.md) had appeared to refute H2e (the
claim that the cortical pathway cannot reach behaviour at all), but
[E038](experiments/E038-h2e-depletion-audit.md)'s clean, equally-powered re-measurement
found that result did not survive controlling for `food_deplete_rate`
(+0.390, t=2.55 → −0.890, t=1.60, not significant). H2e reverts to `UNDER TEST`. See H2e
below for the full correction.

**The pathway learning acts through cannot move the metric H2 is measured on.** Deleting
`W_out` entirely and multiplying it tenfold both leave feeding unchanged, while the same
metric detects a halved innate peck reflex at t=4.32:

| condition | fed % | vs fixed |
|---|---|---|
| fixed | 3.671 | — |
| lesioned (`W_out` = 0) | 3.531 | −0.139 ± 0.204, **t=0.68** |
| amplified (`W_out` × 10) | 3.561 | −0.110 ± 0.203, **t=0.54** |
| peck reflex × 0.5 *(positive control)* | 3.213 | −0.458 ± 0.106, **t=4.32** |

So "the rule does not produce improvement" and "the route from the rule to the metric is
inert" are indistinguishable on the evidence, and every null from E001 onward is
compatible with the second. **This does not refute H2 — it withdraws the standing of the
null.** See H2e.

**Caveat that must travel with this:** `W_out` above is *untrained*, and random drive is
roughly zero-mean, so it may simply average out. A learned `W_out` is structured and
could in principle do what a random one cannot. The experiment cannot separate those, and
the test that would — lesioning a **trained** flock — is `docs/backlog.md` §5's causal
efficacy check, still never run.

**The credit window is struck as the explanation.** Cited since E022, promoted to the top
of the queue by E028, and refuted the first time it was measured
([E031](experiments/E031-the-credit-window-is-not-the-blocker.md) §3): a hen feeds every
**0.3 s**, reward moves every step, and two thirds of the peak peck–reward correlation
sits at lag 0, inside the rule's 0.2 s window. Do not sweep `tau_slow`.

~~**Status: UNDER TEST — a clean null** —
[E020](experiments/E020-h2-after-the-e019-fixes.md), re-run after the E019 fixes.~~

A hen who learns is **statistically indistinguishable** from one who cannot.
~~**+0.011 ± 0.012, t=0.95**, pooled across **24 matched seeds** in two independent blocks
([E020](experiments/E020-h2-after-the-e019-fixes.md) seeds 0–11 at +0.001,
[E021](experiments/E021-the-cost-of-exploration.md) seeds 12–23 at +0.021).~~ **Re-measured
on the corrected connectome by [E037](experiments/E037-h2-rebaseline.md):
+0.0003 ± 0.0156, t=0.02, 24 seeds (gain=0.95, E/I-fixed, and — a second confound E037
found and controlled for — `food_deplete_rate=0`, matching E020/E021's actual world; a
mechanic added by E025 for an unrelated hypothesis silently degrades this exact contrast
at the duration and flock size this harness uses).** An even cleaner null than the
original, and worth citing over E020/E021's stale number now. The rule neither helps nor
harms.

**Quoted as the pooled estimate deliberately, still.** E020 reported its own block as
+0.001 ± 0.010 and read that as "the harm is gone". E021 showed a single 12-seed block is
not enough to support that precision — and E037 repeated the lesson inside itself: its
two 12-seed blocks individually read **t=2.96 and t=2.19, significant, in opposite
directions**, and only pooling reveals the null. The null holds; the confidence any
single block would attach to it does not.

**Read this as a withdrawal, not a success.** `REFUTED at this timescale` was a positive
claim that learning makes hens worse, and that claim is gone. Nothing has replaced it.
The rule still does not produce the behavioural improvement H2 asserts.

~~**Status: REFUTED at this timescale** — E013, the first test with no confound left.
Learning does not merely fail to help; it makes hens significantly worse
(+0.062 ± 0.016, t=3.85) and they feed on 4.7% of timesteps against the control's 6.2%.~~

**Superseded by E020.** E013 was not the confound-free test it was recorded as.
[E019](experiments/E019-three-verified-defects.md) found 98.1% of its reward variance
came from the call cost, and the readout it measured could only slide a constant offset.
With both fixed, the harm is gone (+0.062 → +0.001) and so is the erosion (19,088 →
35,480 of 36,373 innate synapses, 48% destroyed → 2.5% lost).

**Attribution is not established.** Four things changed between E013 and E020 — the
strike-units fix (E014) plus E019's three — so E020 says what H2's status *is*, not
which fix moved it. An ablation ladder is owed and is now worth running.

~~**One unpredicted result, deliberately left unexplained.** The noise-only control is
now **significantly worse** than fixed (+0.032, t=3.84), where in E013 it was
indistinguishable (t=0.32). Exploration has become costly and nobody knows why.~~

**Withdrawn by [E021](experiments/E021-the-cost-of-exploration.md).** It did not
replicate. On a fresh seed block the same contrast measures **−0.000 ± 0.035, t=0.01**,
against E020's +0.032 ± 0.008, t=3.84. There is no exploration cost to explain, in either
audio regime.

**This is the most important methodological result the project has, and it is not about
exploration.** The same contrast, same conditions, same n=12, on a different seed block:
the **standard error was 4.4× larger** (0.008 → 0.035) and a p≈0.003 finding became
p≈0.99. The pairing is sound and the t table is correct; what is not safe is assuming a
seed block's *variance* is representative. Seeds 0–11 were homogeneous, which made a
small difference look decisive.

**Standing rule, from this:** no result changes a status in this tree on one seed block.
See `CLAUDE.md`. ~~E004's t=3.93 and E016's staging result are both single-block and both
still cited — they need re-checking.~~ **Closed out.** E016 was marked superseded by
E020 at the time (§ below). E004's t=3.93 turned out to be the most re-checked number in
the tree, just never explicitly retired: E010 re-ran it unchanged except for gain and it
vanished (t=3.93 → t=0.08); the fully corrected connectome and world were re-tested
independently by E013→E020→E021→E023→E037, landing on today's authoritative
+0.0003 ± 0.0156, t=0.02. Both are superseded, not merely "still cited."

~~**The smoking gun is the synapse count.** Learning ends with 19,088 of 36,373
innate synapses, destroying 48% of the connectome. The reward prediction error hovers
near zero, so updates approximate a random walk that the Dale clamp makes
irreversible — a ratchet.~~

**Struck through: E014 disproved this in two ways.** The erosion was a units bug (a
strike contributing −100 to reward), not a random walk — and fixing it recovered the
connectome without moving the behaviour at all. Pruning and harm merely co-occurred.

**E004 is reinterpreted, not withdrawn.** In the saturated regime the readout could
only apply a constant bias to the motor drive, so the same erosion could not reach
behaviour and what learning tuned was an offset that happened to help. A tuned offset
is not a learned policy.

**[E014](experiments/E014-strike-units-bug.md) found the erosion's cause and it did
not rescue H2.** Reward subtracted `struck * strike_penalty / dt` — but being caught
is a discrete *event*, not a rate, so at dt=0.01 a single strike contributed **−100**,
about 150x what the drive terms contribute. Erosion tracked strikes exactly (seeds
with zero strikes lost 17%; seeds with thousands lost 50–75%). Fixed, all seeds
survive 81–86%.

**The behaviour did not move**: +0.081 → +0.082, still significantly worse (t=3.46).
So connectome destruction and behavioural harm were two separate things that merely
appeared together — E013's mechanism story linked them and was wrong twice over.

> **E015 and E016 are SUPERSEDED by [E020](experiments/E020-h2-after-the-e019-fixes.md).**
> Both decompose a harm that no longer exists — E020 measures it at +0.001 (t=0.08). They
> described the old rule accurately and should not be cited about the current one. Kept
> because the route matters, and because a project that deleted its superseded findings
> would be unable to show how it got here.

**[E015](experiments/E015-decomposing-the-harm.md) decomposes the harm between the
two learned pathways:**

| what learns | harm vs fixed | synapses |
|---|---|---|
| recurrent only | +0.010 | 28,383 |
| readout only | +0.021 | 36,369 |
| **both** | **+0.052** | 30,109 |

The readout is about twice as harmful as the recurrent weights — and **the two
together are superadditive**, +0.052 against the +0.031 independence would predict.

~~Plausibly a moving-target problem: the readout chases a pallial representation being
rewritten underneath it.~~ **Withdrawn by
[E016](experiments/E016-staged-learning.md)**, which staged the two pathways and got
the opposite of what that predicts: letting the pallium settle *first* does not help
at all (t=0.25), while doing the readout first cuts the harm 69% (+0.052 → +0.016,
significant). **New reading — labelled a hypothesis:** harm is dominated by whichever
pathway learns *last*, and the readout is the harmful one. Staging does not fix an
interaction; it decides who gets the last word.

**Pruning is nearly free**: the recurrent-only condition loses 22% of the connectome
for +0.010. That retires the last of E013's original story — the harm is not the loss
of synapses. It is not exploration either (t=0.32).

~~**Which makes this a consequence of H2d.** The pallium cannot represent distinctions,
`eta_out` grows a readout from that degenerate representation, and the cortical
pathway transmits structured state-dependent noise into an already-competent motor
system. Exploration noise is harmless because it is zero-mean; this is not.~~

**Withdrawn by [E019](experiments/E019-three-verified-defects.md).** The cortical
pathway does not transmit state-dependent anything. `ΔW_out` is **rank one** (top
singular value share 0.9981), and the cortical contribution to the motor drive varies by
**0.7% of its own magnitude** over three seconds of behaviour. It is a constant offset,
and it is negative on peck (+0.02 → −0.52) and on the call channels. The hen learns to
peck less; that is the whole of the measured harm.

Two further caveats on this node, both from E019:

- **E013's status as the clean test of H2 does not hold.** 98.1% of the reward variance
  in that run came from the vigour term — a cost added for a different hypothesis, moved
  out of the metric by E012 and into the teaching signal, where it was never checked.
  `REFUTED at this timescale` stands only as "refuted under a reward that was 98% call
  cost", which is not the claim the tree is making.
- **E015's superadditivity and E016's "last word" may both be artefacts** of a single
  rank-one offset being scaled two ways. Plausible, unverified, and not yet acted on.

---

### Status history (superseded; kept because the route matters)

Previously `UNDER TEST` — downgraded by E010, whose evidence was then found to be
**confounded**.

E010 reported the effect collapsing from t=3.93 to t=0.08 after the gain correction.
That run had also, silently, given every condition — *including the fixed control* —
the exploration noise added in E007 after E004 was run. It compared *(new gain +
noise)* against *(old gain, no noise)* and blamed the gain. See the invalidity notice
on [E010](experiments/E010-rebaseline-at-corrected-gain.md).

**[E012](experiments/E012-corrected-phase-1-contrast.md) isolated the real cause, and
it was neither the gain nor the noise.** With one variable changed at a time:

| gain | call cost | hunger change |
|---|---|---|
| 0.90 | 0 | +0.060 ← the E004 configuration |
| 0.70 | 0 | **+0.033** ← corrected gain is *better* |
| 0.70 | 8e-4 | +0.224 ← current |

The gain is nearly neutral. **`call_energy_cost` — added in E005 for H3 — explains
essentially all of the degradation**, because it triples the rate hunger accumulates
and so swamps the very metric H2 is measured on.

In the corrected 12-seed contrast, learning does not beat the fixed control
(+0.016 ± 0.040, t=0.40) and roughly cancels the cost of its own exploration
(noise-only: +0.018). But that contrast runs *with* the call cost, so its metric is
compromised too.

At that point no run had yet tested H2 cleanly. E013 did, by moving the call cost to
its own budget — and returned the significant negative result recorded at the top of
this section.

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

**[E010](experiments/E010-rebaseline-at-corrected-gain.md) re-ran E004 unchanged
except for the gain, and the effect vanished:**

| | E004 (gain 0.9) | E010 (gain 0.70) |
|---|---|---|
| learning, no growth | −0.063 ± 0.016, **t=3.93** | −0.002 ± 0.028, **t=0.08** |
| fixed control, final hunger | 0.330 | **0.655** |

**Every hen also got worse**, control included — final hunger roughly doubled. The
change did not selectively remove learning; it degraded the whole flock and took the
signal with it.

**The mechanism is E002's finding in disguise.** At gain 0.9 the saturated pallium
emitted a near-constant output, so the cortical readout acted as a harmless fixed
*bias*. At 0.70 the pallium is responsive, so an *untrained* readout injects real
variability into the motor drive — and E002 already showed that untrained cortical
influence makes behaviour worse. Saturation had been accidentally protecting
behaviour.

So `readout_scale` and `eta_out`, both tuned by E002 **against a saturated network**,
are now stale. E010 is best read as "the readout parameters are wrong", not "learning
does not work" — learning still feeds more (6.1% vs 5.6%) and takes 21% less predator
exposure, neither significantly.

**Next: E011** re-runs E002's readout sweep at gain 0.70, then H2 is re-tested. Only
after that does a null mean anything about learning.

**Standing lesson**: parameters tuned under a defect inherit the defect. E002's sweep
was correct at the time and silently encoded an assumption about the operating point
that outlived it.

---

## H2e — the cortical pathway is behaviourally inert, so H2 is not testable through it

~~**Status: UNDER TEST** — opened by
[E031](experiments/E031-the-credit-window-is-not-the-blocker.md).~~

~~**Status: REFUTED** —
[E033](experiments/E033-e032-second-block-pooled.md), a pre-registered pooled test
across two independent 12-seed blocks.~~

**Status: UNDER TEST — E033's `REFUTED` finding did not survive a clean re-measurement**
— [E038](experiments/E038-h2e-depletion-audit.md). E032/E033 ran with the same
undocumented `food_deplete_rate` confound E037 found for H2 (added by E025 for an
unrelated question, never controlled for here). A properly powered, equally-sized
(24-seed) re-measurement with it removed found the interaction at **−0.890 ± 0.556,
t=1.60, NOT significant** — opposite sign from E033's +0.390, t=2.55. Neither result
confirms the other; the clean one is the one to trust, and it is a null. See below for
what stands from E032/E033 and what does not.

**Claim, still open:** the route from pallium to muscle contributes so little to
behaviour that no change in `W_out`, learned or otherwise, can register on H2's metric.
If true, H2 as constructed could not be falsified and every null it produced would be a
fact about the architecture rather than the learning rule. E032/E033 appeared to refute
this; that appearance did not survive a clean world — see below.

**Evidence that motivated the claim:** an untrained `W_out` at 0×, 1× and 10× gain is
indistinguishable on `fed %` (t=0.68, t=0.54) on a metric that detects a halved reflex at
t=4.32. Independently, E027 and E030 found lesioning `W_out` leaves *predation* outcomes
unchanged (+0.010, t=0.46), and E019 measured a learned `ΔW_out` moving cortical drive by
0.7% of its own magnitude. All of this is evidence about an *untrained* or barely-trained
readout; it never separated "the rule learns nothing" from "nothing learned there could
reach behaviour," which is exactly the ambiguity E032/E033 were built to resolve.

**[E032](experiments/E032-causal-efficacy.md) tested a trained readout directly and
missed by 0.07 in t, on one block; [E033](experiments/E033-e032-second-block-pooled.md)
ran a second block and pooled, clearing threshold — both in a world later found to have
an uncontrolled confound:**

| block | interaction | t |
|---|---|---|
| one (0–11), E032, depleted world | +0.541 ± 0.254 | 2.13 |
| two (12–23), E033, depleted world | +0.240 ± 0.172 | 1.39 |
| ~~pooled (24 seeds), depleted world~~ | ~~+0.390 ± 0.153~~ | ~~2.55~~ (threshold 2.069) |
| **pooled (24 seeds), clean world — [E038](experiments/E038-h2e-depletion-audit.md)** | **−0.890 ± 0.556** | **1.60** (not significant) |

**The depleted-world pooled result does not replicate.** E032/E033 ran with
`food_deplete_rate` at its default (added by E025 for H4's unrelated dispersal question,
never controlled for here) — the same confound E037 found substantially affects H2's own
contrast at this exact duration and flock size. A clean, equally-powered re-measurement
(E038) found the opposite sign and no significant effect. **Do not cite the depleted-
world numbers going forward; the clean 24-seed result is the one that stands, and it is
a null.**

**H2's own question, in the depleted world E032/E033 ran in**, had also read null —
trained vs fixed, both intact, +0.153 ± 0.388, t=0.39 — and this part is **not
contradicted** by the correction: H2's clean null was independently re-confirmed on a
controlled world by E037 (+0.0003 ± 0.0156, t=0.02), for reasons unrelated to E032/E033.
What does not survive is the *causal-efficacy* story built on top of it — that a trained
readout costs something specific to remove where an untrained one does not.

**What this changes for H2.** E031 had withdrawn H2's null to "uninformative" because an
untrained pathway couldn't be shown to reach behaviour at all. ~~E033 shows the route is
not closed for a *trained* readout~~ — **E038 shows E033 did not, in fact, show that.**
Whether the cortical pathway can carry a trained signal to behaviour at all is **open
again**, exactly where E031 left it. The credit window remains ruled out
([E031](experiments/E031-the-credit-window-is-not-the-blocker.md) §3) independent of any
of this.

**E007's multiplicative-gating question stays open, neither promoted nor demoted.** It
was motivated by H2e being *true* — the two pathways summing into one motor drive with
no room for a trained signal to matter. E032/E033 appeared to rule it out; that appearance
is now known to rest on a confounded measurement, so the question returns to genuinely
undecided rather than settled in either direction.

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

**Stale evidentiary basis, flagged rather than re-run.** All three supporting runs
(E001, E003, E004) predate E013–E023's fixes. On the current architecture the ordering
doesn't obviously hold either way: E037's clean rebaseline found `learning + growth` vs
fixed at +0.0052 ± 0.0133, t=0.39 — also a null, not distinguished from `learning, no
growth`'s own null. This node's status stays `NOT STARTED` because the growth-rate sweep
itself has never been run, on any architecture; the three-run ordering that opened it is
old evidence for a question nobody has directly tested since.

**Note:** the default stays `growth_enabled=True` until this is settled. Three runs
of a binary contrast is not enough to overturn a default with a biological
justification behind it.

---

## H2c — a learned cue can recruit an innate response via top-down association

> **Blocker re-scoped ([E081](experiments/E081-separability-vs-decodability.md)).** H2c has
> been recorded as blocked by H2d. It is not blocked by the *representation*: the pallium
> supports 98.8% held-out linear decoding of the hawk-vs-call contrast. What H2d actually
> establishes is that the *distance* between mean pallial states is small. If H2c is
> blocked, it is blocked by whether its learning rule finds a discriminant — a question
> about rule type, not about what the pallium encodes — and that has never been tested.

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

**[E008](experiments/E008-top-down-association.md): implemented, and the first
attempt does not test the hypothesis.** The projection learns — `|W_pred|` saturates
its cap — and comprehension stays at zero. The diagnosis is that the rule as written
maps `rate(t) -> obs(t)`, which is an **autoencoder**, not a predictor. During a hawk
event the brain state is dominated by the hawk percept, so the association it forms is
"when in hawk-state, predict hawk" — circular, and no use for recovering the percept
from a call. Pavlovian association needs `rate(t - delta) -> obs(t)`, mapping a cue to
a later outcome.

Also thin: a 30-minute rearing yields roughly **9 seconds** of usable co-occurrence
(hawks present ~24 s, of which only the head-up fraction counts). E009 must raise
predator density so that a null means the mechanism is wrong rather than the data
absent.

The no-cue baseline did not rise, so the hallucination failure mode has not occurred.

**[E009](experiments/E009-lagged-pallial-association.md): fixed both, still null — and
found the actual blocker.** With a lagged pallial source and up to 90x predator
density, comprehension stayed at zero while baseline crouching with *no cue* nearly
doubled (0.078 → 0.147). The projection learns the **base rate** of aerial threat
rather than a contingency on the call. Measuring the representation says why:

```
                mean|rate|   shift: call     shift: hawk
sensory stub      0.4218     0.0949 (22.5%)  0.0819 (19.4%)
pallium           0.8577     0.0277 ( 3.2%)  0.0235 ( 2.7%)
```

The call reaches the brain with *more* afferent weight than the hawk channel. But the
pallium sits at mean rate 0.86 — deep in the flat part of the sigmoid — and the two
percepts differ from each other there by under 1% of the mean. **No associative rule
can be cue-specific when sourced from a representation that does not distinguish the
cues**, and with nothing to condition on, the base rate is the best available
prediction. See H2d.

**Re-tested after H2d's representational fix, and still null**
([E042](experiments/E042-comprehension-after-density-fix.md)). E041 roughly doubled
pallial separability via `sensory_pallium_density=1.0`. Comprehension after 20 minutes'
rearing at that density: **+0.0023 ± 0.0019, t=1.17** against the default density, not
significant, and both conditions' absolute comprehension (0.005–0.007) is **~1/30th**
the auditory scaffold's hand-wired 0.19 — not a working mechanism at either density.
`|W_pred|` grew to under 1% of its cap in every condition, which is at least as likely
an explanation as insufficient separability: the rule may simply not have had enough
exposure in 20 minutes at this predator density to move meaningfully, independent of
whether the representation is now good enough.

**Exposure escalated directly, and the answer is genuinely mixed**
([E043](experiments/E043-exposure-escalation.md)). Doubling predator density (hawk every
10 s instead of 20) left *mean* `|W_pred|` completely unchanged from E042 — the
registered falsifier fires as written. But the *single largest* `W_pred` entry grew to
30–40% of its cap, up from near-zero, an unregistered but striking observation:
something is learning, concentrated rather than diffuse, and averaged away by the metric
that was supposed to detect it. Comprehension itself still barely moved (0.006–0.008,
still ~1/25th the scaffold). **Status stays `NOT STARTED`.** The open question is no
longer "is it separability or exposure" — it's whether the concentrated growth in
`W_pred` is a real, targeted association (the right pallial neurons predicting the right
channel) or noise drifting in a large mostly-irrelevant matrix, which needs a structural
read of *which* entries moved, not another rearing run at more exposure.

**The structural read is done, and the two metrics disagree with each other**
([E044](experiments/E044-structural-read.md)). `IDX_AERIAL` — the channel a correct
association should predict — ranks an unremarkable 30th of 59 target channels by raw
weight, below average: whatever `W_pred` is mostly doing, it isn't concentrating on the
hawk percept. But *within* the aerial-channel-specific weights, which pallial neurons
get weighted correlates significantly with which neurons actually respond to the call
(mean r=+0.304 across 6 seeds, t=2.61, barely clears 2.571) — real structure, not pure
noise, though inconsistent (2 of 6 seeds near zero). **Status stays `NOT STARTED`.**
Reading: a narrow, real trace of the correct association exists, buried inside a matrix
whose largest-scale behaviour is about something else entirely — what that something
else is has not been identified. Three experiments (E042–E044) have now converged on
"something real but small and partial" without finding a lever that moves it; the
project is stepping back to the wider backlog rather than continuing to escalate this
specific mechanism.

**A structurally different candidate — capacity competition from an unrelated
channel — has now also been tested and failed** ([E054](experiments/E054-food-call-saturation-and-pallium-capacity.md)).
Unlike E042–E044, which all targeted the alarm channel's own representation, this asked
whether the food-call channel's near-constant pre-fix activity (E053: 42.8% of hen-steps)
was crowding out pallium capacity that could otherwise represent the rarer alarm signal.
Comprehension after rearing, discovery-pulse vs. legacy continuous food call (density
held at E041's fix throughout): **−0.0005 ± 0.0007, t=0.70**, not significant, wrong
sign if anything. **Status stays `NOT STARTED`.** With this result, every account this
project has been able to name and test — representational fidelity, density, exposure,
structural capacity, and now competing-channel saturation — has failed to produce
comprehension; the pattern increasingly points toward H2f (the rule may be the wrong
*kind*) rather than any remaining precondition still to be found.

**H2f's now-validated rule was tried directly, and it also fails — but cleanly, with
the reason distinguished from a repeat of the earlier confound**
([E058](experiments/E058-h2c-hebbian-readout.md)). The same non-reward-gated, bounded
readout rule that built H2f's audience effect was tested on crouch-comprehension, no
scaffold (building from nothing, unlike H2f's task, which had a wired-in anchor to
amplify), `pred_gain=0.0` so only the readout could contribute. Crouch rose
nominally significantly (t=2.58) — but so did three unrelated control channels (peck,
scratch, flee), at matched magnitudes (~0.004, two orders of magnitude below H2f's
effect and the scaffold's own comprehension). General excitability, not a targeted
association — the same diagnostic discipline that validated H2f's positive rules this
one out cleanly, the other direction.

**Exposure escalation, the same check E043 ran for `W_pred`, closes the "not enough
data yet" possibility for a stated, checked reason** ([E059](experiments/E059-h2c-readout-exposure-escalation.md)).
Doubling predator exposure (`hawk_period_s` 20 → 10, matching E042→E043's escalation)
reproduced E058's numbers almost exactly (crouch 0.0036 vs 0.0036, controls equally
matched) — no movement at all, unlike `W_pred`, where the equivalent escalation moved
the weights substantially even without producing comprehension (E043). Checked directly:
mean `|W_out|` drift after rearing was **0.054 at both exposure levels** (3 seeds,
0.051–0.057 spread either way) — the readout's `readout_scaling_strength` correction
reaches a dynamic *equilibrium* independent of how much co-occurrence data rearing
supplies, unlike `W_pred`'s hard per-synapse clip, which has real headroom until
saturation. **Status stays `NOT STARTED`.** Narrows what H2f's mechanism generalises to,
now for a mechanistic rather than merely empirical reason: it amplifies an existing
anchor, and no amount of exposure alone will make it build one from nothing, given how
its own stabiliser works.

---

## H2d — ~~the pallium does not form separable representations of distinct stimuli~~ → pallial states are *close in distance* but *linearly decodable*

> **The original title is false, established by
> [E081](experiments/E081-separability-vs-decodability.md).** Pallial states for hawk vs
> alarm call are 0.9928 correlated with `pallial_sep` 0.1113 — and support **98.8%
> held-out linear decoding**. Both facts are true; only the second bears on whether a
> readout can use the representation.
>
> **How to read the rest of this node.** Everything below measured `pallial_sep`, which is
> `RMS(hawk − call) / mean|rest|` — a **distance**. Those measurements are correct and
> replicate. What none of them measured is **decodability**, which is what `W_pred` and
> `W_out`, being linear readouts, actually require. So the series is not wrong; it is
> about a quantity that turned out not to be load-bearing. Read every "separability"
> figure below as "distance between mean states", and do not read any of them as "the
> distinction is unavailable to a readout" — it is available, at 98.8%.
>
> **What this leaves standing:**
> - The **localisation** is solid and replicated: whatever compresses the distance
>   happens at sensory→pallium (E017, E034).
> - Every **structural intervention** is null or negative: E/I identity (E023), modality
>   segregation (E035), density — which *reverses* naturalistically (E041/E078),
>   balanced E/I (E077), recurrent gain — already optimal (E079), removing recurrence —
>   worse (E017/E034).
> - Every **mechanism story** is now measured rather than assumed, and none survives:
>   fan-in dilution explains 1–2% (E080), saturation is wrong-signed (E079), recurrent
>   mixing is wrong-signed (E017/E034), common-mode DC is null (E077).
>
> **Where the problem actually moved.** From *representation* to **rule type**.
> Correlational rules (Hebbian/covariance — `W_out` under `hebbian_readout`) converge
> toward matched-filter-shaped directions, which are poor discriminators on highly
> correlated inputs; delta rules (`W_pred`) converge toward discriminants. E081 measured
> the gap directly on place: matched filter **18.8%**, discriminant **84.6%**. Whether any
> rule in this codebase *finds* the discriminant has never been tested, because H2d was
> believed to have foreclosed it. **That is the open question this node now names**, and
> it is a question about learning rules, not about the pallium.
>
> **Consequence for downstream nodes.** H2c, H3 and T2 were each recorded as blocked by
> H2d. They are not blocked by the representation. T2's associative route is unblocked
> (see its node). H2c and H3 should be re-read against decodability rather than distance
> before any further work on them.

**Status: SUPPORTED as a limitation *on distance, not on decodability* (see the note above) — and after
[E077](experiments/E077-reread-balanced-ei.md)/[E078](experiments/E078-density-under-naturalistic-input.md)/[E079](experiments/E079-gain-under-naturalistic-input.md),
every structural intervention tried is null or negative under naturalistic input**
(E/I identity null, modality segregation null, density *reversed*, balanced E/I null,
recurrent gain *already optimal*, removing recurrence slightly worse). **The saturation
framing is also withdrawn**: reducing live drive by two unrelated mechanisms —
`balanced_ei` (0.73 → 0.12) and gain 0.40 (0.69 → 0.18) — gives 1.05× (null) and 0.35×
(significantly worse) respectively, while raising it also hurts. Separability peaks at
the operating point the model already occupies and falls away in every direction tested,
so saturation is a *correlate* of the best regime, not the cause of separability being
low. The live hypothesis is now that **H2d is a property of random projection without
learned feature extraction rather than a defect with a fix** — a two-channel distinction
in a 138-channel observation, projected randomly, differing by ~7-8% of mean rate. E079
§8 names a cheap test of that (separability against progressively masked observations,
since `OBS_DIM` grew 59 → 138 over the project's life). Its own measurement series also
needs re-reading**
— [E073](experiments/E073-naturalistic-separability-probe.md) found the probe used
since E009 **under-drives the pallium by ~2.7×** relative to live operation (sparse
probe 0.2724 mean rate, naturalistic 0.6019, **live rollout 0.7288**). Two consequences.
**H2d is worse than recorded**: naturalistic baseline separability is 0.0365 against the
sparse probe's 0.0961. **And E009's saturation diagnosis was never actually fixed** —
E023's gain re-baselining reported mean rate 0.189, but on the sparse probe; live, the
network is still saturated at 0.7288, where the sigmoid slope is shallow. The series
(E009/E017/E023/E034/E035/E041) is internally valid but characterises a low-drive regime
and is **specifically blind to interventions acting on common-mode drive** — which is
how [E072](experiments/E072-balanced-ei-and-h2d.md)'s `balanced_ei` appeared to score
null there and 2.13× under naturalistic input — **a difference since withdrawn**.
[E077](experiments/E077-reread-balanced-ei.md) re-ran it against E076's corrected
baseline and the 2.13× collapsed to **1.05×, t=0.35, null**: it had been measuring
E063's place block (25.1% of observation drive, always on), not the pallium. **`balanced_ei` does not improve separability under either probe and is closed as an
H2d intervention.** What survives E073 is the half independent of that block — the
E009-series probe really does under-drive the pallium (live 0.6907 vs sparse 0.2724),
and E009's saturation really was never fixed, E023's 0.189 having been a sparse-probe
number. **This is the blocker
behind H2b, H2c and everything downstream.**

Hearing an alarm call and seeing a hawk drive pallial states that differ by 0.008 —
under 1% of the mean rate. The network runs saturated (mean pallial rate 0.83, where
the sigmoid slope is ~0.12) ~~and the two stimuli project onto overlapping random
subsets of the sensory stub with nothing downstream to decorrelate them~~.

A gain sweep shows saturation is real but not the whole story: dropping the recurrent
gain from 0.9 to 0.6 moves the mean rate to 0.21 and improves *relative* separability
fourfold, but absolute separability barely moves and collapses below 0.6.

**Mechanism corrected by [E017](experiments/E017-where-separability-is-lost.md).** The
struck clause above was inferred rather than measured, and it is false. The sensory
stub separates the two percepts *cleanly* — relative separability 1.055, sharing only
9% of target units, cosine 0.245. They are close to orthogonal when they arrive.

The 17× loss is the single sensory → pallium projection, and it is **not** recurrent
mixing: zeroing pallial recurrence makes separability slightly worse (0.79×), not
better. What is left is dilution by fan-in — each pallial unit sums ~19 stub inputs of
which one or two carry the distinction, so a clean difference lands as a small
perturbation on a large common-mode drive.

Segregating auditory afferents onto their own pallial target, as a real bird has (Field
L via nucleus ovoidalis, separate from the entopallium via rotundus), recovers 2.06×.
Real, in the right direction, and not sufficient: 2× against a 17× loss.

**A separate finding from E017 that may matter more than any of the above.** The innate
arc has *no* response to hearing a call — every auditory entry in `reflex_matrix()` is
zero, against a weight of 8.0 for crouch on seeing a hawk. Comprehension being learned
in real chickens does not mean it is learned from nothing; naive chicks respond
differentially to conspecific fear calls at hatch, and the learned part is association
off an already-arousing stimulus. So this node may be diagnosing a representation
problem where there is also a missing scaffold.

**E002, E007, E008 and E009 were all attempts to fix the routing of learning.** The
blocker is upstream of routing: there is not enough in the pallial state to route.

~~**Caveat this raises for an existing result.** H2's supported finding (E004, t=3.93)
was obtained in the saturated regime. Drive regulation evidently only needs coarse
modulation, which a saturated network can still supply — but it should be re-run once
the operating point is fixed, and may well get stronger.~~ **It was re-run, repeatedly,
and did not get stronger — it vanished.** E010 first (same design, corrected gain,
t=3.93 → t=0.08); the fully corrected connectome and world independently by E037
(+0.0003 ± 0.0156, t=0.02). H2 is a clean null on every operating point this has been
checked at, not a supported finding waiting on saturation to clear.

~~**Not yet changed.** The gain default stays 0.9~~ — **stale**. ~~The re-baselining
happened; `hen/connectome.py:48` has `gain = 0.70`.~~ **That correction had itself gone
stale in both value and line number** — the actual default is **`gain = 0.95`** at
`hen/connectome.py:81` ([E079](experiments/E079-gain-under-naturalistic-input.md)),
which is the exact failure mode the note was written to fix. E079 also **validates**
0.95: naturalistic separability peaks there, declining significantly in both directions.

**H2d was demoted from the critical path by
[E019](experiments/E019-three-verified-defects.md), pending re-measurement.** Its whole
diagnosis rests on contrasting "saw a hawk" against "heard an alarm call". At the time,
in the actual coop at the default flock size, the call channel was **constant at 1.0** and
the aerial channel averaged 0.00 — the contrast this node is built on never occurred.
Separability had been measured on hand-injected observations describing a situation the
hen never experienced. E019 fixed call audibility, which was the precondition for this
contrast to occur at all.

**Partially re-measured since, and never propagated here until now.**
[E023](experiments/E023-ei-fix-and-rebaseline.md) re-ran the same hand-injected
settle-and-separate probe (E009/E017's method — still not a live-coop measurement) on the
corrected, E/I-fixed connectome, across a gain sweep:

| gain | mean pallial rate | separability |
|---|---|---|
| 0.70 | 0.189 | 4.5% |
| **0.95 (default)** | **0.276** | **7.4%** |
| 1.00 | 0.320 | 9.4% (peak) |

Against the old, purely-excitatory network's 7.5% at its own usable default: **identical
to within noise.** A structural review had predicted a 1.4× improvement from fixing
inhibition; it did not appear. **H2d is untouched by the E/I fix** — a hen still barely
distinguishes a heard alarm from a seen hawk, at ~7% of mean rate either way, regardless
of which bug-fixed connectome the number is taken on.

**Both now checked — [E034](experiments/E034-h2d-remeasured.md).**

**Localisation replicates.** The loss is still ~14.5× at the sensory→pallium projection
(E017: 17×), recurrence removed still makes separability slightly *worse*, not better
(0.87×, E017: 0.79×) — **recurrence is not the cause, confirmed on the corrected
connectome.** This part is a within-genome comparison and is on solid footing.

~~One real miss: Field-L-style auditory segregation now recovers **1.45×**, not E017's
2.06× — real, same direction, smaller. Cite 1.45× going forward, not 2.06×.~~
**Neither number survives [E035](experiments/E035-modality-segregation-in-the-prior.md),
run immediately after.** Both were unpaired ratio-of-means across 6 genomes on a quantity
with ~6× genome-to-genome spread (0.039–0.221 in E035's own sample); a properly
re-normalised, *paired* 12-genome test found modality segregation indistinguishable from
no segregation at all (t=0.04 against threshold 2.201). **Do not cite either figure.**
Modality segregation is not an established partial fix for H2d — it is untested again,
this time by a test that actually controls for the confound (segregation was never
separated from "less total input drive to the segregated slice" in either prior number).

**E041's density result does not transfer, and reverses
([E078](experiments/E078-density-under-naturalistic-input.md)).** It reproduces exactly
on its own probe (2.12× at full connectivity vs E041's ~2×) — a transfer failure, not a
replication failure. Under naturalistic input denser connectivity is **significantly
worse**: 0.90× at density 0.60 (t=2.70) and **0.71× at 1.00 (t=5.37)**. The mechanism is
saturation, visible in the mean-rate column: sparse input barely moves drive as density
rises (0.2724 → 0.3040) while naturalistic input goes **0.4602 → 0.6745**, into the
compressive region — and live operation sits at 0.6907, so E041's recommended direction
moves the network *toward* the regime where separability is worst. **There is an optimum
naturalistically and the shipped default is at it** (peak at 0.30, significant decline
above, non-significant below), directly contradicting "no optimum found all the way to
full connectivity". **With this, H2d has no intervention with a positive effect under
naturalistic input**: E/I identity null (E023), modality segregation null (E035),
`balanced_ei` null (E077), density reversed (E078). The original E041 text follows.

**A third candidate tested (E041): sparser sensory→pallium connectivity, on the theory
that fewer inputs means less dilution of the informative signal.** Falsified, clearly,
in the opposite direction — separability gets *worse* as density falls (t=4.08 to 5.37
across four density reductions, 12 paired genomes), and *better* as density rises, with
no optimum found all the way to full connectivity (density 1.0, ~2× the default's
separability). The mechanism is not dilution — it's that random sparse sampling gives
too few pallial units a chance to connect to the 1–2 informative channels at all. More
connections is simply better, everywhere tested. **A promising direction, checked and
not yet adopted**: throughput is unaffected (the dense-with-mask architecture means
density doesn't change compute at all) and an 8-seed check found H2's contrast is not
broken — but that check is not itself evidence of anything, per E037's own demonstrated
block-to-block swings on this exact metric, and full connectivity is still not the
default until something more decisive says so.

**Occurrence is no longer hypothetical, and this is the important part.** A 5-minute live
rollout at H4's standard config (16 hens, hawk every 20 s, fixed hen, 480,000 hen-steps)
found the auditory aerial channel spans its full range (std 0.37, not remotely constant —
E019's fix reaches live operation) and **a hen is blind to the hawk while a flockmate
audibly alarm-calls on 11.9% of all hen-steps** — 40% of every hen-step where an alarm is
audible at all. The scenario H2d's whole diagnosis depends on is not an edge case; it is
routine whenever a predator is present.

**This reverses the reasoning that demoted H2d.** It was moved off the critical path
because its contrast seemed not to occur. It occurs constantly. Combined with the
mechanism replicating unchanged, **H2d is not a stale concern — it is the most direct
remaining lever on H2c, H3 and everything downstream of them**, and belongs back near the
top of the queue.

---

## H3 — learned usage reproduces the audience effect without being programmed

> **Blocker re-scoped ([E081](experiments/E081-separability-vs-decodability.md)).** H3 has
> been recorded as blocked by H2d. It is not blocked by the *representation*: the pallium
> supports 98.8% held-out linear decoding of the hawk-vs-call contrast. What H2d actually
> establishes is that the *distance* between mean pallial states is small. If H3 is
> blocked, it is blocked by whether its learning rule finds a discriminant — a question
> about rule type, not about what the pallium encodes — and that has never been tested.

~~**Status: UNDER TEST** (phase 2/3) — two nulls recorded, and **blocked by H2b**.~~

**Status: UNDER TEST** — three nulls now, and the blocking condition has changed.
[E036](experiments/E036-e018-rerun.md) supplied the precondition H2b said was missing (an
innate response to *hearing* a call, so the comprehension chain can close) and the
audience effect still did not emerge from learning: `S+L − S = −0.005, t=2.25`, wrong
sign, short of a 2.37 threshold at 8 matched seeds. **H3 is no longer blocked by "the
chain can't close" — it can, mechanically. It is blocked by whatever H2f turns out to
be.**

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

~~**H3 is blocked by H2b, not by its own design.**~~ **Was blocked by H2b's missing
precondition; that precondition now exists ([E018](experiments/E018-innate-auditory-reflex.md)/
[E036](experiments/E036-e018-rerun.md)) and H3 still does not emerge.** See H2b and H2f
below.

**E005/E006's original design re-run on the fully corrected system, without a scaffold
— four nulls now, and the last "maybe it was measurement" explanation is closed**
([E047](experiments/E047-h3-rerun.md)). Comprehension is exactly zero (confirming H2b's
original diagnosis is a real architectural fact, not an artefact of E019's since-fixed
inaudible channel), and both `alarm_effect` and `food_effect` stay null across every
condition, before and after 30 minutes' rearing, on 8 seeds with `food_deplete_rate`
controlled for. **E005's `food_effect` trend (+0.032, t=0.64) does not replicate** — the
point estimates here are flat to slightly negative. The required fixed control holds
exactly flat, confirming the assay itself remains sound.

**H3 has now been tested both ways this project could imagine it working, and both
fail.** Without a scaffold: no comprehension, nothing to build audience-sensitivity on
(this result). With a scaffold supplying comprehension directly: comprehension is real,
but no learned audience-sensitivity appears on top of it (E036/E040). H3's null is the
same shape as H2c's (E042–E044): every named precondition has now been supplied at some
point, and the specific new contingency these tasks need still does not emerge. **Status
stays `UNDER TEST`, but "blocked by H2b" is retired as the operative explanation** — the
blocker is whatever is common to H2c, H3 and H2f, not a missing precondition specific to
any one of them.

---

## H2b — the learning rule cannot acquire behaviours outside the innate repertoire

**Status: SUPPORTED as a limitation, and narrowed** — established by
[E006](experiments/E006-audibility-weighted-kin-reward.md), refined by
[E007](experiments/E007-exploration-does-not-rescue-comprehension.md), narrowed by
[E036](experiments/E036-e018-rerun.md). **This blocks H3, H4 and H5.**

**The narrowing, from E036.** H2b's original claim was that H3 fails because nothing
responds to a heard call — no foothold for the rule to retime. E018/E036 supplied that
foothold by construction (a fixed, innate crouch-on-hearing response) and re-ran the
full 2×2: with comprehension no longer missing, the learned, audience-contingent *extra*
calling H3 predicts still did not appear (`S+L − S = −0.005 ± 0.002, t=2.25`, wrong
sign). **"Missing a foothold" is no longer sufficient as the whole explanation.** The
rule can be shown to retime what it already has (the scaffold gives it exactly one
stimulus-response pairing to retime); it does not, from that, build the *specific new
contingency* — call more with an audience present — that H3 needs, even with every
ingredient for that contingency present in the world. See H2f.

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

## H2f — the learning rule is the wrong *kind*, not merely wrongly routed or blocked

**Status: SUPPORTED — narrower than the falsifier's clean ideal, and now carrying an
unresolved caveat about its own control.**
[E074](experiments/E074-balanced-ei-adoption-gate.md) re-ran E057's contrast on the
current codebase and found **the food-channel control firing: +0.1054, t=10.04**, 47% the
size of the audience-specific effect it exists to control for. E057 reported that control
**null**, and that null is precisely what distinguished a targeted audience effect from
indiscriminate elevation — the same distinction E058 used to reject H2c's apparent
result. Something between E057 and now changed it; the candidates are all recent
(`OBS_DIM` 74 → 138, `N_CALLS` 4 → 5 shifting every audio index, and E067's `m_acc` fix,
which altered the reward-gated pathway `W` and so the pallial states `W_out` reads).
**Bisected and repaired ([E075](experiments/E075-bisect-h2f-control.md) →
[E076](experiments/E076-closing-the-bisect.md)). With both causes disabled the control
returns to null (t=0.98) and E057 reproduces to within noise** — general +0.132 against
E057's +0.123, audience-specific +0.241 against +0.232. **E057's result stands, fully
reproduced; only its control had been broken, by two additions made while building T2's
scaffold (E060's contamination and E063's place-cell block, 25.1% of all observation
drive), neither of which had an opt-in. Both defaults are now `False`.** The detail
below records how the bisect got there. E067's `m_acc`
fix is **ruled out** (reverting it moves the control by 0.004 — and the non-effect is
coherent: H2f's reward is `d_drive`-dominated, where snapshot and window mean coincide,
while E067's defect applied only to discrete events that barely occur at the 900 s hawk
default). **E060's contamination accounts for roughly half** — disabling it takes the
control from t=10.04 to t=2.70, halved but still over threshold. The residual is
unexplained; the measured lead is E063's place-cell block, which is **25.1% of all
observation drive and active 100% of the time**, though its causal contribution is not
yet tested. **The audience-specific effect itself is not in question** — it holds at
t=31.7–46.9 across all three arms. What is in question is the control that made it
interpretable as *targeted* rather than *indiscriminate*.
([E057](experiments/E057-separating-audience-from-elevation.md)). A non-reward-gated
readout rule builds a real, twice-replicated, predominantly audience-conditional change
in alarm calling — the falsifier's own terms — where the reward-modulated instrumental
rule was null on the identical task and scaffold (E036/E040). See below for the full
history and the precise scope of what is and is not established.

Opened by [E017](experiments/E017-where-separability-is-lost.md)
as an unnamed possibility, promoted to a node by
[E036](experiments/E036-e018-rerun.md)'s pre-committed falsifier firing. **Checked
against the `food_deplete_rate` confound that broke H2e and found robust**
([E040](experiments/E040-h2f-depletion-audit.md)): a clean re-measurement reproduced
E036's result to three decimal places (−0.005 both), same sign, not significant in
either world.

**Claim:** `hen/plasticity.py` implements reward-modulated three-factor plasticity —
instrumental conditioning: act, get rewarded, strengthen whatever was active. The
biology this project keeps trying to reproduce with it (Curio's mobbing-chain work: naive
birds acquiring predator recognition and alarm responses purely by *observing* a
conspecific respond, no reward, no action of their own, transmitted along a chain of six
birds) is closer to Pavlovian / observational association. If the rule is the wrong kind,
every experiment that assumed the right rule in the wrong place — E002 (routing to the
motor output), E007 (adding exploration), E008/E009 (top-down association, blocked
separately by H2d), and now E018/E036 (supplying the missing innate foothold) — was
addressing symptoms of the same underlying mismatch rather than fixing it.

**Evidence so far — one clean, pre-committed falsifier firing.**
[E036](experiments/E036-e018-rerun.md) built the one precondition H2b's story required
(an innate crouch response to *hearing* an alarm call) directly into the reflex arc by
construction, removing "nothing to retime" as a possible explanation, and re-ran H3's
audience-effect test. Learning still did not add a contingent, audience-sensitive extra
call on top of the wired-in response: `S+L − S = −0.005 ± 0.002, t=2.25`, wrong-signed
relative to the registered prediction and short of a 2.37 threshold at 8 seeds. E018 §4
committed in advance that this exact outcome would promote this explanation from
speculation to leading, and it does.

**What this does not yet show.** It is one experiment on one behaviour (contingent
calling), not a demonstration that an alternative rule would succeed — no Pavlovian /
observational rule has been implemented or tested here. It also does not rule out that
H2d's representational bottleneck is doing some of the work: a rule needs something to
condition *on*, and H2d says the pallium barely represents the relevant distinctions
regardless of which rule reads them. E036's scaffold bypasses H2d for this one
stimulus-response pairing (it is wired into the reflex arc, not learned from a pallial
representation), so this result is not confounded by H2d — but a future Pavlovian rule
sourced from the pallium would still need H2d fixed to have anything to associate.

**Falsifier:** a rule closer to Pavlovian association (e.g. sourced from `W_pred`,
already architecturally positioned for this per H2c) succeeds where the instrumental
rule failed, on the same task and the same scaffold. Absence of any attempt is not
evidence either way; this node opens the work rather than closing it.

**Why this belongs in the tree rather than staying a footnote.** It is a genuine fork in
what "fix the learning" means for every hypothesis below H2: not a parameter, not a
routing change, not a missing precondition, but potentially the entire *class* of rule.
That is a larger claim than anything else currently `UNDER TEST` and needs its own
falsifiable test before anything downstream assumes it.

**First attempt at the falsifier's rule, and it broke before it could be tested**
([E055](experiments/E055-hebbian-readout.md)). A non-reward-gated ("Hebbian") variant
of the readout update (`hen/plasticity.py`'s `hebbian_readout`) produced a numerically
significant audience effect (+0.096, t=2.63) — but the mandatory diagnostic this
project's own discipline requires before trusting a surprising positive found it was an
artifact: cortical drive reached 2.0–2.7× reflex magnitude (the documented "behaviour
gets worse" regime), hunger nearly doubled, and every calling channel rose regardless of
audience condition. Cause identified: `W_out` has no synaptic-scaling correction, unlike
`W` — the reward-modulated rule's zero-mean-over-time property had been incidentally
keeping growth bounded, and removing the reward gate removed that too.

**Second attempt, with the missing stabiliser added, is more targeted but a 3-seed
diagnostic could not cleanly separate two effects mixed together**
([E056](experiments/E056-hebbian-readout-scaled.md)). `readout_scaling_strength`
(unit-tested to actually bound growth) fixed the cortical/reflex ratio (now 0.75–0.9×,
not 2–3×) and roughly halved the hunger cost — but `alarm_alone` also rose from
baseline, mixed with a larger rise in `alarm_audience`, and 3 seeds were not enough to
tell whether the two were separable with any confidence.

**Third pass, done properly on the full sample with a matched design, separates them —
and replicates on an independent seed block**
([E057](experiments/E057-separating-audience-from-elevation.md)). Measuring the
audience-specific effect as a difference-in-differences against the `S` baseline's own
alone/audience gap (rather than a raw before/after read), on 8 seeds: general elevation
**+0.123, t=8.81** (significant, real) and audience-specific **+0.232, t=45.59**
(significant, and significantly larger than general elevation, t=10.39). A food-channel
control — a call type this task gives no mechanistic route to an audience effect —
showed neither component, ruling out indiscriminate dysregulation as the explanation.
**Given the scale of the claim, replicated on a fresh, non-overlapping 8-seed block
before being trusted**: general +0.121 (t=11.37), audience-specific +0.232 (t=21.90),
food channel null again — both blocks agree to three decimal places, a level of
consistency this project has not seen on any prior result.

**The falsifier, applied literally, does not fire.** A non-reward-gated rule produces a
real, replicated, predominantly audience-conditional effect where the instrumental rule
was null, on the identical task and scaffold. The honest caveat: this is not a *pure*
audience effect — a smaller, real, context-specific (not indiscriminate) general
component rides alongside it. Status updated to reflect exactly this scope, not a
cleaner claim than the evidence supports.

---

## H4 — an intact channel beats a shuffled one on a task requiring private information

**Status: SUPPORTED as written — but it is not a result about the brain** —
[E030](experiments/E030-third-block-replication.md). **Checked against the
`food_deplete_rate` confound that broke H2e and found robust**
([E039](experiments/E039-h4-depletion-audit.md)): an 8-seed clean re-measurement
(depletion off) reproduced the same sign at comparable magnitude and was significant
even at that reduced sample (t=5.52). Unlike E032/E033, this status stands unmodified.

**The channel effect is real.** A flock hearing its own present tense is caught
**−0.044 ± 0.012 (t=3.60)** less often per hen per hawk than one hearing the same calls
shifted in time. Thirty-six seeds, three independent blocks, on a denominator no
behaviour can move, with the analysis **fixed and committed before the third block ran**.
None of the three pre-registered falsifiers fires.

| block | L vs C?, caught/dive | control's own risk |
|---|---|---|
| A (36–47) | −0.029 ± 0.020, t=1.42 | 0.115 |
| B (48–59) | −0.076 ± 0.016, t=4.75 | 0.156 |
| C (60–71) | −0.028 ± 0.025, t=1.10 | 0.129 |
| **pooled, 36 seeds** | **−0.044 ± 0.012, t=3.60** | — |

**Two things this does not license, and the second is the important one.**

- **Block B is an outlier.** A and C agree at −0.029 and −0.028; B is 2.7× larger and the
  only block significant on its own. A conservative reading of the typical effect is
  nearer **−0.03**, in a world whose baseline danger varies.
- **`Lx` — the intact channel with `W_out` zeroed — is indistinguishable from `L`**
  (+0.010, t=0.46), for the second time on fresh seeds. **H0's subject is a neural model
  of a chicken, and the pallium contributes nothing measurable here.** What is supported
  is a claim about a channel bolted to a reflex arc: two hand-set weights in
  `hen/innate.py` and a threshold in `coop/world.py`. H0 is *not* satisfied by this, and
  cannot be until a learning rule works — `C−`, the capacity control the whole ladder was
  designed around, is vacuous with plasticity off.

~~**Status: UNDER TEST — smaller than reported, and it does not require the pallium** —
[E027](experiments/E027-third-review-verified.md),
[E028](experiments/E028-instrument-repair.md).~~

**On a repaired instrument the headline does not clear.** E028 rebuilt the metric so its
denominator is every (hen, dive) pair — fixed by the predator schedule and flock size,
unreachable by any behaviour — and re-ran the full ladder on fresh seeds 36–47:

| contrast | metric | result |
|---|---|---|
| **L vs C? (registered)** | caught/dive (ITT) | **−0.029 ± 0.020, t=1.42 — not significant** |
| L vs C? | caught/event (E026's, confounded) | −0.142 ± 0.061 — SIGNIFICANT |
| **L vs Lx (`W_out` lesioned)** | caught/dive | **+0.006 ± 0.013 — noise** |
| L vs C? | fed % (T1's registered metric) | +0.053 ± 0.087 — null, third block running |

Two metrics, one dataset, opposite verdicts. `dives` is flat to **0.0%** across all seven
conditions; `blind risk` — E026's denominator — spans a **60% range**. The previous
headline was substantially an artefact of that.

**−0.198 became −0.029.** The direction still favours the intact channel in every
contrast, so a small real effect is likely; it is not what `SUPPORTED` claimed.

**[E029](experiments/E029-positive-control.md) ran the project's first positive control
and the instrument passed.** Planted effects of every size are detected, including the
unmodified hen — so E028's null was **not** a measurement failure, which was the live
possibility. But the same contrast on a fresh block measures **−0.076 ± 0.016, t=4.75**,
against block A's −0.029 ± 0.020, t=1.42. Spread is comparable (sd ratio 0.80×), so the
*means* differ, by 2.7×; the control's own risk differs too (0.115 vs 0.156), i.e. block
B is a more dangerous world.

| block | L vs C?, caught/dive |
|---|---|
| A (36–47) | −0.029 ± 0.020, t=1.42 |
| B (48–59) | −0.076 ± 0.016, t=4.75 |
| **pooled, 24 seeds** | **−0.052 ± 0.014, t=3.87** |

**Not promoted, for three reasons:** the decision to pool was made *after* seeing the
blocks disagree and was not the registered analysis; block A alone does not clear; and
the `Lx` lesion rung is still noise, so this remains a result about two hand-set reflex
weights rather than the neural model. A third block with pooling declared in advance is
the honest way to settle it.

~~**Status: SUPPORTED** at this capacity and this task —
[E026](experiments/E026-h4-supported.md). **The first status past H1a to be supported,
and the headline.**~~

**Downgraded by [E027](experiments/E027-third-review-verified.md), a third outside
review, verified.** With `W_out` set to exactly zero — a complete lesion of the only
route by which 512 simulated neurons can reach a muscle — **the benefit survives**:
−0.208 ± 0.148 paired, against −0.240 ± 0.110 with the pallium intact (8 fresh seeds,
neither significant). H0 asserts that *a neural model of a chicken* can be given a
channel that changes what the flock can do; what E026 measured survives deleting the
neural model. The causal chain is two hand-set weights in `hen/innate.py` and a threshold
in `coop/world.py`.

**What stands:** a contingent channel beats a non-contingent one, and the yoked control
is sound. **What is withdrawn:** that this is a result about the *brain*. The honest
claim is that a well-timed one-bit interrupt restores the receiver's own vision, in what
is currently a reflex agent.

**Three further corrections from E027**, each measured:

- **The metric's premise is false.** "The denominator is fixed the instant the hawk
  commits, so the treatment cannot move it" appears in four files. The *within-dive*
  denominator is fixed; the number of dives that find a hen blind and at risk is a
  behavioural outcome, and it moves **up to +63%** across conditions.
- **The headline is a mean-of-ratios.** Pooled over the same events it is **−0.150**,
  not −0.198. Both estimators are legitimate; the prose quotes the larger one.
- **The scaffold's work is done by the head-raise**, not the crouch response
  (−0.124 vs −0.067). E018 §8 pre-registered this ablation, `hen/innate.py:146-163`
  predicted it, and `scratchpad/verify_yoked.py`'s docstring claims to have run it. None
  of the three did.

A hen who can hear her flockmates is caught **~20 percentage points less often** in
exactly the moments she could not see the hawk herself — **on the mean-of-ratios
estimator; 15.0 points pooled.**

Metric: P(caught | at risk **and blind** at dive onset) — a denominator fixed the
instant the hawk commits, restricted to hens who could not see it, so the treatment
cannot move it and the subset is the only one where a call carries information the
receiver lacks.

| contrast vs deaf | block A (0–11) | block B (12–23) | **pooled, 24 seeds** |
|---|---|---|---|
| **intact channel** | −0.187 ± 0.071 | −0.208 ± 0.095 | **−0.198 ± 0.059, t=3.33** |
| **yoked control** | +0.017 ± 0.040 | +0.052 ± 0.077 | +0.035 ± 0.043, noise |
| intact, no scaffold | +0.031 ± 0.046 | +0.103 ± 0.043 | +0.067 ± 0.031 |

**Both falsifiers were checked and neither fires.** The yoked control — identical calls,
rate, amplitude and energetic cost, shifted in time so it carries no contingency — is
flat in *both* blocks. The benefit is the **information**, not the arousal. And an
intact channel without comprehension gives no benefit, so the channel needs a receiver.

**Replicated on seeds decided before they were run**, per the E021 rule. −0.187 then
−0.208; block B misses significance alone (t=2.19 vs 2.201) with a *larger* magnitude,
the opposite shape to E021's collapse.

**What this does not license.** No plasticity anywhere — comprehension is innate via the
E018 scaffold, so this shows a working channel helps, not that language is *learned*.
H4's prediction never mentioned learning, so this is H4 as written; H0 wants more, and
H2 remains a clean null. T1 only: the signal means "danger", not *which* feeder. And it
runs on a world that changed the same day (the hawk approach phase), so no other number
in this tree is directly comparable to it.

**Four defects stood in the way and all were measurement errors, not brain problems:**
a control that retained 98% of the information it was meant to destroy; a risk metric
confounded three different ways; a world with no interval in which a warning could
arrive; and an innate response arithmetically incapable of hiding a hen. See E026 §2.

---

## T1 — shared vigilance: does the channel let the flock forage more, not just survive more

**Status: SUPPORTED as a narrower claim than proposed** —
[E045](experiments/E045-t1-pareto-frontier.md). `docs/backlog.md` §3's operationalisation
of the "many eyes" prediction under H4: a working channel should let the flock divide
vigilance, so a flock that hears real danger (L) forages *more* than one hearing an
uninformative channel (C?), at matched risk.

**The original prediction — L forages more than C? — is not supported, checked nine
times.** Three prior single-capacity mean comparisons (E026, E028, E028b), a
zero-compute depletion-controlled re-analysis of E039's cache this session, and now a
five-point capacity sweep (`pallium_scale` 0.5×–4.0×, E045) all found fed % statistically
indistinguishable between L and C?. No point across any of these nine checks came close
to significance. This is the standing characterisation now, not an unreplicated null
awaiting a bigger sample.

**What is supported: a narrower, still-real form of the prediction.** L is safer than
C? at every one of the five capacities tested in E045 — same sign throughout, extending
H4's single-capacity result across the full range for the first time — while fed % stays
flat, not worse. A flock with the channel is not eating more, but it is not eating less
either, and is substantially safer throughout. That is genuine Pareto improvement (better
on one axis, tied on the other), just not the specific "freed-up foraging time" mechanism
the many-eyes literature and the original prediction imagined.

**A non-monotonic pattern in fed % across capacity, reported but not established.**
L−C? fed % is negative at 0.5×, positive at 1.0×–2.0×, negative again at 4.0× — an
inverted U, not scatter with no structure — but no individual point is significant at
n=8, so this is a plausible mechanism (interference from an under- or over-capacity
untrained pallium, per E002's ceiling finding) worth a properly powered follow-up, not a
confirmed curve.

**Falsifier, as it stood:** fed % indistinguishable from C? at every capacity tested —
this fired. **What it means for T1**: the hypothesis moves to `SUPPORTED as a narrower
claim` rather than `REFUTED`, because the mechanism it was built to detect (a real
Pareto-relevant consequence of the channel) is present — just on the risk axis, not the
intake axis the original design emphasised.

**T1's second prediction — per-hen vigilance falls as flock size rises — is falsified
too, and the reason corrects the mechanism, not just the data point**
([E046](experiments/E046-t1-flock-size.md)). `head_down` (the vigilance proxy) *falls*
with flock size — hens forage *less*, not more, as the flock grows — at similar rates
for L and C? (−0.0017 vs −0.0025 per hen), with no channel-specific effect. Most likely
cause: this codebase's auditory channel combines in power, so more hens means a louder,
busier acoustic environment regardless of whether calls carry information, and the
innate call-suppression reflex fires on loudness, not content — a chorus effect
symmetric across L and C?, not a vigilance strategy responding to trust in the channel.

**Structural reading: "shared vigilance" in the classic sense — individuals actively
relaxing their own watchfulness — is not something this architecture can produce, and
there is a principled reason why not.** The reflex arc has no adaptive policy to relax;
behaviour is a fixed function of instantaneous input, not a strategy conditioned on
flock size or trust in flockmates. What the E045 safety result more likely reflects is
a passive, statistical version of many-eyes: more hens means more independent chances
someone sees the hawk in time, a property of the population rather than any individual's
behaviour. Consistent with this: E046 found the safety advantage itself is **not
flock-size-uniform** — indistinguishable from zero at `n_hens=4` and `8` (t=0.03, 0.32),
and only clearly significant at `n_hens=32` (t=4.02) — a threshold-like pattern across
4 points, not the smooth scaling E045 (fixed at `n_hens=16`) could not have revealed.

**Both of T1's original predictions are now settled**, replacing the informal open
status they carried through four scattered checks: no intake benefit at any capacity
(E045), no vigilance-relaxation mechanism at any flock size (E046), and a real,
flock-size-dependent safety benefit whose likely origin is statistical (more
independent chances for a true warning to exist) rather than behavioural (individuals
changing what they do).

**Relationship to H2c/H2d/H2f**: none. T1, like H4, runs entirely on the innate route —
hardwired call production plus the E018 comprehension scaffold, no plasticity anywhere.
It says nothing about whether *learning* ever contributes to this or any other channel
consequence, which remains separately open.

---

## T2 — the rotating poisoned feeder: does the flock learn *which* feeder to avoid

**Status: NOT SUPPORTED instrumentally — and, after a positive control
([E069](experiments/E069-t2-positive-control.md)), for a now-settled architectural
reason rather than an instrumentation one. The *associative* route (T2-revised) has a
chain that demonstrably conducts (E082). Both instrument defects are fixed and measured
(E083/E084/E085). The representation blocker E085 identified is **now also fixed**: routing
place cells to the hippocampus takes parked decodability to 99.5% and gives 90.0% under
movement when read directly ([E086](experiments/E086-place-to-hippocampus.md)). What
remains is narrower and specific — **`W_pred` cannot reach that signal**, because its
source is diluted across 336 units and then high-passed by a centring baseline whose 20 s
corner sits directly on the 17–75 s timescale of the hen's own dwell times.** A `sickness_penalty` sweep across a
thousandfold range produces no learned avoidance at any magnitude: `late−early` runs
+1.00, +0.08, +0.08, +1.25, −0.33 with no trend. The signal does reach the weights
(mean `|W−W₀|` rises 26% at penalty 1000) but produces undirected perturbation, not
behaviour — and the connectome survives intact (97.5% of innate synapses), retiring
the E014 erosion concern. This confirms empirically what E058/E059 + E063 jointly
predict: the rule amplifies existing innate anchors and place cells were deliberately
given none, so no reward magnitude can produce place-specific avoidance. The metric
itself is sound (E069 Part A: resolves ~19–35% of baseline at n=8, far finer than
T2's own predicted effect). **Tuning cannot rescue T2** in its instrumental form. A revised architecture is
designed in [docs/backlog.md](../docs/backlog.md)'s **T2-revised** section: the same
task built *associatively* rather than instrumentally, which sidesteps E067's and
E069's findings entirely because neither applies to a rule that is not reward-gated.
It reuses the whole existing scaffold and needs only two new mechanisms — an innate
withdrawal response to *hearing* a gakel call (an anchor on the call, not on any
place, so nothing location-specific is wired), and a shared allocentric population so
a place learned by testimony is recognisable when visited. The associative machinery
itself already exists and already feeds perception (`W_pred` + `pred_gain`), off by
default. L vs. C? remains a real contrast under it: a yoked listener binds calls to
the wrong places, learning as hard as L but learning nothing true.

**All three Stage 2 runs remain withdrawn as tests of the actual claim.** E065, E066 and E068 measured what the code does, accurately; none of them
delivered a teaching signal large enough to test whether a flock *can* learn this.
[E068](experiments/E068-t2-stage2-fixed-eligibility.md) measured why: the sickness
reward is ~0.007% of the reinforcement a hen receives — 0.04% of consolidation
windows carry one at all, and it supplies 16% of the signal even in those. Turning
`sickness_penalty` on vs off moves the recurrent weights by 0.07%, inside seed noise.
The reward design needs calibrating, and a positive control needs running, before a
fourth run means anything.
([docs/backlog.md](../docs/backlog.md)'s T2 section has the complete mechanical design;
[E060](experiments/E060-t2-contamination-scaffold.md) built and validated the scaffold
in isolation, 12/12 ethogram assays including four new falsifier checks, 74/74 full
suite; [E061](experiments/E061-t2-population-scaffold-check.md) confirmed all three
pieces work at the population level in a free-running 16-hen flock;
[E062](experiments/E062-t2-contamination-period-calibration.md) swept
`contamination_period_s` and found no reason to move it off 300s — no learning
involved in any of Stages 1/1b/1c).

**Prerequisite found and filled before Stage 2 could be designed
([E063](experiments/E063-allocentric-place-cells.md)):** T2's literal claim — durable
avoidance of *this specific feeder*, outlasting the visible/audible cue, recognised
from a later approach in any direction — turned out not to be representable at all.
Every existing sensory channel is egocentric by design (`coop/sensing.py`'s own
docstring); a hen who turns away from a location loses any representation of having
been there. E063 added a fixed grid of innate, allocentric place cells (`hen/regions.py`'s
previously-generic `hippocampus` region's first real function) — `OBS_DIM` 88 → 113,
77/77 full suite including the unchanged ethogram.

**Second prerequisite found and filled
([E064](experiments/E064-gakel-location-cue.md)):** E063 alone only solves the problem
for a hen who directly *witnesses* a sickness event (her own place cells tag the
moment `CLS_SICK` and `CLS_FOOD` co-occur in view). A hen who only *hears* the gakel
call from beyond visual range — precisely the case the call was built for — still gets
nothing spatial: audio here has never carried direction, and self-location doesn't
reveal anyone else's. E064 added a loudness-weighted mixture of gakel callers' own
place-cell patterns rather than asking the pallium to learn trigonometry from a
bearing it isn't even given (this model doesn't expose a hen's own heading as a
channel). Required extending `World` with a `pos_log` ring buffer, matching
`call_log`'s pattern, so `channel_mode='yoked'` hands a listener the caller's position
*when she called*, not her current one — verified directly (a caller who moved 22+ m
between calling and observation), the same class of leak E024's shuffled control had
for plain audibility. `OBS_DIM` 113 → 138, 81/81 full suite. Stage 2 can now be
designed against a model where both the direct-witness and indirect-testimony
pathways have somewhere real to learn.

**Stage 2, the actual test, run ([E065](experiments/E065-t2-stage2-learning-contrast.md)):**
16 hens, H2f's validated rule (`hebbian_readout` + `readout_scaling_strength=0.3`,
E057's own configuration), intact (L) vs. yoked (C?) vs. a fixed-connectome baseline
(S), 8 seeds/condition, 90 minutes (18 contamination rotations). Primary metric:
does L's sickness-per-rotation fall further from early (rotations 1–4) to late
(rotations 15–18) than C?'s does. **It does not — and the sign points the wrong way.**
L: +0.875 (late minus early, i.e. slightly *worse*); C?: −0.250 (slightly better);
primary contrast (L − C?) = +1.125 ± 0.715, t=1.57, not significant (threshold 2.365).
S's own early-to-late change (+1.719) was nominally the largest of the three, none
significant — consistent with a within-run trend common to every condition
(population dynamics unrelated to learning or channel content) rather than anything
channel-specific. `|W_out|` drift diagnostic: C? and L landed on the identical
aggregate value (0.0632) — not proof the rule was inactive (E057's own finding is
that channel content changes *which* weights move, not the whole-matrix mean), but a
real, flagged limit of what this diagnostic can distinguish. Matched water-intake
control: clean null, as it should be given the primary result was already null. No
positive control specific to this exact metric was run — flagged as the next step if
this result is ever treated as more than a first pass.

**E065 withdrawn as untrustworthy, and corrected ([E066](experiments/E066-t2-stage2-corrected-rerun.md)).**
Reviewing where to take T2 next found that `hen/plasticity.py`'s `reward()` had no
term for sickness at all — checked directly, there was none. `hebbian_readout` only
changes how `W_out` updates, and `W_out` reads from the motor region's own rates, not
sensory input; any route from place cells to behaviour has to go through `W`'s
standard reward-gated update, which had nothing to learn T2's outcome from regardless
of channel content. E065's null was very likely not a fair test. Fixed with
`sickness_penalty` (off by default, mirroring `readout_scaling_strength`'s own
precedent so no other experiment's dynamics move). Also added, before re-running: a
pre-registered secondary split of sickness onsets into *witnessed* (another
already-sick hen within `vision_range` — explainable by the innate anchor alone,
identical across all three conditions) versus *testimony-only* (not witnessed — the
only case the auditory channel could plausibly help with), addressing a design review
concern that the innate reflex could dilute a small real effect in the aggregate
metric.

**Result: still null, and the split makes it a stronger one, not a weaker one.**
Primary contrast flipped to the predicted sign but stayed tiny and non-significant
(−0.19 ± 1.25, t=0.15, threshold 2.365) — a materially different, more trustworthy
null than E065's wrong-signed one. Testimony-only onsets were about a third the
volume of witnessed ones (confirming the dilution concern was real and directionally
correct), but showed **no effect there either** (+0.06 ± 0.19, t=0.32, wrong sign
again) — ruling out "the aggregate washed out a real effect" as an explanation, since
the bucket built specifically to isolate that effect doesn't show one. `|W_out|` drift
still identical between C? and L, same caveat as E065 about what that coarse check can
and can't distinguish.

**Consequence**: with a genuine reward signal for sickness and a metric aimed
specifically at the auditory channel's own unique contribution, H2f's rule still shows
no detectable learned avoidance. T2 stays `NOT SUPPORTED`, now on solid methodological
ground — the two live objections to E065 have both been checked, not just noted, and
the conclusion held. Consistent with, not contradicted by, E058/E059's finding that
this rule amplifies existing anchors rather than building new associations from
nothing — T2 asked for something harder than either H2f or H2c did. Not claimed as
fully closed: the local-vs-aggregate weight-change question, a longer run, or a
different rule all remain genuinely open and unpursued, not ruled out.

**Correction, from an adversarial review and independent re-verification
([E067](experiments/E067-reward-eligibility-sampling-defect.md)): "a genuine reward
signal" above overstates what E066 actually built.** `m` (the factor gating
`consolidate()`'s update to the recurrent weights `W`) is not a trace — it is
recomputed fresh every step but only the value at the exact consolidation-boundary
step (`w_next.t % pc.interval == 0`, every 50 steps) is ever used. `sickness_penalty`
is a discrete, single-step event, and a swept check over every possible offset within
one interval found it reaches `consolidate()` on only **2% of occurrences** —
confirmed independently with a fresh script against the real `hen/plasticity.py`
functions, not merely read off the review. `strike_penalty` — used since ~E014,
throughout H2/H4/T1's history — shares the identical defect, confirmed the same way;
that broader implication is named in E067 but deliberately **not acted on** here, per
this project's own red-team rule against rewriting the tree on an unverified
reinterpretation. For T2 specifically: this is a third, independent, structural
reason the rule could not plausibly have succeeded — on top of `hebbian_readout`'s
`W_out` update never reading sensory input at all (already known) and place cells
having no pre-existing motor correlate to amplify (already known). T2's *numbers*
in E065/E066 stand as genuine measurements of what the code as written does; the
*interpretation* changes from "a fair test the rule failed" to "the instrument could
not have detected success even if the flock were capable of it." Status unchanged
(`NOT SUPPORTED`), for a now more precisely understood reason.

**Claim:** a flock with a working communication channel converges toward roughly one
hen's mistake per contamination rotation (the discoverer eats the bad feeder, gets
sick, and warns the rest); a flock without one pays roughly N times that, each hen
discovering independently. This is `docs/backlog.md` §3's original T2 design,
unchanged in substance — what changed is that building it surfaced two real gaps that
needed their own design rather than being assumed away.

**What had to be added, in brief** (full reasoning in the backlog): no call in this
model, or in the real-chicken literature this project already cites, is specific to
bad food — the gap is filled with the gakel-call, a real, documented
frustration/negative-expectation vocalisation, not an invented one. No contamination
mechanic existed at all — added as a new `World` state, rotating on a period, invisible
until eaten. Discovering the bad feeder produces a physiological consequence stated
directly by design instruction: **visibly slowed movement for a duration, then
recovery** — mechanical, like crouching, not a learned or reflex *choice*. The
acoustic-only "which feeder" problem already flagged for the ordinary food call applies
identically here, and is fixed the same way any referential-alarm case in this model
is fixed: a call (marks that something happened, carries information past visual
range) paired with a visual cue (`CLS_SICK`, a sick flockmate's location, the same
mechanism `CLS_FLOCKMATE` already uses) that actually carries *where*.

**Innate vs. learned, stated precisely — both, the same split this project has used
throughout.** Contamination, the sickness state and its physiological slowdown, gakel
production, the `CLS_SICK` sense, and a proposed innate turn-away-from-`CLS_SICK`
reflex (the anchor) are all wired, none of it learned — matching how every other
reflex, sense and call in this model is built. **What is learned, and is the entire
point of the task**: whether the flock can turn that innate anchor's momentary
reaction into a *durable*, *location-specific* avoidance that outlasts the visible cue
— continuing to avoid the feeder after the sick hen has recovered and moved on. That is
precisely the "build a new, durable, referential contingency" claim H2f's own result
(E055–E057) showed this project's only working learning mechanism can only do by
*amplifying* an anchor, never from nothing (E058, E059) — so the anchor here is not
optional scaffolding, it is the specific, tested precondition for T2 to have any chance
of succeeding at all, chosen for that reason and stated as such.

**The associative chain conducts, and fails on one wrong reflex
([E082](experiments/E082-t2-chain-control-redone.md)).** E082 redid E070's whole-chain
positive control with a *discriminative* plant instead of a correlational one (the
distinction E081 established). This time the plant fires — pre-flight 1.000 at the
planted place per seed, 0.86–0.96 live — and **forward drive falls 17%** (0.622 → 0.519)
as `pred_gain` rises. So every link works end to end: place cells → pallium →
discriminant → `W_pred` → `relu` → `reflex_in` → gakel scaffold → motor. E070 could
establish none of that, because its plant never fired. **But occupancy at the planted
feeder does not fall** (0.4501 → 0.4339, non-monotonic, ~3% of a 45% baseline), and the
reason is mechanical rather than neural: `_add_gakel_scaffold` suppresses `M_FORWARD`,
and `coop/actuation.py` derives speed from it — so a hen already *at* the bad feeder who
slows down **stays there**. The anchor produces lingering where avoidance requires
leaving. Its own docstring explicitly declines to borrow the anti-predator response
("*no crouch or flee — this is bad food, not a predator*") and the implementation is
functionally a freeze regardless. Hunger stays flat (0.427 → 0.437) and control-feeder
occupancy stays flat, so neither the hallucination nor the smearing falsifier fired —
**selectivity was the risk flagged as most likely to bind, and it did not.** Next:
redesign mechanism 1 so the response produces *leaving* rather than *stopping*
(suppressing `M_PECK` alone is the cheapest candidate, and its ethogram assay already
exists), and **do not re-run the L vs. C? contrast until a redesigned anchor passes this
control.** *(The first run of E082 was invalid and is recorded as such: it planted
against raw `z_lag` while the runtime reads a converged `z_lag − z_lag_bar`, giving
`pred@gakel` 0.04 instead of 1.0 — the identical timescale error E071 documented, made
two experiments after writing it down. Corrected with a 300 s tour-settle and a
pre-flight assertion.)*

**The plant is anti-selective, and three experiments' behavioural readouts go with it
([E083](experiments/E083-gakel-anchor-produces-leaving.md)).** E083 redesigned mechanism 1
per E082's recommendation — `_add_gakel_scaffold` now suppresses `M_PECK` only, never
`M_FORWARD` — and re-ran the matched control. The redesign works as a redesign: the freeze
is gone (forward drive flat at 0.633 → 0.626 where E082 had it falling 17%) and the reflex
has its proximal effect (pecking at the planted feeder falls monotonically, 0.550 → 0.504).
**Occupancy at P still did not fall**; it rose 9%, non-monotonically.

Then the diagnostic. `pred@gakel` was averaging 0.90 over a run in which the hen is at P
only ~42% of the time — the wrong shape for a selective plant. Splitting it by position
shows the plant fires at **0.656 at P and 1.244 elsewhere** (ratio 0.53), with live
magnitude varying **9-fold across seeds** while pre-flight reads exactly 1.000 on all of
them. Profiling against distance rules out the disc simply being wider than the place
code: the innermost bin (0–1 m) is the **lowest of seven** at 0.655 and the peak sits in a
ring 5–7 m away at 2.128. **The plant is inverted.** Both E082 and E083 drove the gakel
channel hardest where the hen was *not* meant to be avoiding — which explains both results
in the right direction, E083's +9% included, as the mechanism working correctly on a
signal with the wrong sign.

The cause is **the third instance of one error**, the one `CLAUDE.md` names as a quantity
verified in the place it had just been moved *from*: the discriminant is fitted on a hen
**parked** at a grid centre and read back on a hen **moving**. E071 was the same shape on
timescales; E082's own invalid first run was the same shape on centring. And the
amplitude-only pre-flight added *after* E082 does not catch it — it asserts the plant
fires at P and asserts nothing about elsewhere, exactly the gap that let E024's "shuffled"
control retain 98% of what it claimed to destroy.

This also **re-scopes E081**, without withdrawing it. Its 84.6% was measured on hens
parked at five cell centres under 0.35 m of jitter, holding one static observation for 200
steps, reading raw `rate(x)`. That number is correct for what it measured. What is
withdrawn is the inference E082 and E083 both drew from it — that a readout separating
five parked point-locations will separate *where the hen is* during free movement. **Next:
fit the discriminant on live trajectory states labelled by position, and make the
pre-flight assert selectivity (fires at P *and* near-silent elsewhere, profile decreasing
with distance) before any behavioural contrast runs.** If a live-fitted discriminant also
fails, the finding moves from the instrument to the representation — mechanism 2's shared
allocentric population would be insufficient as built, which would be the first result in
this arc that is genuinely about the hen.

**The metric could not have resolved the effect, and both falsifiers were guaranteed to
fire ([E084](experiments/E084-live-place-decoding.md)).** E084 set out to fit the
discriminant on live states instead of parked ones. It crashed on seed 1 with a degenerate
split: **0 of 19 200 samples** within 3.33 m of the planted feeder — sixteen hens, twenty
simulated minutes, none of them near it — while the same connectome's other run had a base
rate of 0.424.

Two diagnostics followed, both post-hoc and labelled as such. **The flock aggregates.**
`approach_flockmates` is innate and works, so sixteen hens behave as roughly one clump
(spread 1.66–7.21 m in a 20 m arena); the clump settles where it starts, and occupancy of
a fixed cell inherits the full between-run variance of clump location — 0.000 to 0.481
with nothing changed but the world key. The effective sample size for a spatial metric is
far nearer *one animal per seed* than sixteen.

**And the metric is under-resolved.** At E083's exact metric across 8 seeds, baseline
occupancy is 0.4244 with sd 0.2751 (individual seeds 0.0948 to 0.9651, a tenfold range).
Pairing helps — the within-seed difference has sd 0.0487 — but the minimum detectable
effect at n=4 is **18.3% of baseline**, and **E082 and E083 both pre-registered 15%**. A
real, exactly-as-predicted avoidance effect would have been reported as a null both times.
Observed difference at n=8: +6.4%, t=+1.57, not significant and positive. Two seeds also
sit at the boundary (0.9651 and 0.0948), where the dependent variable has little room to
move — `CLAUDE.md` check 5, alongside check 6.

So **four behavioural experiments in this arc (E070, E082, E083, and E084's unrun Part B)
have measured instrument properties, not the hen.** Part A's own question — is position
linearly decodable during free movement — remains **open**: one seed completed, at chance
(44.4% live-fit, 44.8% parked-fit, held-out), which is a hint and not a result.

**Next, in order.** Fix the metric: choose the target feeder **per seed from an
independent baseline run** — the cell that flock actually occupies — so every seed starts
high and has room to fall, selecting on run key A and running both arms on run key B so
regression-to-the-mean cancels in the paired difference. **n=8 minimum**, where the metric
resolves 9.6%. Then Part A, then Part B. **Standing correction:** E082 and E083 both named
a percentage threshold without checking the metric could resolve it. That check is
arithmetic on a variance estimate and costs one baseline run; it should precede any future
falsifier that names a percentage.

**The instrument is repaired, and the route is now blocked on the representation
([E085](experiments/E085-repaired-instrument.md)).** E085 made no behavioural claim; its
output is an instrument and a number saying what it can resolve.

**The metric fix works.** Choosing the target feeder **per seed from an independent
baseline run** holds up: occupancy at the target is 0.600 in the selection run and 0.602
in the test run, minimum 0.405, drift +0.8% and symmetric across seeds. Occupancy at a
per-seed target resolves **9.7% at n=4 and 5.1% at n=8**, against E084's 18.3% at n=4 for
a fixed cell. Measured from a **null–null contrast** — two null runs differing only in
dynamics RNG — so the threshold is not derived from the treatment it will judge. *Mean
dwell per visit was tried as an alternative and is much worse* (MDE 74.3% at n=8): it is
heavy-tailed, because a hen who essentially never leaves produces one enormous visit that
dominates the mean. That prediction was recorded and was wrong.

**And the decodability gate fires.** A discriminant fitted on live trajectory states and
evaluated on a held-out run of the same world scores **59.6%** balanced accuracy, ratio
1.70, profile not decreasing — all three gate conditions fail. That headline is itself
mostly class imbalance: per-seed accuracy correlates with split skew at **r=+0.870**, and
on the **six seeds with balanced splits** it is **54.3%**, +4.3 ± 1.5, t=+2.95. So the
signal is real and about **four percentage points above chance**. E081 measured 84.6% on
parked states; the tasks are not identical and the numbers should not be subtracted, but
the gap and its direction are the finding — **the place code is legible when she is
standing still and nearly illegible while she is moving.** Even the live-fitted
discriminant fires *least* when she is closest to the target (innermost bin 0.653, lowest
of seven): fitting on live states removed the parked-fit *inversion* but did not produce
selectivity.

**E083's diagnosis replicates on data that did not generate it**: the parked-fit plant
scores 48.7% held-out — chance — with ratio 0.60 against E083's 0.53.

**This is the first result in this arc about the model rather than about our measurement
of it.** `W_pred` is a linear readout, so a linear decoder is the relevant class, and the
same estimator recovers 84.6% from the same population when the hen is parked. The place
cells work. What defeats them is that pallial state under free movement is dominated by
hunger, pecking, flockmates and calls, and **E063 deliberately gave the place channels no
innate anchors** — so nothing amplifies them into the variance a linear readout can find.
That was E085's recorded prediction and it held.

**Next, in order: (1)** give the place channels a weak innate anchor — the one structural
asymmetry that made H2f work is that the rule amplifies what innate wiring already
emphasises (E058/E059, E069), and this wires *that there are places*, not *which place is
aversive*; **(2)** widen the place population, 25 cells at `place_sigma` 2.0 in a 20 m
arena is coarse and `OBS_DIM` growth is cheap; **(3)** failing both, re-scope T2 — if
position cannot be made legible under movement without distorting the model, then durable
avoidance of *this specific feeder* is not reachable in this architecture, and that is a
finding rather than a failure. **Do not run the behavioural contrast against the current
representation**; it would be a fifth null with a known cause.

*One limitation recorded: E085's determinism check ran on seed 0 and cleared, while
E084's bit-identical anomaly was on seed 4 under a different design. That anomaly is not
resolved, only shown not to be universal.*

**The hippocampus was missing from the circuit, and putting it back works
([E086](experiments/E086-place-to-hippocampus.md)).** `regions.py` names HIPPOCAMPUS
"place and spatial memory" and E063 was written up as giving it its first real function.
It never had one: `W_in` writes only into the sensory stub, so of the 64 units taking
place afferents, **64 were in the sensory stub and 0 in the hippocampus** — and `pred_src`
excluded the region anyway, so `W_pred` could not have read it. E086 routes place there
(same afferent statistics, a routing change and not a magnitude change) and extends
`pred_src` to cover it. Off by default; guard test at `n_hens=16`.

**Parked decodability 84.6% → 99.5%**, and the **distance profile is decreasing for the
first time in this arc** — the innermost bin goes from the *lowest* of seven (0.653) to
the *highest* (1.604), falling monotonically out to 7–10 m. The off arm reproduces E085
and E081 exactly, so the comparison is matched.

**But the primary falsifier fires**: under movement, 54.3% → 58.9%, +4.6 ± 3.3, t=+1.40,
not significant. A post-hoc diagnostic on identical data found why, and it is two things,
neither about the representation:

| readout (held-out, balanced-split seeds) | |
|---|---|
| `pred_src`, all 336 units, lagged-centred | 63.5% |
| **hippocampus alone, 80 units, lagged-centred** | **73.7%** |
| pallium alone, 256 units, lagged-centred | 54.4% |
| hippocampus, `z_lag` only (low-pass, τ 1.5 s) | **90.0%** |
| hippocampus, raw rate | 90.7% |

**Dilution costs ~10 points** — 80 place-carrying units pooled with 256 that sit at chance.
**The centring costs ~20**, and the lag itself costs nothing (90.7 → 90.0). `z_lag_bar` is
a **20 s running mean** and E085 measured dwell times of **17–75 s**, so the baseline
tracks position and subtracts it: a high-pass whose corner sits on the signal's own
timescale.

**The centring is not a mistake, which is what makes this a real tension.** E070 measured
a planted association predicting **1.0000 at its own place and 0.9637 at another**, because
`z_lag` is strictly positive with mean ~0.23 and the across-stimulus signal is 3.7% of that
DC. E071 added centring for exactly that reason and it worked. The pathway needs DC removed
*and* slow signals preserved. **And the 20 s is not a considered choice** — `z_lag_bar`
shares `baseline_tau_s` with the *reward* baseline (`plasticity.py:355`), two unrelated
quantities on one constant, with nothing in the source stating it as a decision.

**So E085's "mechanism 2 is insufficient as built" narrows rather than withdraws:** the
representation was insufficient and now is not; the *readout* is. **Next, one experiment —
give `z_lag_bar` its own `pred_bar_tau_s`, default unchanged, and sweep it.** Decodability
should climb from 73.7% toward 90.0% as the corner moves below the signal, **while E070's
selectivity failure does not return** — that is the falsifier and it must be measured, not
assumed, since it is the reason centring exists.

**Falsifier:** with the scaffold validated (Stage 1) and the H2f-style learning rule
applied, if `L` (intact channel) does not out-perform `C?` (shuffled/yoked, matching
H4's own control design) on flock-wide sickness per rotation, the anchor was not enough
— either the durability claim itself is beyond this rule (consistent with, not
surprising given, E058/E059), or the scaffold needs a stronger or differently-shaped
anchor. Not a falsifier of H2f, which already stands on its own, independent evidence.

**Staging — four stages, not two, on review.** The original two-stage plan (build the
scaffold, then run the contrast) repeats a shape this project has been burned by
before: jumping from "the reflex fires in an isolated staged test" straight to "test
the hypothesis" skipped exactly the population-level checks that caught real defects in
H4 (E024's shuffled control, verified only when someone actually measured audibility)
and E025→E048 (a reflex validated in isolation, then found not to disperse the flock
until checked at the population level). T2's scaffold is unusually large — a world
mechanic, a call, a vision class and a reflex, all new at once — so skipping the
equivalent checks here would be repeating that mistake with more moving parts, not fewer.

- **Stage 1 — done** ([E060](experiments/E060-t2-contamination-scaffold.md)): built and
  validated each piece in isolation — staged ethogram probes, no learning, matching
  `run/probes.py`'s existing style. All four falsifier checks pass; a real
  contamination-staging bug and a real viz rendering bug were both found and fixed
  during validation.
- **Stage 1b — done** ([E061](experiments/E061-t2-population-scaffold-check.md)):
  population-level check, full 16-hen flock, still no learning. All three falsifiers
  survived: contamination is discovered at a workable rate (22.3 sickness onsets / 20
  min at the untuned `contamination_period_s=300s` default); the gakel call is
  genuinely audible (heard amplitude correlates 0.167 with a nearby sick flockmate,
  an 11× heard\|sick vs. heard\|not ratio — real signal, not assumed); the innate
  anchor produces real population-level dispersal (mean distance to a sick hen 5.44 m
  with the anchor present vs. 2.71 m stripped, the same shape of result E048 found for
  `CLS_CROWDING`). One caveat recorded, not treated as a failure: the sender-shuffle
  control (E024's original, not E026's fixed yoked one) retains 82% of the intact
  correlation, the same architectural shape CLAUDE.md documents for the alarm channel's
  98% — the flock clumps, so shuffling *which* nearby hen is credited mostly preserves
  "someone nearby is calling." Does not affect T2's claim, which only needs "something
  happened nearby" (`CLS_SICK` carries *where* separately), but is worth a future
  reader knowing about before mistaking it for a new defect.
- **Stage 1c — done** ([E062](experiments/E062-t2-contamination-period-calibration.md)):
  swept `contamination_period_s` over {100, 200, 300, 450, 600}s, 16 hens, no learning.
  Audience (mean distinct flockmates within `vision_range` of a sick hen) saturates
  flat at ~14/15 across the *entire* sweep — the strong gregariousness clumping
  (E025) already makes propagation nearly free at this arena size, so a longer period
  buys no additional reachable audience. Overlap fraction (a rotation firing while a
  hen is still visibly sick) stayed high, 43–61%, at *every* period including 600s —
  ten times `sickness_duration_s`. Prediction wrong, reason found: at ~5.6 discovery
  events per 300s rotation (E061's own rate), cumulative sick-time per rotation
  (~335s) exceeds the rotation itself — this is a direct, measured picture of the "C?
  pays roughly N times per rotation" baseline the backlog predicts, not a defect in
  the period. Neither check gives any reason to move off 300s, which stays the
  default — a checked "no change needed," not a skipped calibration. Per-feeder
  sickness attribution (which of the `n_food` feeders caused a given sick episode)
  doesn't exist yet, so the narrower stale-cue-misattribution risk this stage set out
  to check remains unmeasured; flagged for Stage 2 only if results there look
  consistent with it.
- **Stage 2 — done, null ([E065](experiments/E065-t2-stage2-learning-contrast.md)):**
  16 hens, H2f's rule, L vs. C? vs. a fixed baseline S, 8 seeds, 18 rotations. The
  falsifier fired: L's early-to-late sickness-per-rotation change (+0.875) was not
  smaller than C?'s (−0.250) — the primary contrast (+1.125 ± 0.715, t=1.57) was not
  significant, and the sign points the wrong way. `|W_out|` drift identical between C?
  and L (0.0632 both) — not proof the rule was inactive, but a real limit on what this
  coarse diagnostic can distinguish from a small, localised effect. S's own
  early-to-late change was nominally the largest of the three, consistent with a
  within-run trend unrelated to learning or channel content. The anchor being real and
  tested (Stage 1b) was necessary but not sufficient — this specific durable,
  place-based contingency is harder than H2f's own task, and the rule that worked
  there does not clear this bar.

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
| E008 | Top-down association built; first version was an autoencoder, tested nothing. |
| E009 | Lagged/pallial association, still null. Found the pallium saturated: H2d opened. |
| E010 | Gain 0.9 -> 0.70; H2 appears to collapse. **Later found confounded — invalid.** |
| E011 | Readout sweep. ~~Control did not improve as predicted; that tell exposed E010's confound.~~ **The E011 file has no result** — its §6–8 are `_Pending._` (found by E022). This row asserted a finding from an experiment that was never written up, and `readout_scale` is still 0.05, so whatever was run changed nothing. E010's confound is independently established by the E010 file itself. |
| E012 | Isolated it: not the gain, not the noise -- **`call_energy_cost` from E005 swamps H2's metric.** |
| E013 | Call cost moved to its own `vigour` budget. **Clean test: learning is significantly WORSE. H2 refuted at this timescale.** |
| E014 | Units bug found: a strike contributed -100 to reward. Connectome recovers; **behaviour does not**. Harm localised to the learned readout, implicating H2d. |
| E015 | Harm decomposed: readout +0.021, recurrent +0.010, both **+0.052 — superadditive**. Pruning nearly free. |
| E016 | Staging tested. Prediction falsified: pallium-first does nothing, **muscles-first cuts harm 69%**. Moving-target story withdrawn. |
| E017 | H2d's mechanism corrected: the stub separates cleanly (1.055), the loss is fan-in at sensory→pallium, not recurrence. Found the innate arc has **no auditory reflex at all**. |
| E018 | **ABORTED mid-run.** The channel it tests carries no information at n=16. Its pre-registered falsifier would have fired for the wrong reason. |
| E019 | External review, verified here. **Calls are inaudible** (a full alarm moves the receiver by 0.0000), **`W_out` is rank one** (0.7% variability), **reward is 98% call cost**. Withdraws H2's mechanism, caveats E013, demotes H2d. All three fixed. |
| E020 | H2 re-run after the fixes. **The harm is gone** (+0.062 → +0.001, t=0.08) and so is the erosion (48% → 2.5%). H2 returns to a clean null; E013, E015 and E016 superseded. ~~Exploration is now costly~~ (withdrawn by E021). |
| E021 | Both predictions wrong, both deleting claims. Learning does **not** repay its exploration cost (+0.021, wrong sign). The exploration cost **did not replicate** (t=3.84 → t=0.01). **SE was 4.4× larger on a fresh seed block** — no status may now change on one block. |
| E022 | Second outside review, verified. **The pallium has no inhibitory neurons.** The primary metric sits on a knife edge (hens start at hunger 0.30; equilibrium *is* 0.30). Its top-ranked item — food layout as 80% of variance — **did not replicate** (30%, not 80%). |
| E026 | **H4 SUPPORTED.** Intact channel −0.198 ± 0.059 vs deaf on P(caught\|blind), 24 seeds, two blocks; **yoked control flat**. Required a working control, an unmovable metric, and a warning interval — all four fixes were measurement errors. |
| E025 | **File finally written** (retrospective, from preserved commits). Food depletion does **not** disperse the flock (23.0% → 21.9% strike-radius overlap, noise); gregariousness's attraction-only wiring does, confirmed by ablation (21.9% → 6.8% with it removed). `food_deplete_rate` kept anyway on the assumption it "does not run out of food over a 20-minute run" — **shown false by [E037](E037-h2-rebaseline.md)** at the duration and flock size H2's own harness actually uses. |
| E024 | H4 ladder built and run with no plasticity. **The control failed**: the shuffled channel keeps 90% of the intact channel's information, because 38.8% of the flock shares each hawk. No result recorded against H4; T1 retired as its vehicle. |
| E086 | **The hippocampus was never in the circuit; putting it there fixes the representation and exposes the readout as the real blocker.** `regions.py` names HIPPOCAMPUS "place and spatial memory" and E063 was written up as giving it its first real function -- but `W_in` writes only into the sensory stub, so **64 of 64 place-receiving units were in the sensory stub and 0 in the hippocampus**, and `pred_src` excluded the region regardless. E086 routes place there and extends `pred_src`; off by default, guard at n_hens=16. **Parked decodability 84.6% -> 99.5%**, and the **distance profile decreases for the first time in this arc** (innermost bin 0.653 -> 1.604, lowest of seven to highest). Off arm reproduces E085 and E081 exactly. **Primary falsifier fires**: moving decodability 54.3% -> 58.9%, +4.6 +/- 3.3, t=+1.40, ns. Post-hoc diagnostic on identical data splits the cause: **dilution costs ~10 pts** (hippocampus alone 73.7% vs pooled 336 at 63.5%; pallium alone 54.4%, chance) and **the centring costs ~20** (`z_lag` uncentred 90.0%, raw rate 90.7%) while the lag itself costs nothing. `z_lag_bar` is a 20 s running mean against dwell times of 17-75 s -- a high-pass sitting on the signal's own timescale -- and it shares `baseline_tau_s` with the **reward** baseline, two unrelated quantities on one constant. Centring is not a mistake (E070: raw DC gave 1.0000 at P vs 0.9637 elsewhere), so this is a genuine tension. **Narrows E085**: the representation is now sufficient, the readout is not. Next: give `z_lag_bar` its own `pred_bar_tau_s` and sweep it, with E070's selectivity failure as the falsifier. |
| E085 | **T2's instrument repaired and measured -- and the route is now blocked on the representation, not the instrument.** Makes no behavioural claim. **Metric fix works**: choosing the target feeder per seed from an independent baseline run holds up (occ 0.600 selection -> 0.602 test, min 0.405, drift +0.8% and symmetric), and resolves **9.7% at n=4, 5.1% at n=8** against E084's 18.3% at n=4 for a fixed cell -- measured from a **null-null contrast** so the threshold is not derived from the treatment it will judge. Mean dwell per visit was tried and is far worse (MDE 74.3%): heavy-tailed, since a hen who never leaves yields one enormous visit. That prediction was recorded and wrong. **Decodability gate fires**: a discriminant fitted on live states and evaluated held-out scores **59.6%**, ratio 1.70, profile not decreasing. The headline is mostly class imbalance -- accuracy correlates with split skew at **r=+0.870**, and on the six balanced-split seeds it is **54.3%, +4.3 +/- 1.5, t=+2.95**: real, reliable, and about four points above chance, against E081's 84.6% parked. Even the live-fit discriminant fires *least* closest to the target (innermost bin 0.653, lowest of seven). E083's diagnosis replicates on fresh data (parked-fit 48.7%, ratio 0.60 vs 0.53). **First result in this arc about the model rather than our measurement of it**: `W_pred` is linear, the place cells work parked, and E063 gave them no innate anchors, so nothing lifts position into the variance a linear readout can reach. Next: innate place anchor, then a wider place population, then re-scope T2. Also records two engineering faults that both presented as exit 0 with an empty file (unguarded module driver; 3.9 GB/run OOM from emitting state every scan step). |
| E084 | **T2's occupancy metric cannot resolve the effect E082 and E083 predicted -- both falsifiers were guaranteed to fire before either ran.** Set out to fit the place discriminant on live states rather than parked ones; **crashed on seed 1 with 0 of 19 200 samples** within 3.33 m of the planted feeder (16 hens, 20 min) while the same connectome's other run gave 0.424. Diagnostics (post-hoc): **the flock aggregates** -- `approach_flockmates` works, so 16 hens behave as one clump (spread 1.66-7.21 m in a 20 m arena) that settles where it starts, and occupancy of a fixed cell runs **0.000 to 0.481 on world key alone**; effective n for a spatial metric is nearer one animal per seed than sixteen. **And the metric is under-resolved**: 8 seeds give baseline 0.4244 +/- 0.2751 (seeds spanning 0.0948-0.9651), paired-difference sd 0.0487, **minimum detectable effect at n=4 = 18.3% of baseline against a pre-registered 15%**. Observed at n=8: +6.4%, t=+1.57, ns and positive. Two seeds sit at ceiling/floor (CLAUDE.md check 5) on top of check 6. Second independent sufficient reason -- with E083's anti-selective plant -- why this arc produced nulls, neither about the hen. **Part A's question stays open**: one seed completed, at chance (44.4% live-fit vs 44.8% parked-fit, held-out). Next: pick the target cell per seed from an independent baseline run so every seed has room to fall, n=8 minimum. |
| E083 | **T2's gakel anchor redesigned to produce leaving -- and the plant found to be anti-selective, invalidating this arc's behavioural readouts.** `_add_gakel_scaffold` now suppresses `M_PECK` only, never `M_FORWARD`. The redesign works: the freeze is gone (fwd flat 0.633 -> 0.626 vs E082's 17% fall) and pecking at the planted feeder falls monotonically (0.550 -> 0.504). **Occupancy at P still did not fall** -- it rose 9%, non-monotonically, so the primary falsifier fired. Then the diagnostic that matters: `pred@gakel` averaged 0.90 while the hen is at P only ~42% of the time, and splitting by position gives **0.656 at P vs 1.244 elsewhere** (ratio 0.53), live magnitude varying **9-fold across seeds** against a pre-flight of exactly 1.000 on all four. A distance profile rules out disc width: innermost bin (0-1 m) is the **lowest of seven** at 0.655, peak in a 5-7 m ring at 2.128. **The plant is inverted** -- E082 and E083 both drove the channel hardest where she was not meant to be avoiding, which explains both results in the right direction. Cause: the discriminant is fitted on a **parked** hen and read back on a **moving** one -- the third instance of E071's error shape, and the amplitude-only pre-flight added after E082 cannot catch it (it never checks *elsewhere*). Withdraws E082's "she slows and stays" diagnosis (its "the chain conducts" stands) and re-scopes E081's 84.6% to parked states. Next: fit the discriminant on live trajectory states; make the pre-flight assert selectivity, not amplitude. |
| E082 | **T2's whole-chain control redone with a discriminative plant: the chain conducts, and fails on one wrong reflex.** E070's plant never fired; this one does -- pre-flight **1.000 at the planted place per seed**, 0.86-0.96 live -- and **forward drive falls 17%** (0.622 -> 0.519) as `pred_gain` rises, so every link works end to end. **But occupancy at the planted feeder does not fall** (0.4501 -> 0.4339, non-monotonic, ~3% of a 45% baseline). Diagnosed: `_add_gakel_scaffold` suppresses `M_FORWARD` and `actuation.py` derives speed from it, so a hen already at the bad feeder **slows and stays**. The anchor produces lingering where avoidance requires leaving -- and its own docstring explicitly declines to borrow the anti-predator response while the implementation is a functional freeze anyway. Hunger flat (0.427 -> 0.437) and control-feeder occupancy flat, so neither the hallucination nor the smearing falsifier fired; **selectivity was the flagged risk and did not bind**. **The first run was invalid**: planted against raw `z_lag` while the runtime reads a converged `z_lag - z_lag_bar`, giving `pred@gakel` 0.04 instead of 1.0 -- the identical timescale error E071 documented, repeated two experiments later. Next: redesign mechanism 1 to produce *leaving* (suppress `M_PECK` only), and no L vs. C? until it passes. |
| E081 | **H2d measures distance, not decodability -- and that unblocks T2.** `pallial_sep` has been RMS distance since E009. But `W_pred`/`W_out` are linear readouts, and what they need is linear *decodability*; two distributions can be highly correlated, tiny in RMS distance, and perfectly separable by a hyperplane. Measured: hawk-vs-call pallial states are **0.9928 correlated**, `pallial_sep` 0.1113, and **98.8% held-out linearly decodable**. **The title claim of H2d is false as written.** Also re-examined E070's blocker: it planted a **matched filter** (copy P's pattern), which scores **18.8%** -- below chance -- where a **discriminant** on the same states scores **84.6%**. E070's 'the chain does not compose' was about the readout I planted, not the network, so **T2-revised is unblocked**. Relocates the problem from representation to **rule type**: correlational rules (Hebbian/covariance, `W_out` under `hebbian_readout`) converge toward matched filters; delta rules (`W_pred`) toward discriminants. Caveats kept prominent -- these are supervised discriminants using labels the network lacks, and nothing here shows any rule actually learns them, only that the information is available. Fifth instrument-not-bird finding this session; second where the instrument was mine. **Re-scoped by E083, not withdrawn:** every figure here was measured on hens **parked** at five cell centres under 0.35 m of jitter, holding one static observation for 200 steps, reading raw `rate(x)`. Correct for what it measured. What is withdrawn is the inference E082 and E083 drew from it -- that a readout separating five parked point-locations separates *where the hen is* during free movement. E083 is the first evidence on that and points the other way. |
| E080 | **The dilution mechanism, standing since E017, explains 1-2% of H2d.** Progressively zeroed the channels identical between hawk and call -- by construction uninformative -- across 12 paired genomes. Max effect **1.21x** (only significant point 1.20x at half removal, t=2.34) against a **14.5-17x loss**. Non-monotonic, and the pre-registered confound explains why: drive falls 0.4602 -> 0.2724 as channels go, and E079 showed lower drive hurts below the optimum, so the curve is dilution-gain minus drive-penalty. Correcting for that headwind might reach 1.3x, not 14.5x. **E017's localisation stands (loss is at sensory->pallium, replicated E034); its mechanism does not.** Three interventions were built on that explanation -- modality segregation, density, balanced E/I -- all null or reversed, testing a mechanism one cheap graded measurement could have shown was too small to matter. **H2d's magnitude now has no identified mechanism**: dilution too small, saturation wrong-signed (E079), recurrent mixing wrong-signed (E017/E034), common-mode DC null (E077). Remaining lead named precisely, with its counter-evidence: pallial recurrence *strength* (distinct from global gain and from removing recurrence), against E070's finding that place similarity is identical at 1-300 settle steps. |
| E079 | **The gain default was also set on the sparse probe -- but it turns out already optimal, and the saturation framing is withdrawn.** Both probes, 12 genomes, paired. Naturalistic separability **peaks exactly at the shipped 0.95**, declining significantly in *both* directions (0.80 -> 0.80x t=2.52; 1.10 -> **0.27x** t=6.54, far sharper than sparse's null 0.84x). Second default validated rather than moved this session. **The live-rate row matters more**: gain 0.40 pulls live pallial rate 0.6861 -> **0.1796**, comprehensively out of saturation, and separability gets **worse** (0.35x). That is the second independent demonstration -- `balanced_ei` cut live rate 0.73 -> 0.12 for a null. **Reducing drive does not improve separability; raising it hurts.** So E009's saturation diagnosis, and my own E073 write-up that leaned on it, mis-identify the constraint: saturation correlates with the best regime rather than causing the low value. **H2d now has no remaining lead.** New hypothesis: it is a property of random projection without learned feature extraction, not a fixable defect -- with the circularity that the pallium cannot learn separating features because separation is the precondition for having anything to learn from. Also fixes `docs/hypothesis.md`'s claim that `connectome.py:48` has `gain = 0.70`: it is **0.95 at line 81**, a stale correction of a stale doc. |
| E078 | **E041's density result reverses under naturalistic input -- and the shipped default turns out optimal.** Both probes, 12 genomes, paired, on E076's corrected baseline. **Reproduces E041 exactly on its own probe** (density 1.00 = 2.12x vs E041's ~2x) -- a transfer failure, not a replication failure. **Naturalistically it reverses**: 0.90x at density 0.60 (t=2.70), **0.71x at 1.00 (t=5.37)**. Mechanism is saturation, visible in mean rate: sparse input barely moves drive with density (0.2724 -> 0.3040), naturalistic goes **0.4602 -> 0.6745** into the compressive region -- and live sits at 0.6907, so E041's recommended direction moves *toward* the worst regime. Separability peaks at the **shipped 0.30 default**, contradicting E041's 'no optimum found up to full connectivity'. My pre-registered prediction was **wrong on direction** -- I said denser could not be actively harmful. **Falsifier fires: H2d now has no intervention with a positive effect naturalistically** (E/I null, segregation null, `balanced_ei` null, density reversed). Only live lead left is saturation itself -- and E023 set the `gain` default from a sparse-probe sweep, never repeated naturalistically. |
| E077 | **Re-reading `balanced_ei` against E076's corrected baseline: the 2.13x was my own artefact.** E073 measured `balanced_ei` at 2.13x (t=5.75) under a naturalistic probe -- but that probe fed `sensing.observe` while E063's place block sat at 25.1% of observation drive, always on, which is precisely the common-mode term `balanced_ei` acts on. With the block off: **1.05x, t=0.35, null**. Naturalistic baseline separability also goes 0.0365 -> 0.0814, so E073's 'H2d severity understated 2.6x' becomes 1.2x. **Two of E073's four claims fall; two stand** -- the E009 probe genuinely under-drives vs live (0.6907 vs 0.2724) and E009's saturation genuinely was never fixed. **`balanced_ei` is closed as an H2d intervention** and E074's adoption gate is moot. Third distinct harm now traced to one un-opted-in addition: E063's block broke H2f's control, manufactured this false positive across two follow-up experiments, and inflated the probe-vs-live gap. E073 corrected in place. |
| E076 | **Bisect closed: both causes were my own additions, and both defaults flipped.** Fourth arm -- contamination AND place cells disabled -- returns H2f's food control to **null (-0.0293, t=0.98)** and reproduces E057 to within noise (general +0.132 vs +0.123, audience-specific +0.241 vs +0.232). **H2f was never damaged**: the audience effect holds t=31.7-46.9 across all four arms and is *largest* in the clean one; only the control that made it interpretable as targeted rather than indiscriminate had broken. Causes: **E060 put contamination into `DEFAULT_COOP`** (32 sickness onsets per 30-min run, gakel calls, `CLS_SICK`, 0.15x mobility, for every experiment since) and **E063 added 25 always-on channels worth 25.1% of observation drive** -- neither with an opt-in, against a convention stated explicitly in six other places. Both defaults now `False`; T2's experiments opt in. Also checked, since E073 rested on it: live pallial rate without place cells is **0.6907** vs 0.7288 -- still deeply saturated, so E073's conclusion survives. Flipping surfaced a second lesson: seven tests failed (correctly, all about the disabled machinery) but several others would have passed **vacuously**, comparing two all-zero vectors -- those were opted in too. |
| E075 | **Bisecting H2f's broken control: my own `m_acc` fix exonerated, and two of my own scaffold additions found changing the world for every experiment.** Three arms, 8 seeds, 30 min, paired. **E067's `m_acc` ruled out** -- reverting moves the control by 0.004 (coherent: H2f's reward is `d_drive`-dominated where snapshot == window mean, and E067's defect hit only discrete events, which barely occur at the 900s hawk default). **E060's contamination is ~half** -- disabling takes the food control t=10.04 -> t=2.70, halved but still over threshold, so the bisect is **partial**. Residual lead, measured not tested: **E063's place block is 25.1% of all observation drive, always on**. The real finding is the pattern -- **E060 put contamination into `DEFAULT_COOP` and E063 added a quarter of observation drive, both with no opt-in**, against a convention every other addition here follows (`legacy_audio`, `auditory_scaffold`, `pred_enabled`, `readout_scaling_strength`, `gakel_scaffold`, `balanced_ei` are all off by default with comments saying why). Four experiments were built on top, including E074's own gate, whose reference arm was contaminated by exactly this. Adds `contamination_enabled` (default `True`, so the flag alone changes nothing). H2f's audience effect is robust across all arms (t=31.7-46.9); only its control is at issue. |
| E074 | **`balanced_ei` adoption gate: two clear, the third passes its falsifier and surfaces a worse problem.** Gate 1, innate behaviour: **13/13 ethogram** on a balanced connectome, near-identical numbers (the reflex arc is separate from `W`, now verified not assumed; adds `run.probes --balanced-ei` so the gate is re-runnable). Gate 2, throughput: **38.6x -> 42.4x**, above the 5x guard, marginally faster. Gate 3, H2f's audience effect: significant in **both** arms (t=42.46 -> t=13.28) so balancing does not cost the project its only positive learning result -- **but the food control fires in the BASELINE arm at +0.1054, t=10.04**, 47% the size of the effect it controls for, where E057 reported it null. Balancing shrinks everything ~100x (audience 0.2242 -> 0.0023, following live pallial rate 0.7288 -> 0.1209) and the control goes null; signal-to-control ratio 2.1x -> 11.5x favours 'cleanup' over 'proportional shrinkage', but the two are not separated here, and 0.0023 against E057's 0.232 may be behaviourally meaningless. **`balanced_ei` stays unadopted** -- its own gate's reference arm is contaminated. Next: bisect H2f's control (revert `m_acc`, then pre-E060 layout) before revisiting. |
| E073 | **H2d's probe under-drives the pallium 2.7x, and E009's saturation was never actually fixed.** Same `balanced_ei` contrast, same 12 genomes, paired, under two probes: **null under sparse injection** (0.90x, t=0.74) and **significant 2.13x under naturalistic input** (t=5.75). Mean pallial rate decides which to believe -- sparse 0.2724, naturalistic 0.6019, **live rollout 0.7288** (16 hens, 5 min, 3 seeds), so the settle-and-separate probe sits ~2.7x below the regime a hen actually inhabits. Consequence nobody had drawn: E009 diagnosed saturation, E023's gain sweep reported 0.189 and the concern left the record as handled -- but that was the sparse probe; **live, the network is still saturated at 0.7288**. `balanced_ei` brings it to 0.1209. Naturalistic baseline separability 0.0365 vs sparse 0.0961, so H2d's severity was understated 2.6x. The series stays internally valid but characterises a low-drive regime and is blind to common-mode interventions; E041's density result should be re-measured naturalistically. 2.13x against a 14.5-17x loss is a dent, not a fix. |
| E072 | **Balanced E/I does not fix H2d on the classic metric -- but the two probes disagree.** Every recurrent projection carries a positive DC by construction (80% excitatory, both signs drawing magnitude from the same `\|normal\|`); the sensory->pallium block nets **+0.9339 per pallial unit**, the exact quantity E017/E034 localised the 14.5-17x loss to, and the exact defect **E027 already fixed for `W_out`** (which nets -0.000000). Balancing it, with `sum\|W\|` held constant so only the E/I ratio changes: **0.90x, t=0.74, null** on a paired 12-genome test. The gain-matched control failed as a control -- matching mean rate needed `gain=0.05`, collapsing separability to 0.0033 -- so its 26.28x is unquotable. Kept because the same intervention moved place-to-place pallial correlation 0.9807 -> 0.7520, a discrepancy E073 then resolved. `balanced_ei` off by default. |
| E071 | **Centring the prediction pathway: a real fix for a real defect, and not sufficient.** `W_pred` never received E019's centring, in either its learning or its readout. Added `pred_centred` (off by default). Uncentred, a planted place->gakel association predicts flat across places (ratio 1.042); centred, mean ratio 0.180 -- **but that figure is cancellation, not selectivity**, and what reaches behaviour is `relu(predicted)`: 0, 1.3027, 0.3921, so a distractor place drives *stronger* withdrawal than the planted one and the effective ratio is 0.565. Corrects E070's "projection fine, readout broken" claim in place: both are real, and the residual is pallial separability. Established H2d as the evidence-backed critical path blocking T2-revised, H2c and H3 at once, with a concrete target -- place-to-place pallial correlation must come down from 0.94-0.96. |
| E070 | **T2-revised's whole-chain positive control, run BEFORE any contrast -- and it failed in five minutes.** A hand-planted place->gakel association predicts **1.0000 at the planted place and 0.9637 at a different place entirely**. Traced: place block perfectly orthogonal in the observation (0.0000), 0.94 by the sensory stub, **0.9995 in the pallium**. Ruled out by measurement, not argument: not a settling artefact (identical at 1-300 settle steps), not dilution by competing channels (a dedicated place-only stub slice at 10/21/32 units leaves 0.9997). Information is present -- across-place variation has real structure (singular values 0.294/0.117/0.097) -- but is 3.7% of the DC baseline. Two defects found in the experiment's own design: `PlasticConfig(enabled=False)` silently zeroes `z_lag` so `pred_gain` 0.0 and 2.0 gave bit-identical output, and occupancy at P was 0.0000 (metric with no room to move). T2-revised paused, both mechanisms correct and kept. |
| E069 | **The positive control T2 needed since E065: the metric is sound, the rule cannot learn this at any signal strength.** Two questions asked apart. **A (no compute, re-analysis of E065/E066/E068 caches)**: minimum detectable effect is 1.69-2.95 events/rotation, **19-35% of baseline** at n=8 -- pre-registered prediction (~2.1, ~25%) confirmed mid-range. An empirical injection check returns 3.11 vs analytic 2.05 for E068, which agree exactly once you account for E068's observed +1.06 offset. The metric was never the limiting factor; T2's own predicted effect is far larger. **B (`sickness_penalty` sweep {0,1,10,100,1000}, 4 seeds, 12 rotations)**: no learned avoidance at any magnitude -- `late-early` = +1.00, +0.08, +0.08, +1.25, -0.33, no trend, the lone negative ~1/6 of MDE. Mean `\|W-W0\|` **does** rise 26% at penalty 1000, so the signal reaches the weights (E067's fix is live) but yields undirected perturbation, not behaviour. Connectome retains **97.5%** of innate synapses vs 98.0% at zero -- **E014's erosion risk retired, `sickness_penalty` is safe to raise and simply useless**. Confirms empirically what E058/E059 (the rule amplifies anchors, never builds them) + E063 (place cells deliberately given no reflex) jointly predict. **Calibration cannot rescue T2.** Methodological note: this control was deferred through three experiments, each producing a null later explained by a newly found defect; running it first would have settled the matter in one 30-minute sweep. |
| E068 | **T2 Stage 2 re-run under E067's fixed eligibility mechanism -- still null, and the reason is finally identified: the sickness reward is ~4 orders of magnitude too small to matter.** Same design as E066, only the corrected mechanism differs. Primary contrast +1.06+/-0.87 t=1.23 (E066 was -0.19; E065 +1.13 -- three runs of the same comparison straddling zero at ~1 SE, which is what a true zero looks like). S bit-identical to E066, confirming the fix touches only conditions that consolidate. **Two follow-up diagnostics, run because the pre-registered `\|W_out\|` check structurally cannot answer whether the fix took effect** (under `hebbian_readout`, `W_out` is not reward-gated -- that diagnostic has been reading the wrong pathway in E065/E066/E068 alike): measuring `W` directly, `sickness_penalty` 0.0 vs 1.0 changes drift by 0.07%, inside seed noise. Why: sickness fires in **0.0417%** of a hen's consolidation windows and supplies 0.16x ambient reward even in those -- total share ~0.007%. E067's 50x fix turned 0.0001% into 0.007%. **All three Stage 2 runs withdrawn as tests of T2's actual claim** -- accurate measurements of the code, not evidence about the bird. Needs reward calibration + the positive control E065 flagged and never ran, before a fourth run means anything. |
| E067 | **Red-team diagnostic: the reward-modulation factor is sampled, not traced, at consolidation time -- E066's "genuine reward signal" claim overstated.** Adversarial review (no context on this conversation) commissioned after T2's second null; every finding independently re-verified with fresh scripts before adoption, per this project's own red-team rule. `m` (`run/simulate.py`'s `reward - ps.baseline`) is recomputed every step but only the value at the exact consolidation boundary (every `interval=50` steps) reaches `consolidate()`. A discrete single-step event -- `sickness_penalty` or `strike_penalty`, both the same shape -- reaches the boundary on only **2% of occurrences** (swept exhaustively, confirmed independently, not just read off the review). For T2: a third, independent, structural reason the rule couldn't have succeeded, alongside `hebbian_readout`'s `W_out` never reading sensory input and place cells having no pre-existing motor correlate. `strike_penalty` (H2/H4/T1's history) shares the identical defect -- named, explicitly **not acted on**: no specific prior conclusion's actual dependence on it has been checked, and this project's own discipline forbids rewriting the tree on an unverified reinterpretation. Also independently verified two narrower E066 corrections: the witnessed/testimony-only split under-gates "witnessed" (no field-of-view check, unlike the real innate reflex) and S's early-to-late trend is substantially a single-rotation cold-start artifact (excluding rotation 0 roughly halves it, 1.719->0.833) -- neither changes the primary L-vs-C? null. Nothing fixed; not pre-registered, a diagnostic. |
| E066 | **T2 Stage 2, corrected: E065's null re-run with a real reward signal, plus a witnessed-vs-testimony-only split -- still null, more solidly.** `reward()` had no term for sickness at all (checked directly) -- E065's null likely wasn't a fair test. Added `sickness_penalty` (off by default), re-ran identically otherwise. Primary contrast sign flipped to the predicted direction but stayed tiny: -0.19+/-1.25, t=0.15 (threshold 2.365). Also split onsets by whether another already-sick hen was visible at that moment (witnessed, explainable by the innate anchor alone, identical across conditions) vs not (testimony-only, the only case audio could help). Testimony-only was ~1/3 the volume of witnessed (confirms a design-review dilution concern was real) but showed no effect either (+0.06+/-0.19, t=0.32, wrong sign) -- rules out "the aggregate washed out a real effect." `\|W_out\|` drift still identical C?/L. T2 stays `NOT SUPPORTED`, now on solid methodological ground -- both live objections to E065 checked directly, conclusion held. |
| E065 | **T2 Stage 2: the actual L vs C? learning contrast -- run, a clean null.** 16 hens, H2f's validated rule (E057), L (intact) vs C? (yoked) vs S (fixed), 8 seeds, 90min/18 rotations. Primary: does L's sickness-per-rotation fall further early(rot 1-4)->late(rot 15-18) than C?'s. It does not, and the sign is wrong: L +0.875 (worse), C? -0.250 (better), primary contrast +1.125+/-0.715 t=1.57, not significant (threshold 2.365). S's own early-late change (+1.719) was nominally largest of the three -- consistent with a within-run trend common to every condition, not learning-specific. `\|W_out\|` drift identical C?/L (0.0632 both) -- not proof the rule was inactive (E057: content changes *which* weights move, not the mean), a real diagnostic limit, not dismissed. Matched water-intake control: clean null as expected given the primary was null. T2 -> `NOT SUPPORTED`, first time actually tested against data. Not the end of T2: local-vs-aggregate weight change, a longer run, or a different rule all remain open, unpursued. |
| E064 | **T2 Stage 2 prerequisite #2: a gakel-call location cue for listeners beyond visual range.** E063's place cells only solve durable location memory for a hen who directly *witnesses* a sickness event; a hen who only *hears* the gakel call still gets nothing spatial, since audio here has never carried direction and self-location doesn't reveal a caller's. Added a loudness-weighted mixture of gakel callers' own place-cell patterns (reusing E063's grid) rather than asking the pallium to learn trigonometry from a bearing it isn't given -- `OBS_DIM` 113 -> 138. Required a new `World.pos_log` ring buffer (matching `call_log`) so `channel_mode='yoked'` hands a listener the caller's position *when she called*, not her current one -- verified directly with a caller who moved 22+m between calling and observation, the same class of leak E024's shuffled control had. 81/81 full suite (77 prior + 4 new); one test-bug caught and fixed (positions initially beyond `hear_range`, correctly zeroing the very thing the test meant to check). T2 stays `NOT STARTED` -- scaffolding, not a result. |
| E063 | **T2 Stage 2 prerequisite: an innate allocentric place-cell channel.** T2's literal claim (durable avoidance of a specific feeder, outlasting the visible cue, recognised from any approach direction) turned out unrepresentable: every existing channel is egocentric by design, so a hen loses any trace of a location the moment she turns. Added a fixed 5x5 Gaussian place-cell grid (`place_sigma=2.0`) giving `hen/regions.py`'s previously-generic `hippocampus` region its first real function -- `OBS_DIM` 88 -> 113, the biggest single jump yet. No changes needed to `connectome.py` (existing exteroceptive routing) or `innate.py` (deliberately no reflex reads it -- raw location carries no innate meaning alone). 77/77 full suite (74 prior + 3 new: peak-at-centre, heading-independence, location-discrimination), full ethogram unchanged. T2 stays `NOT STARTED` -- this is scaffolding, not a result. |
| E062 | **T2 Stage 1c: calibrate `contamination_period_s` — swept, found no reason to change it.** 16 hens, no learning, {100,200,300,450,600}s x 3 seeds x 40min. Audience saturates flat ~14/15 at every period (gregariousness already clumps the flock tightly, E025) — a longer period buys no more reachable audience. Overlap (rotation firing while a hen is still sick) stayed 43-61% at *every* period including 600s, prediction wrong: at ~5.6 discovery events/rotation (E061), cumulative sick-time (~335s) exceeds the 300s rotation itself — a direct measurement of the "C? pays ~N times" no-learning baseline, not a period defect. `contamination_period_s` stays 300.0. Per-feeder sickness attribution doesn't exist yet, so the narrower stale-cue-misattribution risk remains unmeasured; flagged for Stage 2 only if warranted. |
| E061 | **T2 Stage 1b: the scaffold works at the population level, not just in isolation, no learning involved.** 16-hen free-running flock, 3 checks × 3 seeds, 20 min each. **Discovery**: 22.3 sickness onsets / 20 min at the untuned `contamination_period_s=300s` default — not a bottleneck. **Audibility**: heard gakel amplitude correlates 0.167 with a nearby sick flockmate, 11× heard\|sick vs. heard\|not ratio — real signal. **Dispersal**: mean distance to a sick hen 5.44m with the anchor present vs. 2.71m stripped, matching E048's `CLS_CROWDING` result. One caveat, not a failure: the sender-shuffle control (E024's original, not E026's fixed yoked one) retains 82% of intact's correlation, the same architectural shape E024 found for the alarm channel (98%) — the flock clumps, so shuffling *which* nearby hen is credited mostly preserves "someone nearby is calling." Doesn't affect T2's claim, which only needs "something happened" (`CLS_SICK` carries *where* separately). T2 stays `NOT STARTED` — Stage 1b tests the instrument too, not the hypothesis. |
| E060 | **T2 Stage 1: contamination/sickness/gakel scaffold built and validated, no learning involved.** New `CLS_SICK` vision class + `IDX_SICKNESS_ONSET` (`OBS_DIM` 74 -> 88) and `M_CALL_GAKEL` (`MOTOR_DIM` 11 -> 12, first motor-dim change). All four falsifier checks pass (12/12 ethogram, 70/70 suite): sickness onset is a rising edge (contaminated -> sick=True, clean -> sick=False), sickness mechanically slows movement (0.14m vs 0.93m travelled), the gakel call is a discovery pulse (not continuous, matching E053's fix), and the innate anchor reverses gregariousness's attraction to a sick flockmate (turn_R 0.97 vs turn_L 0.79 sick, vs turn_L 0.79 vs turn_R 0.50 healthy). **Found and fixed two real bugs during validation**: contamination was being unconditionally recomputed every step, silently overriding any staged value (caught by the probe's own negative control); `viz/web/app.js` hardcoded a 4-call stride that would have silently misaligned rendering once N_CALLS grew to 5. T2 stays `NOT STARTED` — Stage 1 tests the instrument, not the hypothesis. |
| E059 | **Exposure escalation closes E058's open question, mechanistically.** Doubled predator exposure (`hawk_period_s` 20 -> 10, matching E042->E043) reproduced E058's numbers almost exactly — no movement. Checked directly: mean `\|W_out\|` drift **0.054 at both exposure levels** (3 seeds) — `readout_scaling_strength` reaches a dynamic equilibrium independent of exposure, unlike `W_pred`'s hard clip (which E043 found *does* respond to exposure). H2c stays `NOT STARTED`, now for a checked mechanistic reason rather than "maybe more data would help." |
| E058 | **H2f's validated rule tried on H2c, and it fails cleanly — general excitability, not comprehension.** Same `hebbian_readout`+`readout_scaling_strength` config, crouch-on-hearing-a-call, no scaffold (build from nothing, unlike H2f's wired-anchor task), `pred_gain=0.0`. Crouch nominally significant (t=2.58) but so are three unrelated control channels — peck (t=3.58), scratch (t=3.29), flee (t=2.74) — at matched tiny magnitudes (~0.004, two orders below H2f's effect). The mandatory diagnostic (pre-registered before running) catches this cleanly: uniform elevation across every channel tested, not a targeted crouch association. H2c stays `NOT STARTED`. Narrows H2f's mechanism to *amplifying an existing anchor*, not building one from nothing. |
| E057 | **H2f's falsifier clears — replicated. First genuine positive result in this project's learning-rule history.** Separated E056's mixed effect via difference-in-differences against the `S` baseline's own alone/audience gap, full 8-seed sample: general elevation **+0.123, t=8.81**, audience-specific **+0.232, t=45.59** (significantly larger than general, t=10.39). Food-channel control (no mechanistic audience route) null on both components, ruling out indiscriminate dysregulation. **Replicated on a fresh 8-seed block** (general +0.121 t=11.37, audience-specific +0.232 t=21.90, food null) — both blocks agree to three decimal places. **H2f -> `SUPPORTED`, narrower than a clean audience-only ideal**: a real, context-specific (not indiscriminate) general component rides alongside the larger targeted one. Opens a fresh, direct pass at H2c under the same scrutiny. |
| E056 | **Bounded Hebbian readout: more targeted than E055, still doesn't clear the falsifier as specified.** `readout_scaling_strength=0.3` fixed E055's runaway (cortical/reflex ratio 2-2.7x -> 0.75-0.9x, hunger 0.728 -> 0.546). Audience effect **+0.2324 ± 0.0051, t=45.59** — but the mandatory diagnostic found `alarm_alone` rose 30-40% from baseline (should stay flat per the pre-registered check) alongside a larger, genuinely disproportionate rise in `alarm_audience`. One of three sanity checks fails, applied literally. H2f stays `UNDER TEST`: not confirmed (not clean), not refuted (not simply broken like E055). Next step identified: separate the audience-specific component from the general-elevation one directly, not another constant sweep. |
| E055 | **First attempt at H2f's own falsifier — broke before it could be tested.** A non-reward-gated ("Hebbian") `W_out` update produced a "significant" audience effect (+0.096, t=2.63) that the mandatory diagnostic (this project's own discipline: a surprising positive gets checked before trusted) found was an artifact: cortical drive 2.0-2.7x reflex magnitude (the documented "behaviour gets worse" regime from `eta_out`'s own docstring), hunger nearly doubled (0.39->0.73), every calling channel elevated regardless of audience. Cause: `W_out` has no synaptic-scaling correction, unlike `W` — the reward-modulated rule's zero-mean property had been incidentally bounding growth, and removing the reward gate removed that too. Fixed in E056. |
| E054 | **Food-call saturation was not crowding out pallium capacity for the alarm channel — a clean null, not a repeat of an old one.** Same instrument as E042 (comprehension after rearing), density held at E041's fix throughout, only `legacy_food_call` varied: discovery pulse (E053) vs. recreated pre-fix continuous calling. **−0.0005 ± 0.0007, t=0.70**, not significant, wrong sign if anything. First test of a *competing-channel capacity* account rather than the alarm channel's own representation (unlike E042–E044); it fails the same way. H2c stays `NOT STARTED`; strengthens the case for H2f (rule-kind) over any remaining representational precondition. Single 8-seed block, not independently replicated — a null consistent with four prior experiments' worth of the same pattern, judged not to need the ~490s a second block would cost. |
| E053 | **Food call fixed to fire on discovery, not continuous sight — closes a long-standing backlog item.** Added `IDX_FOOD_ARRIVAL`: a rising-edge pulse on newly arriving at a food patch (same idiom as `strike_event`), decaying over 4s regardless of dwell time, replacing raw `CLS_FOOD` sight-gating on `M_CALL_FOOD`. `OBS_DIM` 73 → 74. Flock-wide food-calling fraction dropped **42.8% → 4.2%**, no hen left above 50% of steps calling (was 4/16). New probe and unit test both pass; full ethogram (8/8) and test suite (62/62) unaffected. Production stays innate and audience-blind, matching the project's existing design — only the temporal trigger changed. Sets up [E054](experiments/E054-food-call-saturation-and-pallium-capacity.md): does removing this saturation change anything for the rarer alarm channel's representation (H2c/H2d)? |
| E051 | **Wall avoidance: `IDX_WALL` was sensed but never wired to a reflex — nothing turned a hen away from a boundary once some other drive carried her there.** Found via the offline-replay viewer (`viz/`, PR #16): a hen visibly stuck at the edge. Added `IDX_WALL_ESCAPE_L/R`, two directional channels derived from the existing wall-proximity signal and the nearest wall's outward normal, wired to turn the hen away (weight 3.0, no forward suppression — the kinematics let turning happen independent of forward drive, avoiding a deadlock an undirected suppression would cause). `OBS_DIM` 71 → 73. Ablation (3 seeds, 10-min, matching a typical `run.record` session): mean hen-steps near a wall 0.50% → 0.11%, longest single continuous dwell **22.6s → 2.3s** — a ~10× cut in worst-case pinning, the number that determines visibility in a recording. Isolated single-hen check confirms the mechanism directly (180°→0° turn, back out to 0.43 m in ~3.6s). Exposed (not caused) a pre-existing marginal-seed test fragility, fixed by trying a short seed list instead of one. |
| E050 | **E048's 3-seed strike-radius improvement does not replicate — H4's shuffled control is still not viable, unmoved by the personal-space fix.** Re-ran E024's original instrument (`scratchpad/shuffle_info.py`) unmodified, at 8 seeds instead of 3, on two independent blocks. Hawk-targeted clustering ("% of flock in strike radius when a hawk is live") replicates exactly across both blocks at **38.4%**, matching E024's original pre-E023, pre-fix baseline of **38.8%** almost exactly — the fix has not moved this number. Shuffled-channel retained correlation 89–100% across the two blocks, same range as E024/E026's original 90–98%. **E048 §6/§7 corrected in place** with a forward pointer; does not change H4's status, which already rests on the yoked control, not the shuffled one. Flagged as the second experiment in a row written up after the run rather than before — a process discipline slipping under time pressure, worth tightening. |
| E048 | **The E025 flock-clumping fix, built and measured: a personal-space vision channel (`CLS_CROWDING`).** Zero until a flockmate is well inside personal-space range, ramping to 1 at contact; wired to turn the hen *away*, at a weight (4.0) that mathematically must and does exceed `CLS_FLOCKMATE`'s attraction weight (1.2) for repulsion to win rather than merely damp. `OBS_DIM` 59 → 71. ~~Diagnostic (3 seeds, 6-min, matching E025's own methodology, run on the current E023-corrected connectome so only compares against itself): nearest-neighbour distance 0.14 → 0.38 m, strike-radius overlap 26.8% → 21.8%, with cold and fed/hen essentially unchanged — real, measured dispersal, well short of stripping gregariousness entirely (nn 1.70, but cold and fed/hen both distorted there).~~ **The strike-radius number above is a 3-seed artefact — see [E050](experiments/E050-shuffle-info-recheck.md), which found no change at 8 seeds (replicated).** Cold and fed/hen stand as reported; the nn-distance figure was not independently re-checked. A `test_reward_is_not_dominated_by_one_component` regression during implementation (repulsion at the exact huddle-radius boundary was suppressing huddling, shifting reward composition toward hunger) led to tightening the activation threshold from the huddle radius itself to well inside it — caught by the test suite, not the diagnostic. |
| E047 | **H3's original design (E005/E006) re-run on the fully corrected system — closes the last "was it measurement" explanation.** Without a scaffold: comprehension exactly zero (H2b's diagnosis confirmed as architectural, not an E019-era artefact), both `alarm_effect` and `food_effect` null across every condition, 8 seeds, `food_deplete_rate=0`. **E005's promising `food_effect` trend (+0.032, t=0.64) does not replicate.** Combined with E036/E040 (scaffold supplies comprehension, audience-sensitivity still doesn't emerge on top), H3 has now failed both ways this project could imagine it working. "Blocked by H2b" retired as the operative explanation — same underlying limitation as H2c. |
| E046 | **T1's vigilance prediction falsified — flock-size sweep, n_hens {4,8,16,32}.** `head_down` *falls* with flock size for both L and C? (slopes −0.0017 vs −0.0025/hen), no channel-specific effect — likely a chorus effect (power-summed audio, louder with more hens, triggers call-suppression regardless of content) rather than a vigilance strategy. Safety advantage itself is flock-size-dependent, not uniform: null at n=4,8 (t=0.03, 0.32), significant at n=32 (t=4.02) — consistent with a passive, statistical many-eyes mechanism (more independent chances of a true warning) rather than individual behavioural change. Both of T1's original predictions now settled. |
| E045 | **T1 finally given the Pareto treatment `docs/backlog.md` always specified, instead of a fourth single-point mean comparison.** 5-capacity sweep (`pallium_scale` 0.5×–4.0×), `food_deplete_rate=0`. Intake: null at every capacity (max t=0.56) — nine total checks across four experiments now agree. Risk: L safer than C? at all 5 capacities, same sign throughout (t=4.59 at 1.0×). **T1 → `SUPPORTED as a narrower claim`**: no intake benefit, but a real, capacity-robust safety benefit — genuine if partial Pareto improvement. Non-monotonic fed% pattern (negative/positive/positive/positive/negative across the sweep) reported, not established at n=8/point. |
| E044 | **Structural read of `W_pred`: the two metrics disagree, and the honest reading is a narrow real signal inside mostly-unrelated growth.** `IDX_AERIAL` ranks 30th of 59 target channels by weight (below average) — not concentrated on the hawk percept overall. But within the aerial-channel weights, correlation with call-responsiveness is significantly positive across 6 seeds (r=+0.304, t=2.61, barely clears 2.571), though inconsistent (2/6 seeds near zero). H2c stays `NOT STARTED`. Three experiments (E042–E044) now converge on "real but small and partial" without a lever to move it further — stepping back to the wider backlog. |
| E043 | **Exposure escalation: registered falsifier fires, but an unregistered number complicates it.** Doubling predator density (hawk every 10s vs E042's 20s): mean `\|W_pred\|` unchanged (0.00057 vs E042's 0.00058) — falsifier fires as written. But max `\|W_pred\|` reached 30–40% of cap, up from near-zero — concentrated growth an average obscures. Comprehension ticks up slightly (0.006–0.008) but density contrast stays non-significant (t=0.66). H2c stays `NOT STARTED`; next step is a structural read of *which* `W_pred` entries moved, not more exposure. |
| E042 | **H2c re-tested at E041's density fix — still null, and by a lot.** Comprehension after 20-min rearing: full density vs default, **+0.0023 ± 0.0019, t=1.17**, not significant. Both conditions' absolute comprehension (0.005–0.007) is ~1/30th the auditory scaffold's 0.19 — nowhere near a working mechanism. `\|W_pred\|` grew to under 1% of its cap everywhere, at least as plausible a bottleneck as remaining separability gap. H2c stays `NOT STARTED`; H2d's fix was necessary, evidently not sufficient. |
| E041 | **H2d: "fan-in dilution" reframed — sparser connectivity makes separability *worse*, not better.** Paired 12-genome sweep, `sensory_pallium_density` {0.30→0.02}: significant decline at every step (t=4.08–5.37). Post-hoc addendum going the other way (0.30→1.00): separability rises monotonically throughout, no optimum found, full connectivity ~2× the default. Mechanism reframed: not too much noise diluting signal, but too few pallial units getting *any* connection to the 1–2 informative channels at random-sparse density. Regression check, same session: throughput unaffected (dense-with-mask architecture means density doesn't change compute), H2's contrast not broken at n=8 (though not treated as evidence either way, per E037's own demonstrated block-to-block variance on this exact metric). Promising, still not adopted as a default. |
| E040 | **H2f checked against the same confound and found robust, closing the audit trail.** Clean re-measurement of E036's 2×2 (`food_deplete_rate=0`, same seeds): `S+L − S` = **−0.005 ± 0.003, t=1.53** — identical mean to E036's depleted-world −0.005 ± 0.002, t=2.25, same sign, comprehension manipulation check unchanged (0.1921 both). Falsifier does not fire. Final tally across all four checks: confound real and status-changing for H2 (E037) and H2e (E038); present but not consequential for H4 (E039) and H2f (E040). |
| E039 | **H4 checked against the same `food_deplete_rate` confound that broke H2e — and found robust.** 8-seed clean re-measurement (depletion off, 10 min, hawk every 20s, matching E030's block-C seeds 60–67): L vs C? on `caught/dive` **−0.077 ± 0.014, t=5.52, SIGNIFICANT** — same sign as E030's pooled −0.044, comparable magnitude, significant even at reduced n. `dives` denominator unmoved (288.0 across all conditions). Falsifier did not fire; H4's `SUPPORTED` status stands unmodified. Two audits of the same confound, two different outcomes — real for H2/H2e, not consequential here. |
| E038 | **H2e's `REFUTED` verdict does not survive a clean world — reverts to `UNDER TEST`.** E032/E033 shared the same undocumented `food_deplete_rate` confound E037 found for H2. Full 24-seed clean re-measurement (`food_deplete_rate=0`, matching E033's exact design and block structure): interaction **−0.890 ± 0.556, t=1.60, NOT significant** — opposite sign from E033's +0.390, t=2.55, and not replicating it. An interim 8-seed check found the same sign reversal first (−0.954, t=1.36); the full run confirms it rather than being an artefact of the smaller sample. E032 and E033 corrected in place with pointers here, per this project's convention. |
| E037 | **H2's null re-confirmed on the corrected connectome — and a second, undocumented confound found and controlled for.** First pass (today's defaults) gave `fed %` 2.6 against E020's 6.6, hunger change 40× larger — traced to `food_deplete_rate` (added by E025 for H4's dispersal question, never checked against H2's 20-minute/16-hen harness; one feeder ends at 0.97% remaining). Gain/readout ruled out directly (lesioning `W_out` at both gains changes nothing). Re-run with `food_deplete_rate=0`, matching E020's world: **+0.0003 ± 0.0156, t=0.02**, 24 seeds — an even cleaner null. The two 12-seed blocks alone were **individually significant in opposite directions** (t=2.96, t=2.19) — only pooling reveals the null, the E021 lesson recurring inside the experiment meant to move past it. |
| E036 | **E018's aborted 2×2 re-run: the falsifier fires, H2f opened.** An innate crouch-on-hearing scaffold (wired, not learned) supplies the precondition H2b said H3 was missing. Learning still does not add a contingent audience effect on top of it: **S+L − S = −0.005 ± 0.002, t=2.25**, wrong-signed, short of threshold 2.37 at 8 seeds. Manipulation check clean (0.19, matching E018's addendum). Innate floor (`S` vs `N`, zero learning) replicates a third time at +0.066. **New node H2f**: the rule may be the wrong *kind* — instrumental where the biology is closer to Pavlovian. Secondary "strikes/hen" flagged uninterpretable: it reads the pre-E027 `n_struck` exposure-step count, not the event-anchored metric H4's lineage moved to. |
| E035 | **E017/E034's Field-L segregation numbers (2.06×, 1.45×) do not replicate — corrected in both files rather than quietly edited.** Moved modality segregation into `connectome.build(modality_segregated=...)`, properly fan-in-renormalised, instead of the ad hoc probe's post-hoc zeroing. On a **paired** 12-genome sample: structural vs. intact **t=0.04** against threshold 2.201 — no effect. The prior numbers were unpaired ratio-of-means on a quantity with ~6× genome-to-genome spread (0.039–0.221 in this sample) — the same statistical trap E021 caught for seed blocks, uncaught here for genomes across two experiments. Localisation (loss at sensory→pallium, recurrence not the cause) is unaffected — a within-genome comparison, never exposed to the confound. |
| E034 | **H2d re-measured on the corrected (E023) connectome, and reprioritised upward.** Localisation replicates: loss still ~14.5× at sensory→pallium (E017: 17×), recurrence removed still 0.87× (not the cause). ~~Field-L segregation now 1.45×, not E017's 2.06×~~ — **neither number survives E035, run immediately after.** **Occurrence check reverses H2d's demotion**: a 5-min live rollout at H4's config found a hen blind to the hawk while a flockmate audibly alarm-calls on **11.9% of all hen-steps** — the contrast this node depends on is routine, not hypothetical, contra the E019-era reasoning that demoted it. |
| E033 | **H2e → `REFUTED`, on the second pre-registered block E032 §9 asked for.** Block two (seeds 12–23): interaction **+0.240 ± 0.172, t=1.39** — same sign as block one, below its own 12-seed threshold as predicted. Pooled 24 seeds, per method declared in advance: **+0.390 ± 0.153, t=2.55** against threshold 2.069. Falsifier condition 1 fires: a trained readout costs something measurable to remove; the pathway is not inert once structured. H2's own question stays null pooled (fed % +0.153 ± 0.388, t=0.39). Required a fresh venv build and a `--seed-offset` argument added to `scratchpad/e032.py`, which hardcoded `range(args.seeds)` and would have silently re-run block one. |
| E032 | **Causal efficacy, the backlog's unrun test, run at last.** Lesioning a *trained* readout hurts (+0.227) where lesioning an *untrained* one helps (−0.314) — interaction **+0.541 ± 0.254, t=2.13 against 2.201**. Misses. Manipulation check clean (`W_out` moves 9.6%). H2e neither confirmed nor falsified; a second block is owed. A sign error in the pre-registered falsifier is recorded in E032 §6 rather than edited away. |
| E031 | **The credit window is refuted, not deferred.** A hen feeds every **0.3 s** and two thirds of the peck-reward correlation sits at lag 0 — inside the rule's window. But `W_out` at 0x and 10x are both indistinguishable on `fed %` (t=0.68, t=0.54) where a halved peck reflex scores t=4.32, so **H2's null is uninformative**: the pathway may be inert. H2e opened. |
| E030 | **H4 → `SUPPORTED` as written, on a pre-registered pooled test.** Third block (60–71) at −0.028; pooled over 36 seeds across three blocks, **−0.044 ± 0.012, t=3.60** against a threshold of 2.030 declared in advance. No falsifier fires. **But `L vs Lx` is +0.010, t=0.46 for the second time** — the effect survives deleting the pallium's route to the muscles, so H0 is *not* satisfied: this is a result about the reflex arc. |
| E029 | **First positive control in the project's history, and the instrument passes** — every planted effect detected, including the unmodified hen, so E028's null was not a measurement failure. But the same contrast on fresh seeds gives **−0.076, t=4.75** against block A's −0.029, t=1.42; pooled over 24 seeds **−0.052 ± 0.014, t=3.87**. Not promoted: pooling was decided post-hoc, block A alone does not clear, and the `W_out` lesion is still noise. |
| E028b | **The ladder re-run on the repaired instrument (seeds 36–47) does not clear.** L vs C? is **−0.029 ± 0.020, t=1.42** on the unmovable denominator, against **−0.142, SIGNIFICANT** on E026's confounded one — same 12 seeds, opposite verdicts. **L vs Lx = +0.006, noise**: the pallium is not in the causal path, replicated on fresh seeds. T1's registered fed % null for a third time. |
| E028 | **Instrument repair, and a re-baselining.** Strikes become events: the strike share of reward variance goes **87.3% → 0.2%**. Dale's law reaches `W_out` (0/48 → 48/48) — and the naive fix broke the flock, which the measurement caught in one run. Metric moves to intent-to-treat `caught/dive`, whose denominator is flat to **0.0%** where E026's swung **−57.7% to +50.0%**. `Lx lesioned` becomes a permanent rung. Registered contrasts (L vs C?, T1's fed %) reported at last. |
| E027 | **Third review, verified. H4 downgraded: the effect survives lesioning `W_out`**, so it is not a result about the neural model. The metric's denominator moves **up to 63%**; the headline is 15.0 pp pooled, not 19.8. **The reward is 87% `n_struck` at the H4 configuration** and the guard that forbids this runs where no hawk arrives. Dale's law violated on `W_out` (0/48). |
| E023 | E/I fixed, gain re-baselined 0.70 → 0.95. **The knife-edge gain is gone** — usable band 4× wider. **Separability unchanged** (7.4% vs 7.5%), so H2d is untouched and the review's predicted 1.4× did not appear. Invalidates every prior number as *comparable*. |
