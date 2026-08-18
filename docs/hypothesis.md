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

---

## H2d — the pallium does not form separable representations of distinct stimuli

**Status: SUPPORTED as a limitation** — measured in
[E009](experiments/E009-lagged-pallial-association.md). **This is the blocker behind
H2b, H2c and everything downstream.**

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

~~**Not yet changed.** The gain default stays 0.9~~ — **stale**. The re-baselining
happened; `hen/connectome.py:48` has `gain = 0.70`. Flagged by external review as a doc
that describes a state the code left behind.

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

**Status: UNDER TEST** — opened by [E017](experiments/E017-where-separability-is-lost.md)
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
