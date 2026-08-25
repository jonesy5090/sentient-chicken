# E110 — give the update a direction of its own

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **H2**, and **H2b** directly.
The intervention [E109](E109-what-the-rule-writes.md)'s diagnosis names.

---

## 2. Question

E109 measured the obstacle. `Δcortical = m · (dz_slow · stub) · dz_motor`, so `dz_motor`
is the update's entire direction in motor space, and it sits at cosine **0.9822** to the
reflex arc's own deviation — **0.9916** in the windows where the reward fires. The rule
can only write *more of what she was already doing*.

The intervention is to change what the postsynaptic factor traces, so the update has a
direction the arc does not dictate. Two candidates:

**Node perturbation** — trace the **exploration noise** itself. `Δw ∝ m · ξ ⊗ x`, with
`ξ` the injected perturbation, is the standard unbiased estimator of a reward gradient,
and it is what `plasticity.py`'s own docstring already reaches for when it explains why
the noise is placed on the motor drive rather than the membrane: "an exploratory action
that happens to pay off gets credited to the synapses that produced it". The rule as
written does not do that — it credits the *output*, which is the arc plus a 2–5% noise
component. Crediting `ξ` alone is the version the docstring describes.

It also has the right biological anchor for this project specifically. **Birdsong
learning works this way**: LMAN injects variability into RA and reinforcement correlates
that variability with song quality. This is a model of a bird, and the one vertebrate
system where perturbation-based credit assignment is well established is birdsong.

**Cortical-only** — trace `sigmoid(cortical + b_motor)`, crediting the learned pathway's
own contribution. Weaker on theory but a direct read of "what did *I* do", and cheap to
include.

**So: does an update with its own direction produce learning that changes behaviour?**

## 3. Prediction

1. **The intervention will do what it targets.** Cosine between the postsynaptic factor
   and the reflex deviation falls below **0.3** for the noise arm (it is independent
   noise, so near zero) and below 0.6 for the cortical arm, against E109's 0.9822.
2. **Behaviour will not follow.** I predict **no** significant improvement in hunger
   against the frozen-readout control, in any arm. Four explanations have now been
   offered for this null and three are dead; the base rate for "this one is the cause" is
   poor, and node perturbation is a famously high-variance estimator being asked to work
   in 30 minutes of chicken time.
3. **If anything works, it is the noise arm**, because it is the only one that is an
   unbiased gradient estimator rather than a plausible-looking substitution.
4. Stated so a null cannot be reframed later: **I expect this experiment to close the
   line, not to open one.**

## 4. Falsifier

**Primary — and it is written to end the line.** If the cosine falls as predicted (the
intervention demonstrably removes E109's obstacle) **and** hunger is still
indistinguishable from the frozen-readout control in every arm, then removing the
obstacle does not produce learning. E109's finding would stand as a real constraint that
is **not the operative cause**, the "what the rule does" line closes alongside the three
upstream ones, and H2's null would have **no** surviving mechanistic candidate anywhere in
the rule. That is the result, and it should be recorded rather than patched with a sixth
mechanism.

**Instrument falsifier — checked and reported before the headline.** The cosine must
actually fall. If it does not, the flag is not doing what it claims and no behavioural
number in this experiment means anything.

**Magnitude-confound falsifier.** Mean `|ΔW_out|` per consolidation must be within 2× of
the baseline arm's. E109's obstacle is about *direction*; if an arm also changes the
update's *size* by an order of magnitude, any behavioural difference is confounded with
learning rate — the E089 lesson. The design controls for this (§5) and this falsifier
checks the control worked.

**Regression falsifier.** Inertness not bit-identical with the flag at its default, any
ethogram assay changing state, or the suite failing.

**Replication rule.** E021: any significant behavioural result is provisional until it
replicates on a disjoint seed block. Nothing moves the tree on block one.

## 5. Design

**`postsynaptic_factor: str = "motor"`** in `PlasticConfig`. `"motor"` is the current
rule, bit-identical. `"noise"` traces the injected exploration perturbation; `"cortical"`
traces `sigmoid(cortical + b_motor)`.

`Drives` gains a `noise` field so the perturbation is available to the caller — it is
currently added to the drive and discarded, which is why crediting it has never been
possible.

**Magnitude control, and its cost.** A raw swap changes the update's size as well as its
direction: at 30 min `explore_sigma` is ~0.4 while the motor output's deviation is far
smaller, so the noise arm would learn at a different effective rate. So the alternative
factor is **rescaled per hen per step to the norm of `dz_motor`**, leaving direction as
the only difference. `PlasticState` gains `z_post`/`z_post_bar`, allocated always and
updated only when the flag is off its default.

The cost is real and is stated rather than hidden: for a true node-perturbation estimator
the perturbation's *magnitude* carries information, and normalising it away makes this a
direction-only test rather than a faithful implementation of the estimator. That is the
right first cut — E109 identified a direction problem — but a null here is a null about
direction, not a refutation of node perturbation as such.

**Arms**, matched seeds, 8 seeds, 16 hens, 30 min rearing:

| arm | |
|---|---|
| baseline | `postsynaptic_factor="motor"` (the current rule) |
| noise | `postsynaptic_factor="noise"` |
| cortical | `postsynaptic_factor="cortical"` |
| **frozen** | `eta_out=0.0` — the readout cannot learn at all |

The frozen arm is the control the whole experiment turns on: it is what "learning changed
nothing" looks like. It does not depend on the postsynaptic factor, so one arm serves all
three.

**Measured**: cosine between the postsynaptic factor and the reflex deviation (the
instrument check, by the E109 method); mean `|ΔW_out|` per consolidation; hunger and
`caught/dive` after rearing, paired by seed against the frozen control; `|cortical|`; and
the full ethogram at any arm that moves.

### Cost

~30 minutes.

---

## 6. Result

8 seeds, 16 hens, 30 min rearing, `hawk_period_s=60`.
`scratchpad/e110_postsynaptic.py`.

### 6a. The instrument — it clears

| arm | **cos(postsynaptic factor, arc)** | bar |
|---|---|---|
| baseline (motor) | **0.9665** | — (E109 measured 0.9822) |
| **noise** | **−0.0034** | must fall below 0.30 ✓ |
| **cortical** | **−0.0461** | must fall below 0.60 ✓ |
| frozen | 0.9865 | — |

**The intervention does exactly what it was built to do.** E109's obstacle is not
reduced, it is *removed*: the update's direction goes from 97% aligned with the reflex
arc to statistically independent of it, in both arms.

### 6b. The headline — no arm beats a frozen readout

| arm | hunger | caught/dive | \|ΔW_out\| | \|cortical\| |
|---|---|---|---|---|
| baseline (motor) | 0.6268 | 0.1767 | 5.74e-03 | 0.1947 |
| noise | 0.6408 | 0.1592 | 2.36e-03 | 0.0918 |
| cortical | 0.6113 | 0.1696 | 7.33e-03 | 0.2379 |
| **FROZEN** (`eta_out=0`) | 0.6332 | 0.1978 | 1.16e-04 | 0.0599 |

Paired against the frozen readout, df=7:

| contrast | hunger | caught/dive |
|---|---|---|
| baseline vs frozen | −0.0063, t=−0.32 | −0.0211, t=−1.17 |
| noise vs frozen | +0.0077, t=+0.70 | −0.0386, t=−2.10 |
| cortical vs frozen | −0.0218, t=−0.97 | −0.0282, t=−2.09 |

**The primary falsifier fires.** The cosine fell as predicted and hunger is
indistinguishable from frozen in every arm. Predictions 2 and 4 hold: no improvement, and
the experiment closes the line rather than opening one.

**And the baseline arm does not beat frozen either.** This is the cleanest statement of
H2's null the project has: with a proper no-learning control, thirty minutes of learning
produces no measurable benefit over not learning at all — on either metric. The readout
does grow (|cortical| 0.0599 frozen → 0.1947 learned, 3.2×). It just does not help.

**On the two caught/dive near-misses.** t=−2.10 and −2.09 against an uncorrected 2.365.
Applying [E107](E107-red-team-review-2026-08-24.md)'s own recommendation — declare the
contrast count and divide α by it — this experiment runs **6** contrasts, so the
Bonferroni threshold at df=7 is **t=3.636**. Both are far short. They are recorded as a
directional trend that a dedicated experiment could test on a fresh block, and claimed as
nothing. This is the first experiment here to apply that discipline, and it is applied to
a result that would have flattered the hypothesis.

### 6c. My magnitude-confound falsifier fires, for the noise arm

|ΔW_out| ratios against baseline: **noise 0.41×** (outside the pre-registered 0.5–2.0
band), cortical 1.28× (inside).

The per-step rescale did match `||dz_post||` to `||dz_motor||` — the guard test asserts
that on a single consolidation. What it cannot control is *accumulation*: the noise
direction is independent from one consolidation to the next, so successive rank-one
updates partly cancel instead of compounding. That is an intrinsic property of the
estimator rather than a failure of the control, but the falsifier was written against the
accumulated quantity and it fires, so **the noise arm's behavioural null is confounded
with a 2.4× smaller total weight change** and is reported as such. The cortical arm is
clean and gives the same answer.

### 6d. Regression

Inertness bit-identical at the default, 15/15 array digests. The ethogram runs at
`NO_PLASTICITY`, where `postsynaptic_factor` is never read, and the inertness gate covers
`Drives.noise` — so no assay can move, and none was re-run on that reasoning rather than
on a measurement.

## 7. Interpretation

**Removing E109's obstacle does not produce learning.**

E109's finding stands as measured: the default rule's update points 98% along the reflex
arc, and it replicated. E110 removes that completely — cosine −0.003 — and behaviour does
not change. **So the obstacle is real and is not the operative cause.** A constraint can
be genuine, measured, replicated, and still not be what is stopping the thing you care
about.

That distinction is the whole content of this experiment, and it is one this project has
got wrong repeatedly in the other direction: E100–E106 built six mechanisms on a measured
constraint without ever testing whether removing it changed anything. E110 is the first
time the removal was actually done.

**The count is now five, and all five are dead.**

| | proposed cause | fate |
|---|---|---|
| E100–E106 | readout converges on a fixed direction | withdrawn — pooling artefact (E107) |
| E107's reviewer | reward's event destroyed at the first synapse | not adopted — AUC 0.670, not 0.528 |
| E108 | the rule cannot see its teaching event | falsified — 0.731 and 0.955 |
| E109 | the update can only point where the arc points | **real, replicated — and not the cause (E110)** |
| E110 | giving the update its own direction fixes it | falsified here |

**H2's null now has no surviving mechanistic candidate anywhere in the rule.** Both
factors are informative at consolidation time (E108), the update's direction can be made
independent of the arc (E110), the weights move (3.2× growth in cortical magnitude), and
behaviour does not follow. Whatever is wrong is not in the rule's inputs, not in its
direction, and not in whether it writes at all.

**What that leaves, honestly.** Three possibilities, and I do not have evidence to rank
them:

1. **The timescale.** Thirty minutes of chicken time, with `critical_period_s = 3 days`,
   may simply be far too short for a reward-modulated rule to accumulate a policy. This
   is testable and expensive, and it is the possibility the project has never spent
   compute on.
2. **The task.** Hunger equilibrates at ~0.63 in every arm including frozen. If the
   coop's foraging problem has no headroom — if a reflex hen is already near the
   achievable optimum — then no learning rule can demonstrate a benefit, and H2 is
   unanswerable in this environment rather than false. E019's history contains exactly
   this failure once already ("hens start at hunger 0.30, which *is* the equilibrium; the
   metric was a coin flip").
3. **The rule.** A three-factor covariance rule with a 0.2 s credit window may not be
   able to solve a task whose payoff structure spans tens of seconds, which
   `CLAUDE.md` already states as a known limit and which nothing has tested.

**Possibility 2 should be checked first, because it is cheap and because it would make
the other two moot.** It is also the one this project's own history flags: "a null is
only informative if the instrument could have shown a positive."

## 8. Consequence

**Adopted, off by default.** `postsynaptic_factor="motor"`. Neither alternative is
recommended — both are behaviourally null, and the noise arm additionally accumulates 2.4×
less weight change.

**`docs/hypothesis.md`.** H2's node records that no mechanistic explanation for its null
survives, and that the *learning-versus-frozen* contrast is itself null at 8 seeds on both
metrics — which is a stronger and more direct statement of the null than any single
mechanism claim. H2b keeps E109's mechanism, annotated as a real constraint that is not
the operative cause.

**Not adopted.** A sixth mechanism. Five have now been proposed and tested, and the
pattern — each plausible, each measured, each not the cause — is itself the finding. The
next experiment should test whether a positive result is *reachable in this environment*
before proposing anything further about the brain.

### Follow-ups

1. **E111 — is there headroom?** Measure the achievable ceiling on foraging: a
   hand-written optimal-ish policy, or a hen with the food channel wired directly to
   approach, against the reflex baseline. If the gap is small, H2 is unanswerable here and
   the environment needs changing before any rule is tested again. Cheap, and it gates
   everything else.
2. **The trained-flock mute** (backlog §5, open since E032) — still untouched, and it
   tests H0 rather than H2, so it is not blocked by any of this.
3. **E101/E102's permuted-gate control**, outstanding from E107.
