# E113 — did E101/E102's gate learn *which* channels to suppress, or just *that* it should?

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **H2b**. A control the tree has been missing since E101, named as
outstanding by [E107](E107-red-team-review-2026-08-24.md).

---

## 2. Question

[E101](E101-top-down-suppression.md) and [E102](E102-basal-ganglia-selective-release.md)
carry the project's only claim that learning produced a *policy*. E102's version:

> The learned policy replicates near-exactly on disjoint seed blocks — 3 of 12 channels
> closed both times, the same three: TURN_R, TURN_L, PECK, with CROUCH, FLEE and every
> call spared. Pecking and turning drive `head_down`, which blinds her to the sky: **she
> learned to suppress the behaviours that blind her and keep the ones that save her.**

Both experiments rest the claim on an interaction: the gate helps on a *reared* brain and
does nothing on an *untrained* one, so what matters is what was learned.

**E107 showed that interaction is empty.** `W_gate` and `W_str` are initialised to zeros
and written only under `pc.enabled`, and the gate is
`sigmoid(W @ stub + GATE_OPEN_BIAS)` with `GATE_OPEN_BIAS = 4.0`. On an untrained brain it
therefore sits at `sigmoid(4.0) = 0.982` — a **1.8% attenuation** of the reflex arc. The
untrained arm's null is guaranteed by construction, so the interaction carries no
information beyond the main effect. E101 prints the 0.982 two paragraphs below its own
claim.

**[E109](E109-what-the-rule-writes.md) makes this load-bearing rather than tidy-up.** It
found the readout's update confined to the reflex arc's own direction — so an *additive*
pathway can only amplify existing tendencies — and observed that E101's and E102's gates
are the only interventions in this project that ever changed behaviour, precisely because
they **multiply** rather than add. If multiplication is the one operation that can
redirect behaviour here, then whether it learned *which* channels to close is the most
consequential open question in the tree.

**So: is the benefit "suppress the right channels", or just "suppress some channels"?**

## 3. Prediction

1. **A row-permuted gate reproduces most of the benefit.** Take the reared `W_str`, permute
   its rows across motor channels, and the predation effect survives. I expect the
   permuted arm to land within one standard error of the true gate.
2. **A uniform gate matched to the same mean level reproduces it too.** E102 measured the
   mean gate at 0.805; a flat 0.805 on every channel, with no structure at all, should do
   about as well.
3. **Because the mechanism is mundane.** `mobility = 1 − crouch` in `actuation.py`, so a
   hen who crouches less moves more and leaves the strike radius. Suppressing *anything*
   that competes with locomotion helps. E101's own §7 half-says this and then attributes
   the benefit to selectivity anyway.
4. **If I am wrong** — if the true gate beats both controls — then E102's policy claim is
   real, and it is the only demonstration in this project of a learned, structured,
   behaviourally effective policy. That would be worth a great deal, which is why the
   controls have to be run rather than argued about.

## 4. Falsifier

**Primary.** If the true gate beats **both** the permuted and the uniform control by more
than 2× the paired standard error, E101/E102's selectivity claim survives its first real
control and should be strengthened in the tree rather than qualified.

**Instrument falsifier — reported before the headline.** The controls must be *matched*.
Mean gate level across channels and steps must be within 0.02 of the true gate's for both
the permuted and the uniform arm. If a control suppresses more or less overall, any
difference is a level effect and the comparison is void.

**Triviality falsifier.** The no-gate arm must differ from the true gate at all. If E102's
main effect does not reproduce here, there is nothing to attribute and the experiment says
nothing about selectivity.

**Replication rule.** E021. Nothing moves the tree on one seed block.

## 5. Design

**No model changes and no new flag.** `W_str` is a `BrainParams` field; the controls are
constructed by transforming a reared one in a probe script.

Rear 8 flocks with `bg_gate=True` exactly as E102 did — 30 min, `hebbian_readout`,
`readout_scaling_strength=0.3`, `hawk_period_s=60` — then run a 10-minute test rollout
with plasticity off under four gate conditions built from the *same* reared brain:

| arm | `W_str` at test |
|---|---|
| **true** | as learned |
| **permuted** | rows permuted across motor channels, per hen, fixed permutation |
| **uniform** | replaced by a constant chosen so the mean gate matches the true arm's |
| **none** | zeros — the gate sits at `sigmoid(4.0) = 0.982` |

Row permutation keeps every learned value and destroys only the assignment of values to
channels, which is exactly the "which" the claim rests on. The uniform arm destroys
structure entirely while holding the level, and is the stricter of the two.

**Measured**: `caught/dive` and hunger over the test window; the realised mean gate per
arm (the instrument check); and the per-channel gate profile for the true arm, to confirm
this reproduces E102's TURN_R/TURN_L/PECK signature before anything is concluded from it.

### Cost

~25 minutes.

---

## 6. Result

**Three seed blocks, not one.** Blocks 1 and 2 gave opposite verdicts on the
pre-registered primary, which is the E021 pattern exactly, so a third was run before
anything was concluded.

### 6a. The instrument — matched in all three blocks

| block | mean gate: true / permuted / uniform |
|---|---|
| 0–7 | 0.8009 / 0.7979 / 0.8009 |
| 8–15 | 0.7993 / 0.7958 / 0.7993 |
| 16–23 | 0.8187 / 0.8183 / 0.8187 |

All within 0.02. The comparison is a comparison of *structure*, not of level.

**One imprecision found and fixed between blocks.** In block 1 the `none` arm ran with the
gate switched off entirely (multiplier 1.0) while the probe reported the 0.982 the gate
*would* have had. A 1.8% mismatch that flattered the true arm. Blocks 2 and 3 run `none`
with `bg_gate` on and `W_str` zeroed, so it really is 0.982. It does not touch the
selectivity contrast, which never involves that arm.

### 6b. E102's signature reproduces exactly, three times

| block | TURN_R | TURN_L | PECK | SCRATCH | FORWARD |
|---|---|---|---|---|---|
| 0–7 | 0.231 | 0.266 | 0.326 | 0.922 | 0.920 |
| 8–15 | 0.222 | 0.286 | 0.291 | 0.919 | 0.925 |
| 16–23 | 0.254 | 0.315 | 0.448 | 0.926 | 0.951 |

The same three channels closed, the rest spared, on three disjoint seed blocks. E102's
description of *what* is learned is not in doubt.

### 6c. The headline — and my predictions were wrong

| contrast (caught/dive) | block 0–7 | 8–15 | 16–23 | **pooled** (post-hoc) |
|---|---|---|---|---|
| true vs none (E102's main effect) | t=−1.72 | t=−2.68 | t=−2.10 | **−0.0679 ± 0.0179, z=−3.80** |
| **permuted vs none** | t=−1.25 | t=−0.23 | t=−0.61 | **−0.0168 ± 0.0152, z=−1.10** |
| **uniform vs none** | t=+0.96 | t=+1.22 | t=−0.22 | **+0.0186 ± 0.0165, z=+1.13** |
| **true vs permuted** *(selectivity)* | t=−0.89 | **t=−4.12** | t=−1.72 | **−0.0554 ± 0.0126, z=−4.40** |
| true vs uniform | t=−5.74 | t=−3.00 | t=−3.45 | **−0.0774 ± 0.0111, z=−7.00** |

**Prediction 1 was wrong.** I predicted a permuted gate would reproduce most of the
benefit. It reproduces **almost none** of it: permuted-vs-none is −0.017 pooled (z=−1.10),
against the true gate's −0.068. Scrambling *which* channel gets which learned value
destroys roughly three quarters of the effect.

**Prediction 2 was wrong.** I predicted a uniform gate matched in level would do about as
well. It does **nothing at all** — pooled +0.019, if anything slightly worse than no gate.

**Prediction 3 was wrong, and it was the mundane explanation.** "Suppressing anything that
competes with locomotion helps, because `mobility = 1 − crouch`" predicts the uniform arm
should work. It does not. A blanket 20% suppression that raises average mobility gives no
benefit; a *spiky* one does.

**Prediction 4 is the branch we are in** — with an important qualification.

### 6d. What the pre-registered rule actually says

§4 required the true gate to beat both controls by more than 2 SE. Against uniform it does
so in **all three blocks**. Against permuted it does so in **one of three**.

So by the rule as written, on a per-block basis, **selectivity is not established** — that
was the verdict the script printed for blocks 1 and 3. The inverse-variance pooled estimate
across all 24 seeds is −0.0554 ± 0.0126 (z=−4.40), which is decisive, but **pooling was
not pre-registered** and is recorded as post-hoc, following E030's precedent for exactly
this situation. All three block estimates share the same sign and the spread is what
8-seed blocks do.

## 7. Interpretation

**E102's selectivity claim survives its first real control, better than I expected, and
not cleanly.**

Two things are now separable and both are established:

1. **Structure is necessary.** A gate matched in mean level but flat across channels gives
   no benefit whatsoever — pooled +0.019, z=+1.13, in the wrong direction. The benefit is
   not "suppress the arc a bit". Robust across all three blocks.
2. **The assignment carries most of the benefit.** A gate with the *same learned values*
   attached to the *wrong channels* recovers only about a quarter of the effect (−0.017
   against −0.068). Pooled, the true gate beats it at z=−4.40; per block, at the
   pre-registered bar, in one of three.

**So E107's concern was right about the evidence and wrong about the conclusion.** The
untrained-brain control genuinely was inert by construction and genuinely carried no
information — that stands. But the claim it was offered in support of turns out to be
approximately true anyway, for reasons nobody had tested. This is the second time in this
session an unverified reinterpretation has been partly overturned by measuring it, and it
is why the red-team skill's central rule is verification rather than adoption.

**What is still not established**, and E102 asserts it: that she learned to suppress *the
behaviours that blind her*. E113 shows the learned assignment beats a random one. It does
not show the benefit runs through `head_down` specifically, and a test that did would need
to break the head-down link while holding the gate fixed. The narrower claim — **a learned,
structured, channel-specific gate outperforms both a scrambled and a flat one** — is what
the data support.

**Against E109, this is consistent and worth stating.** E109 found the additive readout
confined to the reflex arc's own direction, and noted that the two interventions that ever
changed behaviour were both multiplicative. E113 sharpens that: the multiplicative gate is
not merely *able* to redirect behaviour, it demonstrably *does* — its learned channel
assignment is worth about three quarters of its own effect. **The one place in this project
where learning produces a structured, behaviourally effective policy is the one place the
update is not confined to `dz_motor`'s direction.**

## 8. Consequence

**No code changes.** Controls built from a reared `W_str` in a probe script.

**`docs/hypothesis.md`.** H2b records that E101/E102's selectivity claim has now been
controlled: the untrained-brain interaction was empty (E107) and is withdrawn as evidence,
and it is replaced by the permuted and uniform controls, with the per-block/pooled split
stated. E102's "she learned to suppress what blinds her" is narrowed to "the learned
assignment beats a scrambled one"; the head-down mechanism is marked untested.

**Not adopted.** The stronger reading of E102, and any use of the untrained-brain
interaction as evidence for anything.

### Follow-ups

1. **The head-down test.** Break the `peck/scratch → head_down` link while holding the
   learned gate fixed. If the benefit survives, the story is wrong even though the
   selectivity is real.
2. **A fourth block, if this matters enough.** One of three blocks meeting the
   pre-registered bar is not comfortable, and 8 seeds is where this project has been
   burned repeatedly.
3. **The trained-flock mute** (backlog §5, open since E032) remains the oldest untouched
   item.
