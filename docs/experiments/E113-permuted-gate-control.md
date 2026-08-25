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

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*
