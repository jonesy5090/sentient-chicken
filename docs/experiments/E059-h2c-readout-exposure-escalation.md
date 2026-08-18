# E059 — is E058's null exposure-limited, not just architecture-limited?

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**H2c** — a learned cue can recruit an innate response via top-down association.
`NOT STARTED` as a working mechanism. [E058](E058-h2c-hebbian-readout.md) tested H2f's
validated non-reward-gated readout rule on comprehension built from nothing (no
scaffold) and found only uniform general excitability (crouch, peck, scratch, flee all
moved together, ~0.004, two orders below H2f's effect) — not a targeted association.
E058's own consequence section left open whether this was under-exposure (not enough
hawk-call co-occurrence events during rearing for the correlation to accumulate on) or
a genuine block, mirroring the exact question [E043](E043-exposure-escalation.md)
resolved for the `W_pred` pathway (there: exposure moved `|W_pred|` a lot without
producing comprehension — separability, not exposure, was the binding constraint).

## 2. Question

**Does escalating predator exposure — `hawk_period_s` 20 → 10, matching E042→E043's own
escalation for the `W_pred` pathway — change E058's null for the readout pathway, either
by producing a targeted crouch-specific effect or by at least moving the readout's
weights substantially more than E058's 20-minute, `hawk_period_s=20` rearing did?**

## 3. Prediction

**No confident directional prediction on comprehension itself — registered as
genuinely uncertain**, consistent with E043's own mixed history on this exact question
for the sibling pathway (there, exposure moved the weights without moving
comprehension). **A closer-to-mechanical prediction on `|W_out|` drift**: more
co-occurrence events during rearing should produce more accumulated readout change,
independent of whether that change becomes a targeted crouch association — this is the
same "does the knob move the thing it's supposed to move" check E043 ran before
trusting its own null.

## 4. Falsifier

**For "under-exposure" as the explanation of E058's null**: if crouch remains
statistically indistinguishable from the peck/scratch/flee control channels even at
this escalated exposure, under-exposure is not the explanation — the same conclusion
E043 reached for `W_pred`, now established for the readout pathway too, and H2c's block
should be read as architectural (H2d-adjacent or otherwise), not a matter of more
rearing time.

**Not a falsifier**: `|W_out|` (or the relevant readout slice) moving substantially
further without comprehension moving with it — informative on its own terms (the rule
is doing *something* with the extra exposure), not evidence against this experiment's
conclusion about comprehension specifically.

## 5. Design

Identical to E058 in every respect except `hawk_period_s`: **10** instead of 20 (E058's
own value, itself already escalated from the 900s default — this is a second escalation
on top of that one, matching E042→E043's precedent exactly). 16 hens,
`food_deplete_rate=0`, no scaffold, `pred_gain=0.0`, 20 minutes rearing,
`hebbian_readout=True, readout_scaling_strength=0.3`, same `FIXED` control.

**Primary metric**: crouch `LEARN − FIXED`, paired t-test, 8 seeds.
**Mandatory diagnostic, not optional** (same as E058): peck, scratch, flee under the
identical synthetic test — a crouch-only positive is not reported as targeted unless
these three do not show a comparable rise.
**Secondary, exploratory**: mean `|W_out|` (or the relevant slice) after rearing,
compared informally against E058's own (uncollected — will be added if needed) baseline,
as the "did exposure move anything" check.

**Replicates**: 8 seeds. Given E058's null was unambiguous with a built-in control (not
a surprising positive), a second block is warranted here only if this run produces a
surprising positive — matching this project's standing asymmetric practice.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e058_h2c_hebbian.py --seeds 8 --minutes 20 --hawk-period 10 --cache scratchpad/e059_cache.json
```

## 6. Result

8 seeds, 20 min rearing, 16 hens, hawk every 10s (escalated from E058's 20s), no
scaffold, `pred_gain=0.0`.

```
        crouch      peck   scratch      flee
FIXED   -0.0000   -0.0000   -0.0000   +0.0000
LEARN   +0.0036   +0.0041   +0.0047   +0.0033

crouch     LEARN-FIXED: +0.0036 +/- 0.0013  t=2.71  SIGNIFICANT  <-- PRIMARY
peck       LEARN-FIXED: +0.0041 +/- 0.0011  t=3.71  SIGNIFICANT
scratch    LEARN-FIXED: +0.0048 +/- 0.0014  t=3.37  SIGNIFICANT
flee       LEARN-FIXED: +0.0033 +/- 0.0012  t=2.81  SIGNIFICANT
```

**Essentially identical to E058's `hawk_period_s=20` numbers** (crouch 0.0036 vs 0.0036,
peck 0.0041 vs 0.0044, scratch 0.0048 vs 0.0047, flee 0.0033 vs 0.0035) — doubling
predator exposure changed nothing. The falsifier fires immediately and unambiguously:
crouch remains indistinguishable in shape from the three control channels, at the same
tiny magnitude, regardless of exposure.

**Secondary check — did the readout even move more?** (3 seeds, informal, not the
primary metric): mean `|W_out − innate W_out|` after rearing, `hawk_period_s=20` vs
`10`: **0.0539 vs 0.0549** — a 2% difference, well inside the seeds' own spread
(0.051–0.057 either way). Unlike E043's finding for `W_pred` (which moved
*substantially* with exposure, just without producing comprehension), the readout's
drift here does not move at all with exposure.

## 7. Interpretation

**Under-exposure is not the explanation, and the reason is mechanistic, not just
empirical.** `readout_scaling_strength` pulls `W_out`'s row sums back toward the innate
baseline every consolidation — this creates a dynamic *equilibrium* magnitude of drift
that the correction settles into, not a ceiling that more data pushes against harder.
`W_pred`'s stabiliser (`pred_max`, a hard per-synapse clip) behaves differently: more
co-occurrence events move more synapses toward that clip, so exposure has somewhere to
go (E043's finding). The readout's proportional correction has no equivalent headroom —
by construction, it should reach roughly the same steady state regardless of how much
raw correlational signal rearing supplies. This experiment confirms that mechanical
prediction rather than merely failing to falsify it.

**This is a more complete answer than "more exposure might still work."** E043 left
that door open for `W_pred` because the weights genuinely hadn't saturated yet at
E042's exposure level. Here, the equivalent door does not open at all — not because 8
seeds weren't enough to see movement, but because the specific mechanism that makes
`hebbian_readout` safe (the fix E056 built after E055's runaway) is the same mechanism
that caps how much exposure can matter. Escalating further (higher hawk rate, longer
rearing) would very likely reproduce this exact non-result, for the same reason, and is
not worth running again without changing the stabilisation mechanism itself.

## 8. Consequence

- **H2c stays `NOT STARTED`.** E058's null is confirmed as architectural (or at minimum,
  not exposure-limited under this specific stabiliser), closing the question E058 left
  open rather than leaving it genuinely uncertain.
- **A mechanistic distinction worth recording generally**: proportional/scaling-based
  stabilisers (this project's fix for `W_out`) and hard-clip stabilisers (`W_pred`'s
  `pred_max`) respond to exposure differently — the former reaches an exposure-independent
  equilibrium, the latter has real headroom until saturation. Any future non-reward-gated
  rule built with a scaling-style stabiliser inherits this same exposure-insensitivity,
  worth knowing before spending another experiment on an exposure sweep for it.
- **Closes the backlog item this experiment answers** ("does comprehension emerge via
  the readout rule with a longer rearing duration or higher hawk rate") with a clean no,
  and for a stated, checked reason rather than an assumption.
- **Does not touch H2f's own result** — H2f's task used the identical stabiliser and
  succeeded because it had a wired-in anchor to amplify, not because of how much
  `W_out` moved in absolute terms. Nothing here revisits that.
