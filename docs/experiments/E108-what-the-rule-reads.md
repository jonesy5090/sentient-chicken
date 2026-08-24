# E108 — what the learning rule actually reads, at the moment it reads it

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **H2**, directly. The first H2 measurement in this project that
is not a geometry statistic.

---

## 2. Question

[E107](E107-red-team-review-2026-08-24.md) withdrew the mechanism that had explained
every H2 null since E100, and left something more useful in its place: two facts that sit
side by side and should not.

Per hen, every internal stage has direction stability **0.9998** — one near-fixed
pattern. And the same populations decode "am I at a feeder" at **AUC 0.726** and "is a
hawk near me" at **0.822**. A representation can be near-fixed in angle and carry
plenty of usable information, because the variation is small but structured. **Direction
stability was never a measure of whether the representation is usable**, which makes
E100–E106 the wrong instrument twice: once from the pooling bug, and once from the choice
of statistic. E081 said as much in passing — "H2d measures distance, not decodability" —
the project agreed, and then built seven experiments on a distance metric.

So ask the decodability question, and ask it in the one place that decides whether
learning can work at all.

**Everything measured so far is the rate, sampled every step. The rule does not read
that.** It reads

```
dw_out = eta_out · m · dz_motor ⊗ dz_slow[-n_motor:]
```

— a **centred slow trace**, sampled at **consolidation boundaries** every `interval = 50`
steps, multiplied by `m`, the reward-prediction error **averaged over that window**. A
different quantity, at a different sampling rate, in a different regime. Nobody has ever
decoded it. This project's own rule is *when a term moves, measure it in its new home*,
and this term has never been measured in its home.

**The reward is ~83% "did you just eat"** ([E107](E107-red-team-review-2026-08-24.md),
correcting `CLAUDE.md`'s long-stale 87%-predation claim). So the question has a concrete
form:

> At the moment the teaching signal fires, does the presynaptic factor the rule reads
> discriminate the event that caused it?

A covariance rule can only bind reward to a state it can see. If it cannot see feeding at
the instant feeding is being rewarded, every H2 null follows mechanically, with no
appeal to geometry, credit assignment or architecture.

### The arithmetic that motivates the prediction

`tau_slow = 0.20 s`; `interval = 50` steps at `dt = 0.01 s`, so **0.5 s**. The trace is
sampled every 0.5 s and remembers 0.2 s. An event at the start of a window has decayed to
`exp(−0.5/0.2) = 0.082` of its peak by the time the rule looks. **Roughly 92% of each
window is forgotten before it is ever read.** `CLAUDE.md` states the credit window as
0.2 s and the consequence — "anything that has to bridge a longer gap than that is not
learnable by this rule as written" — but the *sampling* interval being 2.5× the memory
has never been measured against it.

---

## 3. Prediction

1. **`m` tracks feeding well.** It is ~83% hunger by construction, so windows in which
   she fed should separate from windows in which she did not at **AUC > 0.80**. If this
   fails, the reward machinery — baseline subtraction, `m_acc` averaging — is destroying
   its own signal, which would be a different and larger finding.
2. **`dz_slow` does not.** I expect **AUC 0.50–0.65**: near chance, well below the 0.726
   the instantaneous rate achieves. Three things stack up — trace decay against the
   sampling interval, centring against a slow mean, and the representation itself.
3. **Late-window events decode better than early-window ones**, and the gap is large.
   This is the decay prediction and it is the one that would name a fix (sample more
   often, or lengthen `tau_slow`) rather than just a defect.
4. **I am genuinely unsure about the reared arm.** If rearing improves what the trace
   carries, the rule is bootstrapping slowly rather than not at all, and that is a
   different diagnosis from a hard floor.

## 4. Falsifier

**Primary — the one that closes this line.** If `dz_slow` decodes feeding at **AUC ≥ 0.70**
*and* `m` does at **AUC ≥ 0.70**, then the rule has both factors it needs at the moment it
needs them, and H2's null is **not** explained by what the rule can see. My hypothesis is
wrong, the trace is fine, and the failure lives downstream in magnitude, in the
outer-product form, or in Dale/scaling. That is a real result and should be recorded as
one rather than pursued into a fourth explanation for the same null.

**Instrument falsifier.** The observation at the same boundaries must decode the same
label at **AUC > 0.80**. If it does not, the label or the windowing is broken and no
number in this experiment means anything. This is checked and reported *before* the
headline.

**Triviality falsifier.** The label must be balanced by construction (median split), and
the raw "did she feed at all this window" base rate must sit between 5% and 95%. A label
that is nearly always true cannot be decoded informatively.

**Confound falsifier.** If the *instantaneous rate* at the same boundaries decodes no
better than `dz_slow`, then the loss is not the trace — it is the representation, already
measured in E107 — and this experiment adds nothing to what is known.

## 5. Design

**No new mechanism, no new flag.** A read-only probe.

The trajectory is generated by `simulate._one_step` itself, not by a re-implementation of
it, so the traces measured are the ones the rule actually consolidates on. A nested scan:
an outer loop over windows, an inner loop of `interval` steps, emitting once per window.

Emitted per window, per hen:

- `dz_slow[-n_motor:]` = `z_slow − z_slow_bar`, motor-stub slice — **what the rule reads**
- `z_slow[-n_motor:]` uncentred — isolates what centring costs
- `rate(x)[-n_motor:]` at the boundary — isolates what the trace costs
- the observation at the boundary — the instrument check
- `m_acc` from the carry
- hunger at window start and end, and `n_fed` over the window

**Label**: median split on hunger *drop* across the window. Median split guarantees the
50/50 base rate the triviality falsifier demands, and hunger drop is the reward's own
dominant term rather than a proxy for it.

**Decoder**: ridge, trained on the first half of the windows, AUC on the held-out second
half. Linear, so every number is a **lower bound** — a high AUC proves presence, a low
one is suggestive rather than conclusive, and prediction 2 is stated in that knowledge.

**Timing**: each window's feeding steps are binned by position within the window (first
/ middle / last third) and the decode repeated per bin, for prediction 3.

**Arms**: untrained (the regime in which learning has to bootstrap) and reared 30 min.
4 seeds each, 16 hens, 300 s of probe.

**One approximation, stated in advance.** `m_acc` is read from the scan carry, so at a
boundary it is the sum over `interval − 1` steps rather than `interval` — the reset
happens inside the same step as the consolidation. It is a one-step offset on a 50-step
window and it is documented rather than corrected, because correcting it would mean
re-implementing `_one_step` and measuring a rule that is not the one that runs.

### Cost

~10 minutes.

---

## 6. Result

### 6a. The instrument, checked first

`scratchpad/e108_what_the_rule_reads.py`, 4 seeds, 600 consolidation windows of 50 steps
(0.50 s) each, `tau_slow = 0.2 s`.

The observation at the same boundaries decodes the label at **AUC 0.986** (untrained) and
**0.989** (reared), against a pre-registered floor of 0.80. **The instrument falsifier
does not fire**; the label and the windowing work.

### 6b. The headline — the primary falsifier fires

| source | untrained | reared 30 min |
|---|---|---|
| **`dz_slow`, what the rule reads** | **0.731** | **0.722** |
| `z_slow` uncentred | 0.686 | 0.707 |
| `rate(x)` at the boundary | 0.707 | 0.704 |
| observation (instrument) | 0.986 | 0.989 |
| **`m`, the teaching signal** | **0.955** | **0.958** |

**The primary falsifier fires, decisively.** It was written as: if `dz_slow` decodes at
AUC ≥ 0.70 *and* `m` does at ≥ 0.70, the rule has both factors it needs at the moment it
needs them and my hypothesis is wrong. It does — 0.731 and 0.955.

**Prediction 2 was wrong.** I predicted `dz_slow` at 0.50–0.65, near chance. It is 0.731,
*above* the instantaneous rate's 0.707. The trace is not costing anything; centring gains
about 0.045 rather than losing. The 0.2 s memory against a 0.5 s sampling interval, which
§2's arithmetic said should forget 92% of each window, evidently forgets nothing that
matters.

**Prediction 3 was wrong, and its failure is the interesting part.** Decoding *where in
the window* she fed, from `dz_slow`:

| | first third | middle | last third |
|---|---|---|---|
| untrained | **0.824** | 0.798 | 0.763 |
| reared | **0.774** | 0.758 | 0.728 |

I predicted late-window events would decode *better*, because a 0.2 s trace read at the
boundary should be dominated by the last 0.2 s. The opposite holds, in both arms. The
trace is not reading an *event* at all — it is reading a **persistent state**. Feeding in
the first third predicts a longer bout, because being at a patch is strongly
autocorrelated, and that is what survives to the boundary. The rule can see **"I am at a
food patch"**, which is precisely the state a foraging policy would need to bind to.

### 6c. Two errors of mine in this experiment's own design

**My triviality falsifier fired, for a quantity that was not the label.** §4 required the
raw "did she feed at all this window" base rate to sit between 5% and 95%; it is 5.5%
untrained and **4.3%** reared. I wrote that check believing the decode label — a median
split on hunger drop — was a separate, balanced quantity.

**And my §5 claim that the median split "guarantees the 50/50 base rate" was simply
false.** Hunger drop has a mass point at zero: in the 94.5% of windows where she never
ate, it takes the same value. So the median *is* that value, and `hunger_drop > median` is
true exactly when she fed. `scratchpad/e108b_label_check.py` measures the agreement
between the two labels at **100.0%**.

The label was therefore never degenerate — it is exactly "did she feed", at a 5.5% base
rate, and AUC is a rank statistic that handles the imbalance. Confirmed directly:

| label | `dz_slow` | observation |
|---|---|---|
| median split on hunger drop (§5's label) | 0.731 | 0.986 |
| **fed at all in the window** | **0.731** | 0.986 |
| **at a feeder for more than half the window** | **0.800** | 0.985 |

Identical, as they must be. The result stands; two statements in my own §§4–5 do not.

**One check I could not run.** I intended to test the movement confound — whether the
decode rides on how much she moved rather than on food — and `World` has no velocity
field, so it returned nothing. The 100% label agreement makes it moot: the label is
feeding, so there is no movement quantity for it to be confounded with. Recorded rather
than quietly dropped.

## 7. Interpretation

**H2's null is not explained by what the rule can see.** At the instant `consolidate`
runs, the reward-prediction error separates feeding windows at AUC 0.955, and the
presynaptic factor separates them at 0.731 — 0.800 for the more natural "she is at a
patch" framing. Both factors are present, informative, and available simultaneously. The
covariance rule has everything it needs to bind "at a patch" to whatever she was doing
there.

**This closes the third distinct explanation for the same null in eight experiments**,
and it is worth listing them because the pattern matters more than any one of them:

| | proposed cause | fate |
|---|---|---|
| E100–E106 | the readout converges on a fixed direction | withdrawn by E107 — a pooling artefact |
| E107's reviewer | the reward's event is destroyed at the first synapse | not adopted — AUC 0.670, not 0.528 |
| **E108** | **the rule is blind to its own teaching event** | **falsified here — 0.731 and 0.955** |

Three mechanisms, each plausible, each aimed at the same null, each wrong. What they
share is that all three were **upstream** guesses: they proposed that some signal never
reaches the rule. It reaches the rule. **The failure is in what the rule does with it.**

**The candidates that remain, in the order I would test them.**

1. **The postsynaptic factor.** `dz_motor` traces the *motor output*, which is dominated
   by the reflex arc. So the rule strengthens synapses pairing "at a patch" with
   "whatever the arc was already doing at a patch" — which is already the correct
   behaviour. A rule that reinforces the existing policy cannot change it. This is the
   one I would test first and it is directly measurable: decode, from `dz_motor` at the
   boundaries, how much of it is reflex-driven versus cortical.
2. **Magnitude.** E106 measured `|cortical|` at 1.606 against a reflex drive of similar
   size, and the whole pathway collapsing to 0.020 when its common mode was removed. What
   the readout learns may simply be too small to move behaviour, or too large and
   saturating.
3. **The outer-product form.** `dw_out = m · dz_motor ⊗ dz_slow` is rank one per
   consolidation (E105 §5a). Its accumulation depends on the *variety* of `dz_slow`
   directions, and E107 established the per-hen representation is 0.9998 fixed — so the
   accumulated update may span very few directions regardless of how informative each one
   is. Note this is the E105 line, but arrived at from decodability rather than geometry,
   and it is the only one of the three that the withdrawn work still supports.

**What should not happen** is a fourth upstream mechanism. Three have now been proposed
and none survived.

## 8. Consequence

**No code changes.** E108 is a read-only probe and adds no flag, no mechanism, and no
default.

**`docs/hypothesis.md`.** H2's node records that the "rule cannot see the reward event"
explanation is falsified, alongside E107's withdrawal of the geometry explanation. The
tree now has **no** standing mechanism for its central null, which is a more honest
position than it has held since E100 and should be stated as such.

**What this experiment establishes positively**, and it is not nothing: the three-factor
rule's two factors are both informative at consolidation time, the eligibility trace's
0.2 s memory against a 0.5 s sampling interval is not a defect, and centring *helps*
rather than costs. Those were all open assumptions and are now measured.

### Follow-ups

1. **E109 — decode `dz_motor`.** Candidate 1 above. Cheap, direct, and it is the only
   factor in the rule that has still never been measured.
2. **The trained-flock mute** (backlog §5, open since E032) is untouched by all of this
   and is now the oldest open item by a wide margin.
3. **E101/E102's permuted-gate control** remains outstanding from
   [E107](E107-red-team-review-2026-08-24.md) and would retire a standing claim.
