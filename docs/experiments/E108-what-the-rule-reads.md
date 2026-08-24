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

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*
