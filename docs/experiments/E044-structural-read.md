# E044 — is W_pred's growth targeted or diffuse?

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**H2c** — a learned cue can recruit an innate response via top-down association.
`NOT STARTED`. E043 found `W_pred`'s mean magnitude stayed flat under higher exposure
but its *max* entry grew to 30–40% of cap — concentrated growth an average obscures.
This is the structural follow-up both E043 and `docs/backlog.md` flagged as owed: is
that growth on the entries a real association should strengthen, or arbitrary?

## 2. Question

A correct association, if this rule could form one, should predict the aerial-visual
channel (`spec.IDX_AERIAL`) specifically — that's the percept a heard alarm call is
supposed to reconstruct — and should be driven by the pallial neurons that actually
respond to hearing the call, not by neurons that don't. **Does `W_pred`'s learned
structure show either property, or is the growth E043 found unrelated to the
call/hawk distinction it's meant to be about?**

## 3. Prediction

**Genuinely uncertain, registered as such.** Two clean possible outcomes, both
informative:

- **Targeted**: `IDX_AERIAL` ranks near the top of target channels by mean `|W_pred|`,
  and within it, the pallial neurons with the largest weight are the same ones most
  responsive to the call (positive correlation, consistent in sign across seeds). This
  would mean E043's concentrated growth is a real, if tiny, association — and the
  bottleneck to comprehension is magnitude/gain, not correctness of what's being
  learned.
- **Diffuse**: `IDX_AERIAL` is unremarkable among the 59 target channels, and/or the
  correlation with call-responsiveness is near zero or inconsistent in sign. This would
  mean the growth E043 found is closer to a random walk than a structured association —
  the delta rule is moving weights, but not toward anything meaningful yet.

## 4. Falsifier

**For "targeted":** `IDX_AERIAL` at or below the median target channel, or the
call-responsiveness correlation near zero / inconsistent in sign across seeds — the
concentrated growth E043 found is not preferentially about the hawk percept.

**For "diffuse":** `IDX_AERIAL` clearly ranked at or near the top, with a consistently
positive correlation across seeds — the association is real and structured, just still
too weak in magnitude to move behaviour.

Both outcomes are real findings; this section exists so neither can be waved away
after the fact.

## 5. Design

Static analysis on freshly reared connectomes — not reusable from E042/E043's caches,
which only stored scalar summaries, not the full `W_pred` matrix. 6 seeds, full density
(`sensory_pallium_density=1.0`, the condition with the clearest max-growth signal in
E043), `hawk_period_s=10` (matching E043's escalation), 20 minutes rearing, otherwise
identical to E042/E043 (`enabled=True, growth_enabled=False, explore_sigma=0.6,
pred_enabled=True`, `food_deplete_rate=0`).

**Metric 1**: mean `|W_pred|` per target observation channel, averaged over the 16-hen
flock and all pallial sources — where does `IDX_AERIAL` rank among the 59 channels.

**Metric 2**: per seed, the Pearson correlation between (hen 0's `|W_pred|` weight onto
`IDX_AERIAL`, across all pallial neurons) and (hen 0's own call-responsiveness — settled
activation under the alarm call minus rest, the same probe as E017/E034/E041). Hen 0
specifically on both sides, not a flock average: the connectome mask is shared within a
seed but individual weight draws are not, so averaging weight and response across
different hens would compare mismatched pairs.

**Command:** `python -m scratchpad.e044_structural_read --seeds 6 --density 1.0 --hawk-period 10`

## 6. Result

6 seeds, full density, `hawk_period_s=10`, 20 min rearing. Wall clock 142 s.

```
Metric 1 -- top target channels by mean|W_pred| (flock average, all pallial sources):
  channel 50: 0.003525    channel 22: 0.003320    channel 26: 0.003233
  channel 49: 0.002782    channel 30: 0.002719    channel 18: 0.002702
  channel 46: 0.002293    channel  2: 0.002147

  IDX_AERIAL rank: 30 of 59  (mean|W_pred|=0.000679, overall mean=0.000905,
                              BELOW average)

Metric 2 -- per-seed correlation, hen 0's |W_pred[->IDX_AERIAL]| vs hen 0's own
call-responsiveness:
  seed 0: r=+0.505    seed 1: r=+0.451    seed 2: r=+0.279
  seed 3: r=-0.056    seed 4: r=-0.004    seed 5: r=+0.651

  one-sample t-test against zero: mean r=+0.304 +/- 0.117, t=2.61,
  threshold(df=5)=2.571 -> SIGNIFICANT (barely)
```

**Neither pre-registered falsifier condition fires cleanly — the two metrics disagree
with each other.** Metric 1 says diffuse: `IDX_AERIAL` is an unremarkable channel,
ranked 30th of 59, below the average target channel. Metric 2 says targeted, if
narrowly: the correlation between a neuron's call-responsiveness and its weight onto
`IDX_AERIAL` specifically is significantly positive across seeds (t=2.61, just clears
2.571) — but only just, and with real heterogeneity (4 of 6 seeds clearly positive,
0.28–0.65; 2 of 6 indistinguishable from zero).

## 7. Interpretation

**The most defensible reading is that a real but narrow structured signal exists,
buried inside a weight matrix whose largest-scale behaviour is not about this
association at all.** `W_pred`'s overall growth (the channels it puts the most weight
on) is dominated by something other than the hawk/call distinction — seven other
channels rank above `IDX_AERIAL`, and none of section 6's top-8 correspond to it. But
*within* the aerial-channel-specific slice, which neurons get weighted is not random
with respect to which neurons actually respond to the call — the correlation clearing
threshold, even barely and even with two null seeds, is more structure than pure noise
would produce.

**This resolves E043's ambiguity partway, not fully.** E043 found `|W_pred|`'s max grew
substantially while its mean stayed flat, and asked whether that growth was meaningful.
The answer now: whatever grew large is probably not primarily the aerial-prediction
pathway (which isn't even in the top 8 channels by weight) — something else is
absorbing more of the delta rule's capacity. But a smaller, real, and statistically
detectable trace of the *correct* association exists on top of that.

**Why the correlation is inconsistent across seeds is itself worth flagging, not
explaining away.** Two of six seeds show essentially zero relationship. Given `pred_src`
sources every prediction from the same pallial population regardless of genome, and
genome-to-genome separability variance was already large before any of this (E009: 3.5%
– 25.5%; E017/E034/E041's own genome spreads), some seeds may simply start from
representations too poor for even this narrow structure to form — consistent with,
not contradicting, H2d's separability story.

## 8. Consequence

- **H2c stays `NOT STARTED`.** A statistically real trace of correct structure is not a
  working mechanism — E042/E043 already established the behavioural magnitude is
  negligible (1/25th–1/30th the scaffold) regardless of what this file finds about the
  weights underneath it.
- **The next diagnostic question is what's absorbing the *rest* of `W_pred`'s
  capacity** — the seven-plus channels ranked above `IDX_AERIAL` in Metric 1. If that
  turns out to be other, genuinely useful associations (food channels, other call
  types), the rule may be working roughly as intended and simply has many things to
  learn at once, diluting attention to this one. If it's unstructured drift, that
  points back toward the rule's magnitude/gain rather than its correctness. Not
  determined here — would need the same channel-rank analysis repeated for each of the
  top channels, checking whether *their* weight correlates with responsiveness to
  *their* own plausible cue, not assumed from this file's single-channel result.
- **Given the narrowness and inconsistency of this signal, further compute on this
  specific thread (H2c/H2d/W_pred) is not obviously the best next use of it.** Three
  experiments (E042, E043, E044) have now converged on "something real but small and
  partial" without finding a lever that moves it. Worth stepping back to the wider
  backlog rather than continuing to escalate the same mechanism.
