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

*Pending — filled in after the run, not before.*

## 7. Interpretation

*Pending §6.*

## 8. Consequence

*Pending §6.*
