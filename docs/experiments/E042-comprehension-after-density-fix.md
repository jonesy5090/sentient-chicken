# E042 — does E041's density fix unblock comprehension?

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**H2c** — a learned cue can recruit an innate response via top-down association.
`NOT STARTED` as a confirmed mechanism; E008/E009 tested it and found it null, blocked
by H2d. This is the first re-test since E041 found a connectome change that
substantially improves H2d's separability.

## 2. Question

`hen/plasticity.py`'s `W_pred` is a masked delta rule — move the top-down prediction
toward the actual observation, using a lagged pallial trace, no reward involved. It
already exists, was already tested (E008/E009), and already failed for a specific,
measured reason: the pallium's states for "heard a call" and "saw a hawk" differ by
~7% of resting activity, too little to condition on. E041 found `sensory_pallium_density
=1.0` roughly doubles that separability with no throughput cost and no obvious harm to
H2's own contrast. **Does comprehension — crouching to a played-back alarm call with no
predator present — rise above zero when reared at the higher density, where it did not
at the default?**

## 3. Prediction

**Primary.** Comprehension after rearing is higher at `sensory_pallium_density=1.0` than
at the default 0.30, both with association enabled. Confidence: genuinely uncertain,
stated honestly — E041 doubled *separability*, not comprehension directly, and E009's
own diagnosis left open exactly how much separability would be enough. This is why the
question is being tested rather than assumed.

**Secondary, exploratory:** whether full-density comprehension clears the no-association
control by enough margin to be a real finding on its own, not just relatively higher
than the (already-established-null) default-density condition. And whether the
no-cue crouch baseline moves the way E009's failure mode did — learning the base rate of
danger rather than a contingency on the call — which would be visible as `|W_pred|`
saturating its cap regardless of condition, reported alongside.

## 4. Falsifier

**If comprehension at full density is statistically indistinguishable from the default
(the primary contrast doesn't clear threshold), the density fix does not translate into
the thing it was chased for.** That would mean either the remaining separability gap
(full density is better, not perfect) is still too small, or something else besides raw
separability is the actual bottleneck for this specific learning rule — worth knowing
either way, and a genuine possible outcome, not assumed away by running this.

## 5. Design

Three conditions, matched seeds: no-association control (`pred_enabled=False`, default
density — should replicate near-zero), association at default density (`pred_enabled=
True`, density 0.30 — should replicate E009's null), association at full density
(`pred_enabled=True`, density 1.0 — the test). All with `enabled=True,
growth_enabled=False, explore_sigma=0.6` otherwise identical (W_out/W learning stays on
throughout, matching every other rearing condition in the tree; there is no mechanism to
isolate W_pred's learning independently of it).

**World:** 16 hens, hawk every 20 s (H4's established config, already known to produce
frequent blind-and-hearing co-occurrence — E034 measured 11.9% of hen-steps), 20 minutes
rearing, `food_deplete_rate=0` (E037/E038's lesson — this experiment has not been
audited against that confound and there is no reason to introduce it when it costs
nothing to avoid).

**Primary metric:** `run/audience.py`'s existing `comprehension()` — crouch response to
a played-back aerial alarm call minus the no-cue baseline, measured before and after
rearing. Paired across seeds, two-tailed t against `run/experiment.py`'s
`_t_critical()`.

**Replicates:** 8 seeds — a first-pass check on a new question, matching this session's
established two-tier pattern (de-risk first, replicate properly only if promising),
explicitly not a registered 24-seed replacement.

**Command:**
```bash
python -m scratchpad.e042_comprehension --seeds 8 --minutes 20 --hawk-period 20
```

## 6. Result

*Pending — filled in after the run, not before.*

## 7. Interpretation

*Pending §6.*

## 8. Consequence

*Pending §6.*
