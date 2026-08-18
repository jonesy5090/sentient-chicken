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

8 seeds, 20 min rearing, 16 hens, hawk every 20 s, `food_deplete_rate=0`. Wall clock 670 s.

```
condition                                   comp before  comp after  |W_pred|
no assoc, default density                       -0.0001     -0.0005   0.00000
assoc, default density (E009 replication)       -0.0001      0.0047   0.00058
assoc, full density (E041 fix)                  -0.0001      0.0069   0.00047

PRIMARY -- full density vs default density (both assoc):
  +0.0023 +/- 0.0019  t=1.17  threshold(df=7)=2.365  -> NOT SIGNIFICANT

SECONDARY -- full density vs no-association control:
  +0.0074 +/- 0.0027  t=2.73  threshold(df=7)=2.365  -> SIGNIFICANT

SECONDARY -- default density vs no-association (E009 replication check):
  +0.0052 +/- 0.0026  t=2.01  threshold(df=7)=2.365  -> not significant
```

**The falsifier fires.** The registered primary — full density vs default density —
does not clear threshold (t=1.17). E041's separability fix does not translate into a
clear comprehension effect at this sample size.

**The magnitudes are the more important number than any of the significance tests.**
Comprehension after rearing tops out at **0.0069**, against the auditory scaffold's
hand-wired **0.19** (E018/E036/E040) — roughly **1/30th the size**. Whatever is
happening here, even in the condition with a "significant" secondary result, is not
comprehension in any behaviourally meaningful sense; it is a whisper at the edge of
detectability, not a working mechanism.

**The secondary "hint" does not hold up under scrutiny.** Full density clears
significance against the no-association control (t=2.73) while default density does not
(t=2.01) — but `|W_pred|` actually grew *less* at full density (0.00047) than at default
(0.00058), the opposite of what "more separable representation → more learnable
association → more growth" would predict if the secondary result reflected a real
mechanism. That inconsistency, combined with the primary null and the tiny absolute
scale, is more consistent with sampling noise at n=8 than with a genuine effect —
exactly the caution this project's own E021 lesson exists for, and the reason the
falsifier is read from the *primary* registered contrast, not the more favourable
secondary one.

## 7. Interpretation

**H2d's fix (E041) is necessary but evidently not sufficient, at least not alone, at
this rearing duration.** Doubling separability did not produce comprehension anyone
would call working. Candidate explanations, none tested here:

- **Separability, even doubled, may still be too small.** Full density reached ~2× the
  default's separability metric, not a qualitative change in operating regime — the
  representation may need to move much further before a delta rule can use it.
- **20 minutes and one hawk-period-20s schedule may not give enough co-occurrence.**
  `|W_pred|` reached under 1% of its cap in every condition — the rule has barely
  updated at all, which is at least as strong a candidate explanation as "the
  representation still isn't good enough." E009 needed up to 90× predator density to get
  comparable co-occurrence data with a single seed; this run used 1× the H4-standard
  rate.
- **`eta_pred`/`pred_gain` may be tuned for a regime that no longer applies**, the same
  "parameters tuned under a defect inherit the defect" lesson H2 hit with `eta_out` (H2's
  own section, E002/E010).

**This does not reopen H2d or undo E041.** The separability fix is real, measured
independently of this result, and still the strongest lever found for H2d specifically.
What this experiment adds is that fixing separability alone, at these rearing
parameters, is not the whole story for getting comprehension to actually emerge.

## 8. Consequence

- **H2c stays `NOT STARTED` as a working mechanism.** E008/E009's null is not reversed;
  it is narrowed. The representational blocker they hit is measurably smaller now
  (E041), and comprehension still does not appear at a magnitude worth calling a result.
- **Not treating the secondary "significant" result as a finding**, per §6's own
  internal inconsistency (`|W_pred|` moving the wrong way to support it) and this
  project's standing rule against moving statuses on underpowered contrasts.
- **Before spending more compute on rearing runs**: check whether `|W_pred|` growing to
  under 1% of its cap is itself the bottleneck, independent of separability — e.g. a
  longer rearing window or higher predator density (matching E009's own escalation) on
  the *existing* default-density connectome, to see whether comprehension ever exceeds
  noise given enough exposure, before concluding density is even the right axis to push
  further.
- **If pursued further, needs the same two-tier discipline the rest of this session
  used**: a cheap diagnostic isolating exposure/duration from separability, then — only
  if that's promising — a properly powered (24-seed) version of this exact contrast.
  Neither is run here.
