# E065 — T2 Stage 2: does the flock learn to avoid the poisoned feeder?

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**T2** — the rotating poisoned feeder. `NOT STARTED`. Stages 1/1b/1c
(E060–E062) validated the scaffold; E063/E064 filled two representational
prerequisites (self-location via place cells, and a location cue for hens who only
hear the gakel call). This is the actual hypothesis test `docs/backlog.md` §3
originally specified: **a flock with a working communication channel converges
toward roughly one hen's mistake per contamination rotation; a flock without one
pays roughly N times that.**

## 2. Question

With the validated scaffold, the two location prerequisites, and H2f's own validated
learning rule (`hebbian_readout=True`, `readout_scaling_strength=0.3` — the only rule
this project has ever measured building a genuine, replicated positive learning
result, E057), does a flock hearing the real, intact gakel channel (**L**) reduce its
sickness rate over developmental time more than a flock hearing a decorrelated,
yoked version of the same channel (**C?**) — and does either differ from a flock that
cannot learn at all (**S**, a fixed connectome)?

## 3. Prediction

**No confident prediction that this succeeds.** E058/E059 already found this
project's only validated learning rule amplifies an *existing innate anchor*; it does
not build a new stimulus-response association from nothing. T2's anchor (turn away
from `CLS_SICK`) already exists and already produces real dispersal (E061), so the
precondition E058/E059's finding requires is satisfied — closer to H2f's own task
shape than H2c's. But H2f's anchor was audience-conditioned calling, a comparatively
short-latency association; T2 asks for something new even by H2f's standard: a
durable bias that outlasts the visible cue and generalises across a changing set of
egocentric encounters with the same allocentric place. Whether the rule can do that is
genuinely open, which is the entire reason this stage exists rather than being assumed
from E057.

**If it works**, expect: L's sickness-per-rotation rate falls further from early to
late in the run than C?'s does (a difference-in-differences, not a flat average — the
same design E057 used to separate a real effect from ambient drift), and S (no
learning at all) shows no comparable within-run trend in either direction.

## 4. Falsifier

If L's early-to-late reduction in sickness-per-rotation is not clearly larger than
C?'s, the anchor-amplification mechanism validated for H2f does not transfer to T2's
specific durable, place-based contingency — consistent with, not contradicting,
E058/E059's own finding about this rule's limits. Not a falsifier of H2f, which
already stands on independent evidence, nor of E063/E064's prerequisites, which are
validated separately on their own geometric/correctness grounds regardless of what
learning does with them.

**Mandatory diagnostics, run before trusting any positive result** (CLAUDE.md's own
standing rule):
1. **Is the rule actually active?** Report mean `|W_out|` drift for both L and C? over
   the run. If it's ~0, nothing was learned regardless of the metric.
2. **Matched control metric.** Alongside sickness-per-rotation, measure an unrelated
   outcome with no mechanistic route to the gakel channel — total water intake. If
   *this* also shows an L > C? early-to-late difference, the result is general
   excitability or maturation, not a targeted effect (E058's own diagnostic, reused).
3. **S baseline.** If S also shows a within-run sickness-rate trend comparable to L's,
   the effect is population-level habituation/exploration-reduction over developmental
   time, unrelated to either channel or to learning at all.

## 5. Design

**World**: 16 hens, `spec.DEFAULT_COOP` defaults (`contamination_period_s=300s`,
confirmed via E062's sweep as needing no change; `food_deplete_rate=0`, this session's
standing discipline for foraging-adjacent metrics).

**Conditions**, isolating channel content exactly as H4's own ladder does (same
bandwidth, same cost, differing only in whose calls carry real information):

| condition | `PlasticConfig` | `channel_mode` |
|---|---|---|
| **S** | `enabled=False` | `intact` (moot — no learning to bias) |
| **C?** | H2f's validated rule (below) | `yoked` |
| **L** | H2f's validated rule (below) | `intact` |

H2f's validated rule, unchanged from E057: `PlasticConfig(enabled=True,
growth_enabled=False, kin_audible=True, explore_sigma=0.6, hebbian_readout=True,
readout_scaling_strength=0.3)`. `channel_mode='yoked'` requires
`call_log_steps=spec.YOKE_LOG_STEPS`, per `coop/sensing.py`'s own guard.

**Duration**: 90 minutes (5400s) — at the 300s rotation period this gives exactly 18
completed rotations per run, enough to define a clean early window (rotations 1–4)
and late window (rotations 15–18) with ten rotations of runway between them for
learning to accumulate, while keeping compute bounded (540,000 steps/run).

**Metric**: sickness-onset events, summed across the whole flock, bucketed by
`contamination_epoch` (exact rotation boundaries, no detection heuristic needed —
epochs are deterministic from `t` and `contamination_period_s`). Early rate = total
onsets in epochs 0–3 ÷ 4; late rate = total onsets in epochs 14–17 ÷ 4. Primary
statistic: `(late − early)` for L vs. the same quantity for C?, one-sample and
paired t-tests against `run.experiment._t_critical`, matching this project's
established statistical apparatus.

**Replicates**: 8 seeds per condition, matching H2f's own final, validated sample
size (E057) — the standard this project already treats as sufficient to trust a
headline positive result, rather than the 3-seed register used for mechanism-only
diagnostics (E025/E048/E061/E062).

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e065_t2_stage2_contrast.py --seeds 8 --minutes 90
```

## 6. Result

_Not yet run._

## 7. Interpretation

_Pending §6._

## 8. Consequence

_Pending §6._
