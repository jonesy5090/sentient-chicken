# E066 — T2 Stage 2, corrected: does the flock learn to avoid the poisoned feeder?

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**T2** — the rotating poisoned feeder. Status `NOT SUPPORTED` after E065, but that
result is withdrawn as untrustworthy: reviewing `hen/plasticity.py`'s `reward()`
directly found it had no term for sickness at all, so H2f's learning rule had no
signal to learn T2's outcome from regardless of channel content. A prerequisite fix
(`sickness_penalty`, off by default) was added and validated separately. This is the
corrected re-run of E065's exact question, now with a genuine teaching signal.

## 2. Question

Same as E065: does a flock hearing the real, intact gakel channel (L) reduce its
sickness rate over developmental time more than a flock hearing a yoked, decorrelated
version (C?), now that reward actually reflects getting sick?

A second, pre-registered question, raised directly by review of E065/E066's own
design: hens already avoid a *visibly* sick flockmate via a fixed innate reflex
(`hen/innate.py`, validated at the population level in E061) that is identical across
all three conditions — it cannot bias L vs. C?, since anything constant across
conditions cancels out of a difference-in-differences. But it can dilute statistical
power: if most sickness-prevention already happens via direct sight in this
tightly-clumped flock (E025, E048, E062), the auditory channel's unique contribution —
helping hens who did *not* witness the incident directly — only has a small slice of
cases to act on, which an aggregate metric could easily wash out. Does splitting
sickness onsets into *witnessed* (another already-sick hen was within `vision_range`
at that moment) versus *testimony-only* (she was not) reveal a call-specific effect
the aggregate metric misses?

## 3. Prediction

Same structural uncertainty as E065 for the primary question — no confident
prediction either way, now for the right reason (an untested rule on a hard task)
rather than the wrong one (no teaching signal at all). If the mechanism works at all,
expect it to show up more clearly, or exclusively, in the testimony-only split than in
the aggregate — that is precisely the case the gakel-call location cue (E064) was
built for, and the case the innate anchor cannot already explain.

## 4. Falsifier

Same primary falsifier as E065: if L's early-to-late sickness-per-rotation change is
not smaller than C?'s, the rule does not build T2's durable, place-based association
even with a genuine reward signal. Secondary: if the testimony-only split shows no
L-vs-C? difference either, the null is not an artefact of the innate reflex diluting
an aggregate metric — it is a real absence of learning in exactly the cases the
auditory channel was supposed to help.

**Mandatory diagnostics**, unchanged from E065: `|W_out|` drift (rule activity, read
with the same caveat E065 flagged — a coarse, whole-matrix check) and the matched
water-intake control (no mechanistic route to the gakel channel).

## 5. Design

Identical to E065 except:

- **Reward**: `sickness_penalty=1.0` added to all three conditions' `PlasticConfig`
  (harmless for S, which never updates weights regardless).
- **Secondary metric**: at each sickness-onset event, classify by whether any
  flockmate with `sick_on=True` (as of the *previous* step, before this onset) was
  within `cfg.vision_range` of the newly-sick hen at that moment. Sum witnessed and
  testimony-only onsets separately per rotation-chunk, alongside the existing total.
  Same early/late window, same difference-in-differences statistic, computed on each
  split independently as a secondary, pre-registered (not post-hoc) analysis.

Everything else unchanged from E065: 16 hens, `spec.DEFAULT_COOP` defaults
(`contamination_period_s=300s`, `food_deplete_rate=0`), H2f's rule
(`hebbian_readout=True`, `readout_scaling_strength=0.3`), S/C?/L conditions, 8
seeds/condition, 90 minutes (18 rotations), same early (1–4) / late (15–18) windows,
same statistical apparatus.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e066_t2_stage2_corrected.py --seeds 8 --minutes 90
```

## 6. Result

_Not yet run._

## 7. Interpretation

_Pending §6._

## 8. Consequence

_Pending §6._
