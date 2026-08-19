# E068 — T2 Stage 2, re-run under the fixed reward-eligibility mechanism

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**T2** — the rotating poisoned feeder. Status `NOT SUPPORTED` after E065/E066, but
E067 (an adversarial review, independently re-verified) found that the reward signal
those experiments depended on — `sickness_penalty` — reached the learning mechanism
on only ~2% of occurrences, due to a sampling defect in `consolidate()` unrelated to
anything about T2 specifically. That defect is now fixed (`hen/plasticity.py`,
`PlasticState.m_acc`). This re-runs E065/E066's exact question, unchanged, under the
corrected mechanism.

## 2. Question

Identical to E065/E066: does a flock hearing the intact gakel channel (L) reduce its
sickness rate over developmental time more than a flock hearing a yoked, decorrelated
version (C?), now that the reward signal for sickness reliably reaches the weight
update it is supposed to drive?

## 3. Prediction

No confident prediction, for a genuinely new reason this time. E065 and E066's nulls
are no longer good evidence either way about T2's underlying question — a signal that
reached the mechanism 2% of the time is not a fair test of whether more of it would
help. Two of the three structural obstacles named in E066/E067 remain regardless of
this fix (`hebbian_readout`'s `W_out` update never reads sensory input directly;
place cells have no pre-existing motor correlate for it to amplify) — so this is not
expected to suddenly succeed. But the specific objection this run answers is real and
independently verified, and deserves an actual answer rather than an inference.

## 4. Falsifier

Same primary falsifier as E065/E066: if L's early-to-late sickness-per-rotation
change is not smaller than C?'s, the rule does not build T2's durable, place-based
association even with a fully-functioning reward pathway. Same secondary falsifier
for the testimony-only split. Given the two remaining structural obstacles, a null
here would not be surprising — but a positive result, if one appeared, would be
genuinely informative and worth investigating carefully rather than dismissed.

**Mandatory diagnostics**, unchanged: `|W_out|` drift and the matched water-intake
control.

## 5. Design

Identical to E066 in every respect — same world config, same H2f rule with
`sickness_penalty=1.0`, same S/C?/L conditions, same 8 seeds, same 90 minutes / 18
rotations, same early/late windows, same witnessed/testimony-only split, same
statistics — the only change is that this run executes under the corrected
`hen/plasticity.py`/`run/simulate.py` (E067's fix, `PlasticState.m_acc`). Reusing
`scratchpad/e066_t2_stage2_corrected.py` unmodified except for the cache filename, so
any difference in outcome is attributable only to the mechanism fix.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e066_t2_stage2_corrected.py --seeds 8 --minutes 90 --cache scratchpad/e068_cache.json
```

## 6. Result

_Not yet run._

## 7. Interpretation

_Pending §6._

## 8. Consequence

_Pending §6._
