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

8 seeds/condition, 90 min (18 rotations), `threshold(df=7)=2.365`.

| contrast | E066 (broken) | **E068 (fixed)** |
|---|---|---|
| PRIMARY (L−C?) | −0.19 ± 1.25, t=0.15 | **+1.06 ± 0.87, t=1.23** |
| witnessed (L−C?) | −0.25 ± 1.28, t=0.19 | **+1.00 ± 0.81, t=1.23** |
| testimony-only (L−C?) | +0.06 ± 0.19, t=0.32 | **+0.06 ± 0.38, t=0.16** |
| S: late−early | +1.719 | +1.719 (bit-identical) |

Nothing significant anywhere. S came back bit-for-bit identical to E066 — the
expected sanity check that the fix touches only conditions that actually consolidate
weights.

**Two follow-up diagnostics, run because the pre-registered `|W_out|` check cannot
answer whether the fix took effect.** Under `hebbian_readout=True`, `W_out`'s update
uses `m_out = ones_like(m)` — it is not reward-gated at all, so it is structurally
incapable of responding to a change in `m`. The fix affects `W`, the recurrent
weights. That diagnostic has been measuring the wrong pathway for this question in
E065, E066 and E068 alike.

Measuring `W` directly (`scratchpad/e068_w_drift_check.py`, 10 min, 2 seeds):

| condition | mean \|W − W_innate\| |
|---|---|
| `sickness_penalty=0.0` | 3.9313e-04 |
| `sickness_penalty=1.0` | 3.9341e-04 |

A 0.07% difference, with per-seed values overlapping (seed 0 is *lower* with the
penalty on). Turning the sickness reward on or off does not measurably move the
weights it is supposed to gate — **even with E067's fix applied**.

Why (`scratchpad/e068_reward_scale_check.py`, 10 min, 16 hens):

| quantity | value |
|---|---|
| ambient mean \|reward\| per step, per hen | 0.1251 |
| one sickness event's contribution to its window's mean `m` | 0.0200 (0.16× ambient) |
| sickness onsets per hen per 10 min | 0.50 |
| fraction of a hen's consolidation windows carrying one | **0.0417%** |

## 7. Interpretation

**E067's fix works exactly as designed and changes nothing about T2, for a reason
neither E065, E066 nor E067 identified: the sickness reward is roughly four orders of
magnitude too small to compete with ambient homeostatic reward.**

The arithmetic: in 99.96% of a hen's consolidation windows, sickness contributes
exactly zero. In the 0.04% where it does contribute, it supplies 16% of that window's
signal — a minority even there. Its total share of the reinforcement a hen receives
over a run is on the order of `0.0417% × 0.16 ≈ 0.007%`, seven parts in a hundred
thousand. E067's defect was a 50× loss on top of that; correcting it turns 0.0001%
into 0.007%, which is a real 50× improvement and still nowhere near enough to matter.

This supersedes E067's own framing. E067 correctly identified a real defect and
correctly predicted the fix would not rescue T2 — but attributed the remaining
obstacle to architecture (no motor correlate for place cells to amplify). That may
also be true, but it is untestable while the reward term is this small: the weights
never move enough for the architectural question to arise.

**The primary contrast's sign has now flipped twice across three runs of the same
comparison** — E065 +1.13, E066 −0.19, E068 +1.06, all non-significant, spanning both
directions with SEs of 0.87–1.25. Three independent measurements straddling zero at
roughly ±1 SE is what a true effect of zero looks like. The metric is noise-dominated
at this sample size, and the run-to-run swing (≈1.25) exceeds any effect any of the
three runs reported.

**What this does and does not license.** It does license: "T2's reward design cannot
teach the flock anything about sickness at its current magnitude." It does not
license: "the flock cannot learn to avoid the feeder." No experiment in the E065–E068
series has tested the latter, because none of them delivered a teaching signal large
enough to test it with. This is the same shape as the failures tabulated in
`CLAUDE.md` — a null that describes the instrument, not the bird.

## 8. Consequence

**T2 stays `NOT SUPPORTED`, and all three Stage 2 runs (E065, E066, E068) are
withdrawn as tests of T2's actual claim.** They stand as accurate measurements of what
the code does; none is evidence about whether a flock with a working channel learns to
avoid a poisoned feeder. `docs/hypothesis.md`'s T2 node is updated accordingly.

**A fourth run is not worth compute until the reward design is fixed.** Re-running the
same contrast a fourth time would measure the same noise. Two things are needed first,
in this order:

1. **Calibrate `sickness_penalty` against the ambient reward it competes with.** The
   numbers above give a target: to reach parity with ambient reward within the windows
   where it fires, it needs to be ~6× larger; to be a material share of a hen's
   *total* reinforcement it needs far more than that, or sickness needs to be much
   more frequent, or both. This is a design question with real constraints — E014's
   history is precisely a case of one discrete event being scaled so large it
   destroyed the connectome, so "just raise it" is not obviously safe and needs its
   own guard.
2. **Run the positive control E065 flagged and never ran.** Plant a
   guaranteed-detectable effect on the sickness-per-rotation metric and confirm this
   harness reports it. After three nulls on a metric now known to be noise-dominated,
   `CLAUDE.md`'s rule applies squarely: "before concluding a rule cannot learn
   something, show the harness detecting an effect you have deliberately planted."

**Also fix, cheaply, whenever T2 next runs**: the `|W_out|` "is the rule active?"
diagnostic is measuring a pathway that cannot respond to the thing it is checking. It
should read `|W|` instead, or both. It has given a falsely reassuring identical
reading in three consecutive experiments.
