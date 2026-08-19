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

8 seeds/condition, 90 min (18 rotations), `threshold(df=7)=2.365`.

**Primary — total sickness-per-rotation:**

| condition | early | late |
|---|---|---|
| S | 5.906 | 7.625 |
| C? | 8.469 | 8.875 |
| L | 8.750 | 8.969 |

| contrast | mean ± SE | t | |
|---|---|---|---|
| L: late − early | +0.219 ± 0.462 | 0.47 | not significant |
| C?: late − early | +0.406 ± 0.888 | 0.46 | not significant |
| S: late − early | +1.719 ± 1.209 | 1.42 | not significant |
| **PRIMARY: (L late−early) − (C? late−early)** | **−0.188 ± 1.249** | **0.15** | **not significant** |

S's numbers are bit-for-bit identical to E065's — expected and a useful sanity check:
`sickness_penalty` only touches `reward()`, and S never consolidates weights
(`enabled=False`), so nothing about S's own trajectory should move, and nothing did.

**Secondary A — witnessed onsets** (another already-sick hen within `vision_range`,
explainable by the innate anchor alone): early 3.7–6.8, late 5.3–6.9 across
conditions; contrast (L−C?) = −0.250 ± 1.284, t=0.19, not significant.

**Secondary B — testimony-only onsets** (no visible sick hen at that moment, the only
case the auditory channel could plausibly help): early 2.0–2.2, late 2.0–2.3 across
conditions — roughly a third the count of witnessed onsets, confirming the mechanism
directly: most sickness-prevention opportunities in this flock are the kind the innate
reflex already covers. Contrast (L−C?) = **+0.063 ± 0.193, t=0.32, not significant** —
and the sign is wrong here too (L nominally worse than C?, not better).

**Diagnostic 1**: final mean `|W_out|` — C? 0.0632, L 0.0632, identical to four
decimal places, same as E065.

**Diagnostic 2**: (L late−early) − (C? late−early) for water intake = −316.4 ± 364.1,
t=0.87, not significant — clean, as expected given the primary result is null.

Wall clock: 3415s (~57 min).

## 7. Interpretation

**The correction changed the picture, but not the conclusion.** With reward.py
prerequisite gap now fixed, the primary contrast's sign flipped to the direction the
original prediction called for (L's increase smaller than C?'s), but the effect size
is close to zero (−0.19, against a scale where the individual conditions' own SEs are
0.5–1.2) and nowhere near significant. This is a materially different — and more
trustworthy — null than E065's: E065's wrong-signed result is now explained (there was
no teaching signal at all), and this run closes that specific gap while still finding
nothing.

**The witnessed/testimony-only split directly confirms the mechanism the design review
predicted, and gives the null more force, not less.** Testimony-only onsets are about
a third the volume of witnessed ones — most of what "prevents" sickness in this flock
is already covered by the innate visual-avoidance reflex, present identically in all
three conditions. That is exactly the dilution concern raised before this run: if a
real but small call-specific effect existed, it should show up more clearly in this
isolated bucket, where the innate reflex has nothing to contribute and only learned,
auditory-driven avoidance could possibly explain a difference. It does not show up
there either — if anything the (non-significant) point estimate points the wrong way.
This rules out "the aggregate metric washed out a real effect" as an explanation for
the overall null; the null holds in the one bucket built specifically to detect it.

**Diagnostic 1 carries the same caveat as before** — an unchanged aggregate `|W_out|`
does not prove the rule is inactive, only that this coarse check cannot see anything
finer. That limitation is now doing less work, though: with a real reward signal in
place and a null in the specific bucket theory says should show the effect, the burden
has shifted from "was there anything to learn from" (resolved) to "can this rule learn
it at all" (looking increasingly like no, though the local-vs-aggregate weight
question from E065 remains formally open).

## 8. Consequence

**T2 stays `NOT SUPPORTED`, now on solid methodological ground.** The two live
objections to E065's conclusion — no reward signal, and a possible innate-reflex
confound — have both been addressed directly in this single corrected run, and the
result did not change in substance. H2f's rule, with a working teaching signal and
looking specifically at the cases only it could plausibly explain, still shows no
detectable learned avoidance.

This is a stronger, more defensible negative result than E065's, and the T2 line of
work has now had a fair test. Not claimed as fully closed: a longer run, a different
rule, or the finer local-weight diagnostic could still turn up something, and none of
those have been run. But the specific, actionable objections raised during design
review have been checked, not just noted.

Update `docs/hypothesis.md`'s T2 node to record this corrected result alongside
E065's, explaining why the earlier conclusion needed a second look and what changed.
`docs/backlog.md`'s T2 section: mark the corrected re-run done, retire the open
"reward signal" and "innate-reflex confound" questions this run was built to answer.

## 9. Post-hoc corrections (red-team review, E067)

An adversarial review commissioned after this experiment, independently re-verified
before adoption per this project's own red-team discipline, found two real defects
in this experiment's own measurement, alongside the reward-sampling defect recorded
separately in [E067](E067-reward-eligibility-sampling-defect.md):

**The witnessed/testimony-only split (§5) under-gates "witnessed."**
`scratchpad/e066_t2_stage2_corrected.py`'s `nearby_sick` check uses pure Euclidean
distance (`d < cfg.vision_range`), but the actual innate anchor is driven by
`CLS_SICK`, which also requires the sick hen to fall within the observer's field of
view (`coop/sensing.py`: `prox * (jnp.abs(ang) <= _HALF_FOV)`, a 300°/2 half-angle —
a 60° blind cone directly behind her). Confirmed directly by reading both code paths.
This means the "witnessed" bucket includes some cases the innate reflex could not
actually have used (sick flockmate nearby but behind her), so it is not the clean
"explainable by the anchor alone" population §7 describes. This does not weaken the
testimony-only result — correcting for FOV would only move cases *into* that bucket,
never out of it, so the true testimony-only population is at least as large as
reported and shows the same null.

**S's early-to-late trend is substantially a single-rotation warm-up artifact,**
not evidence of a channel/learning-independent population trend as this experiment's
§7 originally claimed. Confirmed directly against `scratchpad/e066_cache.json`:
summed across all 8 seeds, rotation 0 has 26 total onsets against 38–89 at every
other rotation — a clear, isolated dip, most likely the flock not yet having found
the first bad feeder from a cold start. Excluding rotation 0 from S's early window
roughly halves its reported trend (+1.719 → +0.833). **This does not touch the
primary L-vs-C? contrast**, which holds `explore_sigma` and the world seed fixed
across both arms and never uses S in its own computation — the headline null stands
unmodified. It does mean S's trend should not be read as informative about
population-level dynamics independent of channel content, which §7 overstated.

No numbers in §6 are changed by either correction; both are corrections to
interpretation, recorded here rather than silently editing §7's original text.
