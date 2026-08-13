# E021 — what exploration costs, and where the cost came from

> **Pre-registration.** Sections 1–5 written and committed before the run. Sections 6–8
> after.

## 1. Parent hypothesis

**H2** — three-factor plasticity produces measurable behavioural improvement. Currently
`UNDER TEST` with a clean null from
[E020](E020-h2-after-the-e019-fixes.md).

## 2. Questions

Two, sharing four of five conditions and therefore one run.

**A. Does learning repay the cost of its own exploration?** E020 measured learning at
+0.001 and noise-only at +0.037 against the same control, on identical exploration
noise. If that difference is real, learning is buying back the cost of the variability
it requires — which would be **the first positive result H2 has ever produced**.

**B. Why did exploration become costly?** Noise-only went from indistinguishable from
the fixed control (t=0.32, E013) to significantly worse (t=3.84, E020).

Question B has an unusually tight answer available. **For a condition with plasticity
switched off, the audio path is the only thing that changed between E013 and E020.** The
strike-units fix (E014), the reward composition and the covariance rule (E019) all act
exclusively through learning. So either the audio fix caused it, or it was seed
variation. Nothing else is available.

## 3. Predictions

**A. Learning beats noise-only, significantly.** Predict between −0.020 and −0.040, at
p<0.05 on 12 seeds.

> **Tested on fresh seeds.** E020 used seeds 0–11 and that run is what generated this
> hypothesis. E021 uses **seeds 12–23**. Re-reading a post-hoc observation off the data
> that produced it is not a test, and this project has enough forking paths already.

**B. The audio fix is the cause.** Predict the legacy pair reproduces E013 — noise-only
vs fixed under legacy audio comes back **non-significant, |t| < 2.2** — while the modern
pair reproduces E020 at t > 2.2.

**Confidence on B: genuinely uncertain, and the architecture argues against my own
prediction.** With plasticity off and the auditory scaffold off, audio reaches the
motor system only through `W_out` at `readout_scale = 0.05` — the reflex arc has
identically zero weight on every call channel (asserted in
`tests/test_plasticity.py`). That is a weak path for a large behavioural effect. Against
that: the *input statistics* changed enormously (channels went from pinned at 1.0 to
varying across 0.03–0.99), and that does change pallial state. I can argue it both ways,
which is the honest reason to run it.

**If B comes back null in both regimes**, the E020 noise result was seed variation, and
E020's own §8 sentence about exploration being costly must be withdrawn.

## 4. Falsifier

**For A:** if learning is not better than noise-only on fresh seeds, the post-hoc
reading from E020 is dead and should be struck rather than left as a live backlog item.
H2's null stands with no positive component.

**For B:** the two regimes disagreeing localises the cause to the audio fix. The two
regimes agreeing means the cause is not the audio, which — given the deduction in §2 —
leaves only seed variation, and a claim in E020 gets withdrawn.

Both questions have outcomes that remove a claim from the tree. That is the point.

## 5. Design

Five conditions, matched seeds, `explore_sigma` stated explicitly in every one.

| condition | plasticity | σ | audio |
|---|---|---|---|
| fixed (innate only) | off | 0.0 | current |
| noise only | off | 0.6 | current |
| learning, no growth | on | 0.6 | current |
| fixed, legacy audio | off | 0.0 | **pre-E019** |
| noise only, legacy audio | off | 0.6 | **pre-E019** |

A 2×2 of audio × exploration, plus the learning arm. The two legacy conditions exist
only to reconstruct E013's regime; `cfg.legacy_audio` restores both the raw-sigmoid
emission and the linear-sum combination, verified to reproduce the dead channel exactly
(a full-amplitude call moves the receiver by **0.0000**, against +0.932 with the fix).

**Primary contrasts**, both pre-specified as first-class paired comparisons rather than
differences of differences:

1. `learning, no growth` − `noise only` *(question A)*
2. `noise only, legacy audio` − `fixed, legacy audio` *(question B, against the
   already-measured modern pair)*

**Secondary:** fed %, synapses, exposure (reported, not interpreted — retired since
E003/E004).

**Replicates:** 12 matched seeds, **12–23**, matching E020's count.

**Command:** `python -m run.experiment --e021 --minutes 20 --seeds 12 --seed-offset 12`

**Multiple comparisons, stated up front.** This run reports six contrasts. Question A's
prediction is directional and pre-specified, so it is a single test. Question B is a
comparison of two contrasts and is treated as such. Nothing else in the output should be
promoted to a finding without its own pre-registration.

## 6. Result

*To be written after the run.*

## 7. Interpretation

*To be written after the run.*

## 8. Consequence

*To be written after the run.*
