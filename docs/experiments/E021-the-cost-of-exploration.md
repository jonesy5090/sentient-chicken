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

12 matched seeds (**12–23**, fresh), 20 min, 16 hens. Wall clock 63 min.

| condition | change | fed % | synapses |
|---|---|---|---|
| fixed (innate only) | +0.019 | 5.8 | 36414 |
| noise only | +0.018 | 5.6 | 36414 |
| learning, no growth | +0.039 | 5.2 | 35288 |
| fixed, legacy audio | +0.006 | 6.6 | 36414 |
| noise only, legacy audio | +0.039 | 5.0 | 36414 |

**Question A — does learning repay its exploration cost?**

```
learning, no growth  vs  noise only     +0.021 +/- 0.027 SE   t=0.79   noise
```

Predicted −0.020 to −0.040 and significant. Measured **+0.021, wrong sign, not
significant**. The post-hoc reading from E020 does not replicate on seeds that did not
generate it.

**Question B — why did exploration become costly?**

```
current audio:  noise only vs fixed     -0.000 +/- 0.035 SE   t=0.01   noise
legacy audio:   noise only vs fixed     +0.033 +/- 0.017 SE   t=1.96   suggestive
```

Predicted the current-audio pair would reproduce E020's significant cost and the legacy
pair would not. **Neither is significant, and the current-audio pair — the one E020
measured at +0.032, t=3.84 — is now exactly zero.**

### The thing this run actually found

The same contrast, same conditions, same n, different seed block:

| noise-only vs fixed | mean | SE | t | verdict |
|---|---|---|---|---|
| **E020** (seeds 0–11) | +0.032 | 0.008 | **3.84** | SIGNIFICANT |
| **E021** (seeds 12–23) | −0.000 | **0.035** | **0.01** | noise |

**The standard error is 4.4× larger on the second block**, and a p≈0.003 result became
p≈0.99. This is not a small wobble around a threshold. It is a complete non-replication
of a result recorded three commits ago.

## 7. Interpretation

**A is falsified. The one positive thing H2 nearly had is gone.** Learning does not repay
the cost of its own exploration; on fresh seeds it is nominally *worse* than noise alone.
Insisting on seeds 12–23 is the only reason this was caught rather than confirmed — had
E021 re-read seeds 0–11, the same arithmetic would have come back and the claim would
have entered the tree.

**B is moot rather than answered.** There is no exploration cost to explain: it did not
replicate in either audio regime. **E020's claim that exploration became costly is
withdrawn.** The architectural argument in §3 — that audio can barely reach behaviour
with plasticity and the scaffold off — turns out to have been the better guide than the
prediction I registered against it.

**The methodological finding is more consequential than either question, and it cuts at
this project's whole statistical practice.** A t=3.84 on 12 matched seeds did not survive
a fresh block of 12. The pairing is sound and the t table is right (the E003 fix); what
is not safe is the assumption that a seed block's *variance* is representative. Seeds
0–11 happened to be homogeneous — SE 0.008 — which made a +0.032 difference look decisive.
Seeds 12–23 give SE 0.035 for the same comparison.

**This does not overturn E020's headline, and it does reduce its precision.** Pooling
both blocks, learning versus fixed across **24 seeds** is **+0.011 ± 0.012, t=0.95** —
still a clean null, so H2's status is unchanged. But E020 reported +0.001 ± 0.010 and read
it as "the harm is gone"; the pooled estimate is compatible with a residual harm up to
about +0.035, and E013's +0.062 sits roughly 2 SE away rather than being firmly excluded.
The correct statement is weaker than the one E020 made.

**What this does not license.** It does not say every significant result in this project
is a fluke — E013's +0.062 was measured at t=3.85 with SE 0.016 and was independently
corroborated by a mechanism (a reward that was 98% call cost) that was found and fixed.
It says that a single 12-seed block is not sufficient evidence to change a status, which
is a narrower and more actionable claim.

**Predator exposure** behaved as it always does: five of six contrasts "suggestive", none
significant, SEs of 660–2300 on means of 90–2900. It remains retired, and this run is
further evidence for that.

## 8. Consequence

- **The "learning repays its exploration cost" item is struck from the backlog**, not
  left open. It was tested and it failed.
- **E020's exploration-cost claim is withdrawn** in `docs/hypothesis.md`.
- **H2's status is unchanged** — a clean null — but its supporting number is restated as
  the 24-seed pooled estimate, +0.011 ± 0.012, rather than E020's block alone.
- **New standing rule, added to `CLAUDE.md`: a result may not change a status in the
  hypothesis tree on one seed block.** Significant findings need replication on a fresh
  block before the tree moves. This run is the argument for it.
- **`run/experiment.py` should report the per-seed spread**, not just the SE, so a
  homogeneous block is visible while the run is happening rather than two experiments
  later.
- **Re-examine which existing tree statuses rest on a single block.** E004's t=3.93 and
  E016's staging result are both single-block and both currently cited.
