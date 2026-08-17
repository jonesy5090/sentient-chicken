# E030 — the third block, with pooling declared in advance

> **Pre-registered.** Sections 1–5 are written and committed *before* the run starts.
> This exists because E029's pooled result was arrived at post-hoc, and that is exactly
> the move E021 showed can go wrong.

## 1. Parent hypothesis

**H4** — an intact channel beats an uninformative one on a task requiring private
information. Currently `UNDER TEST`.

## 2. Question

Two blocks disagree about the size of the L vs C? effect on the intent-to-treat metric:

| block | contrast | control's own risk |
|---|---|---|
| A (36–47), [E028](E028-instrument-repair.md) | −0.029 ± 0.020, t=1.42 | 0.115 |
| B (48–59), [E029](E029-positive-control.md) | −0.076 ± 0.016, t=4.75 | 0.156 |

Pooling them gives −0.052 ± 0.014, t=3.87 — but **that pooling was decided after seeing
the disagreement**, which is not a registered analysis. A third independent block, with
the analysis fixed beforehand, settles whether the effect is real and roughly how big.

## 3. Prediction

The channel effect is real and of order −0.05 on `caught/dive`. Concretely:

- **Primary:** pooled across all three blocks (36 seeds), L vs C? is negative and clears
  the threshold, t > 2.030 at 35 df.
- **Block C alone** is negative. Its magnitude is not predicted — the A/B spread suggests
  block-to-block variation of at least 2×, and predicting a point estimate would be
  false precision.
- **Secondary:** L vs Lx remains indistinguishable from zero. The pallium is not in the
  causal path, and nothing in this design changes that.

## 4. Falsifier

**H4's channel effect is not supported if any of these fire:**

1. Pooled across 36 seeds, |t| < 2.030 — the effect does not survive three blocks.
2. Block C's contrast comes out **positive** (yoked better than intact), which would make
   the A/B pattern a two-block coincidence rather than an effect.
3. L vs Lx becomes significant *in the direction of L being better*, which would mean the
   lesion result was itself a fluke and the whole E027/E028 reading needs revisiting.

**Declared in advance so it cannot be chosen later:** the primary analysis is the
**pooled 36-seed paired contrast on `caught/dive`**, threshold t=2.030. Per-block results
are reported for transparency but the pooled figure is the one that moves the tree. If
the pooled result clears and block C alone does not, that is *not* grounds to report
block C separately — the pooling is the registered test.

## 5. Design

Seeds **60–71**, untouched by any prior experiment. 12 seeds × 10 min × 16 hens, 1.5×
pallium, hawk every 20 s, **no plasticity anywhere**.

Three conditions, the minimum that carries the argument:

| condition | channel | note |
|---|---|---|
| **C? yoked** | real calls, time-shifted | the registered control |
| **L language** | intact | the hypothesis |
| **Lx lesioned** | intact, `W_out` ≡ 0 | the standing brain check |

`N`, `C−`, `C0` and `Cs` are omitted deliberately: they were measured in E028 on the same
world and none is load-bearing for this question. `C−` in particular is vacuous without
plasticity, as E028 records.

**Metric:** `caught/dive` — every (hen, dive) pair, a denominator fixed by the predator
schedule, run length and flock size, which no behaviour can reach. `caught/event` is
reported alongside **only** to show the two still disagree; it is confounded and is not
the test.

**Command:** `python -m run.h4 --minutes 10 --seeds 12 --seed-offset 60 --hawk-period 20`
restricted to the three conditions above.

## 6. Result

```
block                  mean      SE      t   n   control's own risk
A (36-47) E028      -0.0286  0.0201   1.42  12      0.115
B (48-59) E029      -0.0763  0.0160   4.75  12      0.156
C (60-71) E030      -0.0275  0.0251   1.10  12      0.129
POOLED (36 seeds)   -0.0441  0.0123   3.60  36
```

**The registered primary test clears**: pooled t=3.60 against a threshold of 2.030.

**No falsifier fires.**

1. Pooled |t| < 2.030 — no, t=3.60.
2. Block C positive — no, −0.0275, negative as predicted.
3. L vs Lx significant in L's favour — no: **+0.0097 ± 0.0209, t=0.46**, against a
   threshold of 2.201.

## 7. Interpretation

**The channel effect is real.** A flock that hears its own present tense is caught about
**4.4 percentage points less often per hen per hawk** than one hearing the same calls
shifted in time — identical bandwidth, identical rate, identical energetic cost, no
plasticity anywhere. Thirty-six seeds, three independent blocks, analysis fixed before
the third ran.

**Block B is the outlier and this should temper the point estimate.** A and C agree
closely (−0.029 and −0.028); B is 2.7× larger at −0.076 and is the only block
individually significant. The pooled −0.044 is pulled upward by it. A conservative
reading of the typical effect is **nearer −0.03**, with one unusually dangerous world —
B's control catch rate was 0.156 against A's 0.115 and C's 0.129 — where the channel was
worth more. That the effect scales with baseline danger is coherent, not suspicious, but
it means the pooled figure is an average over worlds rather than a constant.

**It is still not a result about the brain.** `Lx` — the intact channel with `W_out`
zeroed, severing every route from 512 simulated neurons to a muscle — is
indistinguishable from `L` for the second time on fresh seeds. The causal chain is two
hand-set weights in `hen/innate.py` and a threshold in `coop/world.py`.

## 8. Consequence

- **H4 moves to `SUPPORTED` as written** — "an intact channel beats an uninformative one
  on a task requiring private information" — on a pre-registered pooled test across three
  blocks, on a metric whose denominator no behaviour can move, with a control measured
  rather than argued.
- **H0 is NOT satisfied by this, and the distinction is the whole point.** H0's subject
  is *a neural model of a chicken*. What is supported here is a claim about a channel
  attached to a reflex arc. The pallium is present, is 1.5× its default size, and
  contributes nothing measurable. Until a learning rule works, the ladder cannot ask H0's
  question — `C−`, the capacity control the whole design turns on, is vacuous without
  plasticity.
- **The honest headline is smaller than E026's and better earned:** −0.198 (confounded
  denominator, mean-of-ratios, one block) became **−0.044 ± 0.012** (intent-to-treat,
  pooled, three blocks, pre-registered).
- **Next is the credit window**, the oldest owed item: `hen/plasticity.py:34-37` states in
  its own docstring that the rule cannot bridge gaps longer than 0.2 s, against a task
  that pays off over minutes. Every H2 null is uninterpretable until that is tested.
