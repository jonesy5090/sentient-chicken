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

_Pending._

## 7. Interpretation

_Pending._

## 8. Consequence

_Pending._
