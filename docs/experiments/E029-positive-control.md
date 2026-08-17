# E029 — the first positive control this project has ever run

> **Pre-registered interpretation.** The three possible readings were written into
> `run/poscontrol.py` *before* the numbers arrived, which is the step E026 skipped. Seeds
> **48–59**, a block no prior experiment has touched.

## 1. Parent hypothesis

**H4**, `UNDER TEST`. This does not test H4; it tests whether the instrument that
returned E028's null could have shown a positive at all.

## 2. Question

E028's registered contrast came back **−0.029 ± 0.020, t=1.42** — not significant. Two
opposite readings: there is no meaningful channel effect, or there is one and 12 seeds on
this metric cannot resolve it. `CLAUDE.md` says a null is only informative if the
instrument could have shown a positive, and that a positive control is not optional.
**In twenty-eight experiments this project had never run one.**

## 3. Design

Plant effects of increasing size via `scaffold_gain`, which scales the innate response to
hearing an alarm. Gain 1.0 is the hen E028 measured; 2.0 and 4.0 are a deliberately
exaggerated bird that exists only to test the harness. Contrast is the registered one
throughout — **L vs C? yoked** — so the numbers are directly comparable to the −0.029.

## 4. Result

```
planted effect      caught/dive  vs control      SE      t   detected?
C? yoked (control)        0.156          --      --     --          --
L, gain 1                 0.079      -0.076   0.016   4.75         YES
L, gain 2                 0.047      -0.109   0.017   6.53         YES
L, gain 4                 0.054      -0.101   0.019   5.43         YES
```

**The metric is not blind.** It detects every planted effect, including the smallest —
which is the real hen, unmodified.

**Gain 2 and gain 4 are indistinguishable** (−0.109 vs −0.101). The planted effect
saturates, consistent with E027: the crouch half of the scaffold cannot hide a hen at any
weight, and the head-raise half stops paying once she is reliably looking up.

## 5. The part that matters, and it is not the headline

**Gain 1.0 — the unmodified hen — is detected here at t=4.75. E028 measured the same
contrast at t=1.42.**

| block | contrast | control's own risk |
|---|---|---|
| A (36–47), E028 | **−0.029 ± 0.020, t=1.42** | 0.115 |
| B (48–59), E029 | **−0.076 ± 0.016, t=4.75** | 0.156 |
| **pooled, 24 seeds** | **−0.052 ± 0.014, t=3.87** | — |

Per-block spread is *similar* (sd 0.070 vs 0.056, ratio 0.80×), so unlike
[E021](E021-the-cost-of-exploration.md) this is not a variance artefact — **the means
genuinely differ**, by 2.7×. The control's own catch rate differs too, 0.115 against
0.156: block B is simply a more dangerous world, and the channel is worth more there.

## 6. Interpretation

**What is established:** the intent-to-treat metric works. E028's null was **not** a
measurement failure, which was the live possibility this experiment existed to rule out.
That is the deliverable, and it is the first time this project has earned the right to
interpret a null.

**What is suggested but not established:** a real channel effect of roughly −0.05 on
caught/dive. Pooled over 24 seeds it clears comfortably (t=3.87).

**Three reasons not to promote H4 on this.**

1. **The decision to pool was made after seeing that the blocks disagreed.** The seeds
   were fresh and the gain-1 arm was pre-specified, but *pooling* was not the registered
   analysis. E021 is precisely the case where a post-hoc reading of a seed-block
   difference went the wrong way.
2. **Block A alone does not clear**, and the two blocks differ by 2.7× in effect size on
   the same contrast with the same code.
3. **It is still not a result about the brain.** E028's `Lx` rung — `W_out` lesioned
   entirely — was +0.006, noise. Whatever this effect is, it runs through two hand-set
   reflex weights, and no amount of extra seeds changes that.

## 7. Consequence

- **H4 stays `UNDER TEST`.** The instrument is now trusted; the effect is probably real
  and around −0.05; and it is still not about the neural model.
- **A third block, pre-registered as a replication with pooling declared in advance**, is
  the honest way to settle the size. That is a cheap run and it should be the next thing.
- **`scaffold_gain` is now permanent infrastructure.** Any future null on this metric can
  and should be checked against a planted effect before it is interpreted.
- **Recorded:** the planted effect saturates between gain 2 and gain 4, so the scaffold
  has a ceiling. Consistent with E027's arithmetic and worth remembering before anyone
  proposes "turn the scaffold up" as a fix for anything.
