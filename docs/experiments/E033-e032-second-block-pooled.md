# E033 — E032's second block, pooling declared in advance

> **Pre-registered.** Sections 1–5 are written and committed *before* the run starts.
> This exists because E029's pooled result was arrived at post-hoc, and E032 §9
> explicitly asked for a repeat of the E030 template rather than that mistake again.

## 1. Parent hypothesis

**H2e** — the cortical pathway is behaviourally inert, so H2 is not testable through it.
Opened by [E031](E031-the-credit-window-is-not-the-blocker.md), tested once by
[E032](E032-causal-efficacy.md).

## 2. Question

E032 measured the causal-efficacy interaction — `(trained: intact − lesioned) − (fixed:
intact − lesioned)` — on 12 seeds and got **+0.541 ± 0.254, t=2.13**, against a threshold
of 2.201: a miss by 0.07 in t, on one block, which E021's rule says cannot move a status.
Does a second, independent 12-seed block replicate the sign and magnitude, and does the
pooled 24-seed estimate clear its own threshold?

## 3. Prediction

**Primary:** pooled across both blocks (24 seeds), the interaction is positive and clears
t > 2.069 at 23 df.

**Block two alone** is positive, of the same rough order as block one (+0.54); the E029/E030
pair showed block-to-block magnitude can vary by 2–3× even when the sign replicates, so no
tighter point prediction is made.

**Secondary:** trained-vs-fixed on `fed %`, both intact — H2's own question — pools to
something still indistinguishable from zero. Nothing in this design changes the pathway's
effect on foraging, only on what a lesion costs it.

## 4. Falsifier

**H2e is falsified (interaction is real, in the direction of a trained pathway being
efficacious) if:**

1. Pooled across 24 seeds, t > 2.069 and the sign is positive (trained flocks lose more
   when lesioned than fixed flocks do).

**H2e is supported (the pathway is inert regardless of training) if:**

2. Pooled t < 2.069 — the effect does not survive a second block, the way E021's t=3.84
   became t=0.01 on fresh seeds.
3. Block two's interaction comes out **negative** — the sign itself does not replicate,
   making block one's +0.54 look like one-block noise rather than a real but small effect.

**Declared in advance so it cannot be chosen after the fact:** the primary analysis is the
**pooled 24-seed interaction**, threshold t=2.069 at 23 df. Block one and block two are
each reported individually for transparency, but the pooled figure is the one that moves
H2e's status — following E030 exactly, including the same discipline about not reporting a
block separately if the pooled result and the per-block results disagree.

## 5. Design

Seeds **12–23**, the block `scratchpad/e032.py --seed-offset` was written to reach (see
E032 §9 and the handover this experiment picks up from — `scratchpad/e032.py` previously
hardcoded `range(args.seeds)`, always reproducing seeds 0–11; a `--seed-offset` argument was
added before this run so block two uses genuinely fresh seeds rather than re-measuring
block one under a different label).

Identical design to E032: 20 min rearing + 5 min test, 16 hens, two rearing conditions
(plastic / fixed) × two test conditions (intact `W_out` / lesioned `W_out`), matched seeds,
each test branch forked from the identical end-of-rearing world and brain state. Same
manipulation-check gate: mean |ΔW_out|/|W_out| > 0.05 during rearing, void without it.

**Command:**

```bash
python -m scratchpad.e032 --seeds 12 --seed-offset 12 --rear 20 --test 5 --budget 100000
```

**Metric:** `fed %` over the 5-minute test window, matching E032.

**Pooling method:** the per-seed paired differences (interaction terms) from block one
(seeds 0–11) and block two (seeds 12–23) are concatenated into one 24-length sample; mean,
SE and t computed on the pooled sample directly (same method E030 used), not by averaging
the two blocks' summary statistics.

## 6. Result

*Not yet run. Sections 6–8 are written after the block-two run completes and the pooled
statistic is computed from the actual per-seed data — not before.*

## 7. Interpretation

*Pending §6.*

## 8. Consequence

*Pending §6.*
