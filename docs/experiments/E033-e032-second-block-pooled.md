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

12 fresh seeds (12–23), 20 min rearing + 5 min test, 16 hens.

```
MANIPULATION CHECK: mean |dW_out| / |W_out| during rearing = 0.0957   PASSES (gate 0.05)
                    fixed flocks drift 0.000000                        as required

rearing (block two)   intact  lesioned      drop
trained               11.397    11.209    +0.188
fixed                 11.283    11.335    -0.052

block two interaction (trained drop) - (fixed drop):  +0.2396 +/- 0.1720  t=1.39
  (below its own 12-seed threshold of 2.201, as E021's rule says a single block may be)

secondary, block two, both intact: fed % +0.1144 +/- 0.6544  t=0.17
```

Pooled over both blocks (per-seed paired differences concatenated, 24 seeds, per §5's
declared method — not an average of the two blocks' summary statistics):

```
block one (seeds 0-11)    interaction  +0.5412 +/- 0.2537   t=2.13
block two (seeds 12-23)   interaction  +0.2396 +/- 0.1720   t=1.39

POOLED (24 seeds)         interaction  +0.3904 +/- 0.1531   t=2.55
  threshold (df=23) = 2.069  ->  SIGNIFICANT

secondary, pooled (both intact, H2's own question):
  fed %  +0.1529 +/- 0.3877  t=0.39
```

The manipulation check passes cleanly in both blocks (drift 9.63% and 9.57%; fixed
flocks drift exactly 0 in both), so this is not E001's frozen-`W_out` failure repeating.

## 7. Interpretation

**The sign replicated and the pooled effect clears threshold.** Block two's interaction
is +0.240 — same direction as block one's +0.541, smaller in magnitude — and on its own
does not reach its individual 12-seed threshold (t=1.39 vs 2.201), which is unsurprising
and was allowed for in §3: block-to-block magnitude was expected to vary, only the sign
was predicted with confidence. Pooled across 24 seeds, the interaction is
**+0.390 ± 0.153, t=2.55**, against the pre-registered threshold of 2.069 at 23 df.

This is falsifier condition 1 in §4: **it fires.** Per the pre-registration, that is the
outcome that falsifies H2e rather than supports it — the opposite of the E021 pattern
(a single-block result evaporating on fresh seeds). Here a positive single-block result
(t=2.13, one point below threshold) was followed by a second positive block, and pooling
resolved the ambiguity in favour of the effect being real rather than against it.

**What this licenses and what it does not.** A **trained** `W_out` costs something
measurable when removed; the pathway is not simply inert to lesioning once it has
learned. It does **not** license concluding the pathway is efficacious in the direction
H2 needs — the interaction being positive means *removing a trained readout hurts more
than removing an untrained one*, which is compatible with training doing something
constructive, but is silent on whether that something is *foraging*.

**H2's own question, pooled, stays null.** Trained vs fixed, both intact, over both
blocks: **+0.153 ± 0.388, t=0.39** — indistinguishable from zero, same as block one alone
(t=0.43) and block two alone (t=0.17). Whatever the trained readout is doing that makes
it costly to remove, it does not show up as better feeding when left in place. The two
results are not in tension: a readout can matter to the motor drive it feeds (visible
under lesion, a within-subject before/after comparison with the world and reflexes held
fixed) without moving a noisy population-level outcome like `fed %` (a between-subject
comparison against a different flock's baseline).

## 8. Consequence

- **H2e moves to `REFUTED`.** The claim was that the cortical pathway is behaviourally
  inert regardless of training — that no change in `W_out` can register on H2's metric
  family. A pre-registered, pooled, two-block test (24 seeds) found a trained pathway
  *does* register: t=2.55 against a threshold of 2.069, sign consistent across both
  blocks. The pathway is not inert once structured by learning.
- **H2's null regains standing, with a caveat.** E031 had withdrawn H2's null to
  "uninformative" because an untrained pathway couldn't be shown to reach behaviour at
  all — so a null on `fed %` might just mean the measurement route was closed. E033 shows
  the route is not closed for a trained readout: it can be moved (E023's rearing drift),
  and moving it has a measurable behavioural cost when removed (this experiment). So H2's
  clean null (E020/E021, +0.011 ± 0.012, t=0.95) is a fact about **what the rule learns**,
  not about whether the architecture can express what it learns.
- **Next real work is the rule itself, not the architecture.** E031 already ruled out the
  credit window (lag 0 correlation, 0.3 s feeding interval inside the rule's 0.2 s window).
  What remains open is the rule's *magnitude* or what it is actually optimizing — the
  readout changes in a way that costs something to remove, but not in a way that improves
  feeding, which is consistent with it learning something real but not the thing H2's
  metric rewards. E007's multiplicative-gating question is not ruled back in by this
  result — it was motivated by H2e being *true*, and H2e is now refuted — so it drops in
  priority rather than becoming the next step.
- `scratchpad/e032.py`'s `--seed-offset` argument stays; a third block (or a differently
  designed follow-up on what the trained readout actually encodes) would need it again.
