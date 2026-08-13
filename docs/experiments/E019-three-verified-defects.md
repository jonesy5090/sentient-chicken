# E019 — three defects found by external review, verified independently

> **Diagnostic.** Not pre-registered, and not an experiment: it re-measures claims made
> by an outside reviewer who read this repo with no prior context. Recorded because all
> three replicated, two of them more strongly than reported, and because between them
> they change the status of most of the tree.

## 1. Parent hypothesis

All of H2, H2b, H2c, H2d, H3 — and, through H3, the H4 headline design.

## 2. Question

An adversarial review was commissioned specifically to find what eighteen experiments of
accumulated context had stopped seeing. It returned three defects, each in a different
module, none of which appears anywhere in `docs/experiments/`. Do they replicate?

## 3. Method

Re-measured from scratch, without reusing the reviewer's scripts, at
`scratchpad/verify_audio.py` and `scratchpad/verify_readout_and_reward.py`. Where the
numbers differ from the review, **the numbers here are the ones measured in this repo**
and the discrepancy is noted.

## 4. Result

### Defect 1 — the auditory channel is saturated. It carries no information.

`coop/world.py:192` emits `calls = motor[:, CALL_IDX] * vigour` — the **raw sigmoid**,
unthresholded, while the same file gates feeding, crouching and fleeing at `> 0.5`. A
resting hen's motor floor is `sigmoid(REST_BIAS) = sigmoid(-2.5) = 0.076`, so every hen
emits all four calls continuously. `coop/sensing.py:77` then **linearly sums** across
flockmates before clipping: `audio = clip(atten @ calls, 0, 1)`.

Measured after 120 s of settling, in the configuration the flock actually adopts:

| n_hens | audio min | audio mean | nearest neighbour |
|---|---|---|---|
| 2 | 0.054 | 0.096 | 0.55 m |
| 4 | 0.118 | 0.282 | 0.09 m |
| 8 | 0.272 | 0.667 | 0.72 m |
| **16 (the default)** | **0.707** | **0.991** | 0.42 m |

And the decisive one:

> At `n_hens=16`, the aerial-alarm channel heard by hen 0 reads **1.0000**. When hen 1
> calls at **full amplitude** directly beside her, it reads **1.0000**. Delta: **0.0000**.

15 flockmates × 0.076 = 1.14, clipped to 1.0. The channel is pinned before anyone says
anything.

**Every experiment touching communication has been run on a constant input.** E005,
E006, E007, E008, E009, and H2d's diagnosis all concern a channel that cannot vary at
the flock size they were run at. The unit tests never caught it because
`tests/test_plasticity.py` uses `n_hens=4` — the one band where the channel still works.

### Defect 2 — the learned readout has one degree of freedom

`hen/plasticity.py:269-271` updates `W_out` by an outer product of two non-negative,
slowly-varying traces. Every consolidation adds a scalar multiple of nearly the same
matrix. Measured over 200 s, 16 hens:

| quantity | measured | reviewer | meaning |
|---|---|---|---|
| top-1 singular value share of `ΔW_out` | **0.9981** (min 0.9903) | 0.9998 | 1.0 = exactly rank one |
| cortical drive `sd / |mean|` over time | **0.007** | 0.021 | 0 = a pure constant |
| pairwise cosine across hens | **0.405** | 0.828 | reviewer overstated; hens do differ |

The cortical contribution to the motor drive varies by **0.7% of its own magnitude**
across three seconds of behaviour. `W_out` — the connection from the thinking part of
the brain out to the muscles — is not learning a policy. It is learning a **constant
offset**, and the offset is negative on the channels that matter:

```
peck      +0.02 -> -0.52     flee      -0.01 -> -0.54
c_food    -0.02 -> -0.49     c_ground  +0.02 -> -0.43
```

She learns to peck less. Mechanically, that is the harm E013–E016 spent four
experiments characterising.

### Defect 3 — the reward is 98% call-cost

`hen/plasticity.py:171-174` folds the vigour change into `d_drive` alongside the
homeostatic drives. Measured over 900 steps, 16 hens, at hatch:

| component | mean | sd | share of reward variance |
|---|---|---|---|
| hunger | −0.0333 | 0.0000 | 0.0% |
| thirst | −0.0250 | 0.0000 | 0.0% |
| cold | −0.0126 | 0.0229 | 1.9% |
| **vigour** | **−0.0851** | **0.1634** | **98.1%** |
| strike | 0.0000 | 0.0000 | 0.0% |

Higher than the reviewer's 92.9%. In this window hunger and thirst have **zero
variance** — they are pure drift, and not one feeding event occurred in 14,400
hen-steps. The teaching signal is almost entirely "did you just call".

This is [E012](E012-call-cost-confound.md) happening a second time. E012 correctly found
`call_energy_cost` was destroying H2's *metric* by being charged to hunger. The fix moved
it into `vigour` — out of the metric, and into the **teaching signal**, where nobody
looked. E013's pre-run check verified the thing that had been fixed (hunger is untouched
by calling) and not the thing that had been moved.

## 5. Interpretation

**What this licenses.** All three defects are confirmed in this repo by independent
measurement. Defect 1 is arithmetic and not open to interpretation: a full-amplitude
alarm call moves the receiver's channel by exactly zero at the default flock size.

**What it does not license.** The reviewer's further claim — that E013, E014, E015 and
E016 all reduce to defect 2, with no interaction and no "last word" effect — is
plausible and unverified. It is a *reinterpretation* of four experiments and needs its
own runs before anything in the tree is rewritten on the strength of it. Two of the
reviewer's own numbers came in noticeably off when re-measured here (cosine 0.828 vs
0.405), which is a reason to check the rest rather than adopt it.

**The pattern worth naming**, which is the reviewer's most useful observation and is
not about any single defect: **all three are quantities that were verified in the place
they had just been moved *from*.** The call cost was checked in the metric after being
moved to the reward. The gain is documented as 0.9 in `hypothesis.md:392` and is 0.70 in
`connectome.py:48`. The "readout can only apply a constant bias" diagnosis was made at
gain 0.9 in E013, and never re-measured after the re-baselining — where it is still
true. Relocating a term and re-checking its old home is now a known failure mode here.

## 6. Consequence

- **E018 aborted mid-run.** With `audio ≈ 1.0` constantly, `SCAFFOLD_WEIGHT` is a
  permanent +1.5 on crouch and −1.5 on peck — a posture change, not a call response.
  Its pre-registered falsifier would have fired for the wrong reason and promoted the
  wrong hypothesis. See E018 §6.
- **H2's stated mechanism is withdrawn.** `hypothesis.md` says the cortical pathway
  transmits "structured **state-dependent** noise". Measured `sd/|mean| = 0.007`: it is
  state-*in*dependent.
- **E013's "clean test" claim is caveated**, and with it `H2 REFUTED at this timescale`.
  The run had 98% of its teaching signal coming from a term added for a different
  hypothesis.
- **H2d is demoted from the critical path** pending re-measurement. Its diagnosis rests
  on a hawk-vs-call contrast that never occurs in the coop, where the call channel is
  constant at 1.0 and the aerial channel averages 0.00.
- **Three backlog items promoted above everything else**, in order: fix call emission
  and audibility; give `W_out` more than one degree of freedom; take the vigour term out
  of `reward()`.
