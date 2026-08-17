# E027 — third outside review, verified: H4 does not need the brain

> **Diagnostic, not pre-registered.** Commissioned via the `red-team` skill and written
> after the fact, following `TEMPLATE.md`'s sections for comparability. The verification
> runs in §6 were specified before they were executed and use seeds **24–31**, which no
> prior experiment has touched — so nothing here is re-read off the data that generated
> the observation.

## 1. Parent hypothesis

**H4** — an intact channel beats an uninformative one on a task requiring private
information. Marked `SUPPORTED` by [E026](E026-h4-supported.md). This experiment
re-examines that status and downgrades it.

## 2. Question

E026 promoted the project's first result past H1a. A reader with none of the
conversation's context was pointed at the repository and asked, among other things,
whether the central design can answer its own question. It returned ten findings.

Per the skill's one rule, every finding is re-measured here before anything is acted
on — the founding review was wrong about two things, one of which would have produced a
guard test that failed a *working* learning rule.

## 3. What was checked, and by what method

Eight findings verified independently. Six needed no simulation and were confirmed by
reading the code or the project's own committed logs. Two required runs and are in §6.

## 4. Confirmed without a run

| finding | reviewer's number | measured here |
|---|---|---|
| reward is `n_struck`-dominated at the H4 configuration | 87.3% | **87.3%** |
| the guard test's window contains no strike at all | — | **0 of 3000 steps** |
| Dale's law violated on `W_out` | 0 of 48 columns comply | **0 of 48**, all mixed |
| headline inflated by mean-of-ratios vs pooled counts | 15.0 pp | **−0.150 pooled vs −0.198 reported** |
| the metric's denominator moves with the treatment | −5% / +23% | **400→380, 213→263** |
| `_t_critical(11)` disagrees with the prose | 2.228 vs 2.201 | **2.228** |

**The reward finding is the one that changes what happens next.** Same freeze-one-field
method as the guard test itself (`scratchpad/rt_reward.py`):

```
guard config (hawk every 900 s):  hunger 43.1%  cold 56.9%  n_struck  0.0%   passes
H4 config    (hawk every  20 s):  hunger  5.4%  cold  7.3%  n_struck 87.3%   would fail
```

`tests/test_plasticity.py::test_reward_is_not_dominated_by_one_component` asserts that
no component exceeds 80% of the reward variance. At the configuration H4 actually ran,
`n_struck` carries 87.3% and the assertion fails. It passes today only because a hawk
never arrives during its 30-second window at `hawk_period_s=900`.

**This is the E019 pattern exactly** — a guard blind because it runs at the one
configuration where the defect cannot appear, which is how the dead audio channel
survived eighteen experiments at `n_hens=4`. `CLAUDE.md` states the rule that this test
violates, and the test was written to enforce it.

The underlying cause is live and was flagged as owed by E022 §6, never verified:
`n_struck` increments **every step** a hen is in contact (`coop/world.py:272,316`), so a
single catch during a 12 s dive delivers ~1000 units against per-step drive terms of
0.1–0.5. E014 removed the `/dt` and left the per-step accumulation.

## 5. Process failures confirmed by reading

- **The head-raise mechanism was written down three times and measured none.** E018 §8
  pre-registers the exact ablation — *"scaffold with the crouch term but no peck/scratch
  suppression … separates 'she hides because she was told' from 'she looks up and sees
  it herself'"*. `hen/innate.py:146-163` predicts it in a source comment. And
  `scratchpad/verify_yoked.py`'s docstring advertises it as check B — the file is 73
  lines and stops after check A. That file is what E026 cites as its independent check.
- **The registered headline contrast is never computed.** `docs/backlog.md:31` says
  the headline is L vs C?. `scratchpad/blind_risk.py:72` bases every contrast on deaf.
  It is recoverable from the shared pairing — **−0.204** (block A) and **−0.260** (block
  B), both larger than the reported figure — and was not reported.
- **T1's registered metric is null and was not recorded as such.** `docs/backlog.md:96`
  specifies food intake at matched risk. From E026's own logs, fed %: block A deaf 3.06
  vs intact 3.07; block B deaf 2.54 vs intact **2.41**. E026 reports a different metric
  and does not say the registered one failed.
- **Two rungs of the backlog §1 ladder were dropped** — `Cs` self-only, which is the
  only thing distinguishing communication from private memory, and `C−` capacity. With
  plasticity off in every condition, `C−` is provably vacuous, which is worth stating:
  **the capacity control the entire ladder was designed around cannot do anything in an
  all-innate flock**, so H0's "at any capacity" clause is untested.

## 6. The two claims that needed runs

Seeds 24–31, 8 seeds × 5 min, 16 hens, 1.5× pallium, hawk every 20 s, no plasticity.
`scratchpad/rt_verify2.py`, checkpointed per run after an earlier attempt was killed by
container reclamation and took every paired statistic with it.

```
condition                   blind risks   caught   pooled rate
deaf                                 57       40        0.702
intact                               60       22        0.367
intact, head-raise only              72       42        0.583
intact, crouch only                  93       63        0.677
deaf,   W_out=0                      79       37        0.468
intact, W_out=0                      77       25        0.325
```

| contrast | paired mean-of-ratios | pooled |
|---|---|---|
| intact − deaf, pallium intact | −0.240 ± 0.110 (t=2.19) | −0.335 |
| **intact − deaf, `W_out` lesioned** | **−0.208 ± 0.148 (t=1.41)** | **−0.144** |
| head-raise only − deaf | −0.124 ± 0.121 (t=1.03) | −0.118 |
| crouch response only − deaf | −0.067 ± 0.134 (t=0.50) | −0.024 |

**Nothing in this block reaches significance at n=8.** The magnitudes are what carry the
argument, and they should be read as suggestive.

### Claim 1 — the effect does not need the pallium. **Replicates.**

With `W_out` set to exactly zero — a complete lesion of the only route by which 512
simulated neurons can reach a muscle — the benefit is still there. On the mean-of-ratios
estimator E026 used for its headline it barely moves (−0.240 → −0.208). On pooled counts
it roughly halves (−0.335 → −0.144). The two estimators disagree about the magnitude;
they agree the effect survives.

**The reviewer's qualitative claim holds and its quantitative claim does not.** It
reported the lesioned effect at ~80% of the intact one; that is true on one estimator
and false on the other, and neither is significant here.

A detail the review did not report and which matters: **the lesion moves the deaf
baseline from 0.702 to 0.468.** Removing an untrained random cortical projection makes
deaf hens substantially better at surviving. The pallium is not merely uninvolved — at
this configuration it is actively harmful, which is E002's ceiling finding reappearing.

### Claim 2 — the head-raise carries it. **Direction replicates; magnitude does not.**

Splitting the scaffold, the half that suppresses pecking (she looks up, and her own
visual reflex fires at weight 8.0) outperforms the half that makes her crouch on hearing
a call, −0.124 against −0.067. The reviewer reported the crouch half at essentially
zero; **that is too strong** — here it carries roughly half what the head-raise carries.
The two are approximately additive (−0.191 against the full scaffold's −0.240).

This is what the project's own arithmetic predicted: `sigmoid(1.5 − 2.5) = 0.269`
against a 0.5 hiding threshold.

### An unrequested finding, and it is the worst one

**Denominator movement is far larger than either the review or E026 suspected.**

```
deaf                        57 blind risks     +0.0%
intact                      60                 +5.3%
intact, head-raise only     72                +26.3%
intact, crouch only         93                +63.2%
deaf, W_out=0               79                +38.6%
```

E026 §4 justifies the metric on the grounds that "the denominator is fixed the instant
the hawk commits, so the treatment cannot move it". The *within-dive* denominator is
fixed. The **number of dives that find a hen at risk and blind is a behavioural
outcome**, and it varies by up to 63% across conditions. Crouching zeroes locomotion, so
a hen who crouches lingers where the next dive will find her — which is the exact
confound E026 identified in `caught_rate` and believed it had escaped.

## 7. Interpretation

**H4 as stated is not supported.** H0 asserts that *a neural model of a chicken* can be
given a channel that changes what the flock can do. What E026 measured survives deleting
the neural model. The causal chain is two hand-set weights in `hen/innate.py` and a
threshold in `coop/world.py`, and it can be computed on paper.

What is left is real and worth keeping: **a contingent channel beats a non-contingent
one.** The yoked control still does its job (E026's correlation measurement stands, and
this run does not bear on it). But the honest claim is narrower than the tree's framing
— *a well-timed one-bit interrupt restores the receiver's own vision*, in a reflex agent.

**What was right, and should not be lost.** The control design is sound; the yoked
channel is the thing E024 lacked and E026 correctly built. The pairing is sound. The
replication rule was honoured in form. And the review's most useful finding was not any
single defect but that **the reward guard, the metric's premise, and the scaffold
ablation were each verified in the wrong place** — the same meta-pattern E019 named and
`CLAUDE.md` records.

## 8. Consequence

- **H4 moves from `SUPPORTED` to `UNDER TEST — the effect does not require the
  pallium`.** Not refuted: the channel effect is real. The claim about a *neural model*
  is withdrawn.
- **The reward guard is now the top blocker.** The stated next step is to switch
  plasticity on in the H4 world; doing so today hands the rule a modulator that is 87%
  strike events. Fix `n_struck`'s per-step accumulation and re-point the guard at
  `hawk_period_s=20` **before** anything else runs.
- **The metric needs an intent-to-treat denominator** — every (hen, dive) pair, or
  something anchored to a quantity the treatment provably cannot reach.
- **Dale's law on `W_out`** is violated at initialisation and under learning, was found
  by E022, marked "verified — adopt", and dropped from the action list. Reinstated.
- **Not adopted:** the reviewer's claim that the lesioned effect is ~80% of the intact
  one (estimator-dependent), and that the crouch response carries "essentially none"
  (it carries about half the head-raise's share here). Recorded rather than acted on.
- **Owed:** the whole of §5 — the registered L vs C? contrast, T1's null on its
  registered metric, and the two dropped rungs.
