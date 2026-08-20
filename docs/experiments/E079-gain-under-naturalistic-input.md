# E079 — the recurrent gain default was set on the sparse probe too

> **Pre-registered.** Sections 1–5 written and committed before the run.

## 1. Parent hypothesis

**H2d**. [E078](E078-density-under-naturalistic-input.md) left saturation as the only
live lead, and the falsifier fired: no intervention has a positive effect on H2d under
naturalistic input.

## 2. Question

`hen/connectome.py`'s `gain` default is **0.95**, set by [E023](E023-ei-fix-and-rebaseline.md)'s
sweep:

| gain | mean pallial rate | separability |
|---|---|---|
| 0.70 | 0.189 | 4.5% |
| **0.95 (default)** | **0.276** | **7.4%** |
| 1.00 | 0.320 | 9.4% (peak) |

**That sweep ran on the sparse E009-series probe**, where the network sits at ~0.27 and
is not saturated at all. Under naturalistic input the same connectome sits at 0.4602,
and live operation at **0.6907** (E073/E076). E023 chose a gain by optimising in a
regime the hen is never in — the same error E078 just found in E041's density
recommendation, and E073 found in E023's own saturation claim.

Where does separability peak under naturalistic input, and is 0.95 anywhere near it?

## 3. Prediction

**Gains above 0.95 should hurt.** E078 established that naturalistic separability falls
as drive rises past ~0.46 (0.0814 at rate 0.4602 → 0.0581 at 0.6745). Raising gain
raises drive. This part I hold with reasonable confidence.

**Below 0.95 I genuinely do not know**, and say so rather than pick the flattering
guess. Two mechanisms pull opposite ways: E023's sparse sweep found separability falling
monotonically with gain (0.70 → 4.5%), suggesting lower is worse; but E078's naturalistic
data found the *lowest*-drive condition also worse than the middle (rate 0.3786 → 0.0698
vs 0.4602 → 0.0814). If both hold, there is an interior optimum and 0.95 may already be
close to it — in which case this experiment reproduces E078's density outcome, and
validates a second default rather than moving it.

**The outcome that would matter**: a gain that both improves naturalistic separability
*and* pulls live rate out of saturation. That would be the first positive H2d
intervention in the tree.

## 4. Falsifier

If naturalistic separability peaks at or near 0.95, the gain default is validated and
saturation cannot be addressed by gain alone — leaving H2d with no lead at all, which
should be stated as plainly as E078's falsifier was.

## 5. Design

Both probes, same 12 genomes, paired — the E073/E078 design, so probe-to-probe
comparison is itself paired. On E076's corrected baseline.

- **Gains**: 0.40, 0.60, 0.80, 0.95 (default), 1.10.
- **Metric**: `pallial_sep` verbatim from E041, keeping the E009-series comparable.
- **Also reported**: mean pallial rate per cell, since gain acts *through* drive and the
  whole question is where drive should sit.
- **Then, for the default and the naturalistic best**: **live** mean pallial rate from an
  actual rollout (16 hens, 5 min, 3 seeds), not a settle. Separability improvements that
  leave the network saturated in live operation are not worth having, and the settle
  probes cannot see that.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e079_gain_naturalistic.py
```

## 6. Result

12 genomes, paired, threshold t=2.201.

| probe | gain | separability | settle rate |
|---|---|---|---|
| sparse | 0.40 | 0.0279 | 0.1482 |
| sparse | 0.60 | 0.0439 | 0.1719 |
| sparse | 0.80 | 0.0645 | 0.2114 |
| sparse | **0.95** | **0.0961** | 0.2724 |
| sparse | 1.10 | 0.0811 | 0.5739 |
| naturalistic | 0.40 | 0.0284 | 0.1611 |
| naturalistic | 0.60 | 0.0441 | 0.1994 |
| naturalistic | 0.80 | 0.0648 | 0.2762 |
| naturalistic | **0.95** | **0.0814** | 0.4602 |
| naturalistic | 1.10 | 0.0220 | 0.7752 |

Paired against the 0.95 default — **every** other gain is worse on the naturalistic
probe:

| probe | gain | t | | ratio |
|---|---|---|---|---|
| naturalistic | 0.40 | 5.24 | significant | 0.35× |
| naturalistic | 0.60 | 4.16 | significant | 0.54× |
| naturalistic | 0.80 | 2.52 | significant | 0.80× |
| naturalistic | 1.10 | 6.54 | significant | **0.27×** |

**Live rollout** (16 hens, 5 min, 3 seeds):

| gain | live mean pallial rate |
|---|---|
| 0.40 | **0.1796** |
| 0.95 (default) | 0.6861 |

## 7. Interpretation

**The falsifier fires. Naturalistic separability peaks exactly at the shipped default**,
with significant decline in *both* directions — the second default this session has
validated rather than moved (after E078's density). The decline above is far sharper
naturalistically than sparsely (1.10 gives 0.27× vs sparse's 0.84×, null), which is the
saturation effect E078 identified, now confirmed on a second axis.

**But the live-rate row reframes the whole problem, and this matters more than the
validated default.** Gain 0.40 pulls live pallial rate from 0.6861 to **0.1796** —
comprehensively out of saturation — and separability gets **worse**, 0.35×. That is now
the second independent demonstration of the same thing:

| intervention | live rate | naturalistic separability |
|---|---|---|
| `balanced_ei` (E077) | 0.73 → 0.12 | 1.05×, null |
| gain 0.40 (here) | 0.69 → 0.18 | **0.35×, significantly worse** |

**Reducing drive does not improve separability. Twice, by unrelated mechanisms.** And
raising it hurts too (density 1.00 → 0.71×, gain 1.10 → 0.27×). Separability peaks at the
operating point the model already occupies and falls away in every direction tested.

**That means E009's saturation framing mis-identifies the constraint — and so did my own
E073 write-up, which leaned on it.** "The network is saturated, therefore differences get
squashed" predicts that de-saturating helps. It does not. Saturation is a *correlate* of
the regime where separability is best, not the cause of its being low.

## 8. Consequence

**H2d has no remaining lead, and that should be stated as plainly as E078's falsifier
was.** Every structural intervention tried is null or negative, all paired, all on this
metric:

| intervention | verdict |
|---|---|
| E/I identity fix (E023) | null |
| modality segregation (E035) | null |
| sensory→pallium density (E041/E078) | reversed naturalistically |
| balanced E/I (E072/E077) | null |
| recurrent gain (E023/E079) | **default already optimal** |
| removing recurrence (E017/E034) | slightly worse |

**The hypothesis worth forming next is that H2d is not a defect at all.** Two stimuli
differing in one or two channels out of 138, pushed through a random projection into a
recurrent pool, produce pallial states differing by ~7–8% of mean rate. Nothing tried
changes that, and the operating point is already optimal. That is a plausible *property*
of random projection without learned feature extraction — not a bug with a fix waiting.

If so, the circularity is the real finding: the pallium is supposed to learn features
that separate what matters, but cannot, because separating what matters is the
precondition for having anything to learn from.

**One concrete, cheap test of that reframing before accepting it.** `OBS_DIM` has grown
59 → 138 across this project's life, and every addition dilutes the informative fraction
of a two-channel distinction. If H2d is a random-projection property, **separability
should have degraded monotonically as channels were added** — which is checkable by
measuring it against progressively masked observations, and would convert a plausible
story into a measurement. It also carries an uncomfortable implication worth surfacing
now: several of those channels are ones I added this session.

**Also corrected here**: `docs/hypothesis.md` claimed `hen/connectome.py:48` has
`gain = 0.70`. The actual default is **0.95**, at line 81. That note was itself a
correction of a stale doc, and had gone stale in both the value and the line number —
the precise failure mode it was written to fix.
