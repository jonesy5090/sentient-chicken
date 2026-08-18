# E056 — a bounded Hebbian readout, re-testing H2f's falsifier cleanly

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**H2f** — the learning rule is the wrong *kind*. `UNDER TEST`.
[E055](E055-hebbian-readout.md) attempted the rule H2f's falsifier calls for (a
non-reward-gated readout) but found the result confounded: removing the reward gate
also removed the reward-prediction-error's incidental zero-mean stabilising effect, and
`W_out` — unlike `W` — has no synaptic-scaling correction to replace it. The measured
"significant" audience effect traced to unbounded cortical drive (2–2.7× reflex
magnitude) elevating every calling channel regardless of condition, not a targeted
policy. This experiment adds the missing stabiliser (`readout_scaling_strength`,
implemented and unit-tested in the same session) and re-runs the identical task.

## 2. Question

With `W_out`'s growth bounded the same way `W`'s already is, does the non-reward-gated
readout rule produce a genuine, targeted audience-conditional calling effect — as
opposed to the general dysregulation E055 found?

## 3. Prediction

**No confident directional prediction — registered as genuinely uncertain**, for the
same reasons §3 of E055 gave: a shorter credit-assignment chain than reward-driven
learning needs, but no guarantee H2d's representational bottleneck is any less binding
just because the readout is now stable. **What would distinguish a genuine result from
E055's artifact**, checked directly this time before trusting any significant number:
`alarm_alone` should stay near its pre-rearing/scaffold-only level while `alarm_audience`
moves, not both rising together; hunger should stay near the ~0.30–0.40 range every
other condition in this tree lands in, not spike toward 0.7; `|cortical|` should sit
well under `|reflex|` magnitude, not 2–3× over it.

## 4. Falsifier

If `S+L-hebbian-scaled − S` is not significantly positive, **or** it is significant but
fails any of the three sanity checks in §3 (alone/audience both rising, hunger spiking,
cortical overwhelming reflex), the falsifier does not clear — H2f's specific claim (a
Pavlovian-style rule succeeding where instrumental failed) remains untested-positive on
this task, and the representational/credit-assignment explanations already open (H2d,
H2c) become the more likely account.

## 5. Design

**The fix**: `readout_scaling_strength=0.3` (the value `test_readout_scaling_bounds_hebbian_growth`
verified actually bounds growth in a stress test), added to the same `hebbian_readout`
`PlasticConfig` E055 used. Nothing else changes — same task, same scaffold, same world,
same seeds.

**Task**: identical to E055 — `run/audience.py`'s `_run_cell`, `S` (scaffold, fixed) vs.
`S+L-hebbian-scaled` (scaffold, hebbian readout, readout scaling on). 16 hens,
`food_deplete_rate=0`, default `hawk_period_s`, 30 min rearing, 8 seeds.

**Primary metric**: `alarm_effect`, paired `S+L-hebbian-scaled − S`, identical
definition to E036/E040/E055.

**Mandatory diagnostic before any positive result is reported**: the full
`alarm_alone`/`alarm_audience`/`food_alone`/`food_audience`/`|reflex|`/`|cortical|`
breakdown (`scratchpad/e055b_diagnose_hebbian_result.py`'s method, re-pointed at this
condition), run regardless of whether the primary contrast clears significance. This is
not optional or exploratory — E055 showed the primary number alone is not sufficient
evidence.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e056_hebbian_readout_scaled.py --seeds 8 --minutes 30
```

## 6. Result

8 seeds, 30 min rearing, 16 hens, `food_deplete_rate=0`.

```
condition                                           audience  compreh.  strikes/hen   hunger  synapses
S   (scaffold, fixed)                                 +0.065    0.1922       192.95    0.390     36319
S+L-hebbian-scaled (scaffold, bounded hebbian)        +0.298    0.3017       219.45    0.546     35755

PRIMARY -- audience effect, S+L-hebbian-scaled - S:
  +0.2324 +/- 0.0051  t=45.59  threshold(df=7)=2.365  -> SIGNIFICANT
```

**t=45.59 is far larger than anything else in this project's history** (H4's strongest
pooled result was t=3.60) and, per the pre-registered discipline, that alone demanded
the mandatory diagnostic before trusting it — an effect this size and this consistent
across seeds (SE=0.0051) is more often instrumentation than biology.

**Diagnostic, this condition** (`scratchpad/e056b_diagnose_scaled_result.py`, 3 seeds):

```
 seed  alarm alone  alarm aud.  food alone food aud.  |reflex| |cortical|
    0       0.4218      0.6939      0.5013    0.4968    2.0551     1.6293
    1       0.4463      0.7658      0.3728    0.2468    2.2143     1.7019
    2       0.4082      0.6864      0.6149    0.7388    1.8518     1.5949
```

**True baseline, measured fresh for this comparison** (S condition, same seeds, same
method — not previously cached at this granularity):

```
 seed  alone   audience  |reflex|  |cortical|
    0  0.3199  0.3666    1.8521    0.0503
    1  0.3321  0.4151    1.6968    0.0626
    2  0.3210  0.3808    1.7469    0.0843
```

**Two of three pre-registered sanity checks read as genuine improvement over E055; the
third does not clear as specified.** `|cortical|` (1.59–1.70) now sits *under*
`|reflex|` (1.85–2.21) — the scaling fix worked, this is no longer the "cortical
overwhelms reflex" regime. Hunger (0.546) is elevated above the 0.30–0.40 range every
other condition in this tree lands in, but nowhere near E055's 0.728. But `alarm_alone`
did **not** stay flat: it rose from a baseline of ~0.32 to ~0.42 (+0.09 to +0.13, a
30–40% relative increase), while `alarm_audience` rose from ~0.37–0.42 to ~0.69–0.77
(+0.27 to +0.35) — a real, disproportionately larger rise in the audience condition, but
not the "alone flat, audience moves" pattern §3 specified as distinguishing a genuine
result from a general-elevation artifact.

## 7. Interpretation

**Per the falsifier as literally written in §4, it does not clear.** One of the three
named sanity checks (`alarm_alone` staying near baseline) fails, even though the other
two pass and the effect's disproportionate weighting toward the audience condition is
real and measured, not asserted. Applying the pre-registered criterion honestly rather
than reading it generously: this experiment has not produced a clean confirmation of
H2f's falsifier.

**What it has produced is a genuine step forward from E055, worth stating precisely.**
The bounded rule is materially better-behaved (cortical no longer overwhelms reflex,
hunger cost roughly halved) and shows a real audience-weighted component on top of a
smaller general-elevation one — qualitatively different from E055, where the two were
indistinguishable. The honest reading is a mixture: some of this rule's learning is
genuinely audience-correlated, some is still general excitability that a fully clean
rule would not have. Disentangling the two — e.g. by measuring the effect as a
*within-seed ratio* to its own alone-condition rate, or by finding what in the pallium's
representation the readout is actually keying on — was not attempted here and is the
natural next diagnostic, not a rerun with a different constant.

**This is the second experiment in a row where a naive first implementation of an idea
turned out to be measuring an implementation defect, and the corrected version turned
out to be genuinely informative but still not clean on the first attempt.** That
pattern — E055 clearly broken, E056 partially working — is itself evidence that getting
a non-instrumental rule to behave like a targeted policy rather than a generalized drive
is a real, nontrivial design problem, separate from whether Pavlovian-style learning
*can* work here at all.

## 8. Consequence

- **H2f stays `UNDER TEST`.** Not confirmed (the falsifier as specified did not clear)
  and not refuted (unlike E055, this was not simply broken — a real, disproportionate
  audience-weighted signal exists). The clean test H2f's node calls for has still not
  been run to a decisive conclusion.
- **`readout_scaling_strength` is validated as a real fix for the E055 defect**
  (cortical/reflex ratio, hunger cost both improved substantially) and stays in the
  codebase, off by default, for future non-reward-gated readout experiments.
- **Next diagnostic identified, not yet run**: separate the audience-specific
  component from the general-elevation component directly (e.g. `audience effect /
  alone rate` per seed, or a structural read of what the trained `W_out` actually
  correlates with — the same kind of analysis E044 did for `W_pred`). This is a
  targeted follow-up, not a rerun of this experiment with adjusted constants.
- **Filed as a new backlog item**: whether the general-elevation component reflects a
  ceiling on how "clean" any Hebbian-style rule can be on this architecture without an
  explicit decorrelation or normalization mechanism, versus being fixable with further
  tuning of `readout_scaling_strength` or the trace time constants specifically.
