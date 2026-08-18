# E057 — separating the audience-specific effect from general elevation

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**H2f** — the learning rule is the wrong *kind*. `UNDER TEST`. Direct continuation of
[E056](E056-hebbian-readout-scaled.md), which found a significant audience effect
(t=45.59) under the bounded Hebbian readout that mixed a genuine, disproportionate
audience-specific rise with a smaller general-elevation rise, and could not cleanly
separate the two from a 3-seed spot check. This is that separation, done properly, on
the full sample, with a matched design rather than an eyeballed comparison.

## 2. Question

E056's diagnostic (3 seeds) suggested `alarm_alone` rose ~30–40% from baseline while
`alarm_audience` rose by a larger amount — but this was read off 3 seeds, without a
control for what the *scaffold alone* (`S`, no learning) already does to the
alone/audience gap, and without a significance test on either component. **Does the
audience-specific component of the learned effect — measured as a proper
difference-in-differences against the `S` baseline's own alone/audience gap — remain
significantly positive once the general-elevation component is accounted for, and is
the general-elevation component itself distinguishable from zero?**

## 3. Prediction

Based on the 3-seed spot check (not a proper sample, registered honestly as a weak
prior): the audience-specific component (~+0.27 to +0.35 per seed) should be larger
than and separable from the general-elevation component (~+0.09 to +0.13). If this
holds on the full 8-seed sample with both tested for significance, that is the cleanest
reading yet in this project's history that a non-instrumental rule can build a targeted,
audience-conditional contingency — closer to (though not identical with) clearing H2f's
falsifier as originally specified.

## 4. Falsifier

If the general-elevation component (`L_alone − S_alone`) is not significantly smaller
than the audience-specific component (`(L_audience − L_alone) − (S_audience − S_alone)`),
or if the audience-specific component itself is not significant once properly tested,
the mixed reading from E056 stands: this rule has not been shown to build a targeted
contingency, only a general one with some audience correlation riding on top.

## 5. Design

**Instrument**: `run/audience.py`'s `assay()`, called directly (not through `_run_cell`,
which only returns the scalar `.alarm_effect`) to capture the full `AudienceResult`
(`alarm_alone`, `alarm_audience`, `food_alone`, `food_audience`) for every seed.

**Conditions**: identical to E056 — `S` (scaffold, fixed) and `S+L-hebbian-scaled`
(scaffold, `hebbian_readout=True`, `readout_scaling_strength=0.3`), same seeds (0–7),
same world (16 hens, `food_deplete_rate=0`), same 30-minute rearing. Rerunning rather
than reusing E056's cache because that cache only stored the scalar effect, not the
components this test needs — determinism guarantees an identical reared connectome for
the same seed and config, so this is a re-derivation, not a fresh sample.

**Primary metrics** (both paired across the 8 seeds, per-seed):
- **General elevation**: `alarm_alone(L) − alarm_alone(S)`.
- **Audience-specific effect** (difference-in-differences): `(alarm_audience(L) −
  alarm_alone(L)) − (alarm_audience(S) − alarm_alone(S))`.

Both tested against zero (one-sample paired t, `run/experiment.py`'s `_t_critical`
table), and against each other (is the audience-specific component larger in
magnitude).

**Secondary, exploratory**: the same breakdown for `food_alone`/`food_audience` — the
scaffold and this task give food calling no reason to show an audience effect at all
(unlike alarm calling, which the scaffold's own crouch-response gives a mechanistic
route to), so a *food*-channel audience effect of similar size to the alarm one would
be additional evidence for general elevation rather than anything targeted.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e057_separate_effects.py --seeds 8 --minutes 30
```

## 6. Result

8 seeds (0–7), 30 min rearing, matching E056 exactly.

```
     alarm_alone alarm_aud food_alone food_aud
   S      0.3197    0.3852     0.5005   0.5013
   L      0.4424    0.7402     0.4891   0.4798

General elevation: alarm_alone(L) - alarm_alone(S)     +0.1227 +/- 0.0139  t=8.81  SIGNIFICANT
Audience-specific (diff-in-diff)                       +0.2324 +/- 0.0051  t=45.59  SIGNIFICANT
targeted - general                                     +0.1097 +/- 0.0106  t=10.39 SIGNIFICANT

food_alone(L) - food_alone(S)                          -0.0114 +/- 0.0327  t=0.35  not significant
food audience-specific (diff-in-diff)                  -0.0101 +/- 0.0333  t=0.30  not significant
```

**Given the scale of this claim — the first non-null positive result on a targeted
learned behaviour in this project's entire history — this was replicated on a fresh,
non-overlapping 8-seed block (8–15) before being written up**, per this project's own
standing rule for surprising positives.

```
     alarm_alone alarm_aud food_alone food_aud
   S      0.3208    0.3849     0.5007   0.5007
   L      0.4413    0.7372     0.4866   0.4715

General elevation                                      +0.1205 +/- 0.0106  t=11.37  SIGNIFICANT
Audience-specific (diff-in-diff)                       +0.2317 +/- 0.0106  t=21.90  SIGNIFICANT
targeted - general                                     +0.1112 +/- 0.0024  t=45.88  SIGNIFICANT

food channel, both components                          not significant (both blocks)
```

**Both blocks agree almost to three decimal places** (general: 0.1227 vs 0.1205;
targeted: 0.2324 vs 0.2317) — a level of seed-to-seed consistency this project has not
seen before on any result, positive or null. Pooled across all 16 seeds: general
elevation **+0.1216 ± 0.0085, t=14.38**; audience-specific **+0.2321 ± 0.0057, t=40.90**;
targeted significantly exceeds general in both blocks individually (t=10.39, t=45.88)
and pooled. The food-channel control — a call type this task gives no mechanistic route
to an audience effect — shows neither component in either block, ruling out
indiscriminate motor dysregulation as the explanation for either number.

## 7. Interpretation

**Per §4's falsifier, applied literally: it does not fire.** Both required conditions
hold, replicated: the audience-specific component is significantly larger than the
general-elevation component, and the audience-specific component is itself significant.
This is the cleanest positive result this project's learning-rule series has produced.

**What it does and does not establish, precisely.** The bounded Hebbian readout builds
a real, reproducible, disproportionately audience-conditional change in alarm calling
that the reward-modulated instrumental rule (E036, on the identical task and scaffold)
did not. It is not a *pure* audience effect — a smaller but equally real and replicated
general-elevation component sits alongside it. The food-channel control is the key piece
that makes the general component interpretable rather than just another confound: it is
specific to the alarm-calling context (both conditions of which involve an actual hawk
present, only audience toggling), not a blanket increase in vocal output. The most
economical reading is that the rule learned two things that both plausibly rest on the
same underlying association — hearing/seeing a real threat more strongly potentiating
alarm calling in general (the co-occurrence any hawk encounter provides, regardless of
audience), *and*, on top of that, a further, larger increment specifically when a
flockmate is detectably present. Both are consistent with a covariance-based rule
picking up real, available correlations in the traces — which is what it was built to
do — rather than with the kind of unbounded runaway E055 showed.

**This is evidence for H2f, not merely "not evidence against it."** A non-reward-gated,
correlation-based update rule succeeded, on a task and scaffold where the standard
reward-modulated rule was tested and found null (E036, replicated E040). That is close
to H2f's falsifier as written, with one honest caveat: the falsifier's implicit picture
was a *clean* audience-only effect, and what was found is a *predominantly* audience-weighted
effect with a smaller, real, non-audience-specific component riding along. Whether that
caveat is disqualifying is a matter of how strictly "closer to Pavlovian association
succeeds" is read — this file reports the full, precise breakdown so that judgment is
not made by this file alone.

## 8. Consequence

- **H2f moves to `SUPPORTED — narrower than the falsifier's clean ideal`.** A
  non-instrumental rule produces a real, replicated, predominantly targeted effect
  where the instrumental rule was null. The scope is stated precisely: audience-specific
  component (+0.232, pooled, t=40.90) significantly exceeds a real general-elevation
  component (+0.122, pooled, t=14.38) that is itself specific to the alarm-calling
  context, not indiscriminate. Matches this project's own precedent for this kind of
  qualified positive (T1: "SUPPORTED as a narrower claim").
- **This reopens H2c and the wider comprehension question** — not directly tested here
  (this experiment is about calling, a motor policy, not the perceptual/comprehension
  question H2c is about), but the demonstration that *some* non-reward-gated rule can
  build a targeted contingency on this architecture is the first positive proof of
  concept for the class of rule H2c's own mechanism (`W_pred`) already uses. Worth a
  fresh, direct pass at H2c under the same scrutiny (mandatory diagnostics, replication
  before trusting) this experiment applied.
- **`hebbian_readout` + `readout_scaling_strength=0.3` is now a validated, working
  configuration**, not just a debugging tool — worth documenting as such in
  `PlasticConfig` and considering as a candidate default for future audience/social
  learning tasks, distinct from the general-purpose instrumental rule.
- **The remaining general-elevation component is worth its own explanation**, not just
  its own number — a structural read of what specifically potentiates in `W_out` when a
  real hawk is present (audience or not) would say whether it is a sensible "threat
  salience" association or something less interpretable. Filed to `docs/backlog.md`,
  not required before this result stands.
