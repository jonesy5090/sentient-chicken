# E058 — H2c under the rule that worked for H2f

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**H2c** — a learned cue can recruit an innate response via top-down association.
`NOT STARTED` as a working mechanism (E042–E044, E054: every precondition anyone could
supply to the `W_pred` pathway — density, exposure, structural capacity, reduced
competing-channel noise — left comprehension "nowhere near a working mechanism").
[H2f](experiments/E057-separating-audience-from-elevation.md) has just demonstrated,
for the first time in this project, that a non-reward-gated, bounded readout rule
(`hebbian_readout` + `readout_scaling_strength`) can build a real, replicated, targeted
behavioural contingency on this architecture. H2c has never been tested with a rule
that has actually been shown to work here — every prior H2c attempt used either the
reward-modulated instrumental rule (which H2f found null on a comparable task) or the
unbounded `W_pred` pathway alone (which is architecturally restricted to *perception*,
not motor output, and separately blocked by H2d's representational bottleneck).

## 2. Question

Comprehension — crouching to a heard alarm call with no predator visible — is normally
measured through `W_pred`'s perceptual route (augmenting the observation the fixed
reflex arc reads) or is absent when that route is disabled. **Does the readout pathway
itself, trained with the same non-reward-gated rule that built H2f's audience effect,
learn to crouch on hearing a call directly** — a motor association bypassing `W_pred`
and its H2d bottleneck entirely, measured with `pred_gain=0.0` so no perceptual route
contributes at all?

## 3. Prediction

**No confident directional prediction — registered as genuinely uncertain**, but for a
specific, statable reason unlike prior H2c registrations: unlike the audience-effect
task (H2f), which had a wired-in scaffold providing an innate anchor for learning to
amplify, this test uses **no scaffold** — comprehension must be built from nothing.
Whether the correlation exists for the rule to find is itself an open question: a hen
who is head-up when a hawk arrives both sees it (crouching via the strong innate
reflex) and, if flockmates are calling, hears the alarm at the same time — a real,
available co-occurrence in the world dynamics, not a wired one. Whether 8-16 seeds
of rearing supply enough such moments for a Hebbian rule to find that correlation is
unknown.

## 4. Falsifier

If comprehension (measured with `pred_gain=0.0`, so purely through the readout) is not
significantly positive relative to a fixed control, H2c remains unbuilt via this route
too — consistent with, not contradicting, H2f's result, since H2f's task had a scaffold
this one deliberately does not. **A positive result requires the same scrutiny H2f's
did**: a control across other, unrelated motor channels (peck, scratch, flee) under the
identical synthetic call-vs-silence comparison. If those *also* show a "comprehension"-shaped
effect, the result is general excitability, not a targeted crouch association, exactly
the pattern E055 (pre-fix) showed and E057's food-channel control ruled out for H2f.

## 5. Design

**World**: 16 hens, `food_deplete_rate=0`, `hawk_period_s=20` (matching E042/E043's
precedent for this hypothesis line — H2c's association needs real hawk-call
co-occurrence events during rearing, which the audience-effect task's 900s default
would supply too rarely to be a fair test).

**Scaffold**: off (`auditory_scaffold=False`) — the cleanest test of whether the rule
builds the association from nothing, not whether it amplifies a wired-in one.

**Conditions**: `FIXED` (`enabled=False`, no learning — required control) vs.
`LEARN_HEBBIAN` (`enabled=True, growth_enabled=False, explore_sigma=0.6,
hebbian_readout=True, readout_scaling_strength=0.3` — the exact configuration E056/E057
validated, minus the scaffold).

**Instrument**: `run/audience.py`'s `comprehension()` logic, reimplemented locally to
return the full motor breakdown (not just the crouch scalar): for each of `M_CROUCH`,
`M_PECK`, `M_SCRATCH`, `M_FLEE`, the mean rate under a held call-present observation
minus the mean rate under a held call-absent observation, `pred_gain=0.0` throughout so
only the readout pathway can contribute.

**Primary metric**: crouch comprehension, `LEARN_HEBBIAN − FIXED`, paired t-test,
`run/experiment.py`'s `_t_critical` table.

**Mandatory diagnostic, not optional**, per E055's lesson: the same comparison on
`M_PECK`, `M_SCRATCH`, `M_FLEE` under the identical synthetic test. A positive crouch
result is only reported as a targeted association if these three do not show a
comparable rise.

**Replicates**: 8 seeds first pass, matching this project's standing default. **Any
significant positive result gets a fresh, non-overlapping second block before being
written up as a finding**, per the standing rule this session applied to E057.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e058_h2c_hebbian.py --seeds 8 --minutes 20
```

## 6. Result

8 seeds, 20 min rearing, 16 hens, hawk every 20s, no scaffold, `pred_gain=0.0`.

```
        crouch      peck   scratch      flee
FIXED   -0.0000   -0.0000   -0.0000   +0.0000
LEARN   +0.0036   +0.0044   +0.0048   +0.0035

crouch     LEARN-FIXED: +0.0036 +/- 0.0014  t=2.58  SIGNIFICANT  <-- PRIMARY
peck       LEARN-FIXED: +0.0044 +/- 0.0012  t=3.58  SIGNIFICANT
scratch    LEARN-FIXED: +0.0048 +/- 0.0015  t=3.29  SIGNIFICANT
flee       LEARN-FIXED: +0.0035 +/- 0.0013  t=2.74  SIGNIFICANT
```

**The falsifier fires.** Crouch is nominally significant on its own (t=2.58) — but so
are all three control channels, at closely matched magnitudes (0.0035–0.0048 across all
four) and comparable or larger t-statistics. There is no crouch-specific signal here:
whatever the rule built, it built at essentially the same small size on every motor
channel tested, including three with no mechanistic connection to hearing an alarm
call. This is precisely the general-excitability pattern the pre-registered diagnostic
was designed to catch, and precisely the pattern E057's food-channel control ruled out
for H2f — the two results look superficially similar (both "significant") and are
mechanistically opposite.

**Scale, for context.** These effects (~0.004) are roughly two orders of magnitude
smaller than H2f's audience effect (~0.23) and the auditory scaffold's own wired-in
comprehension (~0.19, E036/E040) — consistent with E042–E044's characterisation of
this pathway as "nowhere near a working mechanism," not a new, larger signal.

Not replicated on a second seed block: this is a null with an unambiguous, built-in
control ruling out the one confound that mattered, not a surprising positive — this
project's standing practice reserves replication for the latter (E057), not the former
(E054 and others).

## 7. Interpretation

**Building comprehension from nothing, via the readout pathway alone, does not work —
and the reason this time is legible rather than another dysregulation artefact.** H2f's
task had a wired-in scaffold providing an innate anchor for the rule to *amplify*; this
task had none, asking the rule to build the entire association from naturally-occurring
correlation in rearing data. The uniform, tiny rise across every channel looks like
ordinary connectome drift under 20 minutes of a non-zero learning rate — not nothing,
but not a targeted association either.

**This narrows what H2f's result generalises to, honestly.** It is not "this rule kind
solves associative learning on this architecture" — it is "this rule kind can amplify
an existing, wired-in anchor into a targeted, disproportionate policy." Building an
anchor from scratch is evidently a different, harder problem, unsolved by this attempt.
Whether that is because the co-occurrence signal genuinely is too sparse at this
rearing duration and hawk rate, or because H2d's representational bottleneck still
applies to whatever the readout would need to condition on even via this route, was not
distinguished here.

## 8. Consequence

- **H2c stays `NOT STARTED` as a working mechanism.** This closes the specific version
  of the backlog item E057 opened ("a fresh, direct pass at H2c... using this now-validated
  non-instrumental rule family") with a clean negative, not a further-open question.
- **H2f's result is not weakened by this null** — the two tasks differ in exactly the
  way that matters (wired anchor to amplify vs. nothing to build from), and this
  experiment's own control channels demonstrate the diagnostic discipline that made
  H2f's positive trustworthy is doing its job here too, the other direction.
- **New, more precise open question for `docs/backlog.md`**: does comprehension emerge
  via this rule with a *longer* rearing duration or *higher* hawk rate (more
  co-occurrence events for the correlation to accumulate on), or does H2d's bottleneck
  block it regardless of exposure — the same escalation E043 already ran for the
  `W_pred` pathway, not yet tried for the readout pathway.
