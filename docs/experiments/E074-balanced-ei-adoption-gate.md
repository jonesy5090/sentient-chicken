# E074 — `balanced_ei` adoption gate

> **A gate, not a pre-registered experiment.** [E073](E073-naturalistic-separability-probe.md)
> §8 named three checks that must clear before `balanced_ei` can become a default.
> Gates 1 and 2 were run before this file was written and are reported as run; gate 3's
> design is stated below before its result. Labelled honestly rather than back-dated
> into pre-registration format.

## Why a gate at all

E072/E073 established that balanced E/I gives **2.13× separability** under naturalistic
input (t=5.75, paired 12 genomes) and brings live pallial rate from **0.7288 to 0.1209**
— out of saturation for the first time since E009 identified it. That is a
representation result. It is not on its own a reason to change the connectome every
experiment in the project runs on.

Three things could each independently disqualify it, and all three are cheap relative to
the cost of adopting a bad default:

## Gate 1 — does innate behaviour survive? **PASS**

Full ethogram re-run on an E/I-balanced connectome (`python -m run.probes --balanced-ei`,
a flag added for this purpose so the gate is re-runnable rather than a one-off script).

**13/13 assays pass, with near-identical numbers.** Expected, and now verified rather
than assumed: the reflex arc is a fixed matrix applied to `reflex_in`, entirely separate
from `W`, so balancing `W` cannot touch it. What does shift slightly is the cortical
pathway's small contribution — e.g. approach-flockmates left-bias +0.13 → +0.11, closing
distance 0.196 → 0.181 m; head-up aerial 0.65 → 0.66. All well inside each assay's
margin.

## Gate 2 — throughput? **PASS**

Throughput is a correctness constraint in this project, not a nicety.

| | real-time factor |
|---|---|
| baseline | 38.6× |
| `balanced_ei` | 42.4× |

Above the suite's 5× guard either way, and marginally *faster* balanced — plausibly
fewer saturated units, though the difference is small enough to be noise. No cost.

## Gate 3 — does the one positive learning result survive?

**The check that matters.** H2f's audience-sensitive calling (E057) is this project's
only genuine, replicated positive learning result. If balancing breaks it, the
intervention is disqualified regardless of what it does for representation.

**Run internally, not against E057's recorded numbers.** The codebase has moved a long
way since: `OBS_DIM` 74 → 138 across three new sensory blocks, and E067's `m_acc` fix
changed the learning rule itself. E057's figures describe a different program. Both arms
are therefore run here, same code, same seeds, paired: S (fixed) vs L (H2f's rule),
8 seeds, 30 minutes, with `balanced_ei` off and on.

**Reported**: general elevation, the audience-specific difference-in-differences (E057's
headline), and the food-channel control that should stay null in both arms.

**Falsifier**: if the audience-specific effect is significant with `balanced_ei=False`
and not significant with it on, balancing costs the project its only positive learning
result and is not adopted.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e074_gate3_h2f.py
```

### Result

_Running._

## Consequence

_Pending gate 3._
