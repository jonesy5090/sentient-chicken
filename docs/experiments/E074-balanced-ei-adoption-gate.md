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

8 seeds, 30 min, paired, threshold t(7)=2.365.

| | general elevation | **audience-specific** | food control *(must be null)* |
|---|---|---|---|
| `balanced_ei=False` | +0.1500 ± 0.0219, t=6.85 | **+0.2242 ± 0.0053, t=42.46** | **+0.1054 ± 0.0105, t=10.04** |
| `balanced_ei=True` | +0.0038 ± 0.0006, t=6.69 | **+0.0023 ± 0.0002, t=13.28** | +0.0002 ± 0.0002, t=1.10 *(null)* |

**The stated falsifier does not fire.** It required the audience-specific effect to be
significant with balancing off and *not* significant with it on. It is significant in
both (t=42.46 → t=13.28). `balanced_ei` does not cost the project its positive learning
result.

**But the food control fires in the baseline arm, and that is a bigger problem than the
gate was looking for.** E057's design uses the food channel precisely because the task
gives it no mechanistic audience route; E057 reported it null, and that null is what
made the audience-specific effect trustworthy rather than indiscriminate elevation. On
the current codebase it comes back at **+0.1054, t=10.04** — 47% the size of the
audience effect it is supposed to be a control for.

## Consequence

**Gate 3 passes as specified; `balanced_ei` is still not adopted, for a different
reason than the gate anticipated.**

**Reading the two arms.** Balancing shrinks every effect by roughly two orders of
magnitude (audience 0.2242 → 0.0023), which follows directly from live pallial rate
dropping 0.7288 → 0.1209: `W_out` reads pallial rates, so a smaller input produces
smaller learned changes in behaviour. Two readings compete and this experiment does not
separate them:

- **Proportional shrinkage.** Everything scales down together, the control included, and
  the control's apparent "fix" is just it falling below detectability.
- **Cleanup.** The baseline's control effect is a spurious common-mode-driven component
  inflating *both* channels, and balancing removes it.

The signal-to-control ratio favours the second reading: **2.1× baseline (0.2242 vs
0.1054) against 11.5× balanced (0.0023 vs 0.0002)**. The audience effect is far better
separated from its own control after balancing. That is suggestive, not decisive — a
proportional-shrinkage account with a floor effect could produce the same pattern, and
distinguishing them needs a designed test rather than an inference from two numbers.

**Against that: 0.0023 may be behaviourally meaningless.** E057's 0.232 was a call
amplitude; 0.0023 is a hundredth of it. A statistically bulletproof effect (t=13.28)
that never changes what a hen audibly does is not obviously worth having, and nothing
here measures behaviour — only the assay's readout of `W_out`.

**The blocking finding is about H2f, not about `balanced_ei`.** E057's food control was
null; on the current codebase it is t=10.04. Something between E057 and now changed it,
and the candidates are all recent: `OBS_DIM` 74 → 138 across three new sensory blocks;
`N_CALLS` 4 → 5 shifting every audio index; and **E067's `m_acc` fix, which changed the
reward-gated pathway itself**. That last one is mine, and it is the most likely
candidate on mechanism — it altered what `W` learns from, and `W` feeds the pallial
states `W_out` reads.

**This must be resolved before `balanced_ei` is adopted or rejected**, because the
baseline arm of this very gate is the contaminated one. Adopting a connectome change on
the strength of a comparison whose reference arm has a firing control would be exactly
the "quantity verified in the place it was moved from" pattern `CLAUDE.md` catalogues.

**Next, in order:**
1. **Bisect H2f's food control.** Re-run E057's contrast with `m_acc` reverted, then
   with the pre-E060 observation layout, to identify what broke the control. Cheap: the
   same 8×30 min contrast, two extra arms.
2. Only then revisit `balanced_ei` adoption, against an uncontaminated baseline.

**H2f's status should carry a caveat immediately**, ahead of that bisect: its headline
result was validated under a control that no longer holds on current code.
