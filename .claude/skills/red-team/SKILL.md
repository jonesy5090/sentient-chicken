---
name: red-team
description: Commission an adversarial outside review of the project from an agent with no prior context, then verify its findings before acting on any of them. Use when work feels stuck in a loop, when successive experiments keep generating explanations for the same null, when a result is surprising enough to be suspicious, before committing to an expensive run, or when someone asks for a second opinion, sanity check, critique, or "are we going in circles". Also invoked as /red-team.
---

# Red team

Commission a critique from an agent that has none of this conversation's context, then
**verify everything it says before acting on any of it.**

The reason this exists: on 2026-08-13 an outside review found three defects that had
survived eighteen experiments. Every communication experiment in the project had been
running on an auditory channel that carried no information — a full-amplitude alarm call
from an adjacent hen moved the receiver's input by exactly 0.0000. Nothing had caught it
because the test suite ran at `n_hens=4`, the one flock size where the channel still
worked, and because the people looking had spent eighteen experiments building
increasingly sophisticated explanations for the resulting nulls.

**Accumulated context is what makes a project productive and what makes it blind.** This
skill buys a reader who has neither.

## The one rule

**Verify before you act. The reviewer will be wrong about some of it.**

In the founding case the review returned five findings. Three replicated exactly and two
of those were *worse* than reported. But one of its numbers was off by a factor of two on
re-measurement, and one of its central inferences was simply wrong — it argued that a
rank-one weight update meant the readout could only apply a constant offset, when a
rank-one `ΔW = u vᵀ` contributes `u (v · x)`, which varies perfectly well with `x`. A
guard test written on the reviewer's framing would have failed a *working* rule.

So: re-measure each finding independently, in this repo, without reusing the reviewer's
scripts. Adopt what replicates. Record what does not, and say so. An outside reviewer
with no context is unusually good at spotting what you have stopped seeing and unusually
bad at knowing which of the things they spot actually matter here.

Never rewrite the tree on an unverified reinterpretation. In the founding case the
reviewer further claimed four experiments all reduced to a single defect — plausible,
never tested, and recorded as *not adopted* rather than acted on.

## Commissioning the review

Spawn **one** agent. Not several — the value is one reader forming an independent view,
and parallel agents re-derive the same context at multiplied cost.

**Withhold your conclusions.** This is the part that is tempting to skip and the part
that does the work. Do not tell it your current hypothesis, your suspected mechanism, or
what you expect it to find. Point it at the repo and let it form its own view. If you
seed it, you get your own reasoning back with a second byline.

**Do tell it:**

- Where to start — `docs/hypothesis.md`, `docs/backlog.md`, `CLAUDE.md`, `README.md`,
  then `docs/experiments/`, then the actual code.
- That the docs are the authors' *account* of the code, and checking whether that account
  is accurate is part of the job.
- That this project's stated culture is that overturned claims are its most valuable
  output, so it should not soften.
- Resource constraints — what is currently running, and not to start long jobs. Short
  numerical checks are encouraged; verifying a claim beats speculating about one.
- To rank findings by how much they would change what the project does next.
- To return a written review, cite `file:line`, modify nothing.

**The five standing questions.** Ask all of them; the last is the one that pays.

1. **Claims the code does not support.** Anywhere the docs assert a mechanism the source
   does not implement, or a measurement whose method would not produce that number.
2. **Confounds and invalid inferences.** Do the conditions being compared truly differ in
   one thing? Check the matched-seed logic and the assay staging specifically.
3. **Can the central design answer its own question?** Not whether it runs — whether the
   comparison distinguishes the hypothesis from the alternatives.
4. **Statistical practice.** Sample sizes, the t machinery, multiple comparisons across
   the whole experiment series, whether pre-registration is real or ceremonial.
5. **The thing nobody has thought of.** Name the persistent null explicitly and ask
   whether a simpler explanation is being missed, or whether something structural — the
   environment, the reward, the timescale, the neuron model, the observation encoding —
   makes the whole line of work moot.

## After it reports

In this order.

1. **Triage by consequence**, not by how alarming it sounds. Which findings would change
   the next action?
2. **Stop work that a finding invalidates, immediately.** In the founding case an
   experiment was killed ~40 minutes into an 8-seed run. Its pre-registered falsifier
   would have fired for the wrong reason and promoted the wrong hypothesis. A run whose
   instrument is broken does not become useful by finishing, and having the number in
   hand only creates temptation.
3. **Re-measure independently.** Own script, this repo, from scratch. Report *your*
   numbers and note where they differ from the reviewer's.
4. **Record it as an experiment file**, honestly labelled a diagnostic rather than
   pre-registered, following `docs/experiments/TEMPLATE.md`. Include what you did not
   adopt and why.
5. **Push every consequence back into `docs/hypothesis.md`** — withdraw mechanisms, strike
   claims, mark superseded experiments. Strike through rather than delete; the route is
   the record.
6. **Then fix.** Each fix needs a guard test, and the guard must run at the configuration
   where the defect appeared. The founding defects were invisible at the suite's default
   flock size, which is exactly why they lasted eighteen experiments.

## Watch for the failure mode the review will probably find

The founding review's most useful observation was not any single defect but the pattern
behind all three: **each was a quantity verified in the place it had just been moved
*from*.** A call cost checked in the metric after being moved into the reward. A gain
documented at one value and running at another. A diagnosis made under one setting and
never re-checked after re-baselining, where it happened to still be true.

When a term is relocated, measure it in its new home. Look for this pattern first.

## Length

The review itself should be as long as it needs to be. **Your report of it to the user
should be roughly a page**, and should lead with what changes rather than with what was
found.

## When the answer is unflattering

That is the expected case and the reason for calling this. If the review finds the
project has been measuring a constant for eighteen experiments, say so plainly, in the
first paragraph, with the number. Do not open with what survived.

Two things to keep in proportion. Do not over-correct into treating the reviewer as an
oracle — see the one rule. And do not bury a genuinely sound practice that the review
confirms: in the founding case the control design caught a pathway its own author had not
predicted, and that was worth stating alongside the defects, briefly, after them.
