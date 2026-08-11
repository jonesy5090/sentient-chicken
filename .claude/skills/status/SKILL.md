---
name: status
description: Answer the three standing project questions — what we are trying to achieve, what has been built and what has actually been proven, and what is next. Use whenever someone asks for a project status, briefing, recap, catch-up, "where are we", "what have we done", "what's left", or invokes /status. Also use before starting significant new work, to check it ladders up to the hypothesis tree.
---

# Project status

Answer three questions, in this order, every time. The value is in the consistency —
someone should be able to ask this at any point and get a comparable answer.

## Sources of truth

Read these before answering. Do not answer from memory or from the conversation:

| file | gives you |
|---|---|
| `docs/hypothesis.md` | the tree, statuses, evidence, falsifiers — **start here** |
| `docs/experiments/*.md` | what was actually run and what it returned |
| `docs/backlog.md` | what is proposed, ranked, with reasoning |
| `README.md` | current phase, measured envelope |
| `docs/ethics.md` | tripwire state, last review |

If they disagree, `docs/hypothesis.md` wins and the discrepancy is worth reporting —
a drifted README is itself a finding.

## The one rule

**Built is not proven.** This project's characteristic failure is conflating "the code
runs" with "the hypothesis is supported". Phase 1 plasticity restructured thousands of
synapses for three experiments before any of it changed behaviour — E001 was a null
with a busy connectome. Those are different facts and a status report that blurs them
is worse than none.

Do not use a worked example from this file as evidence of current state; it will go
stale. Read the tree.

So: every claim in section 2 is tagged as either **built** (code exists, tests pass)
or **proven** (an experiment with a pre-registered prediction returned a result).
Never let a `SUPPORTED` status stand on anything other than a recorded experiment.

## Output format

### 1. What are we trying to achieve

- State **H0** from the hypothesis tree in one or two sentences, in plain language.
- Then the route: the sub-hypotheses H1…Hn as a chain, with each one's status. Make
  the dependency visible — which ones block which.
- Name the headline experiment and its decisive comparison, so the reader knows what
  would actually settle it.

### 2. What have we done so far

Two separate lists. Do not merge them.

**Built** — components that exist and are tested. One line each, with the module.
Include the measured performance envelope, since it constrains everything.

**Proven** — hypotheses with `SUPPORTED` status, each with the number that supports
it and a link to the experiment. If a hypothesis is `UNDER TEST` with a null recorded,
say so explicitly here rather than leaving it out; nulls are results.

Give the test count and assay count as a health line, not as evidence of progress.

### 3. What's next

- **Immediate blocker**, if any — what stops the next hypothesis being testable.
  Usually there is exactly one and it should be first.
- **Backlog**, in the order `docs/backlog.md` ranks it, with the reasoning for the
  order preserved. Do not re-rank silently.
- **Behaviours not yet in the model** — spatial memory, social hierarchy, the
  language channel, and anything else the tree needs but the code lacks.
- **Open items from experiments** — the follow-ups recorded in section 8 of each
  experiment file.

## Length

Roughly a page. This is a briefing, not an audit. If a section runs long, the tree has
grown branches nobody is working on and that is worth saying rather than enumerating.

## When the answer is unflattering

Report it plainly. A status that always sounds like progress is not tracking
anything. If the last experiment was a null, if a hypothesis has been open for weeks,
if the backlog is growing faster than it is being cleared — those are the useful
parts of the answer.
